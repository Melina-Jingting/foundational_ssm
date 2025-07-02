from foundational_ssm.data_utils.loaders import get_nlb_train_val_loaders
from foundational_ssm.models import SSMFoundational
from omegaconf import OmegaConf
import jax
import equinox as eqx
import jax.random as jr
import matplotlib.pyplot as plt 
import optax
import jax.numpy as jnp
from jax.tree_util import tree_map
import wandb
import json
import os
from collections import defaultdict
from foundational_ssm.constants import DATASET_IDX_TO_GROUP_SHORT
from foundational_ssm.metrics import compute_r2_standard
from foundational_ssm.utils import save_model_wandb
from foundational_ssm.utils.training import get_filter_spec, mse_loss, make_step, load_model_and_state, get_finetune_mode
from foundational_ssm.constants import MC_MAZE_CONFIG
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import numpy as np


@hydra.main(config_path="../configs", config_name="finetune")
def main(cfg: DictConfig):
    train_dataset, train_loader, val_dataset, val_loader = get_nlb_train_val_loaders(
        task=cfg.train_dataset.task,
        held_in_trial_types=MC_MAZE_CONFIG.CENTER_OUT_HELD_IN_TRIAL_TYPES if cfg.train_dataset.holdout_angles else None,
    )

    model, state = load_model_and_state(cfg.wandb_pretrained_model_id, cfg.model)

    key = jr.PRNGKey(cfg.rng_seed)
    train_key, val_key = jr.split(key, 2)

    filter_spec = get_filter_spec(model, freeze_ssm=cfg.training.freeze_ssm, freeze_mlp=cfg.training.freeze_mlp)

    lr_scheduler = lambda step: cfg.optimizer.lr
    opt = optax.chain(
        optax.adamw(learning_rate=lr_scheduler, weight_decay=cfg.optimizer.weight_decay)
    )
    opt_state = opt.init(eqx.filter(model, filter_spec))
    loss_fn = mse_loss

    finetune_mode = get_finetune_mode(cfg.wandb_pretrained_model_id, cfg.training.freeze_ssm)
    run_name = f'{finetune_mode}_holdout-{cfg.train_dataset.holdout_angles}'
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Initialize wandb
    wandb.init(project=cfg.wandb.project, name=run_name, config=config_dict)  # type: ignore
    wandb.define_metric("epoch", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("epoch_train_loss", step_metric="epoch")

    # Compute held-in and held-out trial types from the dataset
    all_trial_types = set([t for batch in train_loader for t in batch['trial_type']]) | set([t for batch in val_loader for t in batch['trial_type']])
    heldin_types = set(MC_MAZE_CONFIG.CENTER_OUT_HELD_IN_TRIAL_TYPES)
    heldout_types = all_trial_types - heldin_types

    best_r2_score = 0
    for epoch in range(cfg.training.epochs):
        epoch_loss = 0
        train_preds = []
        train_targets = []
        train_trial_types = []
        for batch in train_loader:
            inputs = batch["neural_input"]
            targets = batch["behavior_input"]
            trial_types = batch["trial_type"]
            dataset_group_idx = batch["dataset_group_idx"][0]
            key, subkey = jr.split(train_key)
            model, state, opt_state, loss_value, grads = make_step(
                model,
                state,
                filter_spec,
                inputs,
                targets,
                dataset_group_idx,
                loss_fn,
                opt,
                opt_state,
                subkey)
            epoch_loss += loss_value
            wandb.log({"train/loss": loss_value})
            # Get model predictions for R2
            batch_keys = jr.split(subkey, inputs.shape[0])
            preds, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0, None), out_axes=(0, None))(inputs, state, batch_keys, dataset_group_idx)
            train_preds.append(preds)
            train_targets.append(targets)
            train_trial_types.extend(trial_types)
            
        if epoch % cfg.training.log_every == 0:
            wandb.log({"epoch": epoch})
            wandb.log({"train/epoch_loss": epoch_loss})
            train_preds = jnp.concatenate(train_preds, axis=0)
            train_targets = jnp.concatenate(train_targets, axis=0)
            train_trial_types = np.array(train_trial_types)
            for group, group_name in [(heldin_types, "heldin"), (heldout_types, "heldout")]:
                mask = np.isin(train_trial_types, list(group))
                if np.any(mask):
                    r2 = compute_r2_standard(train_preds[mask], train_targets[mask])
                    wandb.log({f"train/r2_{group_name}": r2})
            # Validation
            val_preds = []
            val_targets = []
            val_trial_types = []
            for batch in val_loader:
                inputs = batch["neural_input"]
                targets = batch["behavior_input"]
                trial_types = batch["trial_type"]
                dataset_group_idx = batch["dataset_group_idx"][0]
                key, subkey = jr.split(val_key)
                batch_keys = jr.split(subkey, inputs.shape[0])
                preds, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0, None), out_axes=(0, None))(inputs, state, batch_keys, dataset_group_idx)
                val_preds.append(preds)
                val_targets.append(targets)
                val_trial_types.extend(trial_types)
            val_preds = jnp.concatenate(val_preds, axis=0)
            val_targets = jnp.concatenate(val_targets, axis=0)
            val_trial_types = np.array(val_trial_types)
            
            avg_r2_score = 0
            n_groups = 0
            for group, group_name in [(heldin_types, "heldin"), (heldout_types, "heldout")]:
                mask = np.isin(val_trial_types, list(group))
                if np.any(mask):
                    r2 = compute_r2_standard(val_preds[mask], val_targets[mask])
                    wandb.log({f"val/r2_{group_name}": r2})
                    avg_r2_score += r2
                    n_groups += 1
            avg_r2_score = avg_r2_score / n_groups
            if avg_r2_score > best_r2_score:
                best_r2_score = avg_r2_score
                save_model_wandb(model, run_name, OmegaConf.to_container(cfg.model), wandb.run)
            
            print(f"Epoch {epoch}/{cfg.training.epochs}, Loss: {epoch_loss:.4f}")
    wandb.finish()
    
if __name__ == "__main__":
    main()