import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import wandb
from foundational_ssm.utils.wandb_utils_jax import load_checkpoint_wandb, save_model_wandb
from foundational_ssm.models.decoders import SSMDownstreamDecoder
from foundational_ssm.utils.training import get_filter_spec, mse_loss, make_step, get_finetune_mode
from foundational_ssm.data_utils.loaders import get_nlb_train_val_loaders
import optax
from foundational_ssm.metrics import compute_r2_standard
import numpy as np


@hydra.main(config_path="../configs", config_name="finetune")
def main(cfg: DictConfig):
    train_dataset, train_loader, val_dataset, val_loader = get_nlb_train_val_loaders(
        task=cfg.train_dataset.task,
        holdout_angles=cfg.train_dataset.holdout_angles,
    )
    key = jr.PRNGKey(cfg.rng_seed)
    train_key, val_key = jr.split(key, 2)
    
    model = SSMDownstreamDecoder(
        **cfg.model
    )
    state = eqx.nn.State(model)
    
    lr_scheduler = lambda step: cfg.optimizer.lr
    filter_spec = get_filter_spec(model, freeze_ssm=cfg.training.freeze_ssm, freeze_mlp=cfg.training.freeze_mlp)
    
    opt = optax.chain(
        optax.adamw(learning_rate=lr_scheduler, weight_decay=cfg.optimizer.weight_decay)
    )
    opt_state = opt.init(eqx.filter(model, filter_spec))

    # Load checkpoint (model, state, opt_state, epoch, step)
    model, state, opt_state, epoch, step = load_checkpoint_wandb(
        path=None,  # path is ignored, wandb artifact is used
        model_template=model,
        state_template=state,
        opt_state_template=opt_state,
        wandb_run_name=cfg.wandb_pretrain_run_name,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
    )
    
    loss_fn = mse_loss

    finetune_mode = get_finetune_mode(cfg.wandb_pretrained_model_id, cfg.training.freeze_ssm)
    run_name = f'{finetune_mode}_holdout-{cfg.train_dataset.holdout_angles}'
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Initialize wandb
    wandb.init(project=cfg.wandb.project, name=run_name, config=config_dict)  # type: ignore
    wandb.define_metric("epoch", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("epoch_train_loss", step_metric="epoch")

    # No need to compute held-in and held-out trial types from the dataset anymore
    best_heldout_r2 = -float('inf')
    for epoch in range(cfg.training.epochs):
        epoch_loss = 0
        train_preds = []
        train_targets = []
        train_held_out = []
        for batch in train_loader:
            inputs = batch["neural_input"]
            targets = batch["behavior_input"]
            held_out_flags = batch["held_out"]
            dataset_group_idx = batch["dataset_group_idx"][0]
            key, subkey = jr.split(train_key)
            model, state, opt_state, loss_value, grads = make_step(
                model,
                state,
                filter_spec,
                inputs,
                targets,
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
            train_held_out.extend(held_out_flags)
        if epoch % cfg.training.log_every == 0:
            wandb.log({"epoch": epoch})
            wandb.log({"train/epoch_loss": epoch_loss})
            train_preds = jnp.concatenate(train_preds, axis=0)
            train_targets = jnp.concatenate(train_targets, axis=0)
            train_held_out = np.array(train_held_out)
            for held_out_value, group_name in [(0, "heldin"), (1, "heldout")]:
                mask = train_held_out == held_out_value
                if np.any(mask):
                    r2 = compute_r2_standard(train_preds[mask], train_targets[mask])
                    wandb.log({f"train/r2_{group_name}": r2})
            # Validation
            val_preds = []
            val_targets = []
            val_held_out = []
            for batch in val_loader:
                inputs = batch["neural_input"]
                targets = batch["behavior_input"]
                held_out_flags = batch["held_out"]
                dataset_group_idx = batch["dataset_group_idx"][0]
                key, subkey = jr.split(val_key)
                batch_keys = jr.split(subkey, inputs.shape[0])
                preds, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0, None), out_axes=(0, None))(inputs, state, batch_keys, dataset_group_idx)
                val_preds.append(preds)
                val_targets.append(targets)
                val_held_out.extend(held_out_flags)
            val_preds = jnp.concatenate(val_preds, axis=0)
            val_targets = jnp.concatenate(val_targets, axis=0)
            val_held_out = np.array(val_held_out)
            avg_r2_score = 0
            n_groups = 0
            heldout_r2 = None
            for held_out_value, group_name in [(0, "heldin"), (1, "heldout")]:
                mask = val_held_out == held_out_value
                if np.any(mask):
                    r2 = compute_r2_standard(val_preds[mask], val_targets[mask])
                    wandb.log({f"val/r2_{group_name}": r2})
                    avg_r2_score += r2
                    n_groups += 1
                    if group_name == "heldout":
                        heldout_r2 = r2
            if n_groups > 0:
                avg_r2_score = avg_r2_score / n_groups
                if heldout_r2 is not None and heldout_r2 > best_heldout_r2:
                    best_heldout_r2 = heldout_r2
                    save_model_wandb(model, run_name, OmegaConf.to_container(cfg.model), wandb.run)
            print(f"Epoch {epoch}/{cfg.training.epochs}, Loss: {epoch_loss:.4f}")
    wandb.finish()
    
if __name__ == "__main__":
    main()