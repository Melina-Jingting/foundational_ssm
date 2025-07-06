import os
import sys
import warnings
import logging
from collections import defaultdict


# Typing
from typing import List, Dict

# Hydra & config
import hydra
from omegaconf import OmegaConf, DictConfig

# JAX & Equinox
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import wandb
import jax.profiler
import numpy as np

# Foundational SSM core imports
from foundational_ssm.data_utils import get_brainset_train_val_loaders, get_dataset_config
from foundational_ssm.models import SSMFoundational
from foundational_ssm.utils import save_model_wandb
from foundational_ssm.constants import DATASET_IDX_TO_GROUP_SHORT
from foundational_ssm.utils.training import get_filter_spec, create_cosine_annealing_scheduler, mse_loss, make_step, predict_batch
from foundational_ssm.metrics import compute_r2_standard

import warnings
import traceback
import sys

import multiprocessing as mp

import h5py
import torch


WARNING_LOG_FILE = "warnings.log"

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    with open(WARNING_LOG_FILE, "a") as log:
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))



@hydra.main(config_path="../configs", config_name="pretrain", version_base="1.3")
def main(cfg: DictConfig):
    warnings.showwarning = warn_with_traceback
    # mp.set_start_method("spawn", force=True)

    print(OmegaConf.to_yaml(cfg))

    # Load dataset
    train_dataset, train_loader, val_dataset, val_loader = get_brainset_train_val_loaders(
        train_config=get_dataset_config(
            cfg.train_dataset.name,
            subjects=cfg.train_dataset.subjects
        ),
        val_config=get_dataset_config(
            cfg.val_dataset.name,
            subjects=cfg.val_dataset.subjects
        ),
        batch_size=cfg.train_dataset.batch_size,
        num_workers=cfg.training.num_workers
    )
    
    key = jr.PRNGKey(cfg.rng_seed)
    model_key, train_key, val_key = jr.split(key, 3)

    model = SSMFoundational(
            **cfg.model
        )
    state = eqx.nn.State(model)
    
    param_leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    total_params = sum(x.size for x in param_leaves)
    print(model)
    print(f"Total parameters: {total_params}")

    filter_spec = get_filter_spec(
        model,
        freeze_ssm=cfg.training.freeze_ssm,
        freeze_mlp=cfg.training.freeze_mlp
    )
    
    # Calculate total training steps for scheduler
    total_steps = len(train_loader) * cfg.training.epochs
    
    # Create scheduler based on config
    use_cosine_scheduler = getattr(cfg.optimizer, 'use_cosine_scheduler', True)  # Default to True for backward compatibility
    
    if use_cosine_scheduler:
        # Create cosine annealing scheduler
        lr_scheduler = create_cosine_annealing_scheduler(
            initial_lr=cfg.optimizer.lr,
            total_steps=total_steps,
            min_lr=getattr(cfg.optimizer, 'min_lr', 0.0),  # Default to 0.0 if not specified
            warmup_steps=getattr(cfg.optimizer, 'warmup_steps', 0)  # Default to 0 if not specified
        )
    else:
        # Use constant learning rate
        lr_scheduler = lambda step: cfg.optimizer.lr
    
    # Load JAX optimizer with scheduler
    opt = optax.chain(
        optax.adamw(learning_rate=lr_scheduler, weight_decay=cfg.optimizer.weight_decay)
    )
    opt_state = opt.init(eqx.filter(model, filter_spec))
    
    # Load JAX loss function
    loss_fn = mse_loss
    
    run_name = f"{cfg.wandb.run_prefix}_sub-{''.join(cfg.train_dataset.subjects)}_l{cfg.model.ssm_num_layers}_d{cfg.model.ssm_dim}"
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project=cfg.wandb.project, name=run_name, config=config_dict)  # type: ignore
    
    # Define metrics with custom x-axis
    wandb.define_metric("epoch", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("epoch_train_loss", step_metric="epoch")
    
    best_r2_score = 0
    save_model_wandb(model, run_name, OmegaConf.to_container(cfg.model), wandb.run)
    
    # Track current step for scheduler
    current_step = 0
    
    jax.profiler.start_trace("/tmp/jax_trace")
    for epoch in range(cfg.training.epochs):
        epoch_loss = 0
        for batch in train_loader:
            batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}
            inputs = batch["neural_input"]
            targets = batch["behavior_input"]
            dataset_group_idx = batch["dataset_group_idx"][0]
            
            key, subkey = jr.split(train_key)
            
            # model, state, opt_state, loss_value, grads = make_step(
            #     model,
            #     state,
            #     filter_spec,
            #     inputs, 
            #     targets, 
            #     dataset_group_idx,
            #     loss_fn,
            #     opt,
            #     opt_state,  
            #     subkey
            # )
            loss_value = 0
            
            # Get current learning rate from scheduler
            current_lr = lr_scheduler(current_step)
            epoch_loss += loss_value
            current_step += 1
            
            wandb.log({
                "train/loss": loss_value,
                "train/learning_rate": current_lr,
            })
            
        wandb.log({"train/epoch_loss": epoch_loss,
                   "epoch": epoch})
        if epoch % cfg.training.log_every == 0:
            total_r2_score = 0
            group_preds = defaultdict(list)
            group_targets = defaultdict(list)
            for batch in val_loader:
                dataset_group_idx = int(batch["dataset_group_idx"][0])
                dataset_group_key = DATASET_IDX_TO_GROUP_SHORT[dataset_group_idx]
                
                batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}
                inputs = batch["neural_input"]
                targets = batch["behavior_input"]
                
                key, subkey = jr.split(val_key)
                batch_keys = jr.split(subkey, inputs.shape[0])
                preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0, None), out_axes=(0, None))(inputs, state, batch_keys, dataset_group_idx)
                group_preds[dataset_group_key].append(preds)
                group_targets[dataset_group_key].append(targets)
                
            for group_key, preds in group_preds.items():
                preds = jnp.concatenate(preds, axis=0)
                targets = jnp.concatenate(group_targets[group_key], axis=0)
                r2_score = compute_r2_standard(preds, targets)
                wandb.log({f"val/r2_{group_key}": r2_score})
                total_r2_score += r2_score
            avg_r2_score = total_r2_score / len(group_preds)
            
            if avg_r2_score > best_r2_score:
                best_r2_score = avg_r2_score
                save_model_wandb(model, run_name, OmegaConf.to_container(cfg.model), wandb.run)
            
            print(f"Epoch {epoch}/{cfg.training.epochs}, Loss: {epoch_loss:.4f}")
    
    jax.profiler.stop_trace()
    wandb.finish()
    
    print(jax.devices())
    print(jax.default_backend())
            
if __name__ == "__main__":
    main()
    


