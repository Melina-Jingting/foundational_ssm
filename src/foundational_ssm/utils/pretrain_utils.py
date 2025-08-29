import time


import numpy as np

# JAX & Equinox
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import equinox as eqx
from omegaconf import OmegaConf
import optax

import wandb

from foundational_ssm.constants import DATASET_IDX_TO_GROUP_SHORT
from foundational_ssm.metrics import compute_r2_standard
from foundational_ssm.models import SSMFoundationalDecoder
from .wandb_utils_jax import load_checkpoint_wandb
from .training_utils import log_batch_metrics, track_batch_timing, create_optimizer_and_state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def mse_loss_foundational(model, state, inputs, targets, mask, dataset_group_idxs, key, dataset_group_weights=None, skip_timesteps=0):
    """MSE loss for foundational model (takes dataset_group_idx and mask)"""
    batch_keys = jr.split(key, inputs.shape[0])
    preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0, 0), out_axes=(0, None))(inputs, state, dataset_group_idxs, batch_keys)
    
    # Only evaluate loss on timesteps > skip_timesteps
    preds = preds[:, skip_timesteps:, :]  # Shape: (batch, seq_len - skip_timesteps, output_dim)
    targets = targets[:, skip_timesteps:, :]  # Shape: (batch, seq_len - skip_timesteps, output_dim)
    mask = mask[:, skip_timesteps:]  # Shape: (batch, seq_len - skip_timesteps)

    # Only compute loss on unmasked elements
    squared_error = (preds - targets) ** 2
    mask = mask[..., None]
    masked_squared_error = jnp.where(mask, squared_error, 0.0)
    
    # dataset_group_weights = dataset_group_weights[..., None, None]  # shape (batch, 1, 1) to broadcast
    weighted_squared_error = squared_error #* dataset_group_weights
        
    masked_squared_error = jnp.where(mask, weighted_squared_error, 0.0)
    mse = masked_squared_error.sum() / mask.sum()
    return (mse, state)

@eqx.filter_jit
def make_step_foundational(model, state, inputs, targets, mask, key, dataset_group_idxs, loss_fn, opt, opt_state, skip_timesteps=0):
    """Make step for foundational model (takes dataset_group_idx and mask)"""
    # dataset_group_weights = dataset_group_weights[dataset_group_idxs]
    (value, state), grads = loss_fn(model=model, 
                                    state=state, 
                                    inputs=inputs, 
                                    targets=targets, 
                                    mask=mask, 
                                    dataset_group_idxs=dataset_group_idxs, 
                                    key=key, 
                                    # dataset_group_weights=dataset_group_weights, 
                                    skip_timesteps=skip_timesteps)
    updates, opt_state = opt.update([grads], opt_state, [model])
    model = eqx.apply_updates(model, updates[0])
    return model, state, opt_state, value, grads


def train_one_batch(batch, model, state, loss_fn, opt, opt_state, train_key, lr_scheduler, current_step, skip_timesteps=0):
    batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}
    inputs = batch["neural_input"]
    targets = batch["behavior_input"]
    dataset_group_idxs = batch["dataset_group_idx"]
    mask = batch["mask"]
    
    model, state, opt_state, loss_value, grads = make_step_foundational(
        model, state, inputs, targets, mask, train_key, dataset_group_idxs, loss_fn, opt, opt_state, skip_timesteps=skip_timesteps
    )
    
    current_lr = lr_scheduler(current_step)
    wandb.log({
        "train/loss": loss_value,
        "train/learning_rate": current_lr,
    }, step=current_step)
    return model, state, opt_state, loss_value

def train_one_epoch(train_loader, model, state, loss_fn, opt, opt_state, rng_key, lr_scheduler, current_step, epoch, skip_timesteps=0):    
    epoch_loss = 0
    batch_count = 0
    minute_start_time = time.time()
    prev_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        data_load_time = time.time() - prev_time
        batch_process_start = time.time()
        
        model, state, opt_state, loss_value = train_one_batch(
            batch, model, state, loss_fn, opt, opt_state, rng_key, lr_scheduler, current_step, skip_timesteps=skip_timesteps
        )
        batch_process_end = time.time()
        batch_process_time = batch_process_end - batch_process_start
        
        log_batch_metrics(data_load_time, batch_process_time, epoch, current_step)
        epoch_loss += loss_value
        batch_count += 1
        current_time = time.time()
        batch_count, minute_start_time = track_batch_timing(batch_count, minute_start_time, current_time, current_step)
        prev_time = time.time()
        current_step += 1
    
    wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch}, step=current_step)
    return model, state, opt_state, current_step, epoch_loss
    

def validate_one_epoch(val_loader, model, state, skip_timesteps=0):
    metrics = {}
    all_preds = []
    all_targets = []
    all_dataset_group_idxs = []
    all_session_dates = []  # 1. Add list to store session dates
    val_start_time = time.time()
    
    inference_model = eqx.nn.inference_mode(model)
    
    for batch in val_loader:
        batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}
        dataset_group_idxs = batch["dataset_group_idx"]
        inputs = batch["neural_input"]
        targets = batch["behavior_input"]
        session_dates = batch["session_date"] # Get session dates from batch
        
        mask = batch["mask"][..., None]
        preds, state = jax.vmap(
            inference_model, axis_name="batch", 
            in_axes=(0, None, 0, None), out_axes=(0, None)
        )(inputs, state, dataset_group_idxs, jr.PRNGKey(0))

        preds = preds[:, skip_timesteps:, :]
        targets = targets[:, skip_timesteps:, :]
        mask = mask[:, skip_timesteps:, :]

        all_preds.append(jnp.where(mask, preds, 0))
        all_targets.append(jnp.where(mask, targets, 0))
        all_dataset_group_idxs.append(dataset_group_idxs)
        all_session_dates.append(session_dates) # 2. Append session dates from batch

    # Concatenate all collected data
    all_preds = jnp.concatenate(all_preds, axis=0) 
    all_targets = jnp.concatenate(all_targets, axis=0)
    all_dataset_group_idxs = jnp.concatenate(all_dataset_group_idxs, axis=0)
    all_session_dates = jnp.concatenate(all_session_dates, axis=0) # 3. Concatenate session dates
    
    # --- 4. New Nested Metric Calculation Logic ---
    unique_dataset_group_idxs = jnp.unique(all_dataset_group_idxs)
    for dataset_group_idx in unique_dataset_group_idxs:
        dataset_group_idx = int(dataset_group_idx)
        dataset_group_short_name = DATASET_IDX_TO_GROUP_SHORT[dataset_group_idx]
        
        # Create a mask for the current dataset group
        group_mask = (all_dataset_group_idxs == dataset_group_idx)
        
        # Filter data to only this group
        group_preds = all_preds[group_mask]
        group_targets = all_targets[group_mask]
        group_session_dates = all_session_dates[group_mask]
        
        unique_session_dates = jnp.unique(group_session_dates)
        session_r2_scores = []

        # Inner loop: iterate through each session within the group
        for session_date in unique_session_dates:
            session_date = int(session_date)
            
            # Create a mask for the current session within the filtered group data
            session_mask = (group_session_dates == session_date)
            
            # Get predictions and targets for this specific session
            session_preds = group_preds[session_mask]
            session_targets = group_targets[session_mask]
            
            # Compute and store the R2 score for the session
            r2_score = compute_r2_standard(session_preds, session_targets)
            metrics[f"val/r2/{dataset_group_short_name}/{session_date}"] = float(r2_score)
            session_r2_scores.append(float(r2_score))

        # After iterating through all sessions, compute mean and std for the group
        if session_r2_scores: # Avoid division by zero if a group has no sessions
            metrics[f"val/r2/{dataset_group_short_name}/mean"] = np.mean(session_r2_scores)
            metrics[f"val/r2/{dataset_group_short_name}/std"] = np.std(session_r2_scores)
            
    # --- End of New Logic ---

    # Keep the overall R2 calculations
    r2_score_all = compute_r2_standard(all_preds, all_targets)
    metrics['val/r2/avg'] = float(np.mean([v for k, v in metrics.items() if "/mean" in k]))
    metrics['val/r2/all'] = float(r2_score_all)

    # Log validation timing
    metrics['val/time'] = time.time() - val_start_time
    return metrics


def load_training_state(cfg, model_cls=SSMFoundationalDecoder, wandb_resume_run_id=None):    
    start_epoch = 0
    current_step = 0
    best_r2_score = 0.0
    checkpoint_metadata = {}
    
    if wandb_resume_run_id is not None:
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, id=wandb_resume_run_id, resume="allow")
        wandb_run_name = wandb.run.name
        cfg = OmegaConf.create(dict(wandb.run.config))
    else: 
        dataset_name = cfg.dataset_args.config.split("/")[-1].split(".")[0]
        model_name = cfg.model_cfg.split("/")[-1].split(".")[0]
        wandb_run_name = f"{model_name}_{dataset_name}{getattr(cfg.wandb, 'run_name_postfix', '')}"
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(project=cfg.wandb.project, name=wandb_run_name, config=dict(config_dict)) 

    model_cfg = OmegaConf.load(cfg.model_cfg)
    wandb.config.update(OmegaConf.to_container(model_cfg))
    
    model, state = eqx.nn.make_with_state(model_cls)(
            **model_cfg.model
        )
    opt, opt_state, lr_scheduler = create_optimizer_and_state(model, model_cfg.optimizer, model_cfg.model)
        
    if wandb_resume_run_id is not None:
        artifact_full_name = f"{cfg.wandb.entity}/{cfg.wandb.project}/{wandb_run_name}_checkpoint:latest"
        model, state, opt_state, checkpoint_metadata = load_checkpoint_wandb(
            model_template=model,
            state_template=state,
            opt_state_template=opt_state,
            artifact_full_name=artifact_full_name
        )
        start_epoch = checkpoint_metadata.get('epoch', 0)
        current_step = checkpoint_metadata.get('step', checkpoint_metadata.get('current_step', 0))
        best_r2_score = checkpoint_metadata.get('best_r2_score', 0.0)
        
        
    return cfg, model, state, opt, opt_state, start_epoch, lr_scheduler, current_step, best_r2_score