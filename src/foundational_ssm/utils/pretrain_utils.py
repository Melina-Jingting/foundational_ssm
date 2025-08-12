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
from foundational_ssm.constants.constants import get_dataset_group_weights_array
from foundational_ssm.metrics import compute_r2_standard
from foundational_ssm.models import SSMFoundationalDecoder
from .wandb_utils_jax import load_checkpoint_wandb
from .training_utils import log_batch_metrics, track_batch_timing, create_optimizer_and_state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def mse_loss_foundational(model, state, inputs, targets, mask, dataset_group_idxs, key, dataset_group_weights, skip_timesteps=0):
    """MSE loss for foundational model (takes dataset_group_idx and mask)"""
    batch_keys = jr.split(key, inputs.shape[0])
    preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0, 0), out_axes=(0, None))(inputs, state, dataset_group_idxs, batch_keys)
    
    # Only evaluate loss on timesteps > skip_timesteps
    preds = preds[:, skip_timesteps:, :]  # Shape: (batch, seq_len - skip_timesteps, output_dim)
    targets = targets[:, skip_timesteps:, :]
    mask = mask[:, skip_timesteps:]  # Shape: (batch, seq_len - skip_timesteps)

    # Only compute loss on unmasked elements
    squared_error = (preds - targets) ** 2
    mask = mask[..., None]
    masked_squared_error = jnp.where(mask, squared_error, 0.0)
    
    dataset_group_weights = dataset_group_weights[..., None, None]  # shape (batch, 1, 1) to broadcast
    weighted_squared_error = squared_error * dataset_group_weights
        
    masked_squared_error = jnp.where(mask, weighted_squared_error, 0.0)
    mse = masked_squared_error.sum() / mask.sum()
    return (mse, state)

@eqx.filter_jit
def make_step_foundational(model, state, inputs, targets, mask, key, dataset_group_idxs, loss_fn, opt, opt_state, skip_timesteps=0):
    """Make step for foundational model (takes dataset_group_idx and mask)"""
    dataset_group_weights = get_dataset_group_weights_array()
    dataset_group_weights = dataset_group_weights[dataset_group_idxs]
    (value, state), grads = loss_fn(model=model, 
                                    state=state, 
                                    inputs=inputs, 
                                    targets=targets, 
                                    mask=mask, 
                                    dataset_group_idxs=dataset_group_idxs, 
                                    key=key, 
                                    dataset_group_weights=dataset_group_weights, 
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
    

def validate_one_epoch(val_loader, model, state, epoch, current_step, skip_timesteps=0):
    metrics = {}  # New: store metrics per group
    all_preds = []
    all_targets = []
    all_dataset_group_idxs = []
    val_start_time = time.time()
    prev_time = time.time()
    inference_model = eqx.nn.inference_mode(model)
    for batch_idx, batch in enumerate(val_loader):
        data_load_time = time.time() - prev_time
        batch_process_start = time.time()
        batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}
        dataset_group_idxs = batch["dataset_group_idx"]
        inputs = batch["neural_input"]
        targets = batch["behavior_input"]
        
        mask = batch["mask"]
        mask = mask[..., None]
        preds, state = jax.vmap(inference_model, axis_name="batch", in_axes=(0, None, 0, None), out_axes=(0, None))(inputs, state, dataset_group_idxs, jr.PRNGKey(0))

        preds = preds[:, skip_timesteps:, :]
        targets = targets[:, skip_timesteps:, :]
        mask = mask[:, skip_timesteps:, :]

        all_preds.append(jnp.where(mask, preds, 0))
        all_targets.append(jnp.where(mask, targets, 0))
        all_dataset_group_idxs.append(dataset_group_idxs)
        batch_process_end = time.time()
        batch_process_time = batch_process_end - batch_process_start
        prev_time = time.time()

    all_preds = jnp.concatenate(all_preds, axis=0)
    all_targets = jnp.concatenate(all_targets, axis=0)
    all_dataset_group_idxs = jnp.concatenate(all_dataset_group_idxs, axis=0)
    unique_dataset_group_idxs = jnp.unique(all_dataset_group_idxs)
    for dataset_group_idx in unique_dataset_group_idxs:
        dataset_group_idx = int(dataset_group_idx)
        dataset_group_short_name = DATASET_IDX_TO_GROUP_SHORT[dataset_group_idx]
        dataset_group_mask = all_dataset_group_idxs == dataset_group_idx
        preds = all_preds[dataset_group_mask]
        targets = all_targets[dataset_group_mask]
        r2_score = compute_r2_standard(preds, targets)
        metrics[f"val/r2_{dataset_group_short_name}"] = float(r2_score)
    
    r2_score = compute_r2_standard(all_preds, all_targets)
    metrics['val/r2_avg'] = float(np.mean([metrics[key] for key in metrics.keys() if "r2" in key]))
    metrics['val/r2_all'] = float(r2_score)

    # Log validation timing and resources
    val_end_time = time.time()
    val_time = val_end_time - val_start_time
    metrics['val/time'] = val_time
    metrics['epoch'] = epoch

    wandb.log(metrics, step=current_step)
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
        dataset_name = cfg.dataset_cfg.split("/")[-1].split(".")[0]
        model_name = cfg.model_cfg.split("/")[-1].split(".")[0]
        wandb_run_name = f"{model_name}_{dataset_name}{getattr(cfg.wandb, 'run_name_postfix', '')}"
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(project=cfg.wandb.project, name=wandb_run_name, config=dict(config_dict)) 

    model_cfg = OmegaConf.load(cfg.model_cfg)
    model, state = eqx.nn.make_with_state(model_cls)(
            **model_cfg.model
        )
    opt, opt_state, lr_scheduler = create_optimizer_and_state(model, model_cfg)
        
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