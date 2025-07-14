import os
import sys
import warnings
import logging
from collections import defaultdict
import signal
import atexit


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
from foundational_ssm.data_utils import get_brainset_train_val_loaders_tfds, get_dataset_config
from foundational_ssm.models import SSMFoundationalDecoder
from foundational_ssm.utils import save_model_wandb
from foundational_ssm.constants import DATASET_IDX_TO_GROUP_SHORT
from foundational_ssm.utils.training import get_filter_spec, create_cosine_annealing_scheduler, mse_loss_foundational, make_step_foundational, predict_batch
from foundational_ssm.metrics import compute_r2_standard
from foundational_ssm.utils.wandb_utils_jax import save_checkpoint_wandb, load_checkpoint_wandb
from foundational_ssm.utils.training_utils import (
    log_batch_metrics, track_batch_timing, 
    setup_wandb_metrics, log_epoch_summary, compute_r2_by_groups,
    prepare_batch_for_training, extract_batch_data
)

import warnings
import traceback
import sys

import multiprocessing as mp

import h5py
import torch
import time
import psutil
import tensorflow as tf


WARNING_LOG_FILE = "warnings.log"

# Global variables for signal handling
interrupted = False
current_training_state = None  # Will store (model, state, opt_state, epoch, current_step, run_name)

def signal_handler(signum, frame):
    """Handle interruption signals gracefully"""
    global interrupted
    print(f"[WARNING] Received signal {signum}. Saving checkpoint before exit...")
    interrupted = True

def save_interrupted_checkpoint():
    """Save checkpoint when interrupted"""
    global current_training_state
    if current_training_state is not None:
        model, state, opt_state, epoch, current_step, run_name, metrics = current_training_state
        metadata = {}
        if metrics:
            metadata.update(metrics)
        metadata.update({
            'train_loss': 0.0,  # Will be updated if we have it
            'interrupted': True,
            'interruption_epoch': epoch,
            'interruption_step': current_step
        })
        try:
            save_checkpoint_wandb(model, state, opt_state, epoch, current_step, metadata, run_name)
            print(f"[INFO] Checkpoint saved at epoch {epoch}, step {current_step} due to interruption")
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    with open(WARNING_LOG_FILE, "a") as log:
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

def convert_tf_batch_to_jax(batch):
    """
    Convert TensorFlow batch to JAX-compatible format.
    
    Args:
        batch: Tuple of (neural_input, behavior_input, dataset_group_idx) from TF dataset
        
    Returns:
        Dict with JAX arrays
    """
    neural_input, behavior_input, dataset_group_idx = batch
    
    # Convert TF tensors to numpy arrays, then to JAX arrays
    return {
        "neural_input": jnp.array(neural_input.numpy()),
        "behavior_input": jnp.array(behavior_input.numpy()),
        "dataset_group_idx": jnp.array(dataset_group_idx.numpy()),
    }

def train_one_batch(batch, model, state, filter_spec, loss_fn, opt, opt_state, train_key, lr_scheduler, current_step):
    # Convert TF batch to JAX format
    batch = convert_tf_batch_to_jax(batch)
    batch = prepare_batch_for_training(batch)
    inputs = batch["neural_input"]
    targets = batch["behavior_input"]
    dataset_group_idxs = batch["dataset_group_idx"]
    
    key, subkey = jr.split(train_key)
    model, state, opt_state, loss_value, grads = make_step_foundational(
        model, state, filter_spec, inputs, targets, 
        loss_fn, opt, opt_state, subkey, dataset_group_idxs,
    )
    
    current_lr = lr_scheduler(current_step)
    wandb.log({
        "train/loss": loss_value,
        "train/learning_rate": current_lr,
    }, step=current_step)
    return model, state, opt_state, loss_value

def train_one_epoch(train_loader, model, state, filter_spec, loss_fn, opt, opt_state, train_key, lr_scheduler, current_step, epoch):    
    epoch_loss = 0
    batch_count = 0
    minute_start_time = time.time()
    prev_time = time.time()
    
    # Convert TF dataset to iterator
    train_iter = iter(train_loader)
    
    for batch_idx, batch in enumerate(train_iter):
        data_load_time = time.time() - prev_time
        batch_process_start = time.time()
        
        model, state, opt_state, loss_value = train_one_batch(
            batch, model, state, filter_spec, loss_fn, opt, opt_state, train_key, lr_scheduler, current_step
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

def validate_one_epoch(val_loader, model, state, val_key, DATASET_IDX_TO_GROUP_SHORT, compute_r2_standard, epoch, current_step):
    from collections import defaultdict
    import time
    import psutil
    print("[INFO] Validating one epoch")
    total_r2_score = 0
    all_preds = []
    all_targets = []
    all_group_idxs = []
    metrics = {}  # New: store metrics per group
    val_start_time = time.time()

    # Convert TF dataset to iterator
    val_iter = iter(val_loader)

    for batch_idx, batch in enumerate(val_iter):
        
        # Convert TF batch to JAX format
        batch = convert_tf_batch_to_jax(batch)
        inputs = batch["neural_input"]
        targets = batch["behavior_input"]
        dataset_group_idxs = batch["dataset_group_idx"]

        key, subkey = jr.split(val_key)
        batch_keys = jr.split(subkey, inputs.shape[0])
        preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0, 0), out_axes=(0, None))(inputs, state, batch_keys, dataset_group_idxs)
        
        all_preds.append(preds)
        all_targets.append(targets)
        all_group_idxs.append(dataset_group_idxs)

    all_preds = jnp.concatenate(all_preds, axis=0)
    all_targets = jnp.concatenate(all_targets, axis=0)
    all_group_idxs = jnp.concatenate(all_group_idxs, axis=0)
    
    for group_key in jnp.unique(all_group_idxs):
        group_key_int = int(group_key)  # Convert JAX array to Python int
        group_name = DATASET_IDX_TO_GROUP_SHORT[group_key_int]
        group_preds = all_preds[all_group_idxs == group_key]
        group_targets = all_targets[all_group_idxs == group_key]
        r2_score = compute_r2_standard(group_preds, group_targets)
        metrics[f"val/r2_{group_name}"] = float(r2_score)
        
    r2_score = compute_r2_standard(all_preds, all_targets)
    metrics["val/r2_all"] = float(r2_score)

    # Log validation timing and resources
    val_end_time = time.time()
    val_time = val_end_time - val_start_time
    metrics['val/runtime'] = val_time

    return metrics

@hydra.main(config_path="../configs", config_name="pretrain", version_base="1.3")
def main(cfg: DictConfig):
    warnings.showwarning = warn_with_traceback
    mp.set_start_method("spawn", force=True)

    # Set up signal handling for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(save_interrupted_checkpoint)

    print(OmegaConf.to_yaml(cfg))

    # Load dataset using TFDS loader instead of PyTorch DataLoader
    train_dataset, train_loader, val_dataset, val_loader = get_brainset_train_val_loaders_tfds(
        train_config=get_dataset_config(
            **cfg.train_dataset
        ),
        val_config=get_dataset_config(
            **cfg.val_dataset
        ),
        **cfg.dataloader
    )

    key, train_key, val_key = jr.split(jr.PRNGKey(cfg.rng_seed), 3)

    model = SSMFoundationalDecoder(
            **cfg.model
        )
    state = eqx.nn.State(model)

    filter_spec = get_filter_spec(
        model,
        **cfg.filter_spec
    )
    
    # Calculate total steps - for TF datasets, we need to estimate the number of batches
    # This is a rough estimate since TF datasets don't have a fixed length like PyTorch DataLoaders
    estimated_batches_per_epoch = 1000  # This should be calculated based on your dataset size
    total_steps = estimated_batches_per_epoch * cfg.training.epochs
    
    if cfg.optimizer.use_cosine_scheduler:
        lr_scheduler = create_cosine_annealing_scheduler(
            initial_lr=cfg.optimizer.lr,
            total_steps=total_steps,
            min_lr=getattr(cfg.optimizer, 'min_lr', 0.0),  # Default to 0.0 if not specified
            warmup_steps=getattr(cfg.optimizer, 'warmup_steps', 0)  # Default to 0 if not specified
        )
    else:
        lr_scheduler = lambda step: cfg.optimizer.lr
    
    opt = optax.chain(
        optax.adamw(learning_rate=lr_scheduler, weight_decay=cfg.optimizer.weight_decay)
    )
    opt_state = opt.init(eqx.filter(model, filter_spec))
    
    loss_fn = mse_loss_foundational
    
    run_name = f"{cfg.wandb.run_prefix}_sub-{''.join(cfg.train_dataset.subjects)}_l{cfg.model.ssm_num_layers}_d{cfg.model.ssm_dim}"
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Track current step for scheduler and best r2 score
    current_step = 0
    best_r2_score = 0.0
    # jax.profiler.start_trace("/tmp/jax_trace")

    if cfg.wandb.resume_run_id is not None:
        # Resume existing wandb run
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, id=cfg.wandb.resume_run_id, resume="allow")
        setup_wandb_metrics()
        
        # Load checkpoint
        model, state, opt_state, last_epoch, current_step, checkpoint_metadata = load_checkpoint_wandb(
            path=None,  # path is ignored, wandb is used
            model_template=model,
            state_template=state,
            opt_state_template=opt_state,
            wandb_run_name=run_name,
            wandb_project=cfg.wandb.project,
            wandb_entity=cfg.wandb.entity,
        )
        start_epoch = last_epoch + 1
        # Load best R² score from checkpoint metadata if available
        best_r2_score = checkpoint_metadata.get('best_r2_score', 0.0)
    else:
        wandb.init(project=cfg.wandb.project, name=run_name, config=config_dict)  # type: ignore            
        setup_wandb_metrics()
        start_epoch = 0

    for epoch in range(start_epoch, cfg.training.epochs):
        # Check for interruption
        if interrupted:
            print("[INFO] Training interrupted. Saving checkpoint...")
            break
        
        model, state, opt_state, current_step, epoch_loss = train_one_epoch(
                train_loader, model, state, filter_spec, loss_fn, opt, opt_state, train_key, lr_scheduler, current_step, epoch
            )
    
        if epoch % cfg.training.checkpoint_every == 0:
            print(f"[DEBUG] main: Running validation for epoch {epoch}")
            metrics = validate_one_epoch(
                val_loader, model, state, val_key, DATASET_IDX_TO_GROUP_SHORT, compute_r2_standard, epoch, current_step
            )
            wandb.log(metrics, step=current_step)
            
            # Track best R² score
            current_r2_avg = metrics.get('r2_avg', 0.0)
            if current_r2_avg > best_r2_score:
                best_r2_score = current_r2_avg
                print(f"[DEBUG] main: New best R² score: {best_r2_score:.4f} at epoch {epoch}")
            
            metadata = metrics
            metadata.update({
                'train_loss': epoch_loss,
                'best_r2_score': best_r2_score,
                'interrupted': False
            })
            
            # Update global state for signal handling
            global current_training_state
            current_training_state = (model, state, opt_state, epoch, current_step, run_name, metrics)
            
            print(f"[DEBUG] main: Saving checkpoint for epoch {epoch}")
            save_checkpoint_wandb(model, state, opt_state, epoch, current_step, metadata, run_name)
    
    # jax.profiler.stop_trace()
    wandb.finish()
    
    print("[DEBUG] main: Training completed successfully")
            
if __name__ == "__main__":
    main() 