#!/usr/bin/env python3
"""
Simplified training script that bypasses Hydra to test if the issue is with Hydra configuration processing.
"""

import os
import sys
import warnings
import logging
from collections import defaultdict
import signal
import atexit
import time

# Typing
from typing import List, Dict

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
from omegaconf import OmegaConf

WARNING_LOG_FILE = "warnings.log"

# Global variables for signal handling
interrupted = False
current_training_state = None  # Will store (model, state, opt_state, epoch, current_step, run_name)

def signal_handler(signum, frame):
    """Handle interruption signals gracefully"""
    global interrupted
    print(f"\nReceived signal {signum}. Saving checkpoint before exit...")
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
            print(f"Checkpoint saved at epoch {epoch}, step {current_step} due to interruption")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    with open(WARNING_LOG_FILE, "a") as log:
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

def train_one_batch(batch, model, state, filter_spec, loss_fn, opt, opt_state, train_key, lr_scheduler, current_step):
    print(f"[DEBUG] train_one_batch: Starting batch processing for step {current_step}")
    batch = prepare_batch_for_training(batch)
    inputs, targets, _ = extract_batch_data(batch)
    dataset_group_idx = batch["dataset_group_idx"][0]
    print(f"[DEBUG] train_one_batch: Batch shape - inputs: {inputs.shape}, targets: {targets.shape}, dataset_group_idx: {dataset_group_idx}")
    
    key, subkey = jr.split(train_key)
    print(f"[DEBUG] train_one_batch: Calling make_step_foundational...")
    model, state, opt_state, loss_value, grads = make_step_foundational(
        model, state, filter_spec, inputs, targets, 
        loss_fn, opt, opt_state, subkey, dataset_group_idx,
    )
    print(f"[DEBUG] train_one_batch: make_step_foundational completed, loss_value: {loss_value}")
    
    current_lr = lr_scheduler(current_step)
    print(f"[DEBUG] train_one_batch: Logging to wandb - loss: {loss_value}, lr: {current_lr}, step: {current_step}")
    wandb.log({
        "train/loss": loss_value,
        "train/learning_rate": current_lr,
    }, step=current_step)
    print(f"[DEBUG] train_one_batch: wandb.log completed for step {current_step}")
    return model, state, opt_state, loss_value

def train_one_epoch(train_loader, model, state, filter_spec, loss_fn, opt, opt_state, train_key, lr_scheduler, current_step, epoch):
    print(f"[DEBUG] train_one_epoch: Starting epoch {epoch} with {len(train_loader)} batches")
    print(f"[DEBUG] train_one_epoch: About to start iterating over train_loader...")
    
    epoch_loss = 0
    batch_count = 0
    minute_start_time = time.time()
    prev_time = time.time()
    
    print(f"[DEBUG] train_one_epoch: Creating iterator...")
    train_iter = iter(train_loader)
    print(f"[DEBUG] train_one_epoch: Iterator created successfully")
    
    for batch_idx, batch in enumerate(train_loader):
        print(f"[DEBUG] train_one_epoch: Processing batch {batch_idx + 1}/{len(train_loader)} at step {current_step}")
        data_load_time = time.time() - prev_time
        batch_process_start = time.time()
        
        model, state, opt_state, loss_value = train_one_batch(
            batch, model, state, filter_spec, loss_fn, opt, opt_state, train_key, lr_scheduler, current_step
        )
        batch_process_end = time.time()
        batch_process_time = batch_process_end - batch_process_start
        print(f"[DEBUG] train_one_epoch: Batch {batch_idx + 1} completed - loss: {loss_value}, process_time: {batch_process_time:.4f}s")
        
        log_batch_metrics(data_load_time, batch_process_time, epoch, current_step)
        epoch_loss += loss_value
        batch_count += 1
        current_time = time.time()
        batch_count, minute_start_time = track_batch_timing(batch_count, minute_start_time, current_time, current_step)
        prev_time = time.time()
        current_step += 1
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_loss_so_far = epoch_loss / (batch_idx + 1)
            print(f"[DEBUG] train_one_epoch: Progress - {batch_idx + 1}/{len(train_loader)} batches, avg_loss: {avg_loss_so_far:.4f}")
    
    print(f"[DEBUG] train_one_epoch: Epoch {epoch} completed - total_loss: {epoch_loss}, avg_loss: {epoch_loss/len(train_loader):.4f}")
    wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch}, step=current_step)
    print(f"[DEBUG] train_one_epoch: Logged epoch_loss to wandb: {epoch_loss}")
    return model, state, opt_state, current_step, epoch_loss

def validate_one_epoch(val_loader, model, state, val_key, DATASET_IDX_TO_GROUP_SHORT, compute_r2_standard, epoch, current_step):
    print(f"[DEBUG] validate_one_epoch: Starting validation for epoch {epoch}")
    from collections import defaultdict
    import time
    import psutil
    print("Validating one epoch")
    total_r2_score = 0
    group_preds = defaultdict(list)
    group_targets = defaultdict(list)
    metrics = {}  # New: store metrics per group
    val_start_time = time.time()

    print(f"[DEBUG] validate_one_epoch: Processing {len(val_loader)} validation batches")
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx % 10 == 0:
            print(f"[DEBUG] validate_one_epoch: Processing validation batch {batch_idx + 1}/{len(val_loader)}")
        
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

    print(f"[DEBUG] validate_one_epoch: Computing R2 scores for {len(group_preds)} groups")
    for group_key, preds in group_preds.items():
        preds = jnp.concatenate(preds, axis=0)
        targets = jnp.concatenate(group_targets[group_key], axis=0)
        r2_score = compute_r2_standard(preds, targets)
        print(f"[DEBUG] validate_one_epoch: Group {group_key} R2: {r2_score:.4f}")
        wandb.log({f"val/r2_{group_key}": r2_score, "epoch": epoch}, step=current_step)
        total_r2_score += r2_score
        metrics[group_key] = float(r2_score)  # Store as float for serialization

    avg_r2_score = total_r2_score / len(group_preds) if group_preds else 0
    metrics['r2_avg'] = avg_r2_score
    print(f"[DEBUG] validate_one_epoch: Average R2 score: {avg_r2_score:.4f}")

    # Log validation timing and resources
    val_end_time = time.time()
    val_time = val_end_time - val_start_time
    metrics['val_time'] = val_time
    print(f"[DEBUG] validate_one_epoch: Validation completed in {val_time:.2f} seconds")

    return metrics

def main():
    print("[DEBUG] main: Starting training script (without Hydra)")
    warnings.showwarning = warn_with_traceback
    mp.set_start_method("spawn", force=True)

    # Set up signal handling for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(save_interrupted_checkpoint)

    # Load config directly (bypassing Hydra)
    config_path = "configs/pretrain.yaml"
    cfg = OmegaConf.load(config_path)
    print(f"[DEBUG] main: Config loaded directly: {type(cfg)}")
    print(OmegaConf.to_yaml(cfg))

    # Load dataset
    print("[DEBUG] main: Loading dataset...")
    train_dataset, train_loader, val_dataset, val_loader = get_brainset_train_val_loaders(
        train_config=get_dataset_config(
            **cfg.train_dataset
        ),
        val_config=get_dataset_config(
            **cfg.val_dataset
        ),
        **cfg.dataloader
    )
    print(f"[DEBUG] main: Dataset loaded - train_loader: {len(train_loader)} batches, val_loader: {len(val_loader)} batches")
    
    # Test dataloading in isolation
    print("[DEBUG] main: Testing dataloading in isolation...")
    try:
        test_iter = iter(train_loader)
        print("[DEBUG] main: Created iterator successfully")
        
        # Try to get the first batch
        print("[DEBUG] main: Attempting to get first batch...")
        first_batch = next(test_iter)
        print(f"[DEBUG] main: Successfully loaded first batch with keys: {list(first_batch.keys())}")
        
        # Test the transform
        print("[DEBUG] main: Testing transform on first batch...")
        if hasattr(train_dataset, 'transform') and train_dataset.transform is not None:
            transformed_batch = train_dataset.transform(first_batch)
            print(f"[DEBUG] main: Transform completed successfully, output keys: {list(transformed_batch.keys())}")
        else:
            print("[DEBUG] main: No transform set on dataset")
            
    except Exception as e:
        print(f"[ERROR] main: Failed to load first batch: {e}")
        import traceback
        traceback.print_exc()
        raise e

    # Continue with the rest of the training setup...
    print("[DEBUG] main: Setting up training components...")
    
    # Initialize random keys
    key, train_key, val_key = jr.split(jr.PRNGKey(cfg.rng_seed), 3)
    print(f"[DEBUG] main: Random keys generated with seed {cfg.rng_seed}")

    # Create model
    print("[DEBUG] main: Creating model...")
    model = SSMFoundationalDecoder(**cfg.model)
    state = eqx.nn.State(model)
    print(f"[DEBUG] main: Model created with state: {type(state)}")

    # Create filter spec
    filter_spec = get_filter_spec(model, **cfg.filter_spec)
    print(f"[DEBUG] main: Filter spec created")
    
    # Create optimizer and scheduler
    if cfg.optimizer.use_cosine_scheduler:
        total_steps = len(train_loader) * cfg.training.epochs
        lr_scheduler = create_cosine_annealing_scheduler(
            initial_lr=cfg.optimizer.lr,
            total_steps=total_steps,
            min_lr=getattr(cfg.optimizer, 'min_lr', 0.0),
            warmup_steps=getattr(cfg.optimizer, 'warmup_steps', 0)
        )
        print(f"[DEBUG] main: Cosine scheduler created with total_steps: {total_steps}")
    else:
        lr_scheduler = lambda step: cfg.optimizer.lr
        print(f"[DEBUG] main: Constant LR scheduler created with lr: {cfg.optimizer.lr}")
    
    opt = optax.chain(
        optax.adamw(learning_rate=lr_scheduler, weight_decay=cfg.optimizer.weight_decay)
    )
    opt_state = opt.init(eqx.filter(model, filter_spec))
    print(f"[DEBUG] main: Optimizer initialized")
    
    loss_fn = mse_loss_foundational
    
    # Setup wandb
    run_name = f"{cfg.wandb.run_prefix}_sub-{''.join(cfg.train_dataset.subjects)}_l{cfg.model.ssm_num_layers}_d{cfg.model.ssm_dim}"
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    print(f"[DEBUG] main: Run name: {run_name}")
    
    # Initialize wandb
    print(f"[DEBUG] main: Starting new wandb run with project: {cfg.wandb.project}, name: {run_name}")
    try:
        wandb.init(project=cfg.wandb.project, name=run_name, config=config_dict)
        print(f"[DEBUG] main: Wandb initialized successfully")
        
        if wandb.run is not None:
            print(f"[DEBUG] main: Wandb run ID: {wandb.run.id}")
            print(f"[DEBUG] main: Wandb run URL: {wandb.run.get_url()}")
        else:
            print(f"[DEBUG] main: Wandb run is None")
        
        # Test wandb logging
        test_log = {"test/debug": 1.0}
        wandb.log(test_log, step=0)
        print(f"[DEBUG] main: Test wandb.log completed successfully")
        
    except Exception as e:
        print(f"[ERROR] main: Failed to initialize wandb: {e}")
        raise e
    
    setup_wandb_metrics()
    print(f"[DEBUG] main: Wandb metrics setup completed")
    
    # Track current step for scheduler and best r2 score
    current_step = 0
    best_r2_score = 0.0
    jax.profiler.start_trace("/tmp/jax_trace")

    print(f"[DEBUG] main: Starting training loop from epoch 0 to {cfg.training.epochs}")
    
    # Test one epoch to see if it works
    print(f"[DEBUG] main: Testing one epoch...")
    try:
        model, state, opt_state, current_step, epoch_loss = train_one_epoch(
            train_loader, model, state, filter_spec, loss_fn, opt, opt_state, train_key, lr_scheduler, current_step, 0
        )
        print(f"[DEBUG] main: Test epoch completed successfully - final_step: {current_step}, epoch_loss: {epoch_loss}")
    except Exception as e:
        print(f"[ERROR] main: Failed during train_one_epoch: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    jax.profiler.stop_trace()
    wandb.finish()
    
    print(jax.devices())
    print(jax.default_backend())
    print("[DEBUG] main: Training test completed successfully")

if __name__ == "__main__":
    main() 