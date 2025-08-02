import os
import sys
import warnings
import logging
from collections import defaultdict
import signal
import atexit
import tempfile
import pickle
import shutil


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
from foundational_ssm.models import SSMFoundationalDecoder
from foundational_ssm.constants import DATASET_IDX_TO_GROUP_SHORT
from foundational_ssm.utils.training import get_filter_spec, create_cosine_annealing_scheduler, mse_loss_foundational, make_step_foundational
from foundational_ssm.metrics import compute_r2_standard
from foundational_ssm.utils.wandb_utils_jax import save_checkpoint_wandb, load_checkpoint_wandb, save_best_model_wandb, add_alias_to_checkpoint
from foundational_ssm.utils.training_utils import (
    log_batch_metrics, track_batch_timing, setup_wandb_metrics
)

# Additional imports for prediction and activation logging
from foundational_ssm.dataset import TorchBrainDataset
from foundational_ssm.samplers import DatasetIndex, TrialSampler
from foundational_ssm.loaders import pad_collate, transform_brainsets_regular_time_series_smoothed, get_brainset_train_val_loaders
from foundational_ssm.constants import DATA_ROOT
import h5py
from tqdm import tqdm

import warnings
import traceback
import sys

import multiprocessing as mp

import h5py
import torch
import time
import psutil
import logging

logging.basicConfig(filename='pretrain_decoding.log', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
WARNING_LOG_FILE = "warnings.log"
tempdir = "/cs/student/projects1/ml/2024/mlaimon/foundational_ssm/wandb_artifacts"
os.environ['WANDB_CACHE_DIR'] = '/cs/student/projects1/ml/2024/mlaimon/foundational_ssm/wandb_cache'

# Global variables for signal handling
interrupted = False
current_training_state = None  # Will store (model, state, opt_state, epoch, current_step, run_name)

def signal_handler(signum, frame):
    """Handle interruption signals gracefully"""
    global interrupted
    logger.warning(f"Received signal {signum}. Saving checkpoint before exit...")
    interrupted = True

def save_interrupted_checkpoint():
    """Save checkpoint when interrupted"""
    global current_training_state
    if current_training_state is not None:
        training_state = current_training_state
        if len(training_state) == 7:
            model, state, opt_state, epoch, current_step, run_name, metrics = training_state
        else:
            # Handle case where metrics might not be present
            model, state, opt_state, epoch, current_step, run_name = training_state[:6]
            metrics = training_state[6] if len(training_state) > 6 else None
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
            logger.info(f"Checkpoint saved at epoch {epoch}, step {current_step} due to interruption")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    with open(WARNING_LOG_FILE, "a") as log:
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    

# Remove this function since it's now imported from training_utils

def train_one_batch(batch, model, state, filter_spec, loss_fn, opt, opt_state, train_key, lr_scheduler, current_step):
    batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}
    inputs = batch["neural_input"]
    targets = batch["behavior_input"]
    dataset_group_idxs = batch["dataset_group_idx"]
    mask = batch["mask"]
    
    key, subkey = jr.split(train_key)
    model, state, opt_state, loss_value, grads = make_step_foundational(
        model, state, inputs, targets, mask, key, dataset_group_idxs, filter_spec, loss_fn, opt, opt_state
    )
    
    current_lr = lr_scheduler(current_step)
    wandb.log({
        "train/loss": loss_value,
        "train/learning_rate": current_lr,
    }, step=current_step)
    return model, state, opt_state, loss_value

def train_one_epoch(train_loader, model, state, filter_spec, loss_fn, opt, opt_state, rng_key, lr_scheduler, current_step, epoch):    
    epoch_loss = 0
    batch_count = 0
    minute_start_time = time.time()
    prev_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        data_load_time = time.time() - prev_time
        batch_process_start = time.time()
        
        model, state, opt_state, loss_value = train_one_batch(
            batch, model, state, filter_spec, loss_fn, opt, opt_state, rng_key, lr_scheduler, current_step
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
    


def validate_one_epoch(val_loader, model, state, epoch, current_step):
    logger.info("Validating one epoch")
    metrics = {}  # New: store metrics per group
    all_preds = []
    all_targets = []
    all_dataset_group_idxs = []
    val_start_time = time.time()
    prev_time = time.time()
    for batch_idx, batch in enumerate(val_loader):
        data_load_time = time.time() - prev_time
        batch_process_start = time.time()
        batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}
        dataset_group_idxs = batch["dataset_group_idx"]
        inputs = batch["neural_input"]
        targets = batch["behavior_input"]
        mask = batch["mask"]
        mask = mask[..., None]
        preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0, None, None), out_axes=(0, None))(inputs, state, dataset_group_idxs, jr.PRNGKey(0), True)

        all_preds.append(jnp.where(mask, preds, 0))
        all_targets.append(jnp.where(mask, targets, 0))
        all_dataset_group_idxs.append(dataset_group_idxs)
        batch_process_end = time.time()
        batch_process_time = batch_process_end - batch_process_start
        logger.info(f"Batch size: {inputs.shape[0]}, Batch time dimension: {inputs.shape[1]}, Batch {batch_idx} data load time: {data_load_time:.2f}s, batch process time: {batch_process_time:.2f}s")
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


def generate_predictions_and_activations(model, inputs, state, group_idxs):
    """Generate predictions and activations for a batch"""
    preds, activations_list, state = jax.vmap(model.call_with_activations, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, 0, None))(inputs, state, group_idxs)
    activations = {f'ssm_block_{i}': activations_list[i] for i in range(len(activations_list))}
    return preds, activations


def create_activation_dataset(config):
    """Create a dataset for generating predictions and activations"""
    # Use the same config as validation but with valid_trials split
    activation_dataset = TorchBrainDataset(
        config=config,
        root=DATA_ROOT,
        split='valid_trials',
        transform=transform_brainsets_regular_time_series_smoothed
    )
    sampling_intervals = activation_dataset.get_sampling_intervals()
    return activation_dataset, sampling_intervals


def generate_session_predictions_and_activations(model, state, session_id, trial_intervals, dataset, batch_size=64):
    """Generate predictions and activations for a single session"""
    logger.info(f"Processing session {session_id}")
    
    # Collect all trials for this session
    batch = []
    for trial_interval in trial_intervals:
        trial_idx = DatasetIndex(session_id, trial_interval[0], trial_interval[1])
        sample = dataset[trial_idx]
        batch.append(sample)
        
    batch = pad_collate(batch)
    batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}
    
    dataset_group_idxs = batch["dataset_group_idx"]
    inputs = batch["neural_input"]
    targets = batch["behavior_input"]
    mask = batch["mask"]
    num_trials = inputs.shape[0]
    
    # Split into chunks to avoid memory issues
    preds_list = []
    activations_list = {}
    
    # Initialize activations dict with empty lists
    num_ssm_blocks = len(model.ssm_blocks)
    for i in range(num_ssm_blocks):
        activations_list[f'ssm_block_{i}'] = []
    
    for start_idx in tqdm(range(0, num_trials, batch_size), desc=f"Processing {session_id}"):
        stop_idx = min(start_idx + batch_size, num_trials)
        preds, activations = generate_predictions_and_activations(
            model, inputs[start_idx:stop_idx], state, dataset_group_idxs[start_idx:stop_idx]
        )
        preds_list.append(preds)
        for ssm_block in activations:
            activations_list[ssm_block].append(activations[ssm_block])
    
    preds = np.concatenate(preds_list, axis=0)
    activations = {key: np.concatenate(value, axis=0) for key, value in activations_list.items()}
        
    return {
        'predictions': preds,
        'activations': activations,
        'targets': targets,
        'mask': mask,
        'num_trials': inputs.shape[0]
    }


def log_predictions_and_activations(model, state, cfg, epoch, current_step, run_name):
    """Generate and log predictions and activations as wandb artifact"""
    print(f"[DEBUG] Generating predictions and activations for epoch {epoch}")
    
    # Create activation dataset
    activation_dataset, sampling_intervals = create_activation_dataset(cfg.activations_dataset_config)
    
    # Generate predictions and activations for each session
    session_predictions_and_activations = {}
    for session_id, trial_intervals in sampling_intervals.items():
        session_data = generate_session_predictions_and_activations(
            model, state, session_id, trial_intervals, activation_dataset
        )
        session_predictions_and_activations[session_id] = session_data
    
    # Save to H5 file
    h5_path = f"wandb_artifacts/{run_name}/predictions_and_activations.h5"
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    
    with h5py.File(h5_path, 'w') as f:
        # Create session group
        sessions_group = f.create_group('sessions')
        
        for session_id, session_data in session_predictions_and_activations.items():
            session_group = sessions_group.create_group(session_id)
            
            # Save predictions
            session_group.create_dataset('predictions', data=session_data['predictions'])
            session_group.create_dataset('targets', data=session_data['targets'])
            session_group.create_dataset('mask', data=session_data['mask'])
            session_group.create_dataset('num_trials', data=session_data['num_trials'])
            
            # Save activations
            activations_group = session_group.create_group('activations')
            for block_name, activation_data in session_data['activations'].items():
                activations_group.create_dataset(block_name, data=activation_data)
    
    # Create and log wandb artifact
    artifact = wandb.Artifact(
        name=f"{run_name}_predictions_and_activation",
        type="predictions_and_activations",
        description=f"Model predictions and activations for epoch {epoch}"
    )
    artifact.add_file(h5_path)
    
    # Add metadata
    artifact.metadata.update({
        'epoch': epoch,
        'current_step': current_step,
        'num_sessions': len(session_predictions_and_activations),
        'model_config': OmegaConf.to_container(cfg.model, resolve=True)
    })
    
    wandb.log_artifact(artifact,
        aliases=[f'epoch_{epoch}'])
    print(f"[DEBUG] Logged predictions and activations artifact for epoch {epoch}")
    
    return h5_path


@hydra.main(config_path="../configs", config_name="pretrain", version_base="1.3")
def main(cfg: DictConfig):
    warnings.showwarning = warn_with_traceback
    mp.set_start_method("spawn", force=True)

    logger.info(OmegaConf.to_yaml(cfg))

    # Load dataset
    train_dataset, train_loader, val_dataset, val_loader = get_brainset_train_val_loaders(
            cfg.train_loader,
            cfg.val_loader
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
    
    
    if cfg.optimizer.use_cosine_scheduler:
        total_steps = len(train_loader) * cfg.training.epochs
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
    
    run_name = f"{cfg.wandb.run_prefix}_l{cfg.model.ssm_num_layers}_d{cfg.model.ssm_dim}"
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Track current step for scheduler and best r2 score
    current_step = 0
    best_r2_score = 0.0

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
            filter_spec=filter_spec,
            wandb_run_name=run_name,
            wandb_project=cfg.wandb.project,
            wandb_entity=cfg.wandb.entity,
        )
        start_epoch = last_epoch + 1
        # Load best R² score from checkpoint metadata if available
        best_r2_score = checkpoint_metadata.get('best_r2_score', 0.0)
    else:
        # Start new wandb run
        wandb.init(project=cfg.wandb.project, name=run_name, config=dict(config_dict)) 
        setup_wandb_metrics()
        start_epoch = 0
    

    for epoch in range(start_epoch, cfg.training.epochs):
        # Check for interruption
        train_key, subkey = jr.split(train_key)
        logger.info(f"Running training for epoch {epoch}")
        model, state, opt_state, current_step, epoch_loss = train_one_epoch(
                train_loader, model, state, filter_spec, loss_fn, opt, opt_state, subkey, lr_scheduler, current_step, epoch
            )
        
        if epoch % cfg.training.checkpoint_every == 0:
            metadata = {
                'train_loss': epoch_loss
            }
            logger.info(f"Saving checkpoint for epoch {epoch}")
            checkpoint_artifact = save_checkpoint_wandb(model, state, opt_state, epoch, current_step, metadata, run_name)
    
        if epoch % cfg.training.log_val_every == 0:
            add_alias_to_checkpoint(checkpoint_artifact, f'epoch_{epoch}')
            logger.info(f"Running validation for epoch {epoch}")
            metrics = validate_one_epoch(val_loader, model, state, epoch, current_step)
            # Track best R² score
            current_r2_avg = metrics.get('val/r2_avg', 0.0)
            if current_r2_avg > best_r2_score:
                best_r2_score = current_r2_avg
                logger.info(f"New best R² score: {best_r2_score:.4f} at epoch {epoch}")
                add_alias_to_checkpoint(checkpoint_artifact,  'best', metadata = metrics)
        
        if epoch % cfg.training.log_pred_and_activations_every == 0:
            logger.info(f"Logging predictions and activations for epoch {epoch}")
            log_predictions_and_activations(model, state, cfg, epoch, current_step, run_name)
    
    jax.profiler.stop_trace()
    wandb.finish()
    
    logger.info("Training completed successfully")
            
if __name__ == "__main__":
    main()