
# Standard library imports
from foundational_ssm.utils import h5_to_dict 
import math
import os
import json
import tempfile
import multiprocessing as mp
import logging
import time
import h5py
import hydra
from functools import partial

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import jax
import jax.numpy as jnp
from jax import random as jr
from jax import tree as jt 
from jax import tree_util as jtu
from torch.utils.data import DataLoader
import optax
import equinox as eqx

# Foundational SSM imports
from omegaconf import OmegaConf
from foundational_ssm.utils import transfer_foundational_to_downstream, add_alias_to_checkpoint, load_h5_artifact_with_tempdir
from foundational_ssm.metrics import compute_r2_standard
from foundational_ssm.samplers import SequentialFixedWindowSampler
from foundational_ssm.collate import pad_collate
from foundational_ssm.models import SSMFoundationalDecoder, SSMDownstreamDecoder
from foundational_ssm.transform import smooth_spikes

import multiprocessing as mp


def get_rtt_datasets(dataset_cfg, rng_key):
    if dataset_cfg.phase == 'validation':
        data = h5_to_dict(dataset_cfg.train)
        data['neural_input'] = smooth_spikes(data['neural_input'], kern_sd_ms=20, bin_size_ms=5, time_axis=1)
        
        # Split the data into training and validation sets
        n_samples = data['neural_input'].shape[0]
        
        train_ratio = 0.8
        train_samples = int(n_samples * train_ratio)

        indices = jr.permutation(rng_key, n_samples)
        train_indices = indices[:train_samples]
        val_indices = indices[train_samples:]
        
        train_data = {k: v[train_indices] for k, v in data.items()}
        val_data = {k: v[val_indices] for k, v in data.items()}
        

    if dataset_cfg.phase == 'test':
        train_data = h5_to_dict(dataset_cfg.train)
        train_data['neural_input'] = smooth_spikes(train_data['neural_input'], kern_sd_ms=20, bin_size_ms=5, time_axis=1)
        val_data = h5_to_dict(dataset_cfg.test)
        val_data['neural_input'] = smooth_spikes(val_data['neural_input'], kern_sd_ms=20, bin_size_ms=5, time_axis=1)

        data = {
                'neural_input': np.concatenate((train_data['neural_input'], val_data['neural_input'])),
                'behavior_input': np.concatenate((train_data['behavior_input'], val_data['behavior_input'])),
                'mask': np.concatenate((train_data['mask'], val_data['mask'])),
            }
    return train_data, val_data, data

def create_dataloader(data_dict, batch_size, shuffle=True, rng_key=None):
    """
    Creates a generator for minibatches from a dictionary of data.

    Args:
        data_dict (dict): A dictionary where keys are data names (e.g., 'neural_input')
                          and values are arrays of the same leading dimension (samples).
        batch_size (int): The desired size of each minibatch.
        shuffle (bool): Whether to shuffle the data before creating minibatches.
        rng_key (jr.PRNGKey, optional): JAX PRNGKey for shuffling. Required if shuffle is True.

    Yields:
        dict: A dictionary representing a minibatch.
    """
    n_samples = next(iter(data_dict.values())).shape[0]
    indices = jr.permutation(rng_key, n_samples) if shuffle else jr.arange(n_samples)

    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        minibatch = {k: v[batch_indices] for k, v in data_dict.items()}
        yield minibatch

@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def mse_loss_downstream(model, state, inputs, targets, mask, key, skip_timesteps):
    """MSE loss for downstream model (no dataset_group_idx)"""
    batch_keys = jr.split(key, inputs.shape[0])
    preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None))(inputs, state, batch_keys)
    
    # Only evaluate loss on timesteps > skip_timesteps for causal model
    preds = preds[:, skip_timesteps:, :]  # Shape: (batch, seq_len - skip_timesteps, output_dim)
    targets = targets[:, skip_timesteps:, :]
    mask = mask[:, skip_timesteps:]  # Shape: (batch, seq_len - skip_timesteps)
    
    # Compute squared error only on evaluation timesteps
    squared_error = (preds - targets) ** 2
    mask = mask[..., None]  
    masked_squared_error = jnp.where(mask, squared_error, 0.0)
    mse = masked_squared_error.sum() / mask.sum()
    return (mse, state)


def train_one_batch(batch, model, state, loss_fn, opt, opt_state, rng_key, lr_scheduler, current_step, skip_timesteps):
    batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}
    inputs = batch["neural_input"]
    targets = batch["behavior_input"]
    mask = batch["mask"]
    
    (loss_value, state), grads = loss_fn(model, state, inputs, targets, mask, rng_key, skip_timesteps)
    updates, opt_state = opt.update([grads], opt_state, [model])
    model = eqx.apply_updates(model, updates[0])
    
    current_lr = lr_scheduler(current_step)
    wandb.log({
        "train/loss": loss_value,
        "train/learning_rate": current_lr,
    }, step=current_step)
    
    return model, state, opt_state, loss_value

def train_one_epoch(train_data, model, state, mse_loss_downstream, opt, opt_state, lr_scheduler, current_step, skip_timesteps, batch_size, rng_key):    
    train_loader = create_dataloader(train_data, batch_size, shuffle=True, rng_key=rng_key)
    epoch_loss = 0.0
    num_batches = 0

    for i, minibatch in enumerate(train_loader):
        rng_key, batch_train_key = jr.split(rng_key) # Generate a new key for each batch
        model, state, opt_state, batch_loss = train_one_batch(
            minibatch, model, state,
            mse_loss_downstream, opt, opt_state, batch_train_key, lr_scheduler, current_step,
            skip_timesteps
        )
        epoch_loss += batch_loss
        num_batches += 1
        current_step += 1 # Increment step for each batch

    wandb.log({"train/epoch_loss": epoch_loss}, step=current_step)   
    return model, state, opt_state, current_step, epoch_loss

def validate_one_epoch(batch, model, state, epoch, current_step, skip_timesteps):
    metrics = {} 
    all_preds = []
    all_targets = []
    val_start_time = time.time()
    
    batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}
    inputs = batch["neural_input"]
    targets = batch["behavior_input"]
    mask = batch["mask"]
    mask = mask[..., None]
    
    inf_model = eqx.nn.inference_mode(model)
    preds, state = jax.vmap(inf_model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None))(inputs, state, jr.PRNGKey(0))
    all_preds.append(jnp.where(mask, preds, 0))
    all_targets.append(jnp.where(mask, targets, 0))

    all_preds = jnp.concatenate(all_preds, axis=0)
    all_targets = jnp.concatenate(all_targets, axis=0)
    
    all_preds = all_preds[:, skip_timesteps:, :]  # Shape: (batch, seq_len - skip_timesteps, output_dim)
    all_targets = all_targets[:, skip_timesteps:, :]
    mask = mask[:, skip_timesteps:]  # Shape: (batch, seq_len - skip_timesteps)

    r2_score = compute_r2_standard(all_preds, all_targets)
    metrics[f'val/r2'] = float(r2_score)

    val_end_time = time.time()
    val_time = val_end_time - val_start_time
    metrics[f'val/time'] = val_time
    metrics['epoch'] = epoch

    wandb.log(metrics, step=current_step)
    return metrics


def generate_predictions_and_activations(model, state, batch):
    """Generate predictions and activations for a batch"""
    inputs = batch['neural_input']
    targets = batch['behavior_input']
    mask = batch['mask']
    preds, activations_list, state = jax.vmap(model.call_with_activations, axis_name="batch", in_axes=(0, None, None), out_axes=(0, 0, None))(inputs, state, jr.PRNGKey(0))
    activations = {f'ssm_block_{i}': activations_list[i] for i in range(len(activations_list))}
    
    return {
        'predictions': preds,
        'activations': activations,
        'targets': targets,
        'mask': mask,
        'num_trials': inputs.shape[0]
    }
    
def log_predictions_and_activations(model, state, data, cfg, epoch, current_step, run_name, dataset_name):
    """Generate and log predictions and activations as wandb artifact"""
    print(f"[DEBUG] Generating predictions and activations for epoch {epoch}")
    
    # Save to H5 file
    inf_model = eqx.nn.inference_mode(model)
    h5_path = f"/cs/student/projects1/ml/2024/mlaimon/{run_name}/predictions_and_activations_epoch_{epoch}.h5"
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    session_data = generate_predictions_and_activations(inf_model, state, data)

    with h5py.File(h5_path, 'w') as f:
        # Create session group
        sessions_group = f.create_group('sessions')
        session_group = sessions_group.create_group(dataset_name)
        
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
        name=f"{run_name}_predictions_and_activations",
        type="predictions_and_activations",
        description=f"Model predictions and activations for epoch {epoch}"
    )
    artifact.add_file(h5_path)
    
    # Add metadata
    artifact.metadata.update({
        'epoch': epoch,
        'current_step': current_step,
        'num_sessions': len(session_data['predictions']),
    })
    
    wandb.log_artifact(artifact, aliases=[f'epoch_{epoch}'])
    print(f"[DEBUG] Logged predictions and activations artifact for epoch {epoch}")
    
    return h5_path

def create_model_and_state(model_cfg, ds_mode_cfg, foundational_model_cls=SSMFoundationalDecoder, downstream_model_cls=SSMDownstreamDecoder):
    if hasattr(model_cfg, 'checkpoint'):
        artifact_full_name = model_cfg.checkpoint
        # ===========================================================
        # load_checkpoint 
        # ===========================================================
        artifact = api.artifact(artifact_full_name, type="checkpoint")
        foundational_run = artifact.logged_by()
        foundational_run_cfg = OmegaConf.create(foundational_run.config)
        
        foundational_model = foundational_model_cls(
                **foundational_run_cfg.model
            )
        
        downstream_model_cfg = foundational_run_cfg.model.copy()
        downstream_model_cfg.update({'input_dim':130})
        downstream_model = downstream_model_cls(**downstream_model_cfg)


        # ===================================================================================
        #  Load foundational model from checkpoint and transfer SSM layers to downstream
        # ===================================================================================
        if ds_mode_cfg.from_scratch == False:
            with tempfile.TemporaryDirectory() as temp_dir:
                artifact.download(temp_dir)
                
                # Find the checkpoint file in the downloaded directory
                checkpoint_files = [f for f in os.listdir(temp_dir) if f.endswith('.ckpt')]
                checkpoint_path = os.path.join(temp_dir, checkpoint_files[0])
                print(f"Loading checkpoint from: {checkpoint_path}")
                
                with open(checkpoint_path, 'rb') as f:
                    meta = json.loads(f.readline().decode())
                    foundational_model = eqx.tree_deserialise_leaves(f, foundational_model)
                
            downstream_model = transfer_foundational_to_downstream(foundational_model, downstream_model)
            
    elif hasattr(model_cfg, 'cfg'):
        # ===========================================================
        # load_model_cfg 
        # ===========================================================
        downstream_model_cfg = model_cfg.cfg
        downstream_model = SSMDownstreamDecoder(**downstream_model_cfg)
        if ds_mode_cfg.from_scratch == False:
            raise ValueError("Model configuration provided but no checkpoint specified. Please provide a checkpoint to load the foundational model.")
    downstream_state = eqx.nn.State(downstream_model)        
    return downstream_model, downstream_state, downstream_model_cfg


