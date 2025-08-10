
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

def create_optimizer_and_state(model, cfg):
    optimizer_cfg = cfg.optimizer
    lr_scheduler = lambda step: optimizer_cfg.lr
    opt_mode = getattr(optimizer_cfg, 'mode', 'all')
    
    def _get_param_labels(model):
        flat_model, treedef = jtu.tree_flatten_with_path(model)
        labels = []
        for path, leaf in flat_model:
            path_str = jtu.keystr(path)
            label = "regular"  # Default for params not otherwise specified

            # Check for S5 block parameters first
            if "ssm_blocks" in path_str:
                if "Lambda_re" in path_str or "Lambda_im" in path_str:
                    label = "ssm_A"
                elif "B" in path_str and "weight" in path_str:
                    label = "ssm_B"
                elif "C" in path_str and "weight" in path_str:
                    label = "ssm_C"
                elif ".D" in path_str or "log_step" in path_str:
                    label = "ssm_D_log_step"
            # Then check for encoder/decoder/embedding
            elif ("encoder" in path_str or "encoders") and "dropout" not in path_str: # Catches 'encoder' and 'encoders'
                if "weight" in path_str: label = "encoder_weight"
                elif "bias" in path_str: label = "encoder_bias"
            elif "decoder" in path_str and "dropout" not in path_str:
                if "weight" in path_str: label = "decoder_weight"
                elif "bias" in path_str: label = "decoder_bias"
            elif "context_embedding" in path_str:
                # This handles both eqx.nn.Embedding and a raw learnable array
                if "weight" in path_str or leaf.ndim > 1:
                    label = "embedding"
            
            labels.append(label)
        return treedef.unflatten(labels)

    if opt_mode == "all":
        label_fn = lambda x: jt.map_with_path(
                    lambda k, _: "ssm"
                    if getattr(k[-1], 'name', None) in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                    else ("b" if getattr(k[-1], 'name', None) in ["B"] else "regular"),
                    x
                )
        opt = optax.multi_transform(
            {
                "b": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.lr,
                                                            weight_decay=optimizer_cfg.weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=optimizer_cfg.lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.lr,
                                                                weight_decay=optimizer_cfg.weight_decay),
            },
            [label_fn(model)],
        )
    
    elif opt_mode == "s5":
        # Train all weights with different learning rates for SSM components
        label_fn = lambda x: jt.map_with_path(
                    lambda k, _: "ssm"
                    if getattr(k[-1], 'name', None) in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                    else ("b" if getattr(k[-1], 'name', None) in ["B"] else "regular"),
                    x
                )
        opt = optax.multi_transform(
            {
                "b": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.ssm_lr,
                                                            weight_decay=optimizer_cfg.weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=optimizer_cfg.ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.lr,
                                                                weight_decay=optimizer_cfg.weight_decay),
            },
            [label_fn(model)],
        )
        
    elif opt_mode == "freeze_a":
        # Freeze SSM parameters, train everything else
        label_fn = lambda x: jt.map_with_path(
                    lambda k, _: "frozen"
                    if getattr(k[-1], 'name', None) in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                    else "regular",
                    x
                )
        opt = optax.multi_transform(
            {
                "frozen": optax.set_to_zero(),  
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.lr,
                                                                weight_decay=optimizer_cfg.weight_decay),
            },
            [label_fn(model)],
        )
    
    elif opt_mode == "freeze_ssm":
        # Freeze all SSM block parameters, train everything else
        # This uses a path-based approach that works for both foundational and downstream models
        label_fn = lambda x: jt.map_with_path(
                lambda path, _: "frozen" if any(isinstance(p, str) and p == "ssm_blocks" for p in path) else "regular",
                x
            )
        
        opt = optax.multi_transform(
            {
                "frozen": optax.set_to_zero(),  # No updates for frozen parameters
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.lr,
                                                                weight_decay=optimizer_cfg.weight_decay),
            },
            [label_fn(model)],
    )
        
    elif opt_mode == "encoder_only":
        # Only train encoder, freeze everything else
        label_fn = lambda x: jt.map_with_path(
                    lambda k, _: "trainable"
                    if any(part.name == "encoder" for part in k if hasattr(part, 'name'))
                    else "frozen",
                    x
                )
        opt = optax.multi_transform(
            {
                "frozen": optax.set_to_zero(),  # No updates for frozen parameters
                "trainable": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.lr,
                                                                  weight_decay=optimizer_cfg.weight_decay),
            },
            [label_fn(model)],
        )
        
    elif opt_mode == "muP_SSM":
        base_ssm_io_dim = cfg.optimizer.base_ssm_io_dim  # Base model's H dimension
        base_ssm_dim = cfg.optimizer.base_ssm_dim        # Base model's P dimension
        ssm_io_dim = cfg.model.ssm_io_dim  # Current model's H dimension
        ssm_dim = cfg.model.ssm_dim        # Current model's P dimension
        lr_mult_A = ssm_io_dim / base_ssm_io_dim
        # η_B ~ sqrt(Nx / Nu) (where Nx is ssm_dim or P)
        lr_mult_B = jnp.sqrt(ssm_dim / base_ssm_dim) / jnp.sqrt(ssm_io_dim / base_ssm_io_dim)
        # η_C ~ 1 / (Nx * sqrt(Nu))
        lr_mult_C = (1 / (ssm_dim / base_ssm_dim)) / jnp.sqrt(ssm_io_dim / base_ssm_io_dim)
        # D and log_step are scaled by 1.0 as per standard practice [2, 3]
        lr_mult_D_log_step = 1.0
        
        width_mult = ssm_io_dim / base_ssm_io_dim
        lr_mult_encoder_weight = 1.0
        lr_mult_encoder_bias = width_mult
        lr_mult_decoder_weight = 1.0 / width_mult
        lr_mult_decoder_bias = 1.0
        lr_mult_embedding = 1.0
        
        param_labels = get_param_labels(model)
        base_lr = optimizer_cfg.lr
        weight_decay = optimizer_cfg.weight_decay
        optimizer_map = {
            # µP-SSM parameters
            "s5_A": optax.adam(learning_rate=base_lr * lr_mult_A),
            "s5_B": optax.adam(learning_rate=base_lr * lr_mult_B),
            "s5_C": optax.adam(learning_rate=base_lr * lr_mult_C),
            "s5_D_log_step": optax.adam(learning_rate=base_lr * lr_mult_D_log_step),

            # Canonical µP parameters
            "encoder_weight": optax.adamw(learning_rate=base_lr * lr_mult_encoder_weight, weight_decay=weight_decay),
            "encoder_bias": optax.adamw(learning_rate=base_lr * lr_mult_encoder_bias, weight_decay=weight_decay),
            "decoder_weight": optax.adamw(learning_rate=base_lr * lr_mult_decoder_weight, weight_decay=weight_decay),
            "decoder_bias": optax.adamw(learning_rate=base_lr * lr_mult_decoder_bias, weight_decay=weight_decay),
            "embedding": optax.adamw(learning_rate=base_lr * lr_mult_embedding, weight_decay=weight_decay),

            # Fallback for any other parameters (e.g., layer norms)
            "regular": optax.adamw(learning_rate=base_lr, weight_decay=weight_decay),
        }
        opt = optax.multi_transform(optimizer_map, param_labels)

    else:
        raise ValueError(f"Unknown optimization mode: {opt_mode}. Valid modes are: 'all', 'freeze_ssm', 'encoder_only'")
    
    opt_state = opt.init(eqx.filter([model], eqx.is_array))
    return opt, opt_state, lr_scheduler



    param_labels = get_param_labels(downstream_model)

    # --- 4. Define Optimizer Transformations ---
    base_lr = optimizer_cfg.lr
    weight_decay = optimizer_cfg.weight_decay

    # Map each label to a specific optimizer configuration.
    # We follow your pattern: Adam for core SSM params, AdamW for others.
    optimizer_map = {
        # µP-SSM parameters
        "s5_A": optax.adam(learning_rate=base_lr * lr_mult_A),
        "s5_B": optax.adam(learning_rate=base_lr * lr_mult_B),
        "s5_C": optax.adam(learning_rate=base_lr * lr_mult_C),
        "s5_D_log_step": optax.adam(learning_rate=base_lr * lr_mult_D_log_step),

        # Canonical µP parameters
        "encoder_weight": optax.adamw(learning_rate=base_lr * lr_mult_encoder_weight, weight_decay=weight_decay),
        "encoder_bias": optax.adamw(learning_rate=base_lr * lr_mult_encoder_bias, weight_decay=weight_decay),
        "decoder_weight": optax.adamw(learning_rate=base_lr * lr_mult_decoder_weight, weight_decay=weight_decay),
        "decoder_bias": optax.adamw(learning_rate=base_lr * lr_mult_decoder_bias, weight_decay=weight_decay),
        "embedding": optax.adamw(learning_rate=base_lr * lr_mult_embedding, weight_decay=weight_decay),

        # Fallback for any other parameters (e.g., layer norms)
        "regular": optax.adamw(learning_rate=base_lr, weight_decay=weight_decay),
    }

    # --- 5. Create the Final Optimizer ---
    # `optax.multi_transform` applies the correct optimizer from the map
    # to each parameter based on the labels generated by `get_param_labels`.
    opt = optax.multi_transform(optimizer_map, param_labels)