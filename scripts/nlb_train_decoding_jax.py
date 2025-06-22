import os

import h5py
import numpy as np
import pandas as pd
import wandb

import torch
from torch.utils.data import DataLoader, TensorDataset, default_collate

from jax import random
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_flatten_with_path

import equinox as eqx
import optax

from omegaconf import OmegaConf

from foundational_ssm.models import S5
from foundational_ssm.utils import h5_to_dict
from foundational_ssm.data_preprocessing import smooth_spikes

class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear
    bias: jax.Array

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = random.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)

    def __call__(self, input, state, key):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), None

        out, _ = jax.lax.scan(f, hidden, input)
        # sigmoid because we're performing binary classification
        return jax.nn.sigmoid(self.linear(out) + self.bias), state

@eqx.filter_jit
def predict_batch(model, state, inputs, key):
    """Predict on a batch of inputs using JAX's vmap"""
    batch_keys = random.split(key, inputs.shape[0])
    preds, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0))(inputs, state, batch_keys)
    return preds

def numpy_collate(batch):
  """To allow us to use torch dataloaders with JAX models"""
  return tree_map(np.asarray, default_collate(batch))

def compute_r2_single_dim(pred_single_dim, target_single_dim):
    """Compute R² for a single dimension"""
    corr_matrix = jnp.corrcoef(pred_single_dim, target_single_dim)
    return corr_matrix[0, 1] ** 2 

def compute_r2(preds, targets):
    r2 = jax.vmap(
        compute_r2_single_dim,
        in_axes=(1, 1),
        out_axes=0
    )(preds, targets).mean()
    return r2

def compute_r2_standard(preds, targets):
    """
    Computes the standard coefficient of determination (R²) for each output dimension.
    
    Args:
        preds: Predictions array of shape (num_samples, output_dim)
        targets: Targets array of shape (num_samples, output_dim)
        
    Returns:
        The mean R² across all output dimensions.
    """
    preds_flat = preds.reshape(-1, preds.shape[-1]) 
    targets_flat = targets.reshape(-1, targets.shape[-1])
    ss_res = jnp.sum((targets_flat - preds_flat) ** 2, axis=0) 
    ss_tot = jnp.sum((targets_flat - jnp.mean(targets_flat, axis=0)) ** 2, axis=0)
    zero_variance = ss_tot < 1e-8
    r2_per_dim = 1 - ss_res / (ss_tot + 1e-8) # Add epsilon for stability
    
    return jnp.mean(r2_per_dim)

def compute_mse(preds, targets):
    return jnp.mean((preds - targets) ** 2)


def compute_metrics(preds, targets):
    """Compute comprehensive R² metrics"""
    
    # Flatten: (batch_size, time_steps, output_dim) -> (batch_size*time_steps, output_dim)
    preds_flat = preds.reshape(-1, preds.shape[-1]) 
    targets_flat = targets.reshape(-1, targets.shape[-1])
    
    r2 = compute_r2_standard(preds_flat, targets_flat)
    mse = compute_mse(preds_flat, targets_flat)
    
    return {
        "r2": r2,
        "mse": mse
    }

def log_model_params_and_grads_wandb(model, grads=None):
    model_params = tree_flatten_with_path(model)[0] 
    grads = tree_flatten_with_path(grads)[0] if grads is not None else []
    for path, value in model_params:
        if eqx.is_array(value):
            full_path = "".join(str(p) for p in path)
            hist = wandb.Histogram(value.flatten())
            wandb.log({
                f"params/{full_path}": hist
            })
    for path, value in grads:
        if eqx.is_array(value):
            full_path = "".join(str(p) for p in path)
            hist = wandb.Histogram(value.flatten())
            wandb.log({
                f"grads/{full_path}": hist
            })

@eqx.filter_jit
def mse_loss(model, state, inputs, targets, key):
    # model = eqx.combine(model_params, model_static)
    batch_keys = random.split(key, inputs.shape[0])
    preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None))(inputs, state, batch_keys)
    mse = jnp.mean((preds - targets) ** 2)
    return (mse, state)

@eqx.filter_jit
def make_step(model, inputs, targets, state, opt, opt_state, key):
    # filter_spec = jax.tree_util.tree_map(lambda _: True, model)
    # model_params, model_static = eqx.partition(model, filter_spec)
    (value, state), grads = eqx.filter_value_and_grad(mse_loss, has_aux=True)(model, state, inputs, targets, key)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, value, grads

def train(
    model, 
    train_loader, 
    train_tensors,
    val_tensors,
    loss_fn, 
    state,
    opt,
    epochs,
    log_every,
    key=random.PRNGKey(0),
    wandb_run_name=None
):
    if wandb_run_name is not None:
        wandb.init(
            project="foundational_ssm_nlb",
            name=wandb_run_name,
            config=conf
        )
    
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            key, subkey = random.split(key)
            
            model, state, opt_state, loss_value, grads = make_step(
                model,
                inputs, 
                targets,
                state,
                opt,
                opt_state,  
                subkey
            )
            
            if wandb_run_name is not None:
                wandb.log({
                    "train/loss": float(loss_value)
                })
            
        if epoch % log_every == 0:
            # Generate keys for evaluation
            key, train_key, val_key = random.split(key, 3)
            
            # Get inputs and targets for train and val sets
            train_inputs, train_targets = train_tensors
            train_inputs = jnp.array(train_inputs.numpy())
            train_targets = jnp.array(train_targets.numpy())
            train_preds = predict_batch(model, state, train_inputs, train_key)
            
            val_inputs, val_targets = val_tensors
            val_inputs = jnp.array(val_inputs.numpy())
            val_targets = jnp.array(val_targets.numpy())
            val_preds = predict_batch(model, state, val_inputs, val_key)
            
            # Compute metrics
            train_mse = compute_mse(train_preds, train_targets)
            train_r2 = compute_r2_standard(train_preds, train_targets)
            val_r2 = compute_r2_standard(val_preds, val_targets)
            
            # Log to console
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_mse:.4f}")
            print(f"  Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")
            print(f"  Train MSE: {train_mse:.4f}")
            
            # Extract and prepare parameter/gradient statistics for logging
            if wandb_run_name is not None:
                wandb_log_dict = {
                    "epoch": epoch,
                    # "metrics/train.mse": float(train_mse),
                    "metrics/train.r2": float(train_r2),
                    "metrics/val.r2": float(val_r2),
                }
                wandb.log(wandb_log_dict)
                log_model_params_and_grads_wandb(model, grads)

    return model


# ============================================================ 
# Main script starts here
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processed_data_folder = '/cs/student/projects1/ml/2024/mlaimon/data/foundational_ssm/processed/nlb' 
dataset_name = "mc_maze"
processed_data_path = os.path.join(processed_data_folder,dataset_name + ".h5")
trial_info_path = os.path.join(processed_data_folder,dataset_name + ".csv")

conf = {
    'task':'decoding',
    'dataset': {
        'dataset': dataset_name
    },
    'model': {
        'input_dim': 182,
        'output_dim': 2,
        'd_state': 64,
        'num_layers': 2,
        'hidden_dim': 64,
        'dropout': 0.1,
        'ssm_core':'s5'
    },
    'optimizer': {
        'lr': 0.001,
        'weight_decay': 0.01  # Added common parameter
    },
    'training': {
        'batch_size': 64,
        'epochs': 2000
    },
    'device': 'cuda',
    'framework': 'jax'
}

args = OmegaConf.create(conf)

with h5py.File(processed_data_path, 'r') as h5file:
    dataset_dict = h5_to_dict(h5file)

trial_info = pd.read_csv(trial_info_path)
trial_info = trial_info[trial_info['split'].isin(['train','val'])]
min_idx = trial_info['trial_id'].min()
trial_info['trial_id'] = trial_info['trial_id'] - min_idx

train_ids = trial_info[trial_info['split']=='train']['trial_id'].tolist()
val_ids = trial_info[trial_info['split']=='val']['trial_id'].tolist()

# Concatenate both heldin and heldout spikes since we're using spikes to predict behavior
spikes = np.concat([
    dataset_dict['train_spikes_heldin'], 
    dataset_dict['train_spikes_heldout']],axis=2) 
smoothed_spikes = torch.tensor(
    smooth_spikes(spikes, kern_sd_ms=40, bin_width=5), 
    dtype=torch.float64)
behavior = torch.tensor(
    dataset_dict['train_behavior'],
    dtype=torch.float64)

input_dim = smoothed_spikes.shape[2]
output_dim = behavior.shape[2]

# Split train and val based on splits from nlb
train_dataset = TensorDataset(smoothed_spikes[train_ids], behavior[train_ids])
val_dataset = TensorDataset(smoothed_spikes[val_ids], behavior[val_ids])
full_dataset = TensorDataset(smoothed_spikes, behavior)


train_loader = DataLoader(
    train_dataset, 
    batch_size=args.training.batch_size, 
    shuffle=True,
    collate_fn=numpy_collate
)

wandb_run_name = f"nlb_{args.task}_{args.model.ssm_core}_l{args.model.num_layers}_d{args.model.d_state}"

model_key = random.PRNGKey(0)
model = S5(
    key= model_key,
    num_blocks=args.model.num_layers,
    N=args.model.input_dim,
    ssm_size=args.model.d_state,
    ssm_blocks=1,
    H=args.model.hidden_dim,
    output_dim=args.model.output_dim,
    clip_eigs=True,
    
)
# model = RNN(in_size=182, out_size=2, hidden_size=128, key=model_key)

# Initialize the state
state = eqx.nn.State(model)
batch_size = smoothed_spikes.shape[0]
keys = random.split(random.PRNGKey(0), batch_size)

opt = optax.adamw(
    learning_rate=args.optimizer.lr, 
    weight_decay=args.optimizer.weight_decay
)
loss_fn = mse_loss

model = train(
    model, 
    train_loader,
    train_dataset.tensors, 
    val_dataset.tensors,  # No validation loader in this example
    loss_fn,
    state, 
    opt,
    epochs=args.training.epochs,
    log_every=10,
    wandb_run_name=wandb_run_name
)
