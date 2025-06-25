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

from foundational_ssm.models import S5
from foundational_ssm.data_utils import h5_to_dict
from foundational_ssm.data_utils import smooth_spikes
from foundational_ssm.trainer.decoding import train_decoding_jax

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path




def numpy_collate(batch):
  """To allow us to use torch dataloaders with JAX models"""
  return tree_map(np.asarray, default_collate(batch))


            
def freeze_filter(path, value, freeze_ssm=False, freeze_linear=False):
    if freeze_ssm and "ssm" in path:
        return False
    if freeze_linear and "linear" in path:
        return False
    return eqx.is_inexact_array(value)


@eqx.filter_jit
def predict_batch(model, state, inputs, key):
    """Predict on a batch of inputs using JAX's vmap"""
    batch_keys = jr.split(key, inputs.shape[0])
    preds, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0))(inputs, state, batch_keys)
    return preds

@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def mse_loss(model_params, model_static, state, inputs, targets, key):
    model = eqx.combine(model_params, model_static)
    batch_keys = jr.split(key, inputs.shape[0])
    preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None))(inputs, state, batch_keys)
    mse = jnp.mean((preds - targets) ** 2)
    return (mse, state)

@eqx.filter_jit
def make_step(model, filter_spec, inputs, targets, state, opt, opt_state, key):
    model_params, model_static = eqx.partition(model, filter_spec)
    (value, state), grads = mse_loss(model_params, model_static, state, inputs, targets, key)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, value, grads



# ============================================================ 
# Main script starts here
# ============================================================

@hydra.main(config_path="../configs", config_name="s5_maze")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # For debugging

    processed_data_path = os.path.join(
        to_absolute_path(cfg.dataset.processed_data_folder),
        cfg.dataset.name + ".h5"
    )
    trial_info_path = os.path.join(
        to_absolute_path(cfg.dataset.processed_data_folder),
        cfg.dataset.name + ".csv"
    )

    wandb.init(
        project=cfg.wandb_project,
        name=f"nlb_{cfg.task}_{cfg.model.ssm_core}_l{cfg.model.num_layers}_d{cfg.model.d_state}",
        config= OmegaConf.to_container(cfg, resolve=True)
    )

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

    # Split train and val based on splits from nlb
    train_dataset = TensorDataset(smoothed_spikes[train_ids], behavior[train_ids])
    val_dataset = TensorDataset(smoothed_spikes[val_ids], behavior[val_ids])


    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True,
        collate_fn=numpy_collate
    )
    
    key = random.PRNGKey(cfg.rng_seed)
    model_key, train_key = random.split(key, 2)
    model = S5(
        key= model_key,
        num_blocks=cfg.model.num_layers,
        N=cfg.model.input_dim,
        ssm_size=cfg.model.d_state,
        ssm_blocks=1,
        H=cfg.model.hidden_dim,
        output_dim=cfg.model.output_dim,
        clip_eigs=True,
        
    )
    state = eqx.nn.State(model)
    
    # Filter spec for freezing parts of the model
    filter_spec = tree_map(eqx.is_inexact_array, model)
    
    if cfg.model.freeze_ssm:
        where = lambda fs: tuple(block.ssm for block in fs.blocks)
        filter_spec = eqx.tree_at(where, filter_spec, replace=(False,) * len(filter_spec.blocks))
    if cfg.model.freeze_mlp:
        where = lambda m: (m.linear_decoder, m.linear_encoder)
        filter_spec = eqx.tree_at(where, filter_spec, replace=(False, False))
    
    opt = optax.adamw(
        learning_rate=cfg.optimizer.lr, 
        weight_decay=cfg.optimizer.weight_decay
    )
    opt_state = opt.init(eqx.filter(model, filter_spec))
    
    loss_fn = mse_loss

    model = train_decoding_jax(
        model, 
        filter_spec, 
        train_loader,
        train_dataset.tensors, 
        val_dataset.tensors,  # No validation loader in this example
        loss_fn,
        state, 
        opt,
        opt_state,
        epochs=cfg.training.epochs,
        log_every=10,
        key=train_key,
        wandb_run_name=f"nlb_{cfg.task}_{cfg.model.ssm_core}_l{cfg.model.num_layers}_d{cfg.model.d_state}"
    )

if __name__ == "__main__":
    main()
