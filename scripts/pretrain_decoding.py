import os
import sys
import warnings
import logging

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
from jax.tree_util import tree_map, tree_flatten_with_path
import optax


# Foundational SSM core imports
from foundational_ssm.data_utils import get_train_val_loaders, get_dataset_config
from foundational_ssm.models import SSMFoundational
from foundational_ssm.loss import CombinedLoss

@eqx.filter_jit
def predict_batch(model, state, inputs, key):
    """Predict on a batch of inputs using JAX's vmap"""
    batch_keys = jr.split(key, inputs.shape[0])
    preds, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0))(inputs, state, batch_keys)
    return preds

@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def mse_loss(model_params, model_static, state, inputs, targets, dataset_group_key, key):
    model = eqx.combine(model_params, model_static)
    batch_keys = jr.split(key, inputs.shape[0])
    preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0, None), out_axes=(0, None))(inputs, state, batch_keys, dataset_group_key)
    mse = jnp.mean((preds - targets) ** 2)
    return (mse, state)

@eqx.filter_jit
def make_step(model, state, filter_spec, inputs, targets, dataset_group_key, loss_fn, opt, opt_state, key):
    model_params, model_static = eqx.partition(model, filter_spec)
    (value, state), grads = loss_fn(model_params, model_static, state, inputs, targets, dataset_group_key, key)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, value, grads

@hydra.main(config_path="../configs", config_name="cmt", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Load dataset
    train_dataset, train_loader, val_dataset, val_loader = get_train_val_loaders(
        train_config=get_dataset_config(
            cfg.dataset.name,
            subjects=cfg.dataset.subjects
        ),
        batch_size=cfg.dataset.batch_size
    )
    
    key = jr.PRNGKey(cfg.rng_seed)
    model_key, train_key = jr.split(key, 2)

    model = SSMFoundational(
            ssm_io_dim = cfg.model.ssm_io_dim,
            ssm_dim = cfg.model.ssm_dim,
            ssm_init_diag_blocks = cfg.model.ssm_init_diag_blocks,
            ssm_num_layers = cfg.model.ssm_num_layers,
            output_dim = cfg.model.output_dim,
            key = model_key,
        )
    state = eqx.nn.State(model)
    
    # Filter spec for freezing parts of the model
    filter_spec = tree_map(eqx.is_inexact_array, model)
    
    # Load JAX optimizer
    opt = optax.adam(learning_rate=cfg.optimizer.lr)
    opt_state = opt.init(eqx.filter(model, filter_spec))
    
    # Load JAX loss function
    loss_fn = mse_loss
    
    
    for epoch in range(cfg.training.epochs):
        loss_value = 0.0
        for batch in train_loader:
            inputs = batch["neural_input"]
            targets = batch["behavior_input"]
            dataset_group_key = batch["dataset_group_key"][0]
            
            key, subkey = jr.split(train_key)
            
            model, state, opt_state, loss_value, grads = make_step(
                model,
                state,
                filter_spec,
                inputs, 
                targets, 
                dataset_group_key,
                loss_fn,
                opt,
                opt_state,  
                subkey
            )
            loss_value += loss_value
            
        if epoch % cfg.training.log_every == 0:
            # Generate keys for evaluation
            print(f"Epoch {epoch}/{cfg.training.epochs}, Loss: {loss_value:.4f}")
            
if __name__ == "__main__":
    main()
    


