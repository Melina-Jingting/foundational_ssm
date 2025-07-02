import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from collections import defaultdict
from jax.tree_util import tree_map
import wandb
import os
import json
from foundational_ssm.models import SSMFoundational

# Filter spec for freezing parts of the model
def get_filter_spec(model, freeze_ssm: bool, freeze_mlp: bool):
    filter_spec = tree_map(eqx.is_inexact_array, model)
    if freeze_ssm:
        where = lambda fs: tuple(block.ssm for block in fs.ssm_blocks)
        filter_spec = eqx.tree_at(where, filter_spec, replace=(False,) * len(filter_spec.ssm_blocks))
    if freeze_mlp:
        where = lambda m: (m.decoders, m.encoders)
        filter_spec = eqx.tree_at(where, filter_spec, replace=(False, False))
    return filter_spec



@eqx.filter_jit
def predict_batch(model, state, inputs, key, dataset_group_idx):
    """Predict on a batch of inputs using JAX's vmap"""
    batch_keys = jr.split(key, inputs.shape[0])
    preds, _, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0, None), out_axes=(0, None, 0))(inputs, state, batch_keys, dataset_group_idx)
    return preds

@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def mse_loss(model_params, model_static, state, inputs, targets, dataset_group_idx, key):
    model = eqx.combine(model_params, model_static)
    batch_keys = jr.split(key, inputs.shape[0])
    preds, state, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0, None), out_axes=(0, None, 0))(inputs, state, batch_keys, dataset_group_idx)
    mse = jnp.mean((preds - targets) ** 2)
    return (mse, state)

@eqx.filter_jit
def make_step(model, state, filter_spec, inputs, targets, dataset_group_idx, loss_fn, opt, opt_state, key):
    model_params, model_static = eqx.partition(model, filter_spec)
    (value, state), grads = loss_fn(model_params, model_static, state, inputs, targets, dataset_group_idx, key)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, value, grads


def create_cosine_annealing_scheduler(initial_lr, total_steps, min_lr=0.0, warmup_steps=0):
    """
    Creates a cosine annealing learning rate scheduler with optional warmup.
    
    Args:
        initial_lr: Initial learning rate
        total_steps: Total number of training steps
        min_lr: Minimum learning rate (default: 0.0)
        warmup_steps: Number of warmup steps (default: 0)
        
    Returns:
        optax scheduler function
    """
    if warmup_steps > 0:
        # Warmup followed by cosine annealing
        warmup_scheduler = optax.linear_schedule(
            init_value=0.0,
            end_value=initial_lr,
            transition_steps=warmup_steps
        )
        
        cosine_scheduler = optax.cosine_decay_schedule(
            init_value=initial_lr,
            decay_steps=total_steps - warmup_steps,
            alpha=min_lr / initial_lr  # alpha determines the minimum value
        )
        
        # Combine warmup and cosine annealing
        scheduler = optax.join_schedules(
            schedules=[warmup_scheduler, cosine_scheduler],
            boundaries=[warmup_steps]
        )
    else:
        # Pure cosine annealing
        scheduler = optax.cosine_decay_schedule(
            init_value=initial_lr,
            decay_steps=total_steps,
            alpha=min_lr / initial_lr
        )
    
    return scheduler

def load_model_and_state(wandb_pretrained_model_id, hyperparams):
    """
    either loads a model from wandb or creates a new model from hyperparams
    Args:
        wandb_pretrained_model_id: wandb artifact id of the model to load
        hyperparams: dict of hyperparams to create a new model
    Returns:
        model (SSMFoundational): Loaded model or None if not specified.
    """
    if wandb_pretrained_model_id is not None:
        api = wandb.Api()
        model_artifact = api.artifact(wandb_pretrained_model_id, type="model")
        model_artifact_dir = model_artifact.download()
        model_filename = os.path.join(model_artifact_dir, 'best_model.pt')
        with open(model_filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            if 'model_rng_seed' in hyperparams:
                hyperparams['rng_seed'] = hyperparams.pop('model_rng_seed')
            model = SSMFoundational(**hyperparams)
            model = eqx.tree_deserialise_leaves(f, model)
            state = eqx.nn.State(model)
        return model, state
    else:
        model = SSMFoundational(**hyperparams)
        state = eqx.nn.State(model)
        return model, state

def get_finetune_mode(wandb_pretrained_model_id, freeze_ssm):
    if wandb_pretrained_model_id is None:
        return 'scratch'
    elif freeze_ssm:
        return 'ft_mlp'
    else:
        return 'ft_all'