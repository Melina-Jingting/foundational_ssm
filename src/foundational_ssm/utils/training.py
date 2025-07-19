import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jax.tree_util import tree_map

# Filter spec for freezing parts of the model
def get_filter_spec(model, freeze_ssm: bool, freeze_mlp: bool):
    filter_spec = tree_map(eqx.is_inexact_array, model)
    if freeze_ssm:
        where = lambda fs: tuple(block.ssm for block in fs.ssm_blocks)
        filter_spec = eqx.tree_at(where, filter_spec, replace=(False,) * len(filter_spec.ssm_blocks))
    if freeze_mlp:
        where = lambda m: (m.decoder, m.encoders)
        filter_spec = eqx.tree_at(where, filter_spec, replace=(False, False))
    return filter_spec


@eqx.filter_jit
def predict_batch(model, state, inputs, key, dataset_group_idx):
    """Predict on a batch of inputs using JAX's vmap"""
    batch_keys = jr.split(key, inputs.shape[0])
    preds, _ = jax.vmap(model.call_with_activations, axis_name="batch", in_axes=(0, None, 0, None), out_axes=(0, None))(inputs, state, batch_keys, dataset_group_idx)
    return preds


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def mse_loss_foundational(model_params, model_static, state, inputs, targets, mask, dataset_group_idxs, key):
    """MSE loss for foundational model (takes dataset_group_idx and mask)"""
    model = eqx.combine(model_params, model_static)
    batch_keys = jr.split(key, inputs.shape[0])
    preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0, 0), out_axes=(0, None))(inputs, state, batch_keys, dataset_group_idxs)
    
    # Only compute loss on unmasked elements
    squared_error = (preds - targets) ** 2
    mask = mask[..., None]
    masked_squared_error = jnp.where(mask, squared_error, 0.0)
    mse = masked_squared_error.sum() / mask.sum()
    return (mse, state)


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def mse_loss_downstream(model_params, model_static, state, inputs, targets, key):
    """MSE loss for downstream model (no dataset_group_idx)"""
    model = eqx.combine(model_params, model_static)
    batch_keys = jr.split(key, inputs.shape[0])
    preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None))(inputs, state, batch_keys)
    mse = jnp.mean((preds - targets) ** 2)
    return (mse, state)


# Backward compatibility
mse_loss = mse_loss_foundational


@eqx.filter_jit
def make_step_foundational(model, state, filter_spec, inputs, targets, mask, loss_fn, opt, opt_state, key, dataset_group_idxs):
    """Make step for foundational model (takes dataset_group_idx and mask)"""
    model_params, model_static = eqx.partition(model, filter_spec)
    (value, state), grads = loss_fn(model_params, model_static, state, inputs, targets, mask, dataset_group_idxs, key)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, value, grads


@eqx.filter_jit
def make_step_downstream(model, state, filter_spec, inputs, targets, loss_fn, opt, opt_state, key):
    """Make step for downstream model (no dataset_group_idx)"""
    model_params, model_static = eqx.partition(model, filter_spec)
    (value, state), grads = loss_fn(model_params, model_static, state, inputs, targets, key)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, value, grads


# Backward compatibility
def make_step(model, state, filter_spec, inputs, targets, loss_fn, opt, opt_state, key, dataset_group_idxs=None):
    """Make step with automatic detection of model type based on dataset_group_idx parameter"""
    if dataset_group_idxs is not None:
        return make_step_foundational(model, state, filter_spec, inputs, targets, loss_fn, opt, opt_state, key, dataset_group_idxs)
    else:
        return make_step_downstream(model, state, filter_spec, inputs, targets, loss_fn, opt, opt_state, key)




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



def get_finetune_mode(pretrained_run_name, freeze_ssm):
    if pretrained_run_name is None:
        return 'scratch'
    elif freeze_ssm:
        return 'ft_mlp'
    else:
        return 'ft_all'