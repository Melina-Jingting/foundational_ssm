import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jax.tree_util import tree_map
from foundational_ssm.constants.constants import get_dataset_group_weights_array


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
def predict_batch(model, state, inputs, key, dataset_group_idx, inference=True):
    """Predict on a batch of inputs using JAX's vmap"""
    batch_keys = jr.split(key, inputs.shape[0])
    preds, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0, None), out_axes=(0, None))(inputs, state, dataset_group_idx, batch_keys)
    return preds



@eqx.filter_jit
def make_step_downstream(model, state, inputs, targets, mask, key, filter_spec, loss_fn, opt, opt_state, skip_timesteps):
    """Make step for downstream model (no dataset_group_idx)"""
    model_params, model_static = eqx.partition(model, filter_spec)
    (value, state), grads = loss_fn(model_params, model_static, state, inputs, targets, mask, key, skip_timesteps)
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



def get_finetune_mode(pretrained_run_name, freeze_ssm):
    if pretrained_run_name is None:
        return 'scratch'
    elif freeze_ssm:
        return 'ft_mlp'
    else:
        return 'ft_all'