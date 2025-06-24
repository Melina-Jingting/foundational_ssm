import wandb
import numpy as np 
from sklearn.metrics import r2_score 
from ..utils.wandb_utils_torch import save_model_wandb
from ..utils.wandb_utils_jax import log_model_params_and_grads_wandb
from jax import random as jr
import jax.numpy as jnp
import jax
import equinox as eqx

def train_decoding_torch(model, train_loader, train_tensors, val_tensors, optimizer, loss_fn, num_epochs, wandb_run_name=None, model_metadata=None, device='cuda'):
    model.train()
    best_val_r2 = -np.inf

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            # Forward pass + gradient descent
            optimizer.zero_grad()
            pred = model(input)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        if wandb_run_name != None:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_epoch_loss
            })

        # Log validation metrics and save checkpoints
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            val_spikes, val_behavior = val_tensors
            val_pred = model(val_spikes.to(device))
            val_pred = val_pred.reshape(-1,2).cpu().detach().numpy()
            val_behavior = val_behavior.reshape(-1,2).cpu().detach().numpy()
            val_r2 = r2_score(val_pred, val_behavior)

            train_spikes, train_behavior = train_tensors
            train_pred = model(train_spikes.to(device))
            train_pred = train_pred.reshape(-1,2).cpu().detach().numpy()
            train_behavior = train_behavior.reshape(-1,2).cpu().detach().numpy()
            train_r2 = r2_score(train_pred, train_behavior)

            if wandb_run_name != None:
                wandb.log({
                    "epoch": epoch + 1,
                    "metrics/val.r2": val_r2,
                    "metrics/train.r2": train_r2
                })

                # Save checkpoint if it's the best model so far
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    save_model_wandb(model, wandb_run_name, model_metadata, wandb.run)

            print(f"epoch:{epoch} train loss:{avg_epoch_loss:.4f} val r2:{val_r2:.4f} train r2:{train_r2:.4f}")


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


def train_decoding_jax(
    model, 
    filter_spec,
    train_loader, 
    train_tensors,
    val_tensors,
    loss_fn, 
    state,
    opt,
    opt_state,
    epochs,
    log_every,
    key=jr.PRNGKey(0),
    wandb_run_name=None
):
    
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            key, subkey = jr.split(key)
            
            model, state, opt_state, loss_value, grads = make_step(
                model,
                filter_spec,
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
            key, train_key, val_key = jr.split(key, 3)
            
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
                    "metrics/train.r2": float(train_r2),
                    "metrics/val.r2": float(val_r2),
                }
                wandb.log(wandb_log_dict)
                log_model_params_and_grads_wandb(model, grads)

    return model