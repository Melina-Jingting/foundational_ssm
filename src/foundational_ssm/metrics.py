import torch
import torch.nn.functional as F
import jax.numpy as jnp


def r2_score_torch(y_pred, y_true):
    y_true_mean = torch.mean(y_true, dim=0, keepdim=True)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_total
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



