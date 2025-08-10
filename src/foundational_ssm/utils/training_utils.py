"""
Shared training utilities for both pretrain and finetune scripts.
"""
import psutil
import wandb
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax.tree_util import tree_map
from omegaconf import OmegaConf

def log_batch_metrics(data_load_time, batch_process_time, epoch, current_step):
    """Log timing and system metrics for a batch."""
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_percent = process.cpu_percent(interval=0.0)
    wandb.log({
        "timing/data_load_sec": data_load_time,
        "timing/batch_process_sec": batch_process_time,
        "system/memory_rss_mb": mem_info.rss / 1e6,
        "system/cpu_percent": cpu_percent,
        "epoch": epoch,
    }, step=current_step)


def log_validation_metrics(val_start_time, val_end_time, current_step):
    """Log validation timing and resource metrics."""
    process = psutil.Process()
    mem_info = process.memory_info()
    wandb.log({
        "val/epoch_time_sec": val_end_time - val_start_time,
        "val/memory_rss_mb": mem_info.rss / 1e6,
    }, step=current_step)


def track_batch_timing(batch_count, minute_start_time, current_time, current_step):
    """Track and log batch timing metrics."""
    if current_time - minute_start_time >= 60:
        wandb.log({"timing/batches_per_minute": batch_count}, step=current_step)
        return 0, current_time
    return batch_count, minute_start_time


def setup_wandb_metrics():
    """Setup wandb metric definitions."""
    wandb.define_metric("epoch", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("timing/*", step_metric="step")
    wandb.define_metric("system/*", step_metric="step")


def log_epoch_summary(epoch, total_epochs, epoch_loss, avg_r2_score=None):
    """Log a summary of the epoch."""
    if avg_r2_score is not None:
        print(f"Epoch {epoch}/{total_epochs}, Loss: {epoch_loss:.4f}, Val R2: {avg_r2_score:.4f}")
    else:
        print(f"Epoch {epoch}/{total_epochs}, Loss: {epoch_loss:.4f}")


def compute_r2_by_groups(preds, targets, held_out_flags, prefix="train", current_step=None):
    """Compute R2 scores for held-in and held-out groups."""
    preds = jnp.concatenate(preds, axis=0)
    targets = jnp.concatenate(targets, axis=0)
    held_out_flags = np.array(held_out_flags)
    
    r2_scores = {}
    total_r2 = 0
    n_groups = 0
    
    for held_out_value, group_name in [(0, "heldin"), (1, "heldout")]:
        mask = held_out_flags == held_out_value
        if np.any(mask):
            from foundational_ssm.metrics import compute_r2_standard
            r2 = compute_r2_standard(preds[mask], targets[mask])
            if current_step is not None:
                wandb.log({f"{prefix}/r2_{group_name}": r2}, step=current_step)
            else:
                wandb.log({f"{prefix}/r2_{group_name}": r2})
            r2_scores[f"r2_{group_name}"] = float(r2)
            total_r2 += r2
            n_groups += 1
        else:
            r2_scores[f"r2_{group_name}"] = None
    
    r2_scores["r2_avg"] = float(total_r2 / n_groups)
    
    return r2_scores


def prepare_batch_for_training(batch):
    """Prepare batch data for training by converting to JAX arrays."""
    return {k: jax.device_put(np.array(v)) for k, v in batch.items()}


def extract_batch_data(batch):
    """Extract input, target, and held_out data from batch."""
    inputs = batch["neural_input"]
    targets = batch["behavior_input"]
    held_out_flags = batch["held_out"]
    return inputs, targets, held_out_flags 

def get_filter_spec(model, freeze_ssm: bool, freeze_mlp: bool):
    filter_spec = tree_map(eqx.is_inexact_array, model)
    if freeze_ssm:
        where = lambda fs: tuple(block.ssm for block in fs.ssm_blocks)
        filter_spec = eqx.tree_at(where, filter_spec, replace=(False,) * len(filter_spec.ssm_blocks))
    if freeze_mlp:
        where = lambda m: (m.decoder, m.encoders)
        filter_spec = eqx.tree_at(where, filter_spec, replace=(False, False))
    return filter_spec

