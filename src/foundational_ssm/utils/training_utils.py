"""
Shared training utilities for both pretrain and finetune scripts.
"""

import psutil
import wandb
import numpy as np
import jax.numpy as jnp
import jax.tree as jt
import jax.tree_util as jtu
from jax.tree_util import GetAttrKey, SequenceKey
import equinox as eqx
import optax


def log_batch_metrics(data_load_time, batch_process_time, epoch, current_step):
    """Log timing and system metrics for a batch."""
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_percent = process.cpu_percent(interval=0.0)
    wandb.log(
        {
            "timing/data_load_sec": data_load_time,
            "timing/batch_process_sec": batch_process_time,
            "system/memory_rss_mb": mem_info.rss / 1e6,
            "system/cpu_percent": cpu_percent,
            "epoch": epoch,
        },
        step=current_step,
    )


def log_validation_metrics(val_start_time, val_end_time, current_step):
    """Log validation timing and resource metrics."""
    process = psutil.Process()
    mem_info = process.memory_info()
    wandb.log(
        {
            "val/epoch_time_sec": val_end_time - val_start_time,
            "val/memory_rss_mb": mem_info.rss / 1e6,
        },
        step=current_step,
    )


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
        print(
            f"Epoch {epoch}/{total_epochs}, Loss: {epoch_loss:.4f}, Val R2: {avg_r2_score:.4f}"
        )
    else:
        print(f"Epoch {epoch}/{total_epochs}, Loss: {epoch_loss:.4f}")


def compute_r2_by_groups(
    preds, targets, held_out_flags, prefix="train", current_step=None
):
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


def get_param_labels(model):
    flat_model, treedef = jt.flatten_with_path(model)
    labels = []
    for path, leaf in flat_model:
        path_str = jtu.keystr(path)
        label = "regular"  # Default for params not otherwise specified

        # Check for S5 block parameters first
        if "ssm_blocks" in path_str:
            if "Lambda_re" in path_str or "Lambda_im" in path_str or "log_step" in path_str:
                label = "ssm_A"
            elif "B" in path_str:
                label = "ssm_B"
            elif "C" in path_str:
                label = "ssm_C"
            elif "D" in path_str:
                label = "ssm_D"
            elif "glu" in path_str:
                if ("w1" in path_str) and ("weight" in path_str):
                    label = "glu_w1"
                if ("w2" in path_str) and ("weight" in path_str):
                    label = "glu_w2"
        # Then check for encoder/decoder/embedding
        elif (
            "encoder" in path_str or "encoders"
        ) and "dropout" not in path_str:  # Catches 'encoder' and 'encoders'
            if "weight" in path_str:
                label = "encoder_weight"
            elif "bias" in path_str:
                label = "encoder_bias"
        elif "decoder" in path_str and "dropout" not in path_str:
            if "weight" in path_str:
                label = "decoder_weight"
            elif "bias" in path_str:
                label = "decoder_bias"
        elif "context_embedding" in path_str:
            # This handles both eqx.nn.Embedding and a raw learnable array
            if "weight" in path_str or leaf.ndim > 1:
                label = "embedding"

        labels.append(label)
    return treedef.unflatten(labels)



def create_optimizer_and_state(
    model, optimizer_cfg
):
    """
    Create an Optax optimizer and state, with optional return of a learning-rate tree per parameter.

    Args:
        model: Eqx Module (PyTree) with parameters.
        optimizer_cfg: omegaconf config with fields like lr, weight_decay, mode, ssm_lr, etc.
        model_cfg: optional model config used for muP multipliers.

    Returns:
        opt, opt_state, lr_scheduler[, lr_tree]
    """
    def path_contains(path, *segments: str) -> bool:
        if not segments:
            return True
        keys = [
            k.name if isinstance(k, GetAttrKey)
            else k.idx if isinstance(k, SequenceKey)
            else str(k)
            for k in path
        ]
        pattern = list(segments)
        window = len(pattern)
        if window > len(keys):
            return False
        for start in range(len(keys) - window + 1):
            if keys[start : start + window] == pattern:
                return True
        return False
    
    def path_to_label(path):
        if (path_contains(path, "Lambda_re") or 
            path_contains(path, "Lambda_im") or 
            path_contains(path, "log_step") or 
            path_contains(path, "norm")
            ):
            return "ssm_A"
        elif (path_contains(path, "B") or
              path_contains(path, "C") or
              path_contains(path, "D") 
              ):
            return "ssm_BCD"
        elif (path_contains(path, "glu")):
            return "glu"
        elif (path_contains(path, "encoder")):
            return "encoder"
        elif (path_contains(path, "decoder")):
            return "decoder"
        else:
            return "other"
    
    
    opt_mode = getattr(optimizer_cfg, "mode", "all")
    opt_algorithm_cls = optimizer_cfg.algorithm_cls
    opt_algorithm_kwargs = optimizer_cfg.algorithm_kwargs
    base_lr = opt_algorithm_kwargs.learning_rate
    lr_scheduler = lambda step: base_lr
    
    assert opt_mode in ("all", "freeze_a", "encoder_only"), \
        f"{opt_mode} not implemented, select from (all, s5, freeze_a, freeze_ssm, encoder_only)"
    
    label_fn = lambda x: jt.map_with_path(lambda k, _: path_to_label(k), x)
    label_tree = label_fn(model)
    base_transform = optax.inject_hyperparams(getattr(optax, opt_algorithm_cls))(**opt_algorithm_kwargs)
    ssm_A_transform = optax.adam(learning_rate = base_lr) if opt_algorithm_cls  == "adamw" else base_transform 
    transforms_map = {
        "ssm_A"     : ssm_A_transform if opt_mode in ("all") else optax.set_to_zero(),
        "ssm_BCD"   : base_transform if opt_mode in ("freeze_a", "all") else optax.set_to_zero(),
        "glu"       : base_transform if opt_mode in ("freeze_a", "all") else optax.set_to_zero(),
        "encoder"   : base_transform,
        "decoder"   : base_transform if opt_mode in ("freeze_a", "all") else optax.set_to_zero(),
        "other"     : base_transform
    }

    opt = optax.multi_transform(transforms_map, [label_tree]) 
    opt_state = opt.init(eqx.filter([model], eqx.is_array))
    # we wrap parameters into non-callable pytree for equinox-multitransform compatibility 
    # ref: https://github.com/patrick-kidger/equinox/issues/794

    return opt, opt_state, lr_scheduler
