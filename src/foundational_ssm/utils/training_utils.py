"""
Shared training utilities for both pretrain and finetune scripts.
"""
import psutil
import wandb
import numpy as np
import jax.numpy as jnp
import jax.tree as jt
import jax.tree_util as jtu
import equinox as eqx
import optax
from omegaconf import OmegaConf
from .mup import scale_adam_by_mup

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

def get_param_labels(model):
    flat_model, treedef = jtu.tree_flatten_with_path(model)
    labels = []
    for path, leaf in flat_model:
        path_str = jtu.keystr(path)
        label = "regular"  # Default for params not otherwise specified

        # Check for S5 block parameters first
        if "ssm_blocks" in path_str:
            if "Lambda_re" in path_str or "Lambda_im" in path_str:
                label = "ssm_A"
            elif "B" in path_str:
                label = "ssm_B"
            elif "C" in path_str:
                label = "ssm_C"
            elif ".D" in path_str or "log_step" in path_str:
                label = "ssm_D_log_step"
            elif "glu" in path_str:
                if ("w1" in path_str) and ("weight" in path_str): label = "glu_w1"
                if ("w2" in path_str) and ("weight" in path_str): label = "glu_w2"
        # Then check for encoder/decoder/embedding
        elif ("encoder" in path_str or "encoders") and "dropout" not in path_str: # Catches 'encoder' and 'encoders'
            if "weight" in path_str: label = "encoder_weight"
            elif "bias" in path_str: label = "encoder_bias"
        elif "decoder" in path_str and "dropout" not in path_str:
            if "weight" in path_str: label = "decoder_weight"
            elif "bias" in path_str: label = "decoder_bias"
        elif "context_embedding" in path_str:
            # This handles both eqx.nn.Embedding and a raw learnable array
            if "weight" in path_str or leaf.ndim > 1: label = "embedding"
        
        labels.append(label)
    return treedef.unflatten(labels)

def create_optimizer_and_state(model, optimizer_cfg, model_cfg=None, return_lr_tree: bool = False, mup_meta=None):
    """
    Create an Optax optimizer and state, with optional return of a learning-rate tree per parameter.

    Args:
        model: Eqx Module (PyTree) with parameters.
        optimizer_cfg: omegaconf config with fields like lr, weight_decay, mode, ssm_lr, etc.
        model_cfg: optional model config used for muP multipliers.
        return_lr_tree: when True, also return a PyTree (same structure as `model`) whose
            array leaves are scalar jnp.ndarrays holding the LR applied to that parameter
            (0.0 for frozen params), and None for non-array leaves. This is useful for
            weighting gradients, e.g., in NTK computations.

    Returns:
        opt, opt_state, lr_scheduler[, lr_tree]
    """
    lr_scheduler = lambda step: optimizer_cfg.lr
    opt_mode = getattr(optimizer_cfg, 'mode', 'all')

    base_lr = optimizer_cfg.lr
    weight_decay = getattr(optimizer_cfg, 'weight_decay', 0.0)

    # Containers produced per mode
    transforms_map = {}
    label_tree = None
    label_to_lr = {}

    if opt_mode in ("all", "s5", "freeze_a", "freeze_ssm", "encoder_only"):
        # Build label tree using a path-based function once
        if opt_mode in ("all", "s5"):
            def path_to_label(path):
                last = path[-1]
                if getattr(last, 'name', None) in ["Lambda_re", "Lambda_im", "log_step", "norm"]:
                    return "ssm"
                elif getattr(last, 'name', None) in ["B"]:
                    return "b"
                else:
                    return "regular"
        elif opt_mode == "freeze_a":
            def path_to_label(path):
                last = path[-1]
                return "frozen" if getattr(last, 'name', None) in ["Lambda_re", "Lambda_im", "log_step", "norm"] else "regular"
        elif opt_mode == "freeze_ssm":
            def path_to_label(path):
                return "frozen" if any(hasattr(p, "name") and p.name == "ssm_blocks" for p in path) else "regular"
        else:  # encoder_only
            def path_to_label(path):
                return "trainable" if any(getattr(p, 'name', None) == 'encoder' for p in path) else "frozen"

        label_fn = lambda x: jt.map_with_path(lambda k, _: path_to_label(k), x)
        label_tree = label_fn(model)

        if opt_mode == "all":
            transforms_map = {
                "b": optax.inject_hyperparams(optax.adamw)(learning_rate=base_lr, weight_decay=weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=base_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=base_lr, weight_decay=weight_decay),
            }
            label_to_lr = {"b": base_lr, "ssm": base_lr, "regular": base_lr}
            
        elif opt_mode == "s5":
            ssm_lr = getattr(optimizer_cfg, 'ssm_lr', base_lr)
            transforms_map = {
                "b": optax.inject_hyperparams(optax.adamw)(learning_rate=ssm_lr, weight_decay=weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=base_lr, weight_decay=weight_decay),
            }
            label_to_lr = {"b": ssm_lr, "ssm": ssm_lr, "regular": base_lr}
        elif opt_mode == "freeze_a":
            transforms_map = {
                "frozen": optax.set_to_zero(),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=base_lr, weight_decay=weight_decay),
            }
            label_to_lr = {"frozen": 0.0, "regular": base_lr}
        elif opt_mode == "freeze_ssm":
            transforms_map = {
                "frozen": optax.set_to_zero(),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=base_lr, weight_decay=weight_decay),
            }
            label_to_lr = {"frozen": 0.0, "regular": base_lr}
        else:  # encoder_only
            transforms_map = {
                "frozen": optax.set_to_zero(),
                "trainable": optax.inject_hyperparams(optax.adamw)(learning_rate=base_lr, weight_decay=weight_decay),
            }
            label_to_lr = {"frozen": 0.0, "trainable": base_lr}

        opt = optax.multi_transform(transforms_map, [label_tree])

    elif opt_mode == "muP":
        assert mup_meta is not None, "mup_meta must be provided for muP optimization mode"
        opt = optax.chain(
            optax.adam(learning_rate=base_lr),
            scale_adam_by_mup(mup_meta, axis_convention=getattr(model_cfg, 'mup_axis_convention', 'torch') if model_cfg else "torch"),
        )

    else:
        raise ValueError(f"Unknown optimization mode: {opt_mode}. Valid modes are: 'all', 's5', 'freeze_a', 'freeze_ssm', 'encoder_only', 'muP'")

    # Initialize optimizer state; labels are passed as list-wrapped trees
    opt_state = opt.init(eqx.filter([model], eqx.is_array))

    if not return_lr_tree:
        return opt, opt_state, lr_scheduler

    # Build LR tree aligned with model structure: array leaves get scalar LR, others None
    def leaf_lr(leaf, label):
        if eqx.is_array(leaf):
            lr_val = label_to_lr.get(label, base_lr)
            return jnp.asarray(lr_val)
        else:
            return None

    lr_tree = jtu.tree_map(leaf_lr, model, label_tree)
    return opt, opt_state, lr_scheduler, lr_tree