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



def create_optimizer_and_state(model, cfg):
    optimizer_cfg = cfg.optimizer
    lr_scheduler = lambda step: optimizer_cfg.lr
    opt_mode = getattr(optimizer_cfg, 'mode', 'all')
    
    def _get_param_labels(model):
        flat_model, treedef = jtu.tree_flatten_with_path(model)
        labels = []
        for path, leaf in flat_model:
            path_str = jtu.keystr(path)
            label = "regular"  # Default for params not otherwise specified

            # Check for S5 block parameters first
            if "ssm_blocks" in path_str:
                if "Lambda_re" in path_str or "Lambda_im" in path_str:
                    label = "ssm_A"
                elif "B" in path_str and "weight" in path_str:
                    label = "ssm_B"
                elif "C" in path_str and "weight" in path_str:
                    label = "ssm_C"
                elif ".D" in path_str or "log_step" in path_str:
                    label = "ssm_D_log_step"
            # Then check for encoder/decoder/embedding
            elif ("encoder" in path_str or "encoders") and "dropout" not in path_str: # Catches 'encoder' and 'encoders'
                if "weight" in path_str: label = "encoder_weight"
                elif "bias" in path_str: label = "encoder_bias"
            elif "decoder" in path_str and "dropout" not in path_str:
                if "weight" in path_str: label = "decoder_weight"
                elif "bias" in path_str: label = "decoder_bias"
            elif "context_embedding" in path_str:
                # This handles both eqx.nn.Embedding and a raw learnable array
                if "weight" in path_str or leaf.ndim > 1:
                    label = "embedding"
            
            labels.append(label)
        return treedef.unflatten(labels)

    if opt_mode == "all":
        label_fn = lambda x: jt.map_with_path(
                    lambda k, _: "ssm"
                    if getattr(k[-1], 'name', None) in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                    else ("b" if getattr(k[-1], 'name', None) in ["B"] else "regular"),
                    x
                )
        opt = optax.multi_transform(
            {
                "b": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.lr,
                                                            weight_decay=optimizer_cfg.weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=optimizer_cfg.lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.lr,
                                                                weight_decay=optimizer_cfg.weight_decay),
            },
            [label_fn(model)],
        )
    
    elif opt_mode == "s5":
        # Train all weights with different learning rates for SSM components
        label_fn = lambda x: jt.map_with_path(
                    lambda k, _: "ssm"
                    if getattr(k[-1], 'name', None) in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                    else ("b" if getattr(k[-1], 'name', None) in ["B"] else "regular"),
                    x
                )
        opt = optax.multi_transform(
            {
                "b": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.ssm_lr,
                                                            weight_decay=optimizer_cfg.weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=optimizer_cfg.ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.lr,
                                                                weight_decay=optimizer_cfg.weight_decay),
            },
            [label_fn(model)],
        )
        
    elif opt_mode == "freeze_a":
        # Freeze SSM parameters, train everything else
        label_fn = lambda x: jt.map_with_path(
                    lambda k, _: "frozen"
                    if getattr(k[-1], 'name', None) in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                    else "regular",
                    x
                )
        opt = optax.multi_transform(
            {
                "frozen": optax.set_to_zero(),  
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.lr,
                                                                weight_decay=optimizer_cfg.weight_decay),
            },
            [label_fn(model)],
        )
    
    elif opt_mode == "freeze_ssm":
        # Freeze all SSM block parameters, train everything else
        # This uses a path-based approach that works for both foundational and downstream models
        label_fn = lambda x: jt.map_with_path(
                lambda path, _: "frozen" if any(isinstance(p, str) and p == "ssm_blocks" for p in path) else "regular",
                x
            )
        
        opt = optax.multi_transform(
            {
                "frozen": optax.set_to_zero(),  # No updates for frozen parameters
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.lr,
                                                                weight_decay=optimizer_cfg.weight_decay),
            },
            [label_fn(model)],
    )
        
    elif opt_mode == "encoder_only":
        # Only train encoder, freeze everything else
        label_fn = lambda x: jt.map_with_path(
                    lambda k, _: "trainable"
                    if any(part.name == "encoder" for part in k if hasattr(part, 'name'))
                    else "frozen",
                    x
                )
        opt = optax.multi_transform(
            {
                "frozen": optax.set_to_zero(),  # No updates for frozen parameters
                "trainable": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.lr,
                                                                  weight_decay=optimizer_cfg.weight_decay),
            },
            [label_fn(model)],
        )
        
    elif opt_mode == "muP_SSM":
        base_ssm_io_dim = cfg.optimizer.base_ssm_io_dim  # Base model's H dimension
        base_ssm_dim = cfg.optimizer.base_ssm_dim        # Base model's P dimension
        ssm_io_dim = cfg.model.ssm_io_dim  # Current model's H dimension
        ssm_dim = cfg.model.ssm_dim        # Current model's P dimension
        lr_mult_A = ssm_io_dim / base_ssm_io_dim
        # η_B ~ sqrt(Nx / Nu) (where Nx is ssm_dim or P)
        lr_mult_B = jnp.sqrt(ssm_dim / base_ssm_dim) / jnp.sqrt(ssm_io_dim / base_ssm_io_dim)
        # η_C ~ 1 / (Nx * sqrt(Nu))
        lr_mult_C = (1 / (ssm_dim / base_ssm_dim)) / jnp.sqrt(ssm_io_dim / base_ssm_io_dim)
        # D and log_step are scaled by 1.0 as per standard practice [2, 3]
        lr_mult_D_log_step = 1.0
        
        width_mult = ssm_io_dim / base_ssm_io_dim
        lr_mult_encoder_weight = 1.0
        lr_mult_encoder_bias = width_mult
        lr_mult_decoder_weight = 1.0 / width_mult
        lr_mult_decoder_bias = 1.0
        lr_mult_embedding = 1.0
        
        param_labels = _get_param_labels(model)
        base_lr = optimizer_cfg.lr
        weight_decay = optimizer_cfg.weight_decay
        optimizer_map = {
            # µP-SSM parameters
            "s5_A": optax.adam(learning_rate=base_lr * lr_mult_A),
            "s5_B": optax.adam(learning_rate=base_lr * lr_mult_B),
            "s5_C": optax.adam(learning_rate=base_lr * lr_mult_C),
            "s5_D_log_step": optax.adam(learning_rate=base_lr * lr_mult_D_log_step),

            # Canonical µP parameters
            "encoder_weight": optax.adamw(learning_rate=base_lr * lr_mult_encoder_weight, weight_decay=weight_decay),
            "encoder_bias": optax.adamw(learning_rate=base_lr * lr_mult_encoder_bias, weight_decay=weight_decay),
            "decoder_weight": optax.adamw(learning_rate=base_lr * lr_mult_decoder_weight, weight_decay=weight_decay),
            "decoder_bias": optax.adamw(learning_rate=base_lr * lr_mult_decoder_bias, weight_decay=weight_decay),
            "embedding": optax.adamw(learning_rate=base_lr * lr_mult_embedding, weight_decay=weight_decay),

            # Fallback for any other parameters (e.g., layer norms)
            "regular": optax.adamw(learning_rate=base_lr, weight_decay=weight_decay),
        }
        opt = optax.multi_transform(optimizer_map, param_labels)

    else:
        raise ValueError(f"Unknown optimization mode: {opt_mode}. Valid modes are: 'all', 'freeze_ssm', 'encoder_only'")
    
    opt_state = opt.init(eqx.filter([model], eqx.is_array))
    return opt, opt_state, lr_scheduler