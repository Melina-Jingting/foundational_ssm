import os
import time
import logging
import multiprocessing as mp

import hydra
from omegaconf import OmegaConf, DictConfig

import numpy as np

# JAX & Equinox
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt 
import equinox as eqx
import optax

import wandb
import jax.profiler

# Foundational SSM core imports
from foundational_ssm.models import SSMFoundationalDecoder
from foundational_ssm.constants import DATASET_IDX_TO_GROUP_SHORT
from foundational_ssm.utils.training import mse_loss_foundational, make_step_foundational
from foundational_ssm.metrics import compute_r2_standard
from foundational_ssm.utils.wandb_utils_jax import (
    save_checkpoint_wandb,
    add_alias_to_checkpoint,
    resume_checkpoint_wandb,
)
from foundational_ssm.utils.training_utils import log_batch_metrics, track_batch_timing
from foundational_ssm.loaders import get_brainset_train_val_loaders

WARNING_LOG_FILE = "warnings.log"
logger = logging.getLogger(__name__)
# tempdir = "/cs/student/projects1/ml/2024/mlaimon/foundational_ssm/wandb_artifacts"
# os.environ['WANDB_CACHE_DIR'] = '/cs/student/projects1/ml/2024/mlaimon/foundational_ssm/wandb_cache'
    

def train_one_batch(batch, model, state, loss_fn, opt, opt_state, train_key, lr_scheduler, current_step, skip_timesteps=0):
    batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}
    inputs = batch["neural_input"]
    targets = batch["behavior_input"]
    dataset_group_idxs = batch["dataset_group_idx"]
    mask = batch["mask"]
    
    model, state, opt_state, loss_value, grads = make_step_foundational(
        model, state, inputs, targets, mask, train_key, dataset_group_idxs, loss_fn, opt, opt_state, skip_timesteps=skip_timesteps
    )
    
    current_lr = lr_scheduler(current_step)
    wandb.log({
        "train/loss": loss_value,
        "train/learning_rate": current_lr,
    }, step=current_step)
    return model, state, opt_state, loss_value

def train_one_epoch(train_loader, model, state, loss_fn, opt, opt_state, rng_key, lr_scheduler, current_step, epoch, skip_timesteps=0):    
    epoch_loss = 0
    batch_count = 0
    minute_start_time = time.time()
    prev_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        data_load_time = time.time() - prev_time
        batch_process_start = time.time()
        
        model, state, opt_state, loss_value = train_one_batch(
            batch, model, state, loss_fn, opt, opt_state, rng_key, lr_scheduler, current_step, skip_timesteps=skip_timesteps
        )
        batch_process_end = time.time()
        batch_process_time = batch_process_end - batch_process_start
        
        log_batch_metrics(data_load_time, batch_process_time, epoch, current_step)
        epoch_loss += loss_value
        batch_count += 1
        current_time = time.time()
        batch_count, minute_start_time = track_batch_timing(batch_count, minute_start_time, current_time, current_step)
        prev_time = time.time()
        current_step += 1
    
    wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch}, step=current_step)
    return model, state, opt_state, current_step, epoch_loss
    

def validate_one_epoch(val_loader, model, state, epoch, current_step, skip_timesteps=0):
    logger.info("Validating one epoch")
    metrics = {}  # New: store metrics per group
    all_preds = []
    all_targets = []
    all_dataset_group_idxs = []
    val_start_time = time.time()
    prev_time = time.time()
    for batch_idx, batch in enumerate(val_loader):
        data_load_time = time.time() - prev_time
        batch_process_start = time.time()
        batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}
        dataset_group_idxs = batch["dataset_group_idx"]
        inputs = batch["neural_input"]
        targets = batch["behavior_input"]
        
        mask = batch["mask"]
        mask = mask[..., None]
        preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0, None, None), out_axes=(0, None))(inputs, state, dataset_group_idxs, jr.PRNGKey(0), True)

        preds = preds[:, skip_timesteps:, :]
        targets = targets[:, skip_timesteps:, :]
        mask = mask[:, skip_timesteps:, :]

        all_preds.append(jnp.where(mask, preds, 0))
        all_targets.append(jnp.where(mask, targets, 0))
        all_dataset_group_idxs.append(dataset_group_idxs)
        batch_process_end = time.time()
        batch_process_time = batch_process_end - batch_process_start
        logger.info(f"Batch size: {inputs.shape[0]}, Batch time dimension: {inputs.shape[1]}, Batch {batch_idx} data load time: {data_load_time:.2f}s, batch process time: {batch_process_time:.2f}s")
        prev_time = time.time()

    all_preds = jnp.concatenate(all_preds, axis=0)
    all_targets = jnp.concatenate(all_targets, axis=0)
    all_dataset_group_idxs = jnp.concatenate(all_dataset_group_idxs, axis=0)
    unique_dataset_group_idxs = jnp.unique(all_dataset_group_idxs)
    for dataset_group_idx in unique_dataset_group_idxs:
        dataset_group_idx = int(dataset_group_idx)
        dataset_group_short_name = DATASET_IDX_TO_GROUP_SHORT[dataset_group_idx]
        dataset_group_mask = all_dataset_group_idxs == dataset_group_idx
        preds = all_preds[dataset_group_mask]
        targets = all_targets[dataset_group_mask]
        r2_score = compute_r2_standard(preds, targets)
        metrics[f"val/r2_{dataset_group_short_name}"] = float(r2_score)
    
    r2_score = compute_r2_standard(all_preds, all_targets)
    metrics['val/r2_avg'] = float(np.mean([metrics[key] for key in metrics.keys() if "r2" in key]))
    metrics['val/r2_all'] = float(r2_score)

    # Log validation timing and resources
    val_end_time = time.time()
    val_time = val_end_time - val_start_time
    metrics['val/time'] = val_time
    metrics['epoch'] = epoch

    wandb.log(metrics, step=current_step)
    return metrics


def create_optimizer_and_state(model, optimizer_cfg):
    lr_scheduler = lambda step: optimizer_cfg.lr
    ssm_fn = lambda x: jt.map_with_path(
                lambda k, _: "ssm"
                if  getattr(k[-1], 'name', None) in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if getattr(k[-1], 'name', None) in ["B"] else "regular"),
                x
            )
    opt = optax.multi_transform(
        {
            "none": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.ssm_lr,
                                                        weight_decay=optimizer_cfg.weight_decay),
            "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=optimizer_cfg.ssm_lr),
            "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=optimizer_cfg.lr,
                                                            weight_decay=optimizer_cfg.weight_decay),
        },
        [ssm_fn(model)],
    )
    opt_state = opt.init(eqx.filter([model], eqx.is_array))
    return opt, opt_state, lr_scheduler


@hydra.main(config_path="../configs", config_name="pretrain", version_base="1.3")
def main(cfg: DictConfig):
    mp.set_start_method("spawn", force=True)

    _, train_loader, _, val_loader = get_brainset_train_val_loaders(
            cfg.train_loader,
            cfg.val_loader,
            cfg.dataset_cfg
        )

    model_cfg = OmegaConf.load(cfg.model_cfg)
    model = SSMFoundationalDecoder(
            **model_cfg.model
        )
    state = eqx.nn.State(model)
    
    opt, opt_state, lr_scheduler = create_optimizer_and_state(model, model_cfg.optimizer)
    loss_fn = mse_loss_foundational
    
    dataset_name = cfg.dataset_cfg.split("/")[-1].split(".")[0]
    model_name = cfg.model_cfg.split("/")[-1].split(".")[0]
    run_name = f"{model_name}_{dataset_name}"
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    model, state, opt_state, start_epoch, current_step, checkpoint_metadata, best_r2_score \
        = resume_checkpoint_wandb(model, state, opt_state, config_dict, run_name, cfg.wandb.project, cfg.wandb.entity, cfg.wandb.resume_run_id)
    
    key = jr.PRNGKey(cfg.rng_seed)
    for epoch in range(start_epoch, cfg.training.epochs):
        key, train_key = jr.split(key)
        logger.info(f"Running training for epoch {epoch}") 
        model, state, opt_state, current_step, epoch_loss = train_one_epoch(
                train_loader, model, state, loss_fn, opt, opt_state, train_key, lr_scheduler, current_step, epoch, cfg.skip_timesteps
            )
        
        if epoch % cfg.training.checkpoint_every == 0:
            checkpoint_metadata = {
                'train_loss': epoch_loss
            }
            checkpoint_artifact = save_checkpoint_wandb(model, state, opt_state, epoch, current_step, checkpoint_metadata, run_name)
    
        if epoch % cfg.training.log_val_every == 0:
            add_alias_to_checkpoint(checkpoint_artifact, f'epoch_{epoch}')
            logger.info(f"Running validation for epoch {epoch}")
            metrics = validate_one_epoch(val_loader, model, state, epoch, current_step, cfg.skip_timesteps)
            current_r2_avg = metrics.get('val/r2_avg', 0.0)
            if current_r2_avg > best_r2_score:
                best_r2_score = current_r2_avg
                logger.info(f"New best R² score: {best_r2_score:.4f} at epoch {epoch}")
                add_alias_to_checkpoint(checkpoint_artifact,  'best', metadata = metrics)
        
        # if epoch % cfg.training.log_pred_and_activations_every == 0:
        #     logger.info(f"Logging predictions and activations for epoch {epoch}")
        #     log_predictions_and_activations(model, state, cfg, epoch, current_step, run_name)
    
    wandb.log({
        "final/best_r2_avg": best_r2_score
    }, step=current_step)
    jax.profiler.stop_trace()
    wandb.finish()
    
    logger.info("Training completed successfully")
            
if __name__ == "__main__":
    main()