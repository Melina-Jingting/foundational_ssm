import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import wandb
from foundational_ssm.utils.wandb_utils_jax import load_checkpoint_wandb, save_model_wandb, load_foundational_and_transfer_to_downstream, save_checkpoint_wandb
from foundational_ssm.models.decoders import SSMDownstreamDecoder
from foundational_ssm.utils.training import get_filter_spec, mse_loss_downstream, make_step_downstream, get_finetune_mode
from foundational_ssm.data_utils.loaders import get_nlb_train_val_loaders
import optax
from foundational_ssm.metrics import compute_r2_standard
import numpy as np
from foundational_ssm.utils.training_utils import (
    log_batch_metrics, log_validation_metrics, track_batch_timing, 
    setup_wandb_metrics, log_epoch_summary, compute_r2_by_groups,
    prepare_batch_for_training, extract_batch_data
)
import time
import multiprocessing as mp



def train_one_batch(batch, model, state, filter_spec, loss_fn, opt, opt_state, train_key, current_step):
    """Train on a single batch."""
    batch = prepare_batch_for_training(batch)
    inputs, targets, _ = extract_batch_data(batch)
    key, subkey = jr.split(train_key)
    model, state, opt_state, loss_value, grads = make_step_downstream(
        model, state, filter_spec, inputs, targets, 
        loss_fn, opt, opt_state, subkey
    )
    wandb.log({"train/loss": loss_value}, step=current_step)
    return model, state, opt_state, loss_value


def train_one_epoch(train_loader, model, state, filter_spec, loss_fn, opt, opt_state, train_key, epoch, current_step):
    """Train for one epoch."""
    epoch_loss = 0
    batch_count = 0
    minute_start_time = time.time()
    prev_time = time.time()
    
    train_preds = []
    train_targets = []
    train_held_out = []
    
    for batch_idx, batch in enumerate(train_loader):
        data_load_time = time.time() - prev_time
        batch_process_start = time.time()
        
        model, state, opt_state, loss_value = train_one_batch(
            batch, model, state, filter_spec, loss_fn, opt, opt_state, train_key, current_step
        )
        
        batch_process_end = time.time()
        batch_process_time = batch_process_end - batch_process_start
        log_batch_metrics(data_load_time, batch_process_time, epoch, current_step)
        
        epoch_loss += loss_value
        batch_count += 1
        
        # Get model predictions for R2
        inputs, targets, held_out_flags = extract_batch_data(batch)
        key, subkey = jr.split(train_key)
        batch_keys = jr.split(subkey, inputs.shape[0])
        preds, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None))(inputs, state, batch_keys)
        train_preds.append(preds)
        train_targets.append(targets)
        train_held_out.extend(held_out_flags)
        
        current_time = time.time()
        batch_count, minute_start_time = track_batch_timing(batch_count, minute_start_time, current_time, current_step)
        prev_time = time.time()
        current_step += 1
    
    wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch}, step=current_step)
    return model, state, opt_state, epoch_loss, train_preds, train_targets, train_held_out, current_step


def validate_one_epoch(val_loader, model, state, val_key, epoch, current_step):
    """Validate for one epoch."""
    val_preds = []
    val_targets = []
    val_held_out = []
    val_start_time = time.time()
    
    for batch in val_loader:
        inputs, targets, held_out_flags = extract_batch_data(batch)
        key, subkey = jr.split(val_key)
        batch_keys = jr.split(subkey, inputs.shape[0])
        preds, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None))(inputs, state, batch_keys)
        val_preds.append(preds)
        val_targets.append(targets)
        val_held_out.extend(held_out_flags)
    
    metrics = compute_r2_by_groups(val_preds, val_targets, val_held_out, prefix="val", current_step=current_step)
    
    # Log validation timing and resources
    val_end_time = time.time()
    val_time = val_end_time - val_start_time
    metrics['val_time'] = val_time
    
    return metrics


def compute_train_r2_scores(train_preds, train_targets, train_held_out, current_step):
    """Compute R2 scores for training data."""
    r2_scores = compute_r2_by_groups(train_preds, train_targets, train_held_out, prefix="train", current_step=current_step)
    return r2_scores


def setup_model_and_optimizer(cfg):
    """Setup model, optimizer, and related components."""
    model = SSMDownstreamDecoder(**cfg.model)
    
    # Transfer parameters from pretrained foundational model if specified
    if cfg.finetune.enabled:
        print(f"Transferring parameters from foundational model: {cfg.finetune.run_name}")
        model = load_foundational_and_transfer_to_downstream(
            wandb_run_name=cfg.finetune.run_name,   
            wandb_project=cfg.finetune.project,
            wandb_entity=cfg.wandb.entity,
            downstream_model=model
        )
        print("Successfully transferred SSM blocks and decoder from foundational model")
    
    state = eqx.nn.State(model)
    
    lr_scheduler = lambda step: cfg.optimizer.lr
    filter_spec = get_filter_spec(model, freeze_ssm=cfg.filter_spec.freeze_ssm, freeze_mlp=cfg.filter_spec.freeze_mlp)
    
    opt = optax.chain(
        optax.adamw(learning_rate=lr_scheduler, weight_decay=cfg.optimizer.weight_decay)
    )
    opt_state = opt.init(eqx.filter(model, filter_spec))
    
    return model, state, opt_state, filter_spec, opt


def load_checkpoint_if_specified(cfg, model, state, opt_state, run_name):
    """Load checkpoint if specified in config."""
    if cfg.wandb.resume_run_id is not None: 
        model, state, opt_state, last_epoch, current_step, checkpoint_metadata = load_checkpoint_wandb(
            path=None,  # path is ignored, wandb is used
            model_template=model,
            state_template=state,
            opt_state_template=opt_state,
            wandb_run_name=run_name,
            wandb_project=cfg.wandb.project,
            wandb_entity=cfg.wandb.entity,
        )
        return model, state, opt_state, last_epoch, current_step
    return model, state, opt_state, 0, 0


def setup_wandb(cfg, run_name):
    """Setup wandb logging."""
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project=cfg.wandb.project, name=run_name, config=config_dict)  # type: ignore
    setup_wandb_metrics()


@hydra.main(config_path="../configs", config_name="finetune_rtt", version_base="1.3")
def main(cfg: DictConfig):
    mp.set_start_method("spawn", force=True)
    # Setup data loaders
    train_dataset, train_loader, val_dataset, val_loader = get_nlb_train_val_loaders(
        **cfg.dataset
    )
    
    # Setup random keys
    key = jr.PRNGKey(cfg.rng_seed)
    train_key, val_key = jr.split(key, 2)
    
    # Setup model and optimizer
    model, state, opt_state, filter_spec, opt = setup_model_and_optimizer(cfg)
    
    # Load checkpoint if specified
    # model, state, opt_state, start_epoch, current_step = load_checkpoint_if_specified(cfg, model, state, opt_state, run_name)
    start_epoch = 0
    
    # Setup loss function
    loss_fn = mse_loss_downstream
    
    # Setup run name and wandb
    finetune_mode = get_finetune_mode(cfg.finetune.run_name, cfg.filter_spec.freeze_ssm)
    run_name = f'{finetune_mode}_holdout-{cfg.dataset.holdout_angles}'
    setup_wandb(cfg, run_name)
    
    # Training loop
    best_heldout_r2 = -float('inf')
    current_step = 0
    for epoch in range(start_epoch, cfg.training.epochs):
        # Train one epoch
        model, state, opt_state, epoch_loss, train_preds, train_targets, train_held_out, current_step = train_one_epoch(
            train_loader, model, state, filter_spec, loss_fn, opt, opt_state, train_key, epoch, current_step
        )
        
        # Log training metrics
        if epoch % cfg.training.checkpoint_every == 0:
            wandb.log({"epoch": epoch})
            train_r2_scores = compute_train_r2_scores(train_preds, train_targets, train_held_out, current_step)
            
            # Validate
            val_r2_scores = validate_one_epoch(val_loader, model, state, val_key, epoch, current_step)
            
            # Save best model
            if val_r2_scores["r2_heldout"] is not None and val_r2_scores["r2_heldout"] > best_heldout_r2:
                best_heldout_r2 = val_r2_scores["r2_heldout"]
                save_model_wandb(model, run_name, OmegaConf.to_container(cfg.model), wandb.run)
            
            # Create metadata for checkpoint
            metadata = {
                'train_loss': epoch_loss,
                'best_r2_score': best_heldout_r2
            }
            metadata.update(train_r2_scores)
            metadata.update(val_r2_scores)
            log_epoch_summary(epoch, cfg.training.epochs, epoch_loss, val_r2_scores["r2_avg"])
            save_checkpoint_wandb(model, state, opt_state, epoch, current_step, metadata, run_name)
            
    
    wandb.finish()


if __name__ == "__main__":
    main()