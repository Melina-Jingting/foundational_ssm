import torch
import numpy as np
import wandb
from omegaconf import OmegaConf
from foundational_ssm.metrics import ValidationMetrics
from utils import move_to_gpu
from plotting import plot_training_curves



def train(model, optimizer, train_loader, val_loader, loss_fn, config):
    
    # Check if wandb is already initialized by a sweep
    if wandb.run is None:
        wandb.init(
            project=config.wandb.project,
            name=config.wandb.run_name,
            tags=config.wandb.tags,
            config=OmegaConf.to_container(config, resolve=True)
        )
        wandb_initialized_here = True
    else:
        wandb_initialized_here = False
        wandb.config.update(OmegaConf.to_container(config, resolve=True), allow_val_change=True)
    wandb.watch(model, log="all", log_freq=config.wandb.log_freq)
    
    
    validator = ValidationMetrics(config.device)
    
    # Tracking metrics
    train_losses = []
    val_metrics_history = []

    # Training loop
    for epoch in range(config.training.num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
               

        # Training steps
        for batch_idx, batch in enumerate(train_loader):
            batch = move_to_gpu(batch, config.device)
            loss = training_step(batch, model, optimizer, loss_fn, config.device, config.training.mask_prob)
            train_losses.append(loss.item())
            epoch_loss += loss.item()
            num_batches += 1
            
            wandb.log({
                "batch/loss": loss.item(),
                "batch/step": epoch * len(train_loader) + batch_idx
            })
            
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        wandb.log({
            "train/loss": avg_epoch_loss,
            "train/epoch": epoch + 1
        })

        # Evaluate on validation set periodically
        if (epoch + 1) % config.wandb.log_freq == 0 or epoch == config.training.num_epochs - 1:
            val_metrics = validator.compute_metrics(val_loader, model)
            val_metrics_history.append(val_metrics)
            
            # Log overall validation metrics
            wandb.log({
                "val/encoding_loss": val_metrics["encoding_loss"],
                "val/decoding_loss": val_metrics["decoding_loss"],
                "val/combined_loss": val_metrics["combined_loss"],
                "val/behavior_r2": val_metrics["behavior_r2"],
                "val/epoch": epoch + 1
            })
            
            # Log per-subject metrics
            for subj_id, subj_metrics in val_metrics["per_subject"].items():
                wandb.log({
                    f"val/subject/{subj_id}/encoding_loss": subj_metrics["encoding_loss"],
                    f"val/subject/{subj_id}/decoding_loss": subj_metrics["decoding_loss"],
                    f"val/subject/{subj_id}/combined_loss": subj_metrics["combined_loss"],
                    f"val/subject/{subj_id}/behavior_r2": subj_metrics["behavior_r2"],
                    "epoch": epoch + 1
                })
            
            # Print validation summary
            print(f"Epoch {epoch+1}/{config.training.num_epochs} | " +
                  f"Train Loss: {avg_epoch_loss:.4f} | " +
                  f"Val Decoding Loss: {val_metrics['decoding_loss']:.4f} | " +
                  f"Val Behavior R²: {val_metrics['behavior_r2']:.4f} | " +
                  f"Val Encoding Loss: {val_metrics['encoding_loss']:.4f}")
            
    # Close wandb run
    if wandb_initialized_here:
        wandb.finish()
    return train_losses, val_metrics_history


def training_step(batch, model, optimizer, loss_fn, device, mask_prob=0.5):
    
    # 1. Prepare the masks
    batch_size = batch["neural_input"].shape[0]
    neural_mask = torch.ones(batch_size, device=device)
    behavior_mask = torch.ones(batch_size, device=device)
    for i in range(batch_size):
        mask_type = np.random.choice(['neural', 'behavior', 'none'], p=[mask_prob/2, mask_prob/2, 1-mask_prob])
        if mask_type == 'neural':
            neural_mask[i] = 0.0
        elif mask_type == 'behavior':
            behavior_mask[i] = 0.0
    
    # 2. Forward pass
    optimizer.zero_grad()                  
    pred = model(
        **batch,
        neural_mask=neural_mask,
        behavior_mask=behavior_mask
    )  
    target = batch
    
    # 3. Compute loss
    loss = loss_fn(pred, target)         
    loss.backward()                     
    optimizer.step()    
                       
    return loss