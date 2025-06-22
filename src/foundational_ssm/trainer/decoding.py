import wandb
import numpy as np 
from sklearn.metrics import r2_score 
from ..utils import save_model_wandb

def train_decoding(model, train_loader, train_tensors, val_tensors, optimizer, loss_fn, num_epochs, wandb_run_name=None, model_metadata=None, device='cuda'):
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