import wandb 
import torch 
import numpy as np 

def save_model_wandb(model, model_name, run):
    """Save model as wandb artifact."""
    # Save model locally
    model_path = f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    
    # Create artifact and add file
    model_artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description=f"{model_name}"
    )
    model_artifact.add_file(model_path)
    
    # Log artifact to wandb
    run.log_artifact(model_artifact)
    return model_path

def generate_and_save_predictions_wandb(model, data_tensor, name, run, device):
    """Generate predictions and save as wandb artifact."""
    model.eval()
    with torch.no_grad():
        predictions = model(data_tensor.to(device)).cpu().numpy()
    
    # Save predictions locally
    pred_path = f"{name}_predictions.npy"
    np.save(pred_path, predictions)
    
    # Create artifact and add file
    pred_artifact = wandb.Artifact(
        name=f"{name}_predictions", 
        type="predictions",
        description=f"Model predictions on {name} data"
    )
    pred_artifact.add_file(pred_path)
    
    # Log artifact to wandb
    run.log_artifact(pred_artifact)
    return predictions, pred_path