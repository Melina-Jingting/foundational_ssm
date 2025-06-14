import wandb 
import torch 
import numpy as np 
import h5py
import os

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

def generate_and_save_activations_wandb(model, data_tensor, project_name=None, run_id=None, save_dir="./outputs"):
    """Generate predictions, capture activations, and save as H5 file."""
    os.makedirs(save_dir, exist_ok=True)
    h5_path = os.path.join(save_dir, f"predictions_and_activations.h5")
    spikes, behavior = data_tensor
    
    # Create hooks to capture activations
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks for SSM blocks
    for i, layer in enumerate(model.ssm_block):
        layer.register_forward_hook(get_activation(f'ssm_block_{i}'))
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(spikes)
    
    # Save all data to H5 file
    with h5py.File(h5_path, 'w') as f:
        # Store predictions
        f.create_dataset('predictions', data=predictions.cpu().numpy())
        
        # Store activations
        act_grp = f.create_group('activations')
        for name, tensor in activations.items():
            act_grp.create_dataset(name, data=tensor.cpu().numpy())
    
    # Create and log artifact
    pred_artifact = wandb.Artifact(
        name=f"predictions_and_activations",
        type="predictions",
        description=f"Model predictions and activations"
    )
    pred_artifact.add_file(h5_path)
    
    # Log artifact to wandb
    if project_name is not None and run_id is not None:
        wandb.init(project=project_name, id=run_id, resume="allow")
    
    wandb.run.log_artifact(pred_artifact)
    return predictions, h5_path, activations
