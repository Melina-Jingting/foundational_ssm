import wandb
import equinox as eqx
import jax
from jax.tree_util import tree_flatten_with_path
import json
import os
import tempfile
import shutil
from foundational_ssm.models.decoders import SSMFoundationalDecoder
from .h5py_to_dict import h5_to_dict 
import h5py

def log_model_params_and_grads_wandb(model, grads=None):
    model_params = tree_flatten_with_path(model)[0] 
    grads = tree_flatten_with_path(grads)[0] if grads is not None else []
    for path, value in model_params:
        if eqx.is_array(value):
            full_path = "".join(str(p) for p in path)
            hist = wandb.Histogram(value.flatten())
            wandb.log({
                f"params/{full_path}": hist
            })
    for path, value in grads:
        if eqx.is_array(value):
            full_path = "".join(str(p) for p in path)
            hist = wandb.Histogram(value.flatten())
            wandb.log({
                f"grads/{full_path}": hist
            })
       

def add_alias_to_checkpoint(checkpoint_artifact, alias, metadata=None):
    checkpoint_artifact.wait()
    if alias not in checkpoint_artifact.aliases:
        checkpoint_artifact.aliases.append(alias)
    if metadata is not None:
        checkpoint_artifact.metadata.update(metadata)
    checkpoint_artifact.save()

def load_model_wandb(filename, modelClass):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        # Handle the case where hyperparams might be a string or dict
        if isinstance(hyperparams, str):
            hyperparams = json.loads(hyperparams)
        if 'model_rng_seed' in hyperparams:
            hyperparams['rng_seed'] = hyperparams.pop('model_rng_seed')
            hyperparams['ssm_num_layers'] = 4
        model = modelClass(**hyperparams)
        return eqx.tree_deserialise_leaves(f, model)
    
def save_checkpoint_wandb(model, state, opt_state, metadata):
    """Save model, optimizer state, epoch, and step to a checkpoint file."""
    run_name = wandb.run.name
    
    try:
        current_run = wandb.run
        entity = getattr(current_run, 'entity')
        project = getattr(current_run, 'project')
        api = wandb.Api(overrides={'project': project, 'entity': entity})
        for v in api.artifacts(type_name='checkpoint', name=f'{run_name}_checkpoint'):
            if len(v.aliases) == 0:
                v.delete()
    except Exception as e:
        print(f"No previous checkpoint to delete or deletion failed: {e}")
    
    os.makedirs(f"wandb_artifacts/{run_name}", exist_ok=True)
    eqx.tree_serialise_leaves(f"wandb_artifacts/{run_name}/model.ckpt", model)
    eqx.tree_serialise_leaves(f"wandb_artifacts/{run_name}/state.ckpt", state)
    eqx.tree_serialise_leaves(f"wandb_artifacts/{run_name}/opt_state.ckpt", opt_state)

    artifact = wandb.Artifact(
        name=f'{run_name}_checkpoint',  
        type="checkpoint",                
        description=f"Checkpoint at epoch {metadata['epoch']}",
        metadata=metadata
    )
    artifact.add_file(f"wandb_artifacts/{run_name}/model.ckpt")
    artifact.add_file(f"wandb_artifacts/{run_name}/state.ckpt")
    artifact.add_file(f"wandb_artifacts/{run_name}/opt_state.ckpt")

    wandb.log_artifact(artifact)
    return artifact

def load_checkpoint_wandb(model_template, state_template, opt_state_template, artifact_full_name):
    """Load model, optimizer state, epoch, and step from a checkpoint file."""
    api = wandb.Api()
    try:
        artifact = api.artifact(artifact_full_name, type="checkpoint")
    except Exception as e:
        raise FileNotFoundError(f"Could not find checkpoint artifact: {artifact_full_name}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact.download(temp_dir)
        model = eqx.tree_deserialise_leaves(os.path.join(temp_dir, "model.ckpt"), model_template)
        state = eqx.tree_deserialise_leaves(os.path.join(temp_dir, "state.ckpt"), state_template)
        try:            
            opt_state = eqx.tree_deserialise_leaves(os.path.join(temp_dir, "opt_state.ckpt"), opt_state_template)
        except:
            opt_state = opt_state_template

    meta = artifact.metadata
    return model, state, opt_state, meta
    
    
def load_h5_artifact_with_tempdir(artifact_name, artifact_type='predictions_and_activations'):
    """Load a wandb artifact using a temporary directory and convert to dict
    
    Parameters
    ----------
    artifact_name : str
        Full artifact name (e.g., 'melinajingting-ucl/foundational_ssm_pretrain_decoding/possm_dataset_l1_d64_predictions_and_activations_epoch_300:v0')
    artifact_type : str
        Type of artifact to load
    
    Returns
    -------
    dict
        Dictionary containing the artifact data
    """
    if wandb.run is None:
        wandb.init()
    
    artifact = wandb.run.use_artifact(artifact_name, type=artifact_type)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = artifact.download(temp_dir)
        
        h5_files = [f for f in os.listdir(temp_dir) if f.endswith('.h5')]
        if not h5_files:
            print(f"Available files in {temp_dir}: {os.listdir(temp_dir)}")
            raise FileNotFoundError(f"No H5 file found in {temp_dir}. Available files: {os.listdir(temp_dir)}")
        
        h5_path = os.path.join(temp_dir, h5_files[0])
        
        with h5py.File(h5_path, 'r') as h5obj:
            data_dict = h5_to_dict(h5obj)
        
        return data_dict
    
    
    
def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: An Equinox model
        
    Returns:
        dict: Dictionary containing parameter counts by type and total
    """
    total_params = 0
    param_counts = {}
    
    # Helper function to process each leaf
    def count_leaf_params(path_tuple, leaf):
        if eqx.is_array(leaf):
            param_count = leaf.size
            # Create path string for parameter grouping
            path_str = ".".join(str(p) for p in path_tuple)
            
            # Group by top-level module
            top_level = path_str.split('.')[0] if '.' in path_str else path_str
            if top_level not in param_counts:
                param_counts[top_level] = 0
            param_counts[top_level] += param_count
            
            return param_count
        return 0
    
    # Flatten model and count parameters
    flat_params = jax.tree_util.tree_flatten_with_path(model)[0]
    for path, param in flat_params:
        total_params += count_leaf_params(path, param)
    
    return total_params

def log_model_parameters(model, step=None):
    """
    Count and log model parameters to WandB.
    
    Args:
        model: An Equinox model
        step: Optional step for WandB logging
    """
    param_counts = count_parameters(model)
    
    # Format parameter counts for better readability
    formatted_counts = {}
    for module, count in param_counts.items():
        if module == 'total':
            formatted_counts['model/total_parameters'] = count
        else:
            formatted_counts[f'model/parameters/{module}'] = count
    
    # Also log parameter count in millions for easier interpretation
    formatted_counts['model/total_parameters_M'] = param_counts['total'] / 1e6
    
    # Log to WandB
    wandb.log(formatted_counts, step=step)
    
    # Also print to console for immediate feedback
    print(f"Model Parameter Count:")
    for module, count in param_counts.items():
        if module == 'total':
            print(f"  Total: {count:,} ({count/1e6:.2f}M)")
    
    return param_counts