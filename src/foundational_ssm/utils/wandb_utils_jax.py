import wandb
import equinox as eqx
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


def transfer_foundational_to_downstream(foundational_model, downstream_model):
    """
    Transfer SSM blocks and decoder from a pretrained SSMFoundationalDecoder 
    to a SSMDownstreamDecoder.
    
    Args:
        foundational_model: Pretrained SSMFoundationalDecoder
        downstream_model: SSMDownstreamDecoder to receive the transferred parameters
    
    Returns:
        downstream_model: Updated downstream model with transferred parameters
        downstream_state: New state for the downstream model
    """
    # Transfer SSM blocks
    downstream_model = eqx.tree_at(
        lambda m: m.ssm_blocks, 
        downstream_model, 
        foundational_model.ssm_blocks
    )
    
    # Transfer decoder
    downstream_model = eqx.tree_at(
        lambda m: m.decoder, 
        downstream_model, 
        foundational_model.decoder
    )
    
    return downstream_model

def load_foundational_and_transfer_to_downstream(wandb_run_name, wandb_project, wandb_entity, downstream_model):
    """
    Load a pretrained foundational model from wandb and transfer its SSM blocks 
    and decoder to a downstream model.
    
    Args:
        wandb_run_name: Name of the wandb run containing the foundational model
        wandb_project: Wandb project name
        wandb_entity: Wandb entity name
        downstream_model: SSMDownstreamDecoder to receive the transferred parameters
    
    Returns:
        downstream_model: Updated downstream model with transferred parameters
    """
    api = wandb.Api()
    artifact_full_name = f"{wandb_entity}/{wandb_project}/{wandb_run_name}_best_model:latest"
    artifact = api.artifact(artifact_full_name, type="model")
    
    # Create temporary directory for download
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact.download(temp_dir)
        model_path = os.path.join(temp_dir, 'best_model.pt')
        
        # Load the foundational model
        foundational_model = load_model_wandb(model_path, SSMFoundationalDecoder)
        
        # Transfer parameters to downstream model
        downstream_model, downstream_state = transfer_foundational_to_downstream(foundational_model, downstream_model)
        
        return downstream_model, downstream_state
    
    
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
    
    
    