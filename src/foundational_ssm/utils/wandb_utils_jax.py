import wandb
import equinox as eqx
from jax.tree_util import tree_flatten_with_path
import json
import os
import tempfile
import shutil
from foundational_ssm.models.decoders import SSMFoundationalDecoder
from .h5py_to_dict import h5_to_dict 
import glob
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

def load_model_and_state_wandb(wandb_pretrained_model_id=None, hyperparams=None, model_class=SSMFoundationalDecoder):
    """
    either loads a model from wandb or creates a new model from hyperparams
    Args:
        wandb_pretrained_model_id: wandb artifact id of the model to load
        hyperparams: dict of hyperparams to create a new model
    Returns:
        model (SSMFoundational): Loaded model or None if not specified.
    """
    if wandb_pretrained_model_id is not None:
        api = wandb.Api()
        model_artifact = api.artifact(wandb_pretrained_model_id, type="model")
        with tempfile.TemporaryDirectory() as temp_dir:
            model_artifact_dir = model_artifact.download(temp_dir)
            eqx_files = [f for f in os.listdir(model_artifact_dir) if f.endswith('.eqx')]
            if not eqx_files:
                raise FileNotFoundError(
                    f"No .eqx model file found in artifact directory {model_artifact_dir}. "
                    f"Files: {os.listdir(model_artifact_dir)}"
                )
            model_filename = os.path.join(model_artifact_dir, eqx_files[0])
            with open(model_filename, "rb") as f:
                hyperparams = json.loads(f.readline().decode())
                if 'model_rng_seed' in hyperparams:
                    hyperparams['rng_seed'] = hyperparams.pop('model_rng_seed')
                model = SSMFoundationalDecoder(**hyperparams)
                model = eqx.tree_deserialise_leaves(f, model)
                state = eqx.nn.State(model)
            return model, state
    else:
        model = SSMFoundationalDecoder(**hyperparams)
        state = eqx.nn.State(model)
        return model, state            
            
def save_best_model_wandb(model, run_name, model_metadata, metrics):
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.eqx', delete=False) as temp_file:
        model_path = temp_file.name
        with open(model_path, "wb") as f:
            hyperparam_str = json.dumps(model_metadata)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, model)
    
    try:
        model_artifact = wandb.Artifact(
            name=f"{run_name}_best_model",
            type="model",
            description=f"best model for {run_name}",
            metadata=metrics
        )
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)
        return model_path
    finally:
        # Clean up temporary file
        if os.path.exists(model_path):
            os.unlink(model_path)

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
    
def save_checkpoint_wandb(model, state, opt_state, epoch, step, metadata, run_name):
    """Save model, optimizer state, epoch, and step to a checkpoint file."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.ckpt', delete=False) as temp_file:
        path = temp_file.name
        with open(path, 'wb') as f:
            # Write metadata as JSON in the first line
            meta = json.dumps({'epoch': epoch, 'step': step})
            f.write((meta + '\n').encode())
            eqx.tree_serialise_leaves(f, model)
            eqx.tree_serialise_leaves(f, state)
            eqx.tree_serialise_leaves(f, opt_state)
    
    try:
        # Delete previous checkpoint artifacts if they exist
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
        
        artifact = wandb.Artifact(
            name=f'{run_name}_checkpoint',  # Name for the artifact
            type="checkpoint",                # Artifact type (can be "model", "checkpoint", etc.)
            description=f"Checkpoint at epoch {epoch}",
            metadata=metadata
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        print(f"Saved checkpoint at epoch {epoch}")
        return artifact
    finally:
        # Clean up temporary file
        if os.path.exists(path):
            os.unlink(path)
    

def load_checkpoint_wandb(path, model_template, state_template, opt_state_template, filter_spec, wandb_run_name, wandb_project, wandb_entity):
    """Load model, optimizer state, epoch, and step from a checkpoint file."""
    api = wandb.Api()
    artifact_full_name = f"{wandb_entity}/{wandb_project}/{wandb_run_name}_checkpoint:latest"
    
    try:
        artifact = api.artifact(artifact_full_name, type="checkpoint")
    except Exception as e:
        print(f"Could not find checkpoint artifact: {artifact_full_name}")
        print(f"Error: {e}")
        
        # Try to list available artifacts for debugging
        try:
            print(f"Searching for artifacts in {wandb_entity}/{wandb_project}...")
            # This is a bit hacky but might help debug
            project = api.project(wandb_entity, wandb_project)
            print(f"Available artifacts in project: {[a.name for a in project.artifacts()]}")
        except Exception as debug_e:
            print(f"Could not list artifacts: {debug_e}")
        
        raise FileNotFoundError(f"Checkpoint artifact not found: {artifact_full_name}")
    
    # Create temporary directory for download
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact.download(temp_dir)
        
        # Find the checkpoint file in the downloaded directory
        checkpoint_files = [f for f in os.listdir(temp_dir) if f.endswith('.ckpt')]
        if not checkpoint_files:
            print(f"Available files in {temp_dir}: {os.listdir(temp_dir)}")
            raise FileNotFoundError(f"No checkpoint file found in {temp_dir}. Available files: {os.listdir(temp_dir)}")
        
        checkpoint_path = os.path.join(temp_dir, checkpoint_files[0])
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            meta = json.loads(f.readline().decode())
            model = eqx.tree_deserialise_leaves(f, model_template)
            state = eqx.tree_deserialise_leaves(f, state_template)
            
            # --- Debugging: Print tree structures for opt_state ---
            print("\n--- Debugging opt_state ---")
            print("Structure of opt_state_template:")
            eqx.tree_pprint(opt_state_template)
            import jax.tree_util as jtu
            print("\nPyTreeDef of opt_state_template:")
            print(jtu.tree_structure(opt_state_template))
            
            opt_state = eqx.tree_deserialise_leaves(f, opt_state_template)
        
        return model, state, opt_state, meta['epoch'], meta['step'], meta

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
    
    
    