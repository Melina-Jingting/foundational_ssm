import wandb
import equinox as eqx
from jax.tree_util import tree_flatten_with_path
import json
import os
from foundational_ssm.models.decoders import SSMFoundationalDecoder

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
        model_artifact_dir = model_artifact.download()
        model_filename = os.path.join(model_artifact_dir, 'best_model.pt')
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
            
def save_best_model_wandb(model, run_name, model_metadata):
    model_path = f"wandb_artifacts/{run_name}/best_model.eqx"
    with open(model_path, "wb") as f:
        hyperparam_str = json.dumps(model_metadata)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)
    
    model_artifact = wandb.Artifact(
        name=f"{run_name}_best_model",
        type="model",
        description=f"best model for {run_name}",
        metadata=model_metadata
    )
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)
    return model_path

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
    path = f'wandb_artifacts/{run_name}/checkpoint.ckpt'
    with open(path, 'wb') as f:
        # Write metadata as JSON in the first line
        meta = json.dumps({'epoch': epoch, 'step': step})
        f.write((meta + '\n').encode())
        eqx.tree_serialise_leaves(f, model)
        eqx.tree_serialise_leaves(f, state)
        eqx.tree_serialise_leaves(f, opt_state)
    artifact = wandb.Artifact(
        name=f'{run_name}_checkpoint',  # Name for the artifact
        type="checkpoint",                # Artifact type (can be "model", "checkpoint", etc.)
        description=f"Checkpoint at epoch {epoch}",
        metadata=metadata
    )
    artifact.add_file(path)
    wandb.log_artifact(artifact)
    print(f"Saved checkpoint at epoch {epoch}")
    return path
    

def load_checkpoint_wandb(path, model_template, state_template, opt_state_template, wandb_run_name, wandb_project, wandb_entity):
    """Load model, optimizer state, epoch, and step from a checkpoint file."""
    api = wandb.Api()
    artifact_full_name = f"{wandb_entity}/{wandb_project}/{wandb_run_name}_checkpoint:latest"
    artifact_save_path = os.path.join(os.getcwd(), 'wandb_artifacts', wandb_run_name)
    artifact = api.artifact(artifact_full_name, type="checkpoint")
    dir = artifact.download(artifact_save_path)
    path = os.path.join(dir, 'checkpoint.ckpt')
    with open(path, 'rb') as f:
        meta = json.loads(f.readline().decode())
        model = eqx.tree_deserialise_leaves(f, model_template)
        state = eqx.tree_deserialise_leaves(f, state_template)
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
    artifact_save_path = os.path.join(os.getcwd(), 'wandb_artifacts', wandb_run_name)
    artifact = api.artifact(artifact_full_name, type="model")
    dir = artifact.download(artifact_save_path)
    path = os.path.join(dir, 'best_model.pt')
    
    # Load the foundational model
    foundational_model = load_model_wandb(path, SSMFoundationalDecoder)
    
    # Transfer parameters to downstream model
    downstream_model = transfer_foundational_to_downstream(foundational_model, downstream_model)
    
    return downstream_model