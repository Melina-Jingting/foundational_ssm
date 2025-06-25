import wandb
import equinox as eqx
from jax.tree_util import tree_flatten_with_path
import json

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
            
            
def save_model_wandb(model, run_name, model_metadata, wandb_run):
    model_path = f"best_model.pt"
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
    wandb_run.log_artifact(model_artifact)
    return model_path

def load_model_wandb(filename, modelClass):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = modelClass(**hyperparams)
        return eqx.tree_deserialise_leaves(f, model)