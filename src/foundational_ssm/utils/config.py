import torch
from omegaconf import OmegaConf
import os
import argparse

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default configuration
def get_default_config():
    config = OmegaConf.create({
        "wandb": {
            "project": "foundational_ssm",
            "run_name": "ssm_neural_behavior",
            "tags": ["neural", "behavior", "masking"],
            "log_freq": 1  # Log validation metrics every 10 epochs
        },
        "dataset": {
            "name": "perich_miller_population_2018",
            "subjects": ["j"],
            "batch_size": 64
        },
        "model": {
            "num_neural_features": 192,
            "num_behavior_features": 2,
            "num_context_features": 32,
            "embedding_dim": 64,
            "ssm_projection_dim": 64,
            "ssm_hidden_dim": 64,
            "ssm_num_layers": 1,
            "ssm_dropout": 0.1,
            "pred_neural_dim": 192,
            "pred_behavior_dim": 2,
            "sequence_length": 1.0,
            "sampling_rate": 100,
            "lin_dropout": 0.1,
            "activation_fn": "relu"
        },
        "training": {
            "learning_rate": 2e-3,
            "mask_prob": 0.5,
            "num_epochs": 100,
            "neural_weight": 1.0,
            "behavior_weight": 1.0
        },
        "device": str(device)
    })
    return config




def load_yaml_config(config_name, 
                     config_dir="../../../foundational_ssm/configs"):
    """Load configuration from YAML file."""
    config_path = os.path.join(config_dir, f"{config_name}.yaml")
    config = OmegaConf.load(config_path)

    return config

def override_nested_config(config, param_name, value):
    """Override a nested config parameter using dot notation"""
    if '.' in param_name:
        parts = param_name.split('.')
        nested_config = config
        for part in parts[:-1]:
            if not hasattr(nested_config, part):
                setattr(nested_config, part, OmegaConf.create({}))
            nested_config = getattr(nested_config, part)
        setattr(nested_config, parts[-1], value)
    else:
        setattr(config, param_name, value)
    return config

def parse_args_and_load_config():
    """Parse command-line arguments and load the specified config."""
    parser = argparse.ArgumentParser(description="Train SSM model")
    
    # Config selection
    parser.add_argument("--config", type=str, default="trial1",
                      help="Name of the configuration file (without .yaml extension)")
    
    # Common overrides
    parser.add_argument("--run_name", type=str, default=None,
                      help="Override the run name in the config")
    parser.add_argument("--epochs", type=int, default=None,
                      help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                      help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None,
                      help="Override batch size")
    parser.add_argument("--device", type=str, default=None,
                      help="Override device (cuda, cpu)")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Directory to save output files")
    
    # Parse known args first to get the config
    args, unknown = parser.parse_known_args()
    
    # Load base config
    config = load_yaml_config(args.config)
    
    # Apply standard overrides
    if args.run_name:
        config.wandb.run_name = args.run_name
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.batch_size:
        config.dataset.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    
    # Handle output directory
    if args.output_dir:
        if "output" not in config:
            config.output = OmegaConf.create({})
        config.output.dir = args.output_dir
    
    # Handle custom parameter overrides from wandb sweep
    # Format: --param.subparam=value
    for arg in unknown:
        if arg.startswith('--'):
            if '=' in arg:
                param_name, value = arg[2:].split('=', 1)
            else:
                # Handle boolean flags
                param_name = arg[2:]
                value = True
                
            # Try to convert value to appropriate type
            try:
                # Try as number first
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # If not a number, handle as string or boolean
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                # Otherwise keep as string
            
            # Update config with this parameter
            override_nested_config(config, param_name, value)
    
    return config, args

def print_config_summary(config):
    """Print a summary of the configuration."""
    print("\n=== Configuration Summary ===")
    print(f"Run name: {config.wandb.run_name}")
    print(f"Dataset: {config.dataset.name}, Subjects: {config.dataset.subjects}")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.dataset.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Model: {config.model.ssm_num_layers} SSM layers with {config.model.ssm_hidden_dim} hidden dims")
    print("============================\n")