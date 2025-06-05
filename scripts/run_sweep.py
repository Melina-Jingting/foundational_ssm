import wandb
import argparse
import sys
import os
import subprocess
from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser(description="Run WandB sweep")
    parser.add_argument("--model", type=str, default="cmt",
                      help="Model configuration to sweep over (cmt, c, etc.)")
    parser.add_argument("--count", type=int, default=10,
                      help="Number of runs to execute")
    parser.add_argument("--method", type=str, default="bayes",
                      choices=["bayes", "random", "grid"],
                      help="Search method")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define sweep configuration programmatically
    sweep_config = {
        'method': args.method,
        'metric': {
            'name': 'val/combined_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'config': {
                'value': args.model
            },
            'model.ssm_hidden_dim': {
                'values': [32, 64, 128, 256]
            },
            'model.ssm_num_layers': {
                'values': [1, 2, 3]
            },
            'model.embedding_dim': {
                'values': [32, 64, 128]
            },
            'model.ssm_dropout': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.3
            },
            'model.lin_dropout': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.3
            },
            'training.learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2
            },
            'training.mask_prob': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.8
            },
            'training.neural_weight': {
                'values': [1.0, 5.0, 10.0]
            },
            'training.behavior_weight': {
                'values': [0.1, 1.0, 5.0]
            }
        }
    }
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project=f"foundational_ssm_{args.model}_sweep"
    )
    print(f"Created sweep with ID: {sweep_id}")
    
    # Define a function that will be passed to wandb.agent
    def train_with_sweep():
        wandb.init()
        cmd = ["python", "scripts/pre_train.py"]
        
        for key, value in wandb.config.items():
            if key != "config":  
                cmd.append(f"--{key}={value}")
            else:
                cmd.append(f"--config={value}")
                
        print(f"Running command: {' '.join(cmd)}")
        subprocess.call(cmd)
    
    wandb.agent(sweep_id, function=train_with_sweep, count=args.count)

if __name__ == "__main__":
    main()