#!/usr/bin/env python3
"""
Script to start a wandb sweep for hyperparameter optimization
"""

import wandb
import yaml
import os
from omegaconf import OmegaConf
import argparse


def main():
    # Load sweep configuration
    parser = argparse.ArgumentParser(
        description="Start a wandb sweep for hyperparameter optimization"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="pretrain_sweep_dataset_model",
        help="Path to the sweep configuration YAML file",
    )
    parser.add_argument(
        "--name", type=str, default="pm_sweep_l2", help="Name of the sweep"
    )
    args = parser.parse_args()
    sweep_config_path = f"/cs/student/projects1/ml/2024/mlaimon/foundational_ssm/configs/{args.cfg}.yaml"
    sweep_config = OmegaConf.load(sweep_config_path)

    # Extract the sweep configuration (exclude fixed parameters)
    sweep_config_for_wandb = {
        "program": sweep_config.program,
        "name": args.name,
        "method": sweep_config.method,
        "metric": OmegaConf.to_container(sweep_config.metric, resolve=True),
        "parameters": OmegaConf.to_container(sweep_config.parameters, resolve=True),
        "command": ["python", sweep_config.program, "${args_no_hyphens}"],
    }

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config_for_wandb,
        project=sweep_config.wandb.project,
        entity=sweep_config.wandb.entity,  # Default entity if not specified
    )

    print(f"Created sweep with ID: {sweep_id}")
    print(f"To run the sweep agent, use:")
    print(
        f"wandb agent {sweep_config.wandb.entity}/{sweep_config.wandb.project}/{sweep_id}"
    )

    return sweep_id


if __name__ == "__main__":
    main()
