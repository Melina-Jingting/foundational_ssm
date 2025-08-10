#!/usr/bin/env python3
"""
Script to estimate minimum JAX memory requirements for your model.
Run this before training to determine optimal XLA_PYTHON_CLIENT_PREALLOCATE value.
"""

import os
import jax
import jax.numpy as jnp
from jax import random as jr
import equinox as eqx
from omegaconf import OmegaConf
import psutil
import subprocess

# Import your model
from foundational_ssm.models import SSMDownstreamDecoder
from foundational_ssm.utils.downstream_utils import create_optimizer_and_state

def get_gpu_memory():
    """Get total GPU memory in GB"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        gpu_memory_mb = int(result.stdout.strip())
        return gpu_memory_mb / 1024  # Convert to GB
    except:
        return None

def estimate_model_memory(model_config):
    """Estimate memory usage for model parameters and optimizer state"""
    
    # Create model
    key = jr.PRNGKey(42)
    model, state = eqx.nn.make_with_state(SSMDownstreamDecoder)(**model_config)
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree.leaves(model) if eqx.is_array(x))
    
    # Estimate memory usage (bytes)
    # Each float32 parameter = 4 bytes
    # Optimizer typically stores: parameters + gradients + momentum + variance
    bytes_per_param = 4  # float32
    optimizer_overhead = 3  # AdamW stores param + momentum + variance
    
    model_memory_bytes = param_count * bytes_per_param
    optimizer_memory_bytes = param_count * bytes_per_param * optimizer_overhead
    
    # Add some overhead for activations, JAX compilation cache, etc.
    activation_overhead = 0.5 * (model_memory_bytes + optimizer_memory_bytes)
    
    total_memory_bytes = model_memory_bytes + optimizer_memory_bytes + activation_overhead
    total_memory_gb = total_memory_bytes / (1024**3)
    
    return {
        'param_count': param_count,
        'model_memory_mb': model_memory_bytes / (1024**2),
        'optimizer_memory_mb': optimizer_memory_bytes / (1024**2),
        'total_memory_gb': total_memory_gb,
        'breakdown': {
            'parameters': model_memory_bytes / (1024**2),
            'optimizer_state': optimizer_memory_bytes / (1024**2),
            'activations_overhead': activation_overhead / (1024**2)
        }
    }

def estimate_batch_memory(batch_size, sequence_length, input_dim, output_dim):
    """Estimate memory for a single batch"""
    bytes_per_float = 4
    
    # Input batch
    input_memory = batch_size * sequence_length * input_dim * bytes_per_float
    
    # Output batch  
    output_memory = batch_size * sequence_length * output_dim * bytes_per_float
    
    # Intermediate activations (rough estimate - depends on model depth)
    # For SSM models, intermediate states can be significant
    intermediate_memory = input_memory * 2  # Conservative estimate
    
    total_batch_memory = input_memory + output_memory + intermediate_memory
    return total_batch_memory / (1024**2)  # MB

def recommend_preallocation():
    """Analyze and recommend optimal preallocation setting"""
    
    # Load your model config
    model_config = {
        'input_dim': 130,
        'ssm_io_dim': 128,
        'ssm_dim': 128,
        'ssm_init_diag_blocks': 4,
        'ssm_num_layers': 1,
        'output_dim': 2,
        'rng_seed': 42,
        'dt_min': 0.001,
        'dt_max': 0.1,
        'dropout_p': 0.001
    }
    
    # Get system info
    gpu_memory_gb = get_gpu_memory()
    
    print("=" * 60)
    print("JAX Memory Requirements Analysis")
    print("=" * 60)
    
    # Model memory estimation
    model_mem = estimate_model_memory(model_config)
    print(f"\nModel Memory Breakdown:")
    print(f"  Parameters: {model_mem['param_count']:,}")
    print(f"  Model weights: {model_mem['breakdown']['parameters']:.1f} MB")
    print(f"  Optimizer state: {model_mem['breakdown']['optimizer_state']:.1f} MB")
    print(f"  Activation overhead: {model_mem['breakdown']['activations_overhead']:.1f} MB")
    print(f"  Total estimated: {model_mem['total_memory_gb']:.2f} GB")
    
    # Batch memory estimation
    batch_configs = [
        (32, 100, 130, 2),   # Small batch
        (64, 100, 130, 2),   # Medium batch  
        (128, 100, 130, 2),  # Large batch
    ]
    
    print(f"\nBatch Memory Estimates (sequence_length=100):")
    for batch_size, seq_len, input_dim, output_dim in batch_configs:
        batch_mem = estimate_batch_memory(batch_size, seq_len, input_dim, output_dim)
        print(f"  Batch size {batch_size:3d}: {batch_mem:.1f} MB")
    
    # GPU info and recommendations
    if gpu_memory_gb:
        print(f"\nGPU Information:")
        print(f"  Total GPU memory: {gpu_memory_gb:.1f} GB")
        
        # Calculate recommended preallocation
        base_memory_gb = model_mem['total_memory_gb']
        
        # Add memory for largest expected batch
        max_batch_mem_gb = estimate_batch_memory(128, 100, 130, 2) / 1024
        
        total_needed_gb = base_memory_gb + max_batch_mem_gb
        recommended_fraction = min(0.9, (total_needed_gb / gpu_memory_gb) * 1.2)  # 20% safety margin
        
        print(f"\nRecommendations:")
        print(f"  Minimum needed: {total_needed_gb:.2f} GB ({total_needed_gb/gpu_memory_gb*100:.1f}% of GPU)")
        print(f"  Recommended preallocation: {recommended_fraction:.2f}")
        print(f"  Add to .env file: XLA_PYTHON_CLIENT_PREALLOCATE={recommended_fraction:.2f}")
        
        if recommended_fraction < 0.1:
            print("  → Very efficient! You could use even less memory.")
        elif recommended_fraction < 0.3:
            print("  → Good memory efficiency.")
        elif recommended_fraction < 0.6:
            print("  → Moderate memory usage.")
        else:
            print("  → High memory usage. Consider reducing batch size or model size.")
            
    else:
        print(f"\nCould not detect GPU memory. Install nvidia-ml-py for better estimates.")
        print(f"Estimated minimum needed: {model_mem['total_memory_gb']:.2f} GB")
    
    print(f"\nAlternative approaches:")
    print(f"  XLA_PYTHON_CLIENT_PREALLOCATE=false  # Dynamic allocation (slower)")
    print(f"  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8   # Alternative to preallocate")

if __name__ == "__main__":
    recommend_preallocation()
