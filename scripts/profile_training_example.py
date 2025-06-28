#!/usr/bin/env python3
"""
Example script demonstrating how to use the training profiler.

This script shows how to enable profiling for both JAX and PyTorch training loops
to track accumulated average times for:
1. Data batch production
2. Loss computation and gradient calculation
3. Model updates
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from foundational_ssm.utils.profiling import (
    get_profiler, 
    profile_operation, 
    log_profiling_metrics,
    profile_jax_loss_fn,
    profile_make_step,
    profile_data_loader
)
from foundational_ssm.trainer.decoding import train_decoding_jax, train_decoding_torch


def create_dummy_data(batch_size=32, seq_len=100, input_dim=64, output_dim=2, num_batches=10):
    """Create dummy data for testing."""
    # Create dummy inputs and targets
    inputs = np.random.randn(num_batches * batch_size, seq_len, input_dim).astype(np.float32)
    targets = np.random.randn(num_batches * batch_size, seq_len, output_dim).astype(np.float32)
    
    # Create PyTorch dataset and loader
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create JAX tensors
    jax_inputs = jnp.array(inputs)
    jax_targets = jnp.array(targets)
    
    return loader, (jax_inputs, jax_targets)


def create_dummy_model(input_dim=64, output_dim=2, hidden_dim=32):
    """Create a dummy model for testing."""
    # PyTorch model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # Average over time dimension
            x = x.mean(dim=1)  # (batch, input_dim)
            x = self.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    # JAX model
    class DummyJAXModel(eqx.Module):
        linear1: eqx.nn.Linear
        linear2: eqx.nn.Linear
        
        def __init__(self, key):
            key1, key2 = jr.split(key)
            self.linear1 = eqx.nn.Linear(input_dim, hidden_dim, key=key1)
            self.linear2 = eqx.nn.Linear(hidden_dim, output_dim, key=key2)
            
        def __call__(self, x, state, key):
            # Average over time dimension
            x = jnp.mean(x, axis=1)  # (batch, input_dim)
            x = jax.nn.relu(self.linear1(x))
            x = self.linear2(x)
            return x, state
    
    return DummyModel(), DummyJAXModel(jr.PRNGKey(0))


def demonstrate_pytorch_profiling():
    """Demonstrate PyTorch training with profiling."""
    print("\n=== PyTorch Training Profiling Demo ===")
    
    # Create dummy data and model
    train_loader, (train_inputs, train_targets) = create_dummy_data()
    val_loader, (val_inputs, val_targets) = create_dummy_data()
    
    torch_model, _ = create_dummy_model()
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # Enable profiling
    profiler = get_profiler()
    profiler.reset()
    
    # Profile the data loader
    profiled_loader = profile_data_loader(train_loader)
    
    # Run a few training steps with profiling
    torch_model.train()
    for batch_idx, (inputs, targets) in enumerate(profiled_loader):
        if batch_idx >= 5:  # Only run 5 batches for demo
            break
            
        # Profile the training step
        start_time = profiler.start_timer("pytorch_training_step")
        
        optimizer.zero_grad()
        outputs = torch_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        profiler.end_timer("pytorch_training_step", start_time)
        
        print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    # Log profiling metrics
    log_profiling_metrics()
    
    # Show profiler summary
    summary = profiler.get_summary()
    print("\nPyTorch Profiling Summary:")
    for operation, stats in summary.items():
        print(f"  {operation}: avg={stats['avg_time']:.4f}s, total={stats['total_time']:.2f}s, count={stats['count']}")


def demonstrate_jax_profiling():
    """Demonstrate JAX training with profiling."""
    print("\n=== JAX Training Profiling Demo ===")
    
    # Create dummy data and model
    train_loader, (train_inputs, train_targets) = create_dummy_data()
    val_loader, (val_inputs, val_targets) = create_dummy_data()
    
    _, jax_model = create_dummy_model()
    state = eqx.nn.State(jax_model)
    
    # Create optimizer
    opt = optax.adam(learning_rate=0.001)
    filter_spec = eqx.is_array
    opt_state = opt.init(eqx.filter(jax_model, filter_spec))
    
    # Define loss function
    @eqx.filter_jit
    @eqx.filter_value_and_grad(has_aux=True)
    def mse_loss(model_params, model_static, state, inputs, targets, key):
        model = eqx.combine(model_params, model_static)
        batch_keys = jr.split(key, inputs.shape[0])
        preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None))(inputs, state, batch_keys)
        mse = jnp.mean((preds - targets) ** 2)
        return (mse, state)
    
    # Define make_step function
    @eqx.filter_jit
    def make_step(model, state, filter_spec, inputs, targets, loss_fn, opt, opt_state, key):
        model_params, model_static = eqx.partition(model, filter_spec)
        (value, state), grads = loss_fn(model_params, model_static, state, inputs, targets, key)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, state, opt_state, value, grads
    
    # Enable profiling
    profiler = get_profiler()
    profiler.reset()
    
    # Profile the data loader
    profiled_loader = profile_data_loader(train_loader)
    
    # Create profiled make_step
    profiled_make_step = profile_make_step(make_step)
    
    # Run a few training steps with profiling
    key = jr.PRNGKey(0)
    for batch_idx, (inputs, targets) in enumerate(profiled_loader):
        if batch_idx >= 5:  # Only run 5 batches for demo
            break
            
        key, subkey = jr.split(key)
        
        # Use profiled make_step
        jax_model, state, opt_state, loss_value, grads = profiled_make_step(
            jax_model, state, filter_spec, inputs, targets, mse_loss, opt, opt_state, subkey
        )
        
        print(f"Batch {batch_idx}: Loss = {float(loss_value):.4f}")
    
    # Log profiling metrics
    log_profiling_metrics()
    
    # Show profiler summary
    summary = profiler.get_summary()
    print("\nJAX Profiling Summary:")
    for operation, stats in summary.items():
        print(f"  {operation}: avg={stats['avg_time']:.4f}s, total={stats['total_time']:.2f}s, count={stats['count']}")


def demonstrate_decorator_profiling():
    """Demonstrate using the @profile_operation decorator."""
    print("\n=== Decorator Profiling Demo ===")
    
    profiler = get_profiler()
    profiler.reset()
    
    @profile_operation("expensive_computation")
    def expensive_function(n):
        """Simulate an expensive computation."""
        result = 0
        for i in range(n):
            result += i ** 2
        return result
    
    # Run the function multiple times
    for i in range(10):
        result = expensive_function(10000)
        print(f"Computation {i}: result = {result}")
    
    # Log profiling metrics
    log_profiling_metrics()
    
    # Show profiler summary
    summary = profiler.get_summary()
    print("\nDecorator Profiling Summary:")
    for operation, stats in summary.items():
        print(f"  {operation}: avg={stats['avg_time']:.4f}s, total={stats['total_time']:.2f}s, count={stats['count']}")


if __name__ == "__main__":
    print("Training Profiler Demonstration")
    print("=" * 50)
    
    # Demonstrate different profiling approaches
    demonstrate_decorator_profiling()
    demonstrate_pytorch_profiling()
    demonstrate_jax_profiling()
    
    print("\n" + "=" * 50)
    print("Profiling demonstration completed!")
    print("\nTo use profiling in your training scripts:")
    print("1. Import the profiling utilities: from foundational_ssm.utils.profiling import *")
    print("2. Enable profiling in training functions: enable_profiling=True")
    print("3. Use decorators for custom operations: @profile_operation('operation_name')")
    print("4. Log metrics periodically: log_profiling_metrics()") 