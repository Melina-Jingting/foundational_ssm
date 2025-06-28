# Training Profiling Guide

This guide explains how to use the training profiler to track accumulated average times for key training operations in your foundational SSM training loops.

## Overview

The profiling system tracks timing for:
1. **Data batch production** - Time spent loading and preparing data batches
2. **Loss computation** - Time spent computing loss and gradients
3. **Model updates** - Time spent updating model parameters
4. **Custom operations** - Any other operations you want to profile

## Quick Start

### 1. Enable Profiling in Training Functions

The training functions in `src/foundational_ssm/trainer/decoding.py` now support profiling:

```python
# For JAX training
model = train_decoding_jax(
    model, 
    filter_spec,
    train_loader, 
    train_tensors,
    val_tensors,
    loss_fn, 
    state,
    opt,
    opt_state,
    epochs=100,
    log_every=10,
    key=jr.PRNGKey(0),
    wandb_run_name="my_experiment",
    enable_profiling=True  # Enable profiling
)

# For PyTorch training
train_decoding_torch(
    model, 
    train_loader, 
    train_tensors, 
    val_tensors, 
    optimizer, 
    loss_fn, 
    num_epochs=100, 
    wandb_run_name="my_experiment",
    enable_profiling=True  # Enable profiling
)
```

### 2. View Profiling Results

Profiling metrics are automatically logged to:
- **Console**: Printed every `log_every` steps/epochs
- **Weights & Biases**: If wandb is initialized, metrics are logged with the prefix `profiling/`

Example console output:
```
--- Profiling Metrics (Step 100) ---
data_batch_production: avg=0.0234s, total=2.34s, count=100
loss_fn_computation: avg=0.0456s, total=4.56s, count=100
make_step_total: avg=0.0789s, total=7.89s, count=100
```

## Advanced Usage

### 1. Using the Profiler Directly

```python
from foundational_ssm.utils.profiling import get_profiler, log_profiling_metrics

# Get the global profiler
profiler = get_profiler()

# Start timing an operation
start_time = profiler.start_timer("my_operation")

# ... your code here ...

# End timing
profiler.end_timer("my_operation", start_time)

# Log metrics
log_profiling_metrics()
```

### 2. Using Decorators

```python
from foundational_ssm.utils.profiling import profile_operation

@profile_operation("expensive_function")
def my_expensive_function(data):
    # This function will be automatically profiled
    result = complex_computation(data)
    return result

# Call the function multiple times
for i in range(100):
    result = my_expensive_function(data)
```

### 3. Profiling Data Loaders

```python
from foundational_ssm.utils.profiling import profile_data_loader

# Profile an existing data loader
profiled_loader = profile_data_loader(train_loader)

# Use the profiled loader in training
for batch in profiled_loader:
    # Training code here
    pass
```

### 4. Profiling JAX Loss Functions

```python
from foundational_ssm.utils.profiling import profile_jax_loss_fn

# Wrap your loss function with profiling
profiled_loss_fn = profile_jax_loss_fn(my_loss_fn, "my_loss_computation")

# Use in training
(value, state), grads = profiled_loss_fn(model_params, model_static, state, inputs, targets, key)
```

### 5. Profiling make_step Function

```python
from foundational_ssm.utils.profiling import profile_make_step

# Create a profiled version of make_step
profiled_make_step = profile_make_step(my_make_step)

# Use in training loop
model, state, opt_state, value, grads = profiled_make_step(
    model, state, filter_spec, inputs, targets, loss_fn, opt, opt_state, key
)
```

## Configuration

### Profiler Settings

```python
from foundational_ssm.utils.profiling import TrainingProfiler

# Create a custom profiler with specific settings
profiler = TrainingProfiler(
    log_to_wandb=True,  # Whether to log to wandb
    log_every=100       # How often to log metrics
)
```

### Global Profiler

The module provides a global profiler instance that's used by default:

```python
from foundational_ssm.utils.profiling import get_profiler

profiler = get_profiler()
profiler.reset()  # Reset all timers
summary = profiler.get_summary()  # Get all profiling data
```

## Metrics Explained

For each profiled operation, the system tracks:

- **avg_time**: Average time per operation (seconds)
- **total_time**: Total accumulated time (seconds)
- **count**: Number of times the operation was performed

### Common Operations

1. **data_batch_production**: Time to load and prepare a batch of data
2. **loss_fn_computation**: Time for loss computation and gradient calculation
3. **make_step_total**: Total time for a complete training step
4. **training_step**: Time for PyTorch training step (forward + backward + update)

## Integration with Weights & Biases

When wandb is initialized, profiling metrics are automatically logged with the prefix `profiling/`:

```
profiling/data_batch_production_avg_time
profiling/data_batch_production_total_time
profiling/data_batch_production_count
profiling/loss_fn_computation_avg_time
profiling/loss_fn_computation_total_time
profiling/loss_fn_computation_count
...
```

You can view these metrics in the wandb dashboard to track performance over time.

## Example: Complete Training Script

```python
import jax
import jax.random as jr
import equinox as eqx
import optax
from foundational_ssm.utils.profiling import get_profiler, log_profiling_metrics
from foundational_ssm.trainer.decoding import train_decoding_jax

# Initialize wandb (optional)
import wandb
wandb.init(project="my_project", name="profiled_training")

# Setup your model, data, etc.
# ... (your existing setup code) ...

# Enable profiling and train
model = train_decoding_jax(
    model, 
    filter_spec,
    train_loader, 
    train_tensors,
    val_tensors,
    loss_fn, 
    state,
    opt,
    opt_state,
    epochs=100,
    log_every=10,
    key=jr.PRNGKey(0),
    wandb_run_name="profiled_experiment",
    enable_profiling=True  # This enables all profiling
)

# Get final profiling summary
profiler = get_profiler()
summary = profiler.get_summary()
print("Final profiling summary:", summary)
```

## Troubleshooting

### Profiling Overhead

The profiling system has minimal overhead, but if you notice performance impact:

1. Disable profiling for production runs: `enable_profiling=False`
2. Use profiling only during development/debugging
3. Increase `log_every` to reduce logging frequency

### Missing Metrics

If you don't see expected metrics:

1. Ensure `enable_profiling=True` is passed to training functions
2. Check that wandb is initialized if you expect wandb logging
3. Verify that the operation you want to profile is actually being called

### Custom Operations

To profile custom operations not covered by the default profiling:

```python
from foundational_ssm.utils.profiling import get_profiler

profiler = get_profiler()

# Manual timing
start_time = profiler.start_timer("my_custom_operation")
# ... your code ...
profiler.end_timer("my_custom_operation", start_time)

# Or use decorator
@profile_operation("my_custom_operation")
def my_function():
    # ... your code ...
    pass
```

## Performance Analysis

Use the profiling data to identify bottlenecks:

1. **High data_batch_production time**: Consider data loading optimizations
   - Increase `num_workers` in DataLoader
   - Use `pin_memory=True`
   - Preprocess data offline

2. **High loss_fn_computation time**: Consider model optimizations
   - Reduce model complexity
   - Use mixed precision training
   - Optimize JAX compilation

3. **High make_step_total time**: Look at the overall training pipeline
   - Profile individual components
   - Consider batch size adjustments
   - Check for unnecessary computations

The profiling system helps you make data-driven decisions about where to focus optimization efforts. 