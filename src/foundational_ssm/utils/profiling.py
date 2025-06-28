import time
import functools
from typing import Dict, Any, Optional, Callable
from collections import defaultdict
import jax
import jax.numpy as jnp
import wandb
import numpy as np


class TrainingProfiler:
    """
    A profiler for tracking accumulated average times of key training operations.
    
    This profiler tracks timing for:
    1. Data batch production (data loading)
    2. Loss computation and gradient calculation
    3. Model updates
    4. Any other custom operations
    """
    
    def __init__(self, log_to_wandb: bool = True, log_every: int = 100):
        self.timers = defaultdict(lambda: {"total_time": 0.0, "count": 0, "avg_time": 0.0})
        self.log_to_wandb = log_to_wandb
        self.log_every = log_every
        self.step_count = 0
        
    def start_timer(self, operation_name: str) -> float:
        """Start timing an operation and return the start time."""
        return time.time()
    
    def end_timer(self, operation_name: str, start_time: float):
        """End timing an operation and update accumulated statistics."""
        elapsed_time = time.time() - start_time
        timer = self.timers[operation_name]
        timer["total_time"] += elapsed_time
        timer["count"] += 1
        timer["avg_time"] = timer["total_time"] / timer["count"]
        
    def time_operation(self, operation_name: str):
        """Decorator to time a function or method."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = self.start_timer(operation_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_timer(operation_name, start_time)
            return wrapper
        return decorator
    
    def profile_data_loading(self, data_loader):
        """Profile data batch production from a data loader."""
        original_iter = data_loader.__iter__
        
        def profiled_iter():
            iterator = original_iter()
            while True:
                try:
                    start_time = self.start_timer("data_batch_production")
                    batch = next(iterator)
                    self.end_timer("data_batch_production", start_time)
                    yield batch
                except StopIteration:
                    break
        
        data_loader.__iter__ = profiled_iter
        return data_loader
    
    def log_metrics(self, step: Optional[int] = None):
        """Log current profiling metrics to wandb and console."""
        if step is None:
            step = self.step_count
            
        metrics = {}
        for operation, stats in self.timers.items():
            metrics[f"profiling/{operation}_avg_time"] = stats["avg_time"]
            metrics[f"profiling/{operation}_total_time"] = stats["total_time"]
            metrics[f"profiling/{operation}_count"] = stats["count"]
            
        # Log to console
        print(f"\n--- Profiling Metrics (Step {step}) ---")
        for operation, stats in self.timers.items():
            print(f"{operation}: avg={stats['avg_time']:.4f}s, total={stats['total_time']:.2f}s, count={stats['count']}")
        
        # Log to wandb
        if self.log_to_wandb and wandb.run is not None:
            wandb.log(metrics, step=step)
            
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all profiling data."""
        return dict(self.timers)
    
    def reset(self):
        """Reset all profiling data."""
        self.timers.clear()
        self.step_count = 0


# Global profiler instance
_global_profiler = TrainingProfiler()


def get_profiler() -> TrainingProfiler:
    """Get the global profiler instance."""
    return _global_profiler


def profile_operation(operation_name: str):
    """Decorator to profile a function with the global profiler."""
    return _global_profiler.time_operation(operation_name)


def log_profiling_metrics(step: Optional[int] = None):
    """Log profiling metrics using the global profiler."""
    _global_profiler.log_metrics(step)


# JAX-specific profiling utilities
def profile_jax_loss_fn(loss_fn: Callable, operation_name: str = "loss_computation"):
    """
    Wrap a JAX loss function to profile its execution time.
    
    Args:
        loss_fn: The JAX loss function to profile
        operation_name: Name for the profiling operation
        
    Returns:
        Wrapped loss function that profiles execution time
    """
    def profiled_loss_fn(*args, **kwargs):
        start_time = _global_profiler.start_timer(operation_name)
        try:
            result = loss_fn(*args, **kwargs)
            return result
        finally:
            _global_profiler.end_timer(operation_name, start_time)
    
    return profiled_loss_fn


def profile_make_step(make_step_fn: Callable):
    """
    Profile the make_step function which is a key bottleneck in JAX training.
    
    Args:
        make_step_fn: The make_step function to profile
        
    Returns:
        Wrapped make_step function with profiling
    """
    def profiled_make_step(model, state, filter_spec, inputs, targets, loss_fn, opt, opt_state, key):
        # Profile the entire make_step operation
        start_time = _global_profiler.start_timer("make_step_total")
        
        # Profile the loss computation specifically
        def profiled_loss_fn(model_params, model_static, state, inputs, targets, key):
            loss_start = _global_profiler.start_timer("loss_fn_computation")
            try:
                return loss_fn(model_params, model_static, state, inputs, targets, key)
            finally:
                _global_profiler.end_timer("loss_fn_computation", loss_start)
        
        try:
            result = make_step_fn(model, state, filter_spec, inputs, targets, profiled_loss_fn, opt, opt_state, key)
            return result
        finally:
            _global_profiler.end_timer("make_step_total", start_time)
    
    return profiled_make_step


# Data loading profiling
def profile_data_loader(data_loader):
    """Profile a data loader to track batch production times."""
    return _global_profiler.profile_data_loading(data_loader)


# Training loop integration
def profile_training_loop(training_func: Callable, log_every: int = 100):
    """
    Wrap a training function to automatically log profiling metrics.
    
    Args:
        training_func: The training function to wrap
        log_every: How often to log profiling metrics
        
    Returns:
        Wrapped training function with automatic profiling
    """
    def profiled_training_func(*args, **kwargs):
        # Reset profiler at start of training
        _global_profiler.reset()
        
        # Get the original training loop
        original_loop = training_func(*args, **kwargs)
        
        # If it's a generator (like a training loop), wrap it
        if hasattr(original_loop, '__iter__'):
            def profiled_generator():
                step = 0
                for item in original_loop:
                    step += 1
                    if step % log_every == 0:
                        log_profiling_metrics(step)
                    yield item
                # Log final metrics
                log_profiling_metrics(step)
            return profiled_generator()
        else:
            return original_loop
    
    return profiled_training_func 