#!/usr/bin/env python3
"""
Simple profiling script for transform_brainsets_to_fixed_dim_samples function
"""

import time
import numpy as np
import torch
import sys
import os

# Add the project root to the path
sys.path.append('/cs/student/projects1/ml/2024/mlaimon/foundational_ssm')

from foundational_ssm.constants import parse_session_id, DATASET_GROUP_TO_IDX

def _ensure_dim(arr: np.ndarray, target_dim: int, pad_value: float = 0.0, *, axis: int = 1) -> np.ndarray:
    """
    Crop or pad `arr` along `axis` to match `target_dim`, right-aligning the original data.
    Pads with `pad_value` if needed.
    """
    current_dim = arr.shape[axis]
    
    # If the current dimension is already the target dimension, return the array
    if current_dim == target_dim:
        return arr
    
    shape = list(arr.shape)
    shape[axis] = target_dim
    idx = [slice(None)] * arr.ndim
    
    # If the current dimension is greater than the target dimension, crop the array
    if current_dim > target_dim:
        idx[axis] = slice(None, target_dim)
        return arr[tuple(idx)]
    
    # If the current dimension is smaller than the target dimension, pad the array
    result = np.full(shape, pad_value, dtype=arr.dtype)
    idx[axis] = slice(None, current_dim)
    result[tuple(idx)] = arr
    return result

def transform_brainsets_to_fixed_dim_samples(
    data,
    sampling_rate: int = 100,
    sampling_window_ms: int = 1000
):
    """Original function for comparison"""
    num_timesteps = int(sampling_rate * sampling_window_ms / 1000)
    
    # 1. Bin + smooth spikes
    smoothed_spikes = data.smoothed_spikes.smoothed_spikes

    # 2. Prepare behaviour signal (cursor velocity)
    behavior_input = data.cursor.vel

    # 3. Align channel dimensions
    smoothed_spikes = _ensure_dim(smoothed_spikes, 353, axis=1)
    behavior_input = _ensure_dim(behavior_input, 2, axis=1)
    smoothed_spikes = _ensure_dim(smoothed_spikes, num_timesteps, axis=0)
    behavior_input = _ensure_dim(behavior_input, num_timesteps, axis=0)

    # 4. Pack into torch tensors
    dataset, subject, task = parse_session_id(data.session.id)
    group_tuple = (dataset, subject, task)
    group_idx = DATASET_GROUP_TO_IDX[group_tuple]

    return {
        "neural_input": torch.as_tensor(smoothed_spikes, dtype=torch.float32),
        "behavior_input": torch.as_tensor(behavior_input, dtype=torch.float32),
        "dataset_group_idx": torch.as_tensor(group_idx, dtype=torch.int32),
    }

def transform_brainsets_to_fixed_dim_samples_optimized(
    data,
    sampling_rate: int = 100,
    sampling_window_ms: int = 1000
):
    """Optimized version with potential improvements"""
    num_timesteps = int(sampling_rate * sampling_window_ms / 1000)
    
    # 1. Bin + smooth spikes
    smoothed_spikes = data.smoothed_spikes.smoothed_spikes

    # 2. Prepare behaviour signal (cursor velocity)
    behavior_input = data.cursor.vel

    # 3. Align channel dimensions - optimized to avoid multiple array copies
    # Pre-allocate final arrays
    final_spikes = np.zeros((num_timesteps, 353), dtype=np.float32)
    final_behavior = np.zeros((num_timesteps, 2), dtype=np.float32)
    
    # Copy data efficiently
    if smoothed_spikes.shape[0] > num_timesteps:
        final_spikes[:] = smoothed_spikes[:num_timesteps, :min(353, smoothed_spikes.shape[1])]
    else:
        final_spikes[:smoothed_spikes.shape[0], :min(353, smoothed_spikes.shape[1])] = smoothed_spikes
    
    if behavior_input.shape[0] > num_timesteps:
        final_behavior[:] = behavior_input[:num_timesteps, :min(2, behavior_input.shape[1])]
    else:
        final_behavior[:behavior_input.shape[0], :min(2, behavior_input.shape[1])] = behavior_input

    # 4. Pack into torch tensors - use from_numpy for zero-copy when possible
    dataset, subject, task = parse_session_id(data.session.id)
    group_tuple = (dataset, subject, task)
    group_idx = DATASET_GROUP_TO_IDX[group_tuple]

    return {
        "neural_input": torch.from_numpy(final_spikes),
        "behavior_input": torch.from_numpy(final_behavior),
        "dataset_group_idx": torch.tensor(group_idx, dtype=torch.int32),
    }

def profile_operations():
    """Profile individual operations to identify bottlenecks"""
    print("=== Operation Profiling ===")
    
    # Create realistic test data
    neural_data = np.random.randn(100, 200).astype(np.float32)
    behavior_data = np.random.randn(100, 2).astype(np.float32)
    session_id = "perich_miller_population_2018/c_20131003_center_out_reaching"
    
    timings = {}
    
    # Test 1: _ensure_dim operations
    print("Testing _ensure_dim operations...")
    
    start = time.time()
    for _ in range(1000):
        _ = _ensure_dim(neural_data, 353, axis=1)
    timings['ensure_dim_neural_353'] = (time.time() - start) / 1000
    
    start = time.time()
    for _ in range(1000):
        _ = _ensure_dim(behavior_data, 2, axis=1)
    timings['ensure_dim_behavior_2'] = (time.time() - start) / 1000
    
    start = time.time()
    for _ in range(1000):
        _ = _ensure_dim(neural_data, 100, axis=0)
    timings['ensure_dim_neural_time'] = (time.time() - start) / 1000
    
    # Test 2: parse_session_id
    print("Testing parse_session_id...")
    start = time.time()
    for _ in range(10000):
        _ = parse_session_id(session_id)
    timings['parse_session_id'] = (time.time() - start) / 10000
    
    # Test 3: Dictionary lookup
    print("Testing dictionary lookup...")
    dataset, subject, task = parse_session_id(session_id)
    group_tuple = (dataset, subject, task)
    
    start = time.time()
    for _ in range(10000):
        _ = DATASET_GROUP_TO_IDX[group_tuple]
    timings['dict_lookup'] = (time.time() - start) / 10000
    
    # Test 4: torch.as_tensor vs torch.from_numpy
    print("Testing tensor conversions...")
    final_neural = _ensure_dim(_ensure_dim(neural_data, 353, axis=1), 100, axis=0)
    final_behavior = _ensure_dim(_ensure_dim(behavior_data, 2, axis=1), 100, axis=0)
    
    start = time.time()
    for _ in range(1000):
        _ = torch.as_tensor(final_neural, dtype=torch.float32)
    timings['torch_as_tensor'] = (time.time() - start) / 1000
    
    start = time.time()
    for _ in range(1000):
        _ = torch.from_numpy(final_neural)
    timings['torch_from_numpy'] = (time.time() - start) / 1000
    
    # Print results
    print("\nOperation timing (seconds per operation):")
    for op, timing in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op}: {timing:.8f}")
    
    return timings

def profile_full_functions():
    """Profile the complete functions"""
    print("\n=== Full Function Profiling ===")
    
    # Create mock data object
    class MockData:
        class Session:
            def __init__(self):
                self.id = "perich_miller_population_2018/c_20131003_center_out_reaching"
        
        class SmoothedSpikes:
            def __init__(self):
                self.smoothed_spikes = np.random.randn(100, 200).astype(np.float32)
        
        class Cursor:
            def __init__(self):
                self.vel = np.random.randn(100, 2).astype(np.float32)
        
        def __init__(self):
            self.session = self.Session()
            self.smoothed_spikes = self.SmoothedSpikes()
            self.cursor = self.Cursor()
    
    mock_data = MockData()
    
    # Warm up
    for _ in range(10):
        transform_brainsets_to_fixed_dim_samples(mock_data)
        transform_brainsets_to_fixed_dim_samples_optimized(mock_data)
    
    # Profile original function
    print("Profiling original function...")
    start = time.time()
    for _ in range(1000):
        result_orig = transform_brainsets_to_fixed_dim_samples(mock_data)
    orig_time = (time.time() - start) / 1000
    
    # Profile optimized function
    print("Profiling optimized function...")
    start = time.time()
    for _ in range(1000):
        result_opt = transform_brainsets_to_fixed_dim_samples_optimized(mock_data)
    opt_time = (time.time() - start) / 1000
    
    print(f"\nOriginal function: {orig_time:.8f}s per call")
    print(f"Optimized function: {opt_time:.8f}s per call")
    print(f"Speedup: {orig_time / opt_time:.2f}x")
    
    # Verify results are equivalent
    print(f"\nResults equivalent: {torch.allclose(result_orig['neural_input'], result_opt['neural_input'])}")
    print(f"Results equivalent: {torch.allclose(result_orig['behavior_input'], result_opt['behavior_input'])}")
    print(f"Results equivalent: {result_orig['dataset_group_idx'] == result_opt['dataset_group_idx']}")

def analyze_memory_usage():
    """Analyze memory usage patterns"""
    print("\n=== Memory Usage Analysis ===")
    
    # Create test data
    neural_data = np.random.randn(100, 200).astype(np.float32)
    behavior_data = np.random.randn(100, 2).astype(np.float32)
    
    print(f"Original neural data: {neural_data.nbytes / 1024:.2f} KB")
    print(f"Original behavior data: {behavior_data.nbytes / 1024:.2f} KB")
    
    # Test _ensure_dim memory usage
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create many arrays to see memory impact
    arrays = []
    for _ in range(1000):
        arr = _ensure_dim(neural_data, 353, axis=1)
        arrays.append(arr)
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory usage: {mem_before:.2f} MB -> {mem_after:.2f} MB")
    print(f"Memory increase: {mem_after - mem_before:.2f} MB")
    
    # Clear arrays
    del arrays
    mem_cleared = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory after clearing: {mem_cleared:.2f} MB")

def main():
    """Main profiling function"""
    print("Profiling transform_brainsets_to_fixed_dim_samples function")
    print("=" * 60)
    
    # Profile individual operations
    profile_operations()
    
    # Profile full functions
    profile_full_functions()
    
    # Analyze memory usage
    analyze_memory_usage()
    
    print("\nProfiling complete!")

if __name__ == "__main__":
    main() 