#!/usr/bin/env python3
"""
Profiling script for transform_brainsets_to_fixed_dim_samples function
"""

import time
import cProfile
import pstats
import io
import line_profiler
import memory_profiler
import numpy as np
import torch
from typing import Any, Dict
import sys
import os

# Add the project root to the path
sys.path.append('/cs/student/projects1/ml/2024/mlaimon/foundational_ssm')

from foundational_ssm.constants import parse_session_id, DATASET_GROUP_TO_IDX
from torch_brain.data import Dataset, collate
from foundational_ssm.data_utils import get_dataset_config
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch.utils.data import DataLoader
from foundational_ssm.constants import DATA_ROOT
from omegaconf import OmegaConf

def transform_brainsets_to_fixed_dim_samples(
    data: Any,
    sampling_rate: int = 100,
    sampling_window_ms: int = 1000
) -> Dict[str, torch.Tensor | str]:
    """Convert a *temporaldata* sample to a dictionary of Torch tensors.

    The function takes care of binning & smoothing spikes, cropping/padding neural
    and behavioural features to a globally consistent dimensionality that depends
    on the *(dataset, subject, task)* triple.

    Parameters
    ----------
    data: temporaldata.Data
        Sample returned by **torch-brain**/**temporaldata**.
    sampling_rate: int, default=100
        Target sampling rate *Hz* used for binning.
    sampling_window_ms: int, default=1000   
        Length of the temporal window after binning.
    kern_sd_ms: int, default=20
        Standard deviation of the Gaussian kernel (in ms) for smoothing spikes.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with keys ``neural_input``, ``behavior_input``, ``session_id``
        and ``subject_id``.
    """
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
    
    num_timesteps = int(sampling_rate * sampling_window_ms / 1000)
    
    # ------------------------------------------------------------------
    # 1. Bin + smooth spikes
    # ------------------------------------------------------------------
    smoothed_spikes = data.smoothed_spikes.smoothed_spikes

    # ------------------------------------------------------------------
    # 2. Prepare behaviour signal (cursor velocity)
    # ------------------------------------------------------------------
    behavior_input = data.cursor.vel  # np.ndarray, (timesteps?, features)

    # ------------------------------------------------------------------
    # 3. Align channel dimensions based on (dataset, subject, task)
    # ------------------------------------------------------------------
    smoothed_spikes = _ensure_dim(smoothed_spikes, 353, axis=1)
    behavior_input = _ensure_dim(behavior_input, 2, axis=1)
    smoothed_spikes = _ensure_dim(smoothed_spikes, num_timesteps, axis=0)
    behavior_input = _ensure_dim(behavior_input, num_timesteps, axis=0)

    # ------------------------------------------------------------------
    # 4. Pack into torch tensors
    # ------------------------------------------------------------------
    dataset, subject, task = parse_session_id(data.session.id)
    group_tuple = (dataset, subject, task)
    group_idx = DATASET_GROUP_TO_IDX[group_tuple]

    return {
        "neural_input": torch.as_tensor(smoothed_spikes, dtype=torch.float32),
        "behavior_input": torch.as_tensor(behavior_input, dtype=torch.float32),
        "dataset_group_idx": torch.as_tensor(group_idx, dtype=torch.int32),
    }

def profile_with_cprofile(func, *args, **kwargs):
    """Profile function using cProfile"""
    print("=== cProfile Analysis ===")
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())
    
    return result

def profile_with_line_profiler(func, *args, **kwargs):
    """Profile function using line_profiler"""
    print("=== Line Profiler Analysis ===")
    lp = line_profiler.LineProfiler()
    lp_wrapper = lp(func)
    
    result = lp_wrapper(*args, **kwargs)
    
    lp.print_stats()
    return result

def profile_with_memory_profiler(func, *args, **kwargs):
    """Profile function using memory_profiler"""
    print("=== Memory Profiler Analysis ===")
    mp_wrapper = memory_profiler.profile(func)
    
    result = mp_wrapper(*args, **kwargs)
    return result

def manual_timing_analysis(func, *args, **kwargs):
    """Manual timing analysis of different parts of the function"""
    print("=== Manual Timing Analysis ===")
    
    # Create a mock data object for testing
    class MockData:
        class Session:
            def __init__(self):
                self.id = "perich_miller_population_2018/c_20131003_center_out_reaching"
        
        class SmoothedSpikes:
            def __init__(self):
                # Create realistic neural data
                self.smoothed_spikes = np.random.randn(100, 200).astype(np.float32)
        
        class Cursor:
            def __init__(self):
                # Create realistic behavior data
                self.vel = np.random.randn(100, 2).astype(np.float32)
        
        def __init__(self):
            self.session = self.Session()
            self.smoothed_spikes = self.SmoothedSpikes()
            self.cursor = self.Cursor()
    
    mock_data = MockData()
    
    # Time individual operations
    timings = {}
    
    # Time 1: Accessing smoothed spikes
    start = time.time()
    for _ in range(1000):
        _ = mock_data.smoothed_spikes.smoothed_spikes
    timings['access_smoothed_spikes'] = (time.time() - start) / 1000
    
    # Time 2: Accessing cursor velocity
    start = time.time()
    for _ in range(1000):
        _ = mock_data.cursor.vel
    timings['access_cursor_vel'] = (time.time() - start) / 1000
    
    # Time 3: _ensure_dim operations
    spikes = mock_data.smoothed_spikes.smoothed_spikes
    behavior = mock_data.cursor.vel
    
    start = time.time()
    for _ in range(100):
        _ = _ensure_dim(spikes, 353, axis=1)
    timings['ensure_dim_spikes_353'] = (time.time() - start) / 100
    
    start = time.time()
    for _ in range(100):
        _ = _ensure_dim(behavior, 2, axis=1)
    timings['ensure_dim_behavior_2'] = (time.time() - start) / 100
    
    start = time.time()
    for _ in range(100):
        _ = _ensure_dim(spikes, 100, axis=0)
    timings['ensure_dim_spikes_time'] = (time.time() - start) / 100
    
    start = time.time()
    for _ in range(100):
        _ = _ensure_dim(behavior, 100, axis=0)
    timings['ensure_dim_behavior_time'] = (time.time() - start) / 100
    
    # Time 4: parse_session_id
    start = time.time()
    for _ in range(1000):
        _ = parse_session_id(mock_data.session.id)
    timings['parse_session_id'] = (time.time() - start) / 1000
    
    # Time 5: Dictionary lookup
    dataset, subject, task = parse_session_id(mock_data.session.id)
    group_tuple = (dataset, subject, task)
    
    start = time.time()
    for _ in range(1000):
        _ = DATASET_GROUP_TO_IDX[group_tuple]
    timings['dict_lookup'] = (time.time() - start) / 1000
    
    # Time 6: torch.as_tensor conversions
    final_spikes = _ensure_dim(_ensure_dim(spikes, 353, axis=1), 100, axis=0)
    final_behavior = _ensure_dim(_ensure_dim(behavior, 2, axis=1), 100, axis=0)
    
    start = time.time()
    for _ in range(100):
        _ = torch.as_tensor(final_spikes, dtype=torch.float32)
    timings['torch_as_tensor_spikes'] = (time.time() - start) / 100
    
    start = time.time()
    for _ in range(100):
        _ = torch.as_tensor(final_behavior, dtype=torch.float32)
    timings['torch_as_tensor_behavior'] = (time.time() - start) / 100
    
    start = time.time()
    for _ in range(100):
        _ = torch.as_tensor(0, dtype=torch.int32)
    timings['torch_as_tensor_scalar'] = (time.time() - start) / 100
    
    # Print results
    print("Operation timing (seconds per operation):")
    for op, timing in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op}: {timing:.6f}")
    
    return timings

def analyze_memory_usage():
    """Analyze memory usage patterns"""
    print("=== Memory Usage Analysis ===")
    
    # Create realistic data sizes
    neural_shape = (100, 353)  # (timesteps, channels)
    behavior_shape = (100, 2)  # (timesteps, features)
    
    # Calculate memory usage
    neural_memory = neural_shape[0] * neural_shape[1] * 4  # float32 = 4 bytes
    behavior_memory = behavior_shape[0] * behavior_shape[1] * 4
    
    print(f"Neural data memory: {neural_memory / 1024:.2f} KB")
    print(f"Behavior data memory: {behavior_memory / 1024:.2f} KB")
    print(f"Total data memory: {(neural_memory + behavior_memory) / 1024:.2f} KB")
    
    # Memory overhead of numpy arrays vs torch tensors
    neural_np = np.random.randn(*neural_shape).astype(np.float32)
    neural_torch = torch.as_tensor(neural_np)
    
    print(f"Numpy array memory: {neural_np.nbytes / 1024:.2f} KB")
    print(f"Torch tensor memory: {neural_torch.element_size() * neural_torch.nelement() / 1024:.2f} KB")

def profile_real_data():
    """Profile with real data from the dataset"""
    print("=== Real Data Profiling ===")
    
    try:
        # Load configuration
        config_path = "/cs/student/projects1/ml/2024/mlaimon/foundational_ssm/configs/pretrain.yaml"
        cfg = OmegaConf.load(config_path)
        
        # Create dataset
        train_dataset = Dataset(
            root=DATA_ROOT,
            config=get_dataset_config(
                cfg.train_dataset.name,
                subjects=cfg.train_dataset.subjects
            ),
            split="train",
        )
        train_sampling_intervals = train_dataset.get_sampling_intervals()
        train_dataset.disable_data_leakage_check()
        
        # Create sampler and dataloader
        train_sampler = RandomFixedWindowSampler(
            sampling_intervals=train_sampling_intervals,
            window_length=1.0,
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            sampler=train_sampler,
            collate_fn=collate,
            num_workers=0,  # Use single worker for profiling
            pin_memory=False,  # Disable pin_memory for profiling
            batch_size=1,
        )
        
        # Profile first few samples
        print("Profiling first 5 samples from real dataset...")
        total_time = 0
        for i, batch in enumerate(train_loader):
            if i >= 5:
                break
                
            start_time = time.time()
            result = transform_brainsets_to_fixed_dim_samples(batch)
            end_time = time.time()
            
            sample_time = end_time - start_time
            total_time += sample_time
            
            print(f"Sample {i+1}: {sample_time:.4f}s")
            print(f"  Neural shape: {result['neural_input'].shape}")
            print(f"  Behavior shape: {result['behavior_input'].shape}")
        
        print(f"Average time per sample: {total_time / 5:.4f}s")
        
    except Exception as e:
        print(f"Error profiling real data: {e}")

def main():
    """Main profiling function"""
    print("Profiling transform_brainsets_to_fixed_dim_samples function")
    print("=" * 60)
    
    # Manual timing analysis
    manual_timing_analysis(transform_brainsets_to_fixed_dim_samples)
    print()
    
    # Memory usage analysis
    analyze_memory_usage()
    print()
    
    # Real data profiling
    profile_real_data()
    print()
    
    print("Profiling complete!")

if __name__ == "__main__":
    main() 