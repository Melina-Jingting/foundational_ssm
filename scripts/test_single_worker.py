#!/usr/bin/env python3
"""
Test script with single worker to isolate multiprocessing issues.
"""

import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from foundational_ssm.data_utils.dataset import TorchBrainDataset
from foundational_ssm.data_utils.samplers import GroupedRandomFixedWindowSampler
from foundational_ssm.data_utils.loaders import get_dataset_config, transform_brainsets_to_fixed_dim_samples_with_binning_and_smoothing
from foundational_ssm.constants import DATA_ROOT
import multiprocessing as mp

def test_single_worker():
    """Test with single worker to isolate multiprocessing issues"""
    print("\n" + "="*60)
    print("TEST: Single Worker DataLoader")
    print("="*60)
    
    mp.set_start_method("spawn", force=True)
    
    # Load config
    config_path = "configs/pretrain.yaml"
    cfg = OmegaConf.load(config_path)
    print(f"Config loaded: {type(cfg)}")
    
    # Create dataset
    train_dataset = TorchBrainDataset(
        root=DATA_ROOT,
        config=get_dataset_config(**cfg.train_dataset),
        lazy=cfg.dataloader.lazy,
        keep_files_open=cfg.dataloader.keep_files_open,
    )
    print(f"Dataset created: {len(train_dataset)} samples")
    
    # Set transform
    train_dataset.transform = transform_brainsets_to_fixed_dim_samples_with_binning_and_smoothing
    
    # Create sampler and loader with SINGLE worker
    train_sampling_intervals = train_dataset.get_sampling_intervals()
    train_sampler = GroupedRandomFixedWindowSampler(
        sampling_intervals=train_sampling_intervals,
        window_length=1.0,
        batch_size=cfg.dataloader.batch_size,
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,  # Single worker (main process)
        pin_memory=True,
    )
    
    print(f"DataLoader created: {len(train_loader)} batches")
    
    # Test loading first batch
    print("Testing first batch loading...")
    start_time = time.time()
    try:
        first_batch = next(iter(train_loader))
        load_time = time.time() - start_time
        print(f"✓ First batch loaded successfully in {load_time:.4f}s")
        print(f"  Batch keys: {list(first_batch.keys())}")
        return True
    except Exception as e:
        print(f"✗ Failed to load first batch: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiprocessing_workers():
    """Test with different numbers of workers"""
    print("\n" + "="*60)
    print("TEST: Different Numbers of Workers")
    print("="*60)
    
    mp.set_start_method("spawn", force=True)
    
    # Load config
    config_path = "configs/pretrain.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Create dataset
    train_dataset = TorchBrainDataset(
        root=DATA_ROOT,
        config=get_dataset_config(**cfg.train_dataset),
        lazy=cfg.dataloader.lazy,
        keep_files_open=cfg.dataloader.keep_files_open,
    )
    train_dataset.transform = transform_brainsets_to_fixed_dim_samples_with_binning_and_smoothing
    
    # Test different numbers of workers
    worker_counts = [0, 1, 2, 4, 8, 16]
    
    for num_workers in worker_counts:
        print(f"\nTesting with {num_workers} workers...")
        
        train_sampling_intervals = train_dataset.get_sampling_intervals()
        train_sampler = GroupedRandomFixedWindowSampler(
            sampling_intervals=train_sampling_intervals,
            window_length=1.0,
            batch_size=cfg.dataloader.batch_size,
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        print(f"DataLoader created: {len(train_loader)} batches")
        
        # Test loading first batch
        start_time = time.time()
        try:
            first_batch = next(iter(train_loader))
            load_time = time.time() - start_time
            print(f"✓ {num_workers} workers: First batch loaded in {load_time:.4f}s")
        except Exception as e:
            print(f"✗ {num_workers} workers: Failed - {e}")
            break

def main():
    """Run multiprocessing tests"""
    print("Multiprocessing Isolation Tests")
    print("="*60)
    
    # Test single worker
    single_worker_success = test_single_worker()
    
    # Test different worker counts
    test_multiprocessing_workers()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if single_worker_success:
        print("✓ Single worker works - multiprocessing is the issue")
    else:
        print("✗ Single worker fails - issue is elsewhere")

if __name__ == "__main__":
    main() 