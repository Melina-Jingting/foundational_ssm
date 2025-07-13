#!/usr/bin/env python3
"""
Test script to isolate configuration loading differences between working and non-working scripts.
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
from foundational_ssm.data_utils.loaders import get_dataset_config, transform_brainsets_to_fixed_dim_samples_with_binning_and_smoothing, get_brainset_train_val_loaders
from foundational_ssm.constants import DATA_ROOT
import multiprocessing as mp

def test_direct_config_loading():
    """Test using direct OmegaConf loading (like speed test)"""
    print("\n" + "="*60)
    print("TEST 1: Direct OmegaConf Loading (like speed test)")
    print("="*60)
    
    mp.set_start_method("spawn", force=True)
    
    # Load config directly
    config_path = "configs/pretrain.yaml"
    cfg = OmegaConf.load(config_path)
    print(f"Config loaded: {type(cfg)}")
    print(f"Train dataset config: {cfg.train_dataset}")
    
    # Create dataset manually
    train_dataset = TorchBrainDataset(
        root=DATA_ROOT,
        config=get_dataset_config(**cfg.train_dataset),
        lazy=cfg.dataloader.lazy,
        keep_files_open=cfg.dataloader.keep_files_open,
    )
    print(f"Dataset created")
    
    # Set transform
    train_dataset.transform = transform_brainsets_to_fixed_dim_samples_with_binning_and_smoothing
    
    # Create sampler and loader
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
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
    )
    
    print(f"DataLoader created")
    
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

def test_loader_function():
    """Test using get_brainset_train_val_loaders function (like training script)"""
    print("\n" + "="*60)
    print("TEST 2: Using get_brainset_train_val_loaders Function")
    print("="*60)
    
    mp.set_start_method("spawn", force=True)
    
    # Load config directly
    config_path = "configs/pretrain.yaml"
    cfg = OmegaConf.load(config_path)
    print(f"Config loaded: {type(cfg)}")
    
    # Use the loader function
    try:
        train_dataset, train_loader, val_dataset, val_loader = get_brainset_train_val_loaders(
            train_config=get_dataset_config(**cfg.train_dataset),
            val_config=get_dataset_config(**cfg.val_dataset),
            **cfg.dataloader
        )
        print(f"DataLoader created: {len(train_loader)} batches")
        
        # Test loading first batch
        print("Testing first batch loading...")
        start_time = time.time()
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

def test_hydra_simulation():
    """Test simulating Hydra configuration processing"""
    print("\n" + "="*60)
    print("TEST 3: Simulating Hydra Configuration Processing")
    print("="*60)
    
    mp.set_start_method("spawn", force=True)
    
    # Simulate how Hydra processes the config
    config_path = "configs/pretrain.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Convert to container and back (like Hydra does)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(config_dict)
    print(f"Config processed: {type(cfg)}")
    
    # Create dataset manually
    train_dataset = TorchBrainDataset(
        root=DATA_ROOT,
        config=get_dataset_config(**cfg.train_dataset),
        lazy=cfg.dataloader.lazy,
        keep_files_open=cfg.dataloader.keep_files_open,
    )
    print(f"Dataset created: {len(train_dataset)} samples")
    
    # Set transform
    train_dataset.transform = transform_brainsets_to_fixed_dim_samples_with_binning_and_smoothing
    
    # Create sampler and loader
    train_sampling_intervals = train_dataset.get_sampling_intervals()
    train_sampler = GroupedRandomFixedWindowSampler(
        sampling_intervals=train_sampling_intervals,
        window_length=1.0,
        batch_size=4,
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
    )
    
    print(f"DataLoader created")
    
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

def main():
    """Run all tests to isolate the issue"""
    print("Configuration Loading Isolation Tests")
    print("="*60)
    
    results = {}
    
    # Test 1: Direct config loading (like speed test)
    results['direct'] = test_direct_config_loading()
    
    # Test 2: Using loader function
    results['loader_function'] = test_loader_function()
    
    # Test 3: Hydra simulation
    results['hydra_simulation'] = test_hydra_simulation()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    if all(results.values()):
        print("\nAll tests passed - the issue might be elsewhere")
    else:
        print("\nSome tests failed - this helps isolate the problem")

if __name__ == "__main__":
    main() 