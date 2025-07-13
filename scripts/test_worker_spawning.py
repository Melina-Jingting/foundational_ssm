#!/usr/bin/env python3
"""
Test script to measure worker spawning time with different numbers of workers.
"""

import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from foundational_ssm.data_utils.dataset import TorchBrainDataset
from foundational_ssm.data_utils.samplers import GroupedRandomFixedWindowSampler
from foundational_ssm.data_utils.loaders import get_dataset_config, transform_brainsets_to_fixed_dim_samples_with_binning_and_smoothing
from foundational_ssm.constants import DATA_ROOT
import multiprocessing as mp

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('worker_spawning_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_worker_spawning(num_workers):
    """Test worker spawning with a specific number of workers"""
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING WITH {num_workers} WORKERS")
    logger.info(f"{'='*60}")
    
    # Set up multiprocessing
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
    
    # Create sampler
    train_sampling_intervals = train_dataset.get_sampling_intervals()
    train_sampler = GroupedRandomFixedWindowSampler(
        sampling_intervals=train_sampling_intervals,
        window_length=1.0,
        batch_size=cfg.dataloader.batch_size,
        generator=torch.Generator().manual_seed(42)
    )
    
    # Test DataLoader creation
    logger.info(f"Creating DataLoader with {num_workers} workers...")
    dataloader_start = time.time()
    
    try:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        dataloader_creation_time = time.time() - dataloader_start
        logger.info(f"✓ DataLoader created in {dataloader_creation_time:.4f}s")
        
        # Test iterator creation (triggers worker spawning)
        logger.info("Creating iterator (triggers worker spawning)...")
        iterator_start = time.time()
        
        train_iter = iter(train_loader)
        iterator_creation_time = time.time() - iterator_start
        logger.info(f"✓ Iterator created in {iterator_creation_time:.4f}s")
        
        # Test first batch loading
        logger.info("Loading first batch...")
        first_batch_start = time.time()
        first_batch = next(train_iter)
        first_batch_time = time.time() - first_batch_start
        logger.info(f"✓ First batch loaded in {first_batch_time:.4f}s")
        
        total_time = dataloader_creation_time + iterator_creation_time + first_batch_time
        logger.info(f"✓ Total setup time: {total_time:.4f}s")
        
        return {
            'num_workers': num_workers,
            'dataloader_creation': dataloader_creation_time,
            'iterator_creation': iterator_creation_time,
            'first_batch': first_batch_time,
            'total_time': total_time,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"✗ Failed with {num_workers} workers: {e}")
        import traceback
        traceback.print_exc()
        return {
            'num_workers': num_workers,
            'success': False,
            'error': str(e)
        }

def main():
    """Test worker spawning with different numbers of workers"""
    logger.info("Worker Spawning Test")
    logger.info("="*60)
    
    # Test different numbers of workers
    worker_counts = [0, 1, 2, 4, 8, 16]
    results = []
    
    for num_workers in worker_counts:
        result = test_worker_spawning(num_workers)
        results.append(result)
        
        # If it fails, stop testing higher worker counts
        if not result['success']:
            logger.info(f"Stopping tests after failure with {num_workers} workers")
            break
    
    # Print summary
    print("\n" + "="*80)
    print("WORKER SPAWNING RESULTS")
    print("="*80)
    print(f"{'Workers':<8} {'DataLoader':<12} {'Iterator':<12} {'First Batch':<12} {'Total':<12} {'Status':<10}")
    print("-" * 80)
    
    for result in results:
        if result['success']:
            print(f"{result['num_workers']:<8} {result['dataloader_creation']:<12.4f} {result['iterator_creation']:<12.4f} {result['first_batch']:<12.4f} {result['total_time']:<12.4f} {'✓ PASS':<10}")
        else:
            print(f"{result['num_workers']:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'✗ FAIL':<10}")
    
    print("="*80)
    
    # Find the optimal number of workers
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['total_time'])
        logger.info(f"Optimal number of workers: {best_result['num_workers']} (total time: {best_result['total_time']:.4f}s)")
    else:
        logger.error("No successful worker configurations found")

if __name__ == "__main__":
    main() 