#!/usr/bin/env python3
"""
Script to check dataloading speed using transform_brainsets_to_fixed_dim_samples_with_binning_and_smoothing.
Based on the poyo_test.ipynb notebook.
"""

import os
import sys
import time
import numpy as np
from tqdm import tqdm
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
from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler
import multiprocessing as mp

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataloading_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to test dataloading speed."""
    logger.info("Setting up multiprocessing...")
    mp_start_time = time.time()
    mp.set_start_method("spawn", force=True)
    mp_setup_time = time.time() - mp_start_time
    logger.info(f"Multiprocessing setup completed in {mp_setup_time:.4f}s")
    
    logger.info("Loading configuration...")
    config_start_time = time.time()
    config_path = "configs/pretrain.yaml"
    cfg = OmegaConf.load(config_path)
    config_load_time = time.time() - config_start_time
    logger.info(f"Config loaded in {config_load_time:.4f}s")
    
    logger.info("Creating dataset...")
    dataset_start_time = time.time()
    train_dataset = TorchBrainDataset(
        root= DATA_ROOT,
        config=get_dataset_config(**cfg.train_dataset),
        lazy=cfg.dataloader.lazy,
        keep_files_open=cfg.dataloader.keep_files_open,
    )
    dataset_creation_time = time.time() - dataset_start_time
    logger.info(f"Dataset created in {dataset_creation_time:.4f}s ")
    
    train_sampling_intervals = train_dataset.get_sampling_intervals()
    logger.info(f"Sampling intervals created - {len(train_sampling_intervals)} intervals")
    
    logger.info("Setting up data loader...")
    sampler_start_time = time.time()
    train_sampler = RandomFixedWindowSampler(
        sampling_intervals=train_sampling_intervals,
        window_length=1.0,
        # batch_size=1024,
        # generator=torch.Generator().manual_seed(42)
    )
    sampler_creation_time = time.time() - sampler_start_time
    logger.info(f"Sampler created in {sampler_creation_time:.4f}s")
    
    # Set the transform
    transform_start_time = time.time()
    train_dataset.transform = transform_brainsets_to_fixed_dim_samples_with_binning_and_smoothing
    transform_setup_time = time.time() - transform_start_time
    logger.info(f"Transform set in {transform_setup_time:.4f}s")
    
    logger.info(f"Creating DataLoader with {cfg.dataloader.num_workers} workers...")
    dataloader_start_time = time.time()
    
    # Add detailed logging for DataLoader creation
    logger.info("About to create DataLoader - this is where worker spawning happens")
    
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        batch_size=1024,
    )
    
    dataloader_creation_time = time.time() - dataloader_start_time
    logger.info(f"DataLoader created in {dataloader_creation_time:.4f}s")
    logger.info(f"Number of batches: {len(train_loader)}")
    
    # Test worker spawning by creating iterator
    logger.info("Testing iterator creation (triggers worker spawning)...")
    iterator_start_time = time.time()
    try:
        train_iter = iter(train_loader)
        iterator_creation_time = time.time() - iterator_start_time
        logger.info(f"Iterator created successfully in {iterator_creation_time:.4f}s")
        
        # Test first batch loading
        logger.info("Testing first batch loading...")
        first_batch_start_time = time.time()
        first_batch = next(train_iter)
        first_batch_time = time.time() - first_batch_start_time
        logger.info(f"First batch loaded in {first_batch_time:.4f}s")
        logger.info(f"First batch keys: {list(first_batch.keys())}")
        
    except Exception as e:
        logger.error(f"Failed to create iterator or load first batch: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("Testing dataloading speed...")
    print(f"Number of batches: {len(train_loader)}")
    
    # Measure loading times
    loading_times = []
    num_batches_to_test = min(100, len(train_loader))  # Test up to 100 batches
    
    logger.info(f"Starting batch loading test for {num_batches_to_test} batches...")
    start_time = time.time()
    for i, batch in tqdm(enumerate(train_loader), total=num_batches_to_test):
        batch_time = time.time() - start_time
        loading_times.append(batch_time)
        start_time = time.time()
        
        # Log every 10th batch for monitoring
        if (i + 1) % 10 == 0:
            logger.info(f"Loaded {i + 1}/{num_batches_to_test} batches, last batch took {batch_time:.4f}s")
        
        if i >= num_batches_to_test - 1:
            break
    
    # Calculate statistics
    loading_times = np.array(loading_times)
    avg_time = np.mean(loading_times)
    std_time = np.std(loading_times)
    min_time = np.min(loading_times)
    max_time = np.max(loading_times)
    
    print("\n" + "="*50)
    print("DATALOADING SPEED RESULTS")
    print("="*50)
    print(f"Transform: transform_brainsets_to_fixed_dim_samples_with_binning_and_smoothing")
    print(f"Number of workers: {cfg.dataloader.num_workers}")
    print(f"Batch size: 1024")
    print(f"Number of batches tested: {len(loading_times)}")
    print(f"Average time per batch: {avg_time:.4f} seconds")
    print(f"Standard deviation: {std_time:.4f} seconds")
    print(f"Minimum time: {min_time:.4f} seconds")
    print(f"Maximum time: {max_time:.4f} seconds")
    print(f"Batches per second: {1.0/avg_time:.2f}")
    print("="*50)
    
    # Log timing breakdown
    logger.info("TIMING BREAKDOWN:")
    logger.info(f"  Multiprocessing setup: {mp_setup_time:.4f}s")
    logger.info(f"  Config loading: {config_load_time:.4f}s")
    logger.info(f"  Dataset creation: {dataset_creation_time:.4f}s")
    logger.info(f"  Sampler creation: {sampler_creation_time:.4f}s")
    logger.info(f"  Transform setup: {transform_setup_time:.4f}s")
    logger.info(f"  DataLoader creation: {dataloader_creation_time:.4f}s")
    logger.info(f"  Iterator creation: {iterator_creation_time:.4f}s")
    logger.info(f"  First batch loading: {first_batch_time:.4f}s")
    logger.info(f"  Average batch loading: {avg_time:.4f}s")
    
    # Additional analysis
    print("\nPERCENTILE ANALYSIS:")
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(loading_times, p)
        print(f"{p}th percentile: {value:.4f} seconds")
    
    # Check for outliers
    q1 = np.percentile(loading_times, 25)
    q3 = np.percentile(loading_times, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = loading_times[(loading_times < lower_bound) | (loading_times > upper_bound)]
    print(f"\nOutliers (using IQR method): {len(outliers)} out of {len(loading_times)} batches ({len(outliers)/len(loading_times)*100:.1f}%)")
    
    if len(outliers) > 0:
        print(f"Outlier times: {outliers}")
    
    return loading_times


if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run the test
    loading_times = main()
    
    print("\nTest completed!") 