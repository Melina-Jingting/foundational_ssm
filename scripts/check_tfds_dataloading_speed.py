#!/usr/bin/env python3
"""
Script to check dataloading speed using the TFDS dataloader (tf.data.Dataset).
Mimics foundational_ssm/scripts/check_dataloading_speed.py but uses get_brainset_train_val_loaders_tfds.
"""

import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
from omegaconf import OmegaConf
import logging
import tensorflow as tf

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from foundational_ssm.data_utils.loaders import get_dataset_config
from foundational_ssm.data_utils.tfds_loader import get_brainset_train_val_loaders_tfds_native
from foundational_ssm.constants import DATA_ROOT

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tfds_dataloading_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading configuration...")
    config_start_time = time.time()
    config_path = "configs/pretrain.yaml"
    cfg = OmegaConf.load(config_path)
    config_load_time = time.time() - config_start_time
    logger.info(f"Config loaded in {config_load_time:.4f}s")

    logger.info("Creating Optimized TFDS train loader...")
    loader_start_time = time.time()
    train_dataset, train_loader, val_dataset, val_loader  = get_brainset_train_val_loaders_tfds_native(
        train_config=get_dataset_config(
            **cfg.train_dataset
        ),
        val_config=get_dataset_config(
            **cfg.val_dataset
        ),
        **cfg.dataloader
    )
    
    loader_creation_time = time.time() - loader_start_time
    logger.info(f"TFDS train loader created in {loader_creation_time:.4f}s")

    # Count number of batches (may be infinite, so cap at 100)
    logger.info("Testing iterator creation...")
    iterator_start_time = time.time()
    train_iter = iter(train_loader)
    iterator_creation_time = time.time() - iterator_start_time
    logger.info(f"Iterator created in {iterator_creation_time:.4f}s")

    print("Testing TFDS dataloading speed...")
    tf.profiler.experimental.start('logdir')

    loading_times = []
    num_batches_to_test = 10
    logger.info(f"Starting batch loading test for {num_batches_to_test} batches...")
    start_time = time.time()
    for i in tqdm(range(num_batches_to_test)):
        batch_start = time.time()
        try:
            batch = next(train_iter)
        except StopIteration:
            logger.info(f"End of dataset after {i} batches.")
            break
        batch_time = time.time() - batch_start
        loading_times.append(batch_time)
        # Log every 10th batch
        if (i + 1) % 10 == 0:
            logger.info(f"Loaded {i + 1}/{num_batches_to_test} batches, last batch took {batch_time:.4f}s")

    # Calculate statistics
    loading_times = np.array(loading_times)
    avg_time = np.mean(loading_times)
    std_time = np.std(loading_times)
    min_time = np.min(loading_times)
    max_time = np.max(loading_times)

    print("\n" + "="*50)
    print("TFDS DATALOADING SPEED RESULTS")
    print("="*50)
    print(f"Transform: transform_brainsets_to_fixed_dim_samples_with_binning_and_smoothing")
    print(f"Number of workers: {cfg.dataloader.num_workers}")
    print(f"Batch size: {cfg.dataloader.batch_size}")
    print(f"Number of batches tested: {len(loading_times)}")
    print(f"Average time per batch: {avg_time:.4f} seconds")
    print(f"Standard deviation: {std_time:.4f} seconds")
    print(f"Minimum time: {min_time:.4f} seconds")
    print(f"Maximum time: {max_time:.4f} seconds")
    print(f"Batches per second: {1.0/avg_time:.2f}")
    print("="*50)

    # Log timing breakdown
    logger.info("TIMING BREAKDOWN:")
    logger.info(f"  Config loading: {config_load_time:.4f}s")
    logger.info(f"  Loader creation: {loader_creation_time:.4f}s")
    logger.info(f"  Iterator creation: {iterator_creation_time:.4f}s")
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
    
    tf.profiler.experimental.stop()

    return loading_times

if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Run the test
    loading_times = main()
    print("\nTest completed!") 