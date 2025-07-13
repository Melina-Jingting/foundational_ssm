# Dataloading Speed Test Scripts

This directory contains scripts to test and compare dataloading speeds for the foundational SSM project.

## Scripts

### 1. `check_dataloading_speed.py`

A simple script to test the speed of `transform_brainsets_to_fixed_dim_samples_with_binning_and_smoothing`.

**Usage:**
```bash
cd foundational_ssm/scripts
python check_dataloading_speed.py
```

**Features:**
- Tests the specific transform used in the project
- Provides detailed statistics (mean, std, min, max, percentiles)
- Outlier detection using IQR method
- Configurable number of batches to test

### 2. `check_dataloading_speed_comparison.py`

A comprehensive comparison script that tests multiple transforms and configurations.

**Usage:**
```bash
cd foundational_ssm/scripts

# Basic usage with default settings
python check_dataloading_speed_comparison.py

# Custom configuration
python check_dataloading_speed_comparison.py \
    --num-batches 100 \
    --num-workers 8 \
    --batch-size 512 \
    --config-path /path/to/your/config.yaml
```

**Command line arguments:**
- `--num-batches`: Number of batches to test (default: 50)
- `--num-workers`: Number of workers for DataLoader (default: 16)
- `--batch-size`: Batch size (default: 1024)
- `--config-path`: Path to config file (default: pretrain.yaml)

**Features:**
- Compares multiple transforms:
  - `transform_brainsets_to_fixed_dim_samples_with_binning_and_smoothing`
  - `transform_brainsets_to_fixed_dim_samples`
  - POYO tokenize (if available)
- Detailed comparison table
- Statistical analysis for each transform
- Outlier detection

## Example Output

```
================================================================================
DATALOADING SPEED COMPARISON RESULTS
================================================================================
Transform                                              Avg (s)     Std (s)     Min (s)     Max (s)     Batches/s  
----------------------------------------------------------------------------------------------------
transform_brainsets_to_fixed_dim_samples_with_binning 0.2133      0.0456      0.1234      0.3456      4.69       
transform_brainsets_to_fixed_dim_samples              0.1891      0.0321      0.1123      0.2987      5.29       
POYO tokenize                                         0.4567      0.1234      0.2345      0.7890      2.19       
```

## Configuration

The scripts use the same configuration as your training scripts:
- Dataset configuration from `configs/pretrain.yaml`
- DataLoader settings (workers, batch size, etc.)
- Transform functions from `foundational_ssm.data_utils.loaders`

## Troubleshooting

1. **Import errors**: Make sure you're running from the `foundational_ssm/scripts` directory
2. **CUDA errors**: The scripts will automatically use CPU if CUDA is not available
3. **Memory issues**: Reduce batch size or number of workers if you encounter memory problems
4. **Slow performance**: Try different numbers of workers to find the optimal configuration

## Performance Tips

1. **Number of workers**: Usually 2-4x the number of CPU cores works well
2. **Batch size**: Larger batches are generally more efficient, but require more memory
3. **Pin memory**: Keep `pin_memory=True` for GPU training
4. **Lazy loading**: Use `lazy=True` for large datasets to reduce memory usage

## Expected Performance

Based on the notebook results:
- `transform_brainsets_to_fixed_dim_samples_with_binning_and_smoothing`: ~0.21 seconds/batch
- `transform_brainsets_to_fixed_dim_samples`: ~0.19 seconds/batch
- POYO tokenize: ~0.46 seconds/batch

Your actual performance may vary depending on your hardware and configuration. 