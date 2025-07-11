#!/bin/bash
#SBATCH --job-name=pretrain_decoding
#SBATCH --output=slurm/logs/pretrain_decoding_%j.out
#SBATCH --error=slurm/logs/pretrain_decoding_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gatsby_ws,gpu,gpu_lowp,a100

# Print node information
echo "Job running on node: $SLURMD_NODENAME"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"
echo "SLURM_TMPDIR: $SLURM_TMPDIR"

# Create logs directory if it doesn't exist
mkdir -p slurm/logs

# Load necessary modules (modify as needed for your cluster)
# module load cuda/12.5

unset LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH after unset: '$LD_LIBRARY_PATH'"

# Copy project and data to local storage for faster access
echo "Copying project to local storage..."
if [ -z "$SLURM_TMPDIR" ]; then
    echo "WARNING: SLURM_TMPDIR is not set. Using /tmp instead."
    SLURM_TMPDIR="/tmp"
fi

cp -r /nfs/ghome/live/mlaimon/foundational_ssm $SLURM_TMPDIR/

echo "Copying data to local storage..."
# Check available space in SLURM_TMPDIR
TMPDIR_SIZE=$(df $SLURM_TMPDIR | awk 'NR==2 {print $4}')
DATA_SIZE=$(du -s /nfs/ghome/live/mlaimon/data/foundational_ssm/processed | awk '{print $1}')
PROJECT_SIZE=$(du -s /nfs/ghome/live/mlaimon/foundational_ssm | awk '{print $1}')
TOTAL_NEEDED=$((DATA_SIZE + PROJECT_SIZE))

echo "Available space in SLURM_TMPDIR: ${TMPDIR_SIZE}KB"
echo "Data size: ${DATA_SIZE}KB"
echo "Project size: ${PROJECT_SIZE}KB"
echo "Total needed: ${TOTAL_NEEDED}KB"

if [ $TOTAL_NEEDED -gt $TMPDIR_SIZE ]; then
    echo "WARNING: Not enough space in SLURM_TMPDIR. Using network storage."
    cd /nfs/ghome/live/mlaimon/foundational_ssm
    export DATA_ROOT="/nfs/ghome/live/mlaimon/data/foundational_ssm/processed"
else
    # Copy the data directory to local storage
    cp -r /nfs/ghome/live/mlaimon/data/foundational_ssm/processed/* $SLURM_TMPDIR/foundational_ssm/data/
    cd $SLURM_TMPDIR/foundational_ssm
    export DATA_ROOT="$SLURM_TMPDIR/foundational_ssm/data"
fi

echo "Working directory: $(pwd)"
echo "Data directory: $(ls -la data/ 2>/dev/null || echo 'Using network storage')"

# Debug: Check if data files were copied correctly
if [ -d "data" ]; then
    echo "=== Data directory contents ==="
    ls -la data/
    echo "=== Perich Miller data ==="
    ls -la data/perich_miller_population_2018/ 2>/dev/null || echo "No perich_miller_population_2018 directory found"
    echo "=== Available space in SLURM_TMPDIR ==="
    df -h $SLURM_TMPDIR
    echo "=== DATA_ROOT environment variable ==="
    echo "DATA_ROOT=$DATA_ROOT"
else
    echo "WARNING: Data directory not found in local storage"
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate foundational_ssm  # Replace with your environment name

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# python scripts/pretrain_decoding.py dataloader.num_workers=16 model.ssm_num_layers=1 model.ssm_dim=128 model.ssm_io_dim=128 wandb.resume_run_id=axszehd9
# python scripts/pretrain_decoding.py dataloader.num_workers=16 model.ssm_num_layers=4 model.ssm_dim=128 model.ssm_io_dim=128 wandb.resume_run_id=bvdr2jt7
python scripts/pretrain_decoding.py dataloader.num_workers=16 model.ssm_num_layers=4 model.ssm_dim=64 model.ssm_io_dim=64 wandb.resume_run_id=yjxivxo2

# Copy any important results back to persistent storage
echo "Copying results back to persistent storage..."
if [ -d "logs" ]; then
    cp -r logs /nfs/ghome/live/mlaimon/foundational_ssm/slurm/
fi
if [ -f "*.ckpt" ]; then
    cp *.ckpt /nfs/ghome/live/mlaimon/foundational_ssm/slurm/
fi
if [ -f "*.eqx" ]; then
    cp *.eqx /nfs/ghome/live/mlaimon/foundational_ssm/slurm/
fi

conda deactivate
echo "Job completed"
