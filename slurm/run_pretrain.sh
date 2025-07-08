#!/bin/bash
#SBATCH --job-name=pretrain_decoding
#SBATCH --output=logs/pretrain_decoding_%j.out
#SBATCH --error=logs/pretrain_decoding_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gatsby_ws,gpu,gpu_lowp,a100

# Print node information
echo "Job running on node: $SLURMD_NODENAME"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (modify as needed for your cluster)
# module load cuda/11.7 cudnn/8.4.1 python/3.9

# Activate your conda environment (modify path as needed)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate foundational_ssm  # Replace with your environment name

# Set the PYTHONPATH to include your project root
export PYTHONPATH=/nfs/ghome/live/mlaimon/foundational_ssm/src:$PYTHONPATH
export PYTHONUNBUFFERED=1

# Change to the project directory
cd /nfs/ghome/live/mlaimon/foundational_ssm

# Run the pre-training script
# You can pass additional command-line arguments as needed
export HYDRA_FULL_ERROR=1
python scripts/pretrain_decoding.py dataloader.num_workers=16

# Deactivate the environment
conda deactivate

echo "Job completed"
