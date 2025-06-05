#!/bin/bash
#SBATCH --job-name=ssm_sweep
#SBATCH --output=logs/ssm_sweep_%j.out
#SBATCH --error=logs/ssm_sweep_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Print node information
echo "Job running on node: $SLURMD_NODENAME"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate your conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate foundational_ssm

# Set the PYTHONPATH to include your project root
export PYTHONPATH=/nfs/ghome/live/mlaimon/foundational_ssm:$PYTHONPATH

# Change to the project directory
cd /nfs/ghome/live/mlaimon/foundational_ssm

# Run the sweep
python scripts/run_sweep.py --model cmt --count 20 --method bayes

# Deactivate the environment
conda deactivate
echo "Job completed"