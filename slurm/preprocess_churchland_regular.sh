#!/bin/bash
#SBATCH --job-name=prepare_brainsets
#SBATCH --output=slurm/logs/preprocess_churchland_%j.out
#SBATCH --error=slurm/logs/preprocess_churchland_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

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


# Change to the project directory
cd /nfs/ghome/live/mlaimon/foundational_ssm 

python scripts/preprocess_churchland.py

# Deactivate the environment
conda deactivate

echo "Job completed"
