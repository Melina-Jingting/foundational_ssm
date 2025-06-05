#!/bin/bash
#SBATCH --job-name=nlb_ssm
#SBATCH --output=logs/nlb_ssm_%j.out
#SBATCH --error=logs/nlb_ssm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu  # Change to your cluster's GPU partition name

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

# Ensure wandb API key is available (update with your key)
export WANDB_API_KEY=your_wandb_api_key_here

# Change to the project directory
cd /nfs/ghome/live/mlaimon/foundational_ssm

# Run the NLB script with wandb logging
python scripts/nlb.py \
  --dataset mc_maze_small \
  --phase val \
  --batch_size 32 \
  --epochs 100 \
  --lr 0.001 \
  --hidden_dim 128 \
  --num_layers 2 \
  --wandb

# Deactivate the environment
conda deactivate

echo "Job completed"