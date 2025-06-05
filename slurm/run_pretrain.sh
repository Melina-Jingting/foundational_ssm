#!/bin/bash
#SBATCH --job-name=ssm_pretrain
#SBATCH --output=logs/ssm_pretrain_%j.out
#SBATCH --error=logs/ssm_pretrain_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
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

# Change to the project directory
cd /nfs/ghome/live/mlaimon/foundational_ssm

# Run the pre-training script
# You can pass additional command-line arguments as needed
python scripts/pre_train.py \
  --config cmt \
  --run_name "cmt_run_${SLURM_JOB_ID}" \
  --output_dir results/cmt_${SLURM_JOB_ID} \
  --epochs 200 

# Deactivate the environment
conda deactivate

echo "Job completed"