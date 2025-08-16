#!/bin/bash
#SBATCH --job-name=pretrain_decoding
#SBATCH --output=slurm/logs/pretrain_decoding_%j.out
#SBATCH --error=slurm/logs/pretrain_decoding_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=gatsby_ws,gpu,gpu_lowp,a100
#SBATCH --nodelist=gpu-sr670-[20-23],gpu-xd670-30,gpu-sr675-34

# Print node information
echo "Job running on node: $SLURMD_NODENAME"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

# Create logs directory if it doesn't exist
mkdir -p slurm/logs

# Load necessary modules (modify as needed for your cluster)
module load cuda/12.5

# unset LD_LIBRARY_PATH
# echo "LD_LIBRARY_PATH after unset: '$LD_LIBRARY_PATH'"


source ~/anaconda3/etc/profile.d/conda.sh
conda activate foundational_ssm  # Replace with your environment name

export PYTHONPATH=/nfs/ghome/live/mlaimon/foundational_ssm/src:$PYTHONPATH
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

cd /nfs/ghome/live/mlaimon/foundational_ssm
python scripts/training/pretrain_decoding.py dataset_cfg=configs/dataset/reaching.yaml model_cfg=configs/model/l4.yaml +wandb.run_name_postfix=_normalized


conda deactivate
echo "Job completed"
