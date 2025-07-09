#!/bin/bash
#SBATCH --job-name=debug_env
#SBATCH --output=logs/debug_env_%j.out
#SBATCH --error=logs/debug_env_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gatsby_ws,gpu,gpu_lowp,a100

echo "--- Job running on node: $SLURMD_NODENAME ---"
mkdir -p logs

# --- Environment Setup ---
module load cuda/12.5
source ~/anaconda3/etc/profile.d/conda.sh
conda activate foundational_ssm

# --- DEBUGGING COMMANDS ---
echo -e "\n\n--- 1. LD_LIBRARY_PATH ---"
echo $LD_LIBRARY_PATH | tr ':' '\n'

echo -e "\n\n--- 2. Which Python ---"
which python

echo -e "\n\n--- 3. LDD on libtorch_cpu.so ---"
ldd /nfs/ghome/live/mlaimon/anaconda3/envs/foundational_ssm/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so | grep 'libi'

echo -e "\n\n--- End of Debug ---"