#!/bin/bash

# Robust script to start wandb agents on multiple hosts
# Usage: ./robust_sweep_start.sh <SWEEP_ID> <HOST1> [HOST2 HOST3 ...]

if [ -z "$1" ]; then
    echo "Usage: $0 <SWEEP_ID> <HOST1> [HOST2 HOST3 ...]"
    echo "Example: $0 qn0h7132 dory lamprey zander rudd"
    exit 1
fi

SWEEP_ID="$1"
shift
HOSTS=("$@")

if [ ${#HOSTS[@]} -eq 0 ]; then
    echo "Please provide at least one host."
    exit 1
fi

WANDB_SWEEP="wandb agent melinajingting-ucl/foundational_ssm_rtt_sweep/$SWEEP_ID"

echo "Starting wandb agents on hosts: ${HOSTS[*]} for sweep: $WANDB_SWEEP"
echo ""

# Function to start agent on a host
start_agent() {
    local host=$1
    echo "Starting on $host..."
    
    ssh "$host" << EOF
        tmux kill-session -t wandb_sweep
        cd /cs/student/projects1/ml/2024/mlaimon/foundational_ssm
        tmux new-session -d -s wandb_sweep
        tmux send-keys -t wandb_sweep 'conda activate foundational_ssm' Enter
        tmux send-keys -t wandb_sweep '$WANDB_SWEEP' Enter
        echo "Started wandb agent on \$(hostname)"
EOF
}

# Start agents on all specified hosts
for host in "${HOSTS[@]}"; do
    start_agent "$host"
    sleep 2
done

echo ""
echo "All agents started!"
echo ""
echo "To check on any agent:"
echo "  ssh <hostname>"
echo "  tmux attach -s wandb_sweep"
echo ""
echo "To stop an agent:"
echo "  ssh <hostname>"  
echo "  tmux kill-session -t wandb_sweep"
echo ""
echo "To check if agents are running:"
for host in "${HOSTS[@]}"; do
    echo "  ssh $host 'tmux list-sessions | grep wandb'"
done
