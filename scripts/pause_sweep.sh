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

WANDB_SWEEP="wandb agent $SWEEP_ID"

echo "Stopping wandb agents on hosts: ${HOSTS[*]} for sweep: $WANDB_SWEEP"
echo ""

# Function to start agent on a host
stop_agent() {
    local host=$1
    echo "Stopping on $host..."
    ssh "$host" << EOF
        tmux kill-session -t wandb_sweep
EOF
}

# Start agents on all specified hosts
for host in "${HOSTS[@]}"; do
    stop_agent "$host"
    sleep 2
done

