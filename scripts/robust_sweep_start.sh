#!/bin/bash

# Robust script to start wandb agents on multiple hosts
# Usage: ./robust_sweep_start.sh

SWEEP_ID="wandb agent melinajingting-ucl/foundational_ssm_rtt_sweep/mf8jktbw"

echo "Starting wandb agents on multiple hosts for sweep: $SWEEP_ID"
echo ""

# Function to start agent on a host
start_agent() {
    local host=$1
    echo "Starting on $host..."
    
    ssh "$host" << 'EOF'
        # Kill any existing session first
        tmux kill-session -t wandb_sweep 
        
        # Change to the working directory
        cd /cs/student/projects1/ml/2024/mlaimon/foundational_ssm
        
        # Start new tmux session with proper shell
        tmux new-session -d -s wandb_sweep
        
        # Send commands to activate conda and start wandb agent
        tmux send-keys -t wandb_sweep 'source ~/.bashrc' Enter
        tmux send-keys -t wandb_sweep 'conda activate foundational_ssm' Enter
        tmux send-keys -t wandb_sweep 'wandb agent melinajingting-ucl/foundational_ssm_rtt_sweep/mf8jktbw' Enter
        
        echo "Started wandb agent on $(hostname)"
EOF
}

# Start agents on all hosts
start_agent "javelin"
sleep 2

start_agent "koi"
sleep 2

start_agent "plaice"
sleep 2


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
for host in javelin koi plaice; do
    echo "  ssh $host 'tmux list-sessions | grep wandb'"
done
