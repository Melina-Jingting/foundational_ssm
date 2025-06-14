#!/bin/bash
# Script to download NLB datasets to a configurable directory
# Usage: ./download_nlb_datasets.sh [data_dir]
#   data_dir: Optional target directory for downloads (default: $DATA_DIR env var or hardcoded path)

# Install dependencies
conda init
conda activate dandi
pip install dandi

# Load environment variables from .env file if it exists
ENV_FILE="../.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading configuration from $ENV_FILE"
    # Source the .env file
    source "$ENV_FILE"
fi

# Set data directory with priority:
# 1. Command line argument
# 2. Environment variable from .env
# 3. Hardcoded default
if [ -n "$1" ]; then
    DATA_DIR="$1"
elif [ -n "$FOUNDATIONAL_SSM_RAW_DATA_DIR" ]; then
    DATA_DIR="$FOUNDATIONAL_SSM_RAW_DATA_DIR"
else
    DATA_DIR="/cs/student/projects1/ml/2024/mlaimon/data/foundational_ssm/raw"
fi

# Create the directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Navigate to the target directory
cd "$DATA_DIR"

# Download datasets
echo "Downloading NLB datasets to: $DATA_DIR"
dandi download https://dandiarchive.org/dandiset/000128 
# dandi download https://dandiarchive.org/dandiset/000129
# dandi download https://dandiarchive.org/dandiset/000127
# dandi download https://dandiarchive.org/dandiset/000130

echo "Downloads complete."