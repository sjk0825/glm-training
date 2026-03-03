#!/bin/bash

set -e

echo "========================================"
echo "GLM-4.7 Fine-tuning on RTX 4090 (24GB)"
echo "========================================"

# Check if W&B API key is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY is not set"
    echo "Please set your W&B API key:"
    echo "  export WANDB_API_KEY=your_api_key_here"
    exit 1
fi

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Run training
echo "Starting training..."
cd "$(dirname "$0")"
python -m src.train configs/train.yaml

echo "========================================"
echo "Training complete!"
echo "========================================"
