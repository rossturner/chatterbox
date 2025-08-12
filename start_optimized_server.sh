#!/bin/bash
# Script to start the optimized Chatterbox server

echo "======================================"
echo "Starting Optimized Chatterbox Server"
echo "======================================"
echo ""
echo "This will:"
echo "1. Load the model with optimizations"
echo "2. Apply BFloat16, reduced cache, and torch.compile"
echo "3. Perform warmup compilation (25-30s one-time cost)"
echo "4. Start serving requests with 3-4x better performance"
echo ""
echo "Starting server..."
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Set performance environment variables
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# Start the server
python -m src.server.main