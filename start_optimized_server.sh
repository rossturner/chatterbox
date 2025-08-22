# Script to start the optimized Chatterbox server

echo "======================================"
echo "Starting Optimized Chatterbox Server"
echo "======================================"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Set performance environment variables
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'

# Start the server
python -m src.server.main