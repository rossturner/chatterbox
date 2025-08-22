# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Copy pyproject.toml
COPY pyproject.toml /app/

# Install Python dependencies
RUN pip install --no-cache-dir \
    "numpy~=1.26.0" \
    "resampy==0.4.3" \
    "librosa==0.10.0" \
    "s3tokenizer" \
    "torch==2.6.0" \
    "torchaudio==2.6.0" \
    "transformers==4.46.3" \
    "diffusers==0.29.0" \
    "omegaconf==2.3.0" \
    "conformer==0.3.2" \
    "matplotlib" \
    "whisper-openai" \
    "jiwer" \
    "sounddevice==0.5.2" \
    "faster-whisper>=1.0.0" \
    "fastapi>=0.100.0" \
    "uvicorn[standard]>=0.20.0" \
    "pydantic>=2.0.0" \
    "PyYAML>=6.0.0" \
    "websockets>=11.0.0" \
    "requests>=2.30.0" \
    "tabulate>=0.9.0" \
    "python-multipart"

# Copy source code
COPY src/ /app/src/

# Copy models
COPY models/nicole_v2/lora_v2_2/ /app/models/nicole_v2/lora_v2_2/

# Copy configs
COPY configs/ /app/configs/

# Copy conditionals cache
COPY conditionals_cache/ /app/conditionals_cache/

# Expose port 8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the default command directly
CMD ["python", "-m", "src.server.main"]