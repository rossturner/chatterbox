# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

**⚠️ IMPORTANT: Always use the project's virtual environment (.venv) which has Python 3.10.12**

```bash
# ALWAYS activate the virtual environment first (Python 3.10.12)
source .venv/bin/activate

# Verify you're using the correct Python version
python --version  # Should show Python 3.10.12

# Install for development with all dependencies
pip install -e .
```

**Note**: 
- The virtual environment uses Python 3.10.12 (NOT the system Python)
- Always activate `.venv` before running any commands
- Installation includes PyTorch with CUDA 12.4 support and many ML dependencies (~4GB total)
- The package installs successfully and imports without errors

## Core Architecture

This is a streaming TTS (Text-to-Speech) implementation of Chatterbox, featuring:

- **T3 (Token-to-Token) Model**: Llama-based backbone that converts text tokens to speech tokens
- **S3Gen**: Converts speech tokens to mel spectrograms and then to audio waveforms
- **S3Tokenizer**: Handles speech token encoding/decoding 
- **Voice Encoder**: Generates speaker embeddings for voice cloning
- **Streaming Implementation**: Real-time audio generation with chunked processing

### Key Components

- `src/chatterbox/tts.py`: Main ChatterboxTTS class with streaming support
- `src/chatterbox/vc.py`: Voice conversion functionality 
- `src/chatterbox/models/t3/`: T3 model implementation (text-to-speech-tokens)
- `src/chatterbox/models/s3gen/`: S3Gen model (speech-tokens-to-audio)
- `src/chatterbox/models/s3tokenizer/`: Speech tokenization
- `src/chatterbox/models/voice_encoder/`: Speaker embedding generation

### Training Scripts

- `lora.py`: LoRA fine-tuning on custom voice data (requires 18GB+ VRAM)
- `grpo.py`: GRPO fine-tuning alternative (requires 12GB+ VRAM)  
- `loadandmergecheckpoint.py`: Load and merge trained LoRA checkpoints

Both training scripts expect audio files in `audio_data/` directory and generate training metrics visualizations.

## Available Models

Fine-tuned models are available in `./models/nicole_v2/` directory:
- **lora_v2_2**: LoRA fine-tuned model
- **grpo_v3**: GRPO fine-tuned model

The server configuration (`configs/server_config.yaml`) controls which model is loaded. Models are automatically downloaded from HuggingFace Hub if not found locally.

### Performance Testing

- `performance_test_harness.py`: Repeatable performance testing script for inference generation
  - Tests 3 models: Base Chatterbox, GRPO fine-tuned, Mixed precision quantized  
  - Measures VRAM usage, generation time, and Real-Time Factor (RTF)
  - Generates 9 test audio files per run with comprehensive performance metrics
  - Uses consistent seed (42) for reproducible results

**Running the performance test:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Run performance tests (requires CUDA GPU)
python3 performance_test_harness.py
```

The test harness randomly selects 3 reference audio files and 3 transcripts (mismatched), then tests each model combination. Results are saved to `./output/` directory with detailed performance comparison tables.

## Usage

**The server is the primary way to use this system.** Start the server and use the REST API or WebSocket endpoints for TTS generation.

### Example Files

For development and testing, example files are available:
- `example_tts_stream.py`: Streaming TTS with real-time audio playback
- `example_vc_stream.py`: Voice conversion streaming
- `example_for_mac.py`: macOS-specific audio handling
- `gradio_tts_app.py`: Gradio web interface for TTS
- `gradio_vc_app.py`: Gradio web interface for voice conversion

### Testing Scripts

- `test_optimized_server.py`: Test server performance and functionality
- `websocket_performance_test.py`: Test WebSocket streaming performance

## Server (Primary Usage)

The server is the recommended way to use Chatterbox Streaming, providing a production-ready TTS service with comprehensive API endpoints.

### Starting the Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the optimized server (recommended)
./start_optimized_server.sh

# Or start manually
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'
python -m src.server.main
```

### Server Features

**REST API Endpoints:**
- `/generate` - Main TTS generation endpoint
- `/generate/raw` - Raw audio generation with file upload support
- `/emotions/*` - Emotion management endpoints (create, update, delete emotions)
- `/model/info` - Model information and status
- `/status` - Server status and metrics
- `/health` - Health check endpoint

**WebSocket Streaming:**
- `/ws/generate` - Real-time streaming TTS generation
- `/ws/stats` and `/ws/metrics` - Connection monitoring and metrics

**Production Features:**
- Automatic performance optimizations (BFloat16, torch.compile)
- Request locking for thread safety
- Model warmup and conditionals caching
- Audio trimming with Whisper alignment
- Emotion system with configurable voice samples
- Comprehensive error handling and logging

### Server Configuration

Edit `configs/server_config.yaml` to control:
- Model selection (base, grpo, or local models from `./models/`)
- Performance optimizations and CUDA settings
- WebSocket streaming configuration
- Caching and warmup behavior
- Audio trimming settings

See `SERVER_OPTIMIZATION_GUIDE.md` for detailed performance information.

## Audio Trimming

The server includes an intelligent audio trimming system (`src/server/audio_trimmer.py`) that automatically improves output quality:

- **Whisper-based Alignment**: Uses Faster-Whisper to transcribe generated audio and compare with intended text
- **Babble Tail Removal**: Automatically detects and removes unwanted speech artifacts at the end of audio
- **Adaptive Margins**: Smart margin calculation prevents cutting off words while removing excess
- **Performance Optimized**: Uses tiny.en model on CPU for fast inference without affecting GPU TTS performance

The audio trimmer is automatically applied to all server-generated audio and can be configured in `server_config.yaml`.

## Important Notes

- Context window and chunk size parameters control latency vs quality tradeoffs
- Models load from HuggingFace hub automatically via `from_pretrained()`
- Device support: CUDA (preferred), MPS (macOS), CPU (fallback)
- The server provides production-ready TTS with emotion support and automatic audio optimization