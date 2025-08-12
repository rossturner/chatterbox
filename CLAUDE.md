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

## GRPO Fine-tuned Model

This repository includes a GRPO (Group Relative Policy Optimization) fine-tuned model that has been optimized for improved performance. The GRPO training process enhances the base model's ability to generate high-quality speech while maintaining reasonable computational requirements.

### Quantized Model Variants

The GRPO fine-tuned model has been quantized using multiple techniques to reduce memory usage while maintaining quality:

- **Original GRPO**: Full precision model (~3GB, baseline performance)
- **Float16 Weights**: Half-precision quantization (~1.5GB, 50% size reduction, 27% VRAM reduction)
- **Mixed Precision**: Strategic precision reduction (~1.5GB, similar performance to float16)

### Model Performance Comparison

Performance metrics on RTX 4090 with creative test content:

| Model | Size (MB) | Peak VRAM (GB) | Avg RTF | Avg Gen Time (s) | Quality |
|-------|-----------|----------------|---------|------------------|---------|
| Original GRPO | 3045 | 4.97 | 0.77 | 8.35 | Baseline |
| Float16 Weights | 1523 | 3.62 | 0.74 | 8.17 | Excellent |
| Mixed Precision | 1523 | 3.61 | 0.76 | 8.24 | Excellent |

All quantized variants maintain excellent speech quality while achieving significant memory savings. The quantized models are located in `quantized_models/` directory.

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

## Usage Examples

Key example files demonstrate the streaming capabilities:
- `example_tts_stream.py`: Streaming TTS with real-time audio playback
- `example_vc_stream.py`: Voice conversion streaming
- `example_for_mac.py`: macOS-specific audio handling
- `gradio_tts_app.py`: Gradio web interface for TTS
- `gradio_vc_app.py`: Gradio web interface for voice conversion

## Important Notes

- Streaming implementation achieves ~0.5 RTF (Real-Time Factor) on RTX 4090
- Context window and chunk size parameters control latency vs quality tradeoffs
- Models load from HuggingFace hub automatically via `from_pretrained()`
- Device support: CUDA (preferred), MPS (macOS), CPU (fallback)