# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

```bash
# Create virtual environment (Python 3.8+ required, tested with Python 3.13)
python3 -m venv .venv
source .venv/bin/activate

# Install for development with all dependencies
pip install -e .
```

**Note**: Installation includes PyTorch with CUDA 12.4 support and many ML dependencies (~4GB total). The package installs successfully and imports without errors.

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

### Testing Framework

- `test_models_final.py`: Comprehensive testing script with creative content and multiple reference voices
- `final_model_comparison/`: Generated audio samples and performance metrics
- Testing uses consistent seed (42) across all models for reproducible comparisons

## Usage Examples

Key example files demonstrate the streaming capabilities:
- `example_tts_stream.py`: Streaming TTS with real-time audio playback
- `example_vc_stream.py`: Voice conversion streaming
- `example_for_mac.py`: macOS-specific audio handling
- `gradio_tts_app.py`: Gradio web interface for TTS
- `gradio_vc_app.py`: Gradio web interface for voice conversion

## Important Notes

- All generated audio includes Perth watermarking for responsible AI usage
- Streaming implementation achieves ~0.5 RTF (Real-Time Factor) on RTX 4090
- Context window and chunk size parameters control latency vs quality tradeoffs
- Models load from HuggingFace hub automatically via `from_pretrained()`
- Device support: CUDA (preferred), MPS (macOS), CPU (fallback)