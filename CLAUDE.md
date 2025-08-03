# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chatterbox is an open-source text-to-speech (TTS) model by Resemble AI. It's a production-grade system built on a 0.5B parameter Llama backbone with unique emotion exaggeration control and zero-shot voice cloning capabilities.

## Installation and Setup

**IMPORTANT: Always use a virtual environment to avoid dependency conflicts!**

```bash
# Create and activate virtual environment
python -m venv chatterbox_env
source chatterbox_env/bin/activate  # On Windows: chatterbox_env\Scripts\activate

# Install from PyPI
pip install chatterbox-tts

# Or install from source (recommended for development)
pip install -e .
```

The project requires Python >=3.9 with specific pinned dependencies in `pyproject.toml`. Developed and tested on Python 3.11 on Debian 11. The dependencies are quite heavy (PyTorch, transformers, librosa, etc.) so a virtual environment is essential.

## Core Architecture

The system consists of three main neural network components:

### T3 (Text-to-Token Transformer)
- **Location**: `src/chatterbox/models/t3/`
- **Purpose**: Converts text input to speech tokens using a Llama-based architecture
- **Key class**: `T3` in `t3.py`
- **Features**: Supports emotion conditioning, repetition penalties, sampling controls

### S3Gen (Speech Synthesis Generator) 
- **Location**: `src/chatterbox/models/s3gen/`
- **Purpose**: Generates audio waveforms from speech tokens using flow matching
- **Key class**: `S3Gen` in `s3gen.py`
- **Components**: Flow matching, HiFi-GAN vocoder, F0 predictor, transformer encoders

### Voice Encoder
- **Location**: `src/chatterbox/models/voice_encoder/`
- **Purpose**: Extracts voice embeddings for zero-shot voice cloning
- **Used by**: Both TTS and voice conversion workflows

### Supporting Components
- **S3Tokenizer**: Speech tokenization (semantic tokens)
- **EnTokenizer**: English text tokenization
- **Perth Watermarker**: Automatic neural watermarking of all outputs

## Main APIs

### ChatterboxTTS
- **File**: `src/chatterbox/tts.py`
- **Usage**: Primary TTS interface
- **Key methods**:
  - `from_pretrained(device)`: Load pre-trained model from HuggingFace
  - `generate(text, audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5, ...)`: Generate speech

### ChatterboxVC  
- **File**: `src/chatterbox/vc.py`
- **Usage**: Voice conversion (change speaker of existing audio)
- **Key methods**:
  - `from_pretrained(device)`: Load pre-trained model
  - `generate(audio, target_voice_path)`: Convert voice

## Key Constants

- `S3_SR = 50`: Sample rate for S3 tokenizer (tokens per second)
- `S3GEN_SR = 24000`: Audio sample rate for generated speech
- `REPO_ID = "ResembleAI/chatterbox"`: HuggingFace model repository

## Model Loading Flow

1. Models are loaded from HuggingFace Hub using `hf_hub_download()`
2. Weights are stored as SafeTensors files
3. Each component (T3, S3Gen, VoiceEncoder) loads independently
4. The main classes (`ChatterboxTTS`, `ChatterboxVC`) orchestrate the pipeline

## Conditioning System

The `Conditionals` dataclass in `tts.py` manages the complex conditioning inputs:
- **T3 conditionals**: Speaker embeddings, CLAP embeddings, emotion control
- **S3Gen conditionals**: Speech prompts, acoustic features, voice embeddings

This system enables the emotion exaggeration and voice cloning features.

## Example Scripts

- `example_tts.py`: Basic TTS usage with device auto-detection
- `example_vc.py`: Voice conversion example  
- `example_for_mac.py`: Mac-specific compatibility
- `gradio_tts_app.py` / `gradio_vc_app.py`: Web UI demos

## Audio Processing Pipeline

1. **Text Processing**: Punctuation normalization via `punc_norm()`
2. **Text Encoding**: T3 model converts text to speech tokens
3. **Voice Conditioning**: Voice encoder processes reference audio (if provided)
4. **Speech Generation**: S3Gen converts tokens to mel spectrograms
5. **Vocoding**: HiFi-GAN converts spectrograms to waveforms
6. **Watermarking**: Perth watermarker embeds imperceptible neural watermarks

## Development Notes

- All audio outputs are automatically watermarked using the Perth system
- Models support CUDA, MPS (Apple Silicon), and CPU backends
- The codebase includes extensive attribution to source projects (CosyVoice, HiFT-GAN, etc.)
- Configuration is handled through dataclasses and config files in respective model directories