# CLAUDE.md - Chatterbox TTS HTTP API Server

This file provides guidance to Claude Code when working with the Chatterbox TTS HTTP API server implementation.

## Project Overview

The `src/server/` directory contains a production-ready HTTP API server built on FastAPI that wraps the Chatterbox TTS system with voice precomputation and emotion management capabilities.

## Architecture Overview

### Core Components

- **`server.py`** - Main FastAPI application with REST endpoints
- **`voice_manager.py`** - Manages emotion profiles and precomputed voice conditionals  
- **`models.py`** - Pydantic data models for API requests/responses
- **`config.py`** - Configuration management and environment settings
- **`run.py`** - Server startup script with example setup

### Key Features

1. **Voice Precomputation** - Pre-loads voice conditionals at startup for 2-3x speedup
2. **Emotion Profiles** - Configurable emotions per character with custom exaggeration levels
3. **Web Interface** - Modern responsive UI for emotion management and TTS generation
4. **REST API** - Standard HTTP endpoints for programmatic access
5. **Performance Optimization** - torch.compile and mixed precision support
6. **Persistent Storage** - Emotions, voice samples, and cached conditionals persist between restarts

## Storage Structure

```
src/server/storage/
├── voices/           # Voice sample audio files (.wav)
├── configs/          # Emotion profiles (emotions.json)
├── cache/           # Precomputed voice conditionals (.pt files) [gitignored]
└── outputs/         # Generated TTS audio files [gitignored]
```

## Configuration

The server uses environment variables for configuration:

```bash
# Device settings
CHATTERBOX_DEVICE=cuda          # cuda, mps, or cpu

# Server settings  
CHATTERBOX_HOST=127.0.0.1       # Bind address
CHATTERBOX_PORT=8000            # Port number

# Performance settings
CHATTERBOX_TORCH_COMPILE=true   # Enable model compilation
CHATTERBOX_COMPILE_MODE=reduce-overhead
CHATTERBOX_MIXED_PRECISION=true # Enable AMP

# File limits
CHATTERBOX_MAX_VOICE_SIZE=52428800  # 50MB max upload
```

## API Endpoints

### Generation
- `POST /generate` - Generate TTS audio using emotion profiles
- `POST /emotions/{id}/test` - Test emotion with sample text

### Emotion Management
- `GET /emotions` - List all emotion profiles
- `POST /emotions` - Create new emotion profile  
- `GET /emotions/{id}` - Get specific emotion
- `PUT /emotions/{id}` - Update emotion profile
- `DELETE /emotions/{id}` - Delete emotion profile

### Voice Management
- `POST /emotions/{id}/voices` - Upload voice sample to emotion

### Status & Files
- `GET /health` - Server health and statistics
- `GET /outputs/{filename}` - Serve generated audio files
- `GET /` - Web interface
- `GET /docs` - API documentation

## Voice Manager Deep Dive

The `VoiceManager` class handles the complex voice precomputation system:

### Precomputation Process
1. **Audio Loading** - Load reference audio with librosa
2. **Feature Extraction** - Extract acoustic features for S3Gen vocoder
3. **Speech Tokenization** - Generate speech tokens for T3 conditioning
4. **Speaker Embedding** - Extract voice embedding using voice encoder
5. **Caching** - Store computed conditionals in memory and on disk

### Emotion Profiles
Each emotion profile contains:
- **Metadata** - Name, character, description, timestamps
- **Voice Samples** - List of audio file paths
- **Exaggeration Level** - Float 0.0-1.0 for emotion intensity
- **Cached Conditionals** - Precomputed voice data for instant generation

### Cache Management
- **Memory Cache** - Active conditionals stored in `voice_conditionals` dict
- **Disk Cache** - Persistent `.pt` files in `storage/cache/voice_conditionals/`
- **Cache Keys** - Based on emotion ID, exaggeration, and voice sample hash
- **Safe Copying** - Handles torch.compile compatibility issues with deepcopy

## Performance Optimizations

### Model Compilation
```python
# Applied in server.py during startup
model.t3 = torch.compile(model.t3, mode="reduce-overhead")
model.s3gen = torch.compile(model.s3gen, mode="reduce-overhead")
```

### Mixed Precision
```python
# Used during generation
with torch.cuda.amp.autocast():
    wav = model.generate(text, ...)
```

### Voice Precomputation Benefits
- **Before**: ~17s generation (includes voice processing)
- **After**: ~6-8s generation (voice already cached)
- **Speedup**: 2-3x faster for voice cloning

## Development Patterns

### Adding New Endpoints
1. Define Pydantic models in `models.py`
2. Add endpoint function in `server.py` 
3. Use dependency injection for voice_manager and model
4. Handle errors with HTTPException
5. Update web interface if needed

### Voice Manager Operations
```python
# Create emotion
emotion = await voice_manager.create_emotion(emotion_data)

# Add voice sample  
sample = await voice_manager.add_voice_sample(emotion_id, file_path, filename)

# Use emotion for generation
await voice_manager.use_emotion(emotion_id)
wav = model.generate(text)
```

### Configuration Updates
1. Add new setting to `ServerConfig` in `config.py`
2. Add environment variable with `CHATTERBOX_` prefix
3. Update documentation and examples

## Web Interface

The web interface (`templates/index.html`) provides:

### Generate Tab
- Emotion selection dropdown
- Text input with parameter controls
- Real-time generation with progress feedback
- Audio playback and download

### Manage Emotions Tab  
- Emotion profile cards with metadata
- Test, add voice, delete actions
- Modal forms for creating/editing emotions
- Voice sample upload with drag-and-drop

### Server Status Tab
- Health metrics and uptime
- Memory usage and device info
- Emotion loading statistics

## Testing

The server includes comprehensive testing via Playwright:
- Web interface functionality
- TTS generation with different emotions
- Emotion management (create, test, delete)
- Server health monitoring

## Deployment Notes

### Production Checklist
1. Set appropriate `CHATTERBOX_HOST` (e.g., 0.0.0.0 for network access)
2. Configure `cors_origins` for security
3. Set `enable_docs=false` to disable API docs in production
4. Use process manager (systemd, PM2) for service management
5. Set up reverse proxy (nginx) for HTTPS/domain routing

### Resource Requirements
- **VRAM**: ~3GB for full model with emotions
- **RAM**: ~8GB recommended  
- **Storage**: Voice samples + cache files (varies by usage)
- **CPU**: Multi-core recommended for compilation benefits

### Scaling Considerations
- Each server instance loads full model in memory
- Voice conditionals are not shared between instances
- Consider load balancing for high-traffic scenarios
- Cache warming may be needed after restarts

## Common Issues

### Model Compilation Errors
If torch.compile fails, disable with `CHATTERBOX_TORCH_COMPILE=false`

### Memory Issues  
- Reduce batch sizes in generation
- Use CPU instead of GPU if VRAM limited
- Clear cache periodically in high-usage scenarios

### Voice Sample Issues
- Ensure samples are clean, clear audio
- Recommended: 3-10 seconds of speech
- Supported formats: WAV, MP3, FLAC, OGG
- Check sample rate compatibility

## Integration Examples

### Python Client
```python
import requests

# Generate speech
response = requests.post('http://localhost:8000/generate', json={
    'text': 'Hello world!',
    'emotion': 'nicole_expressive',
    'temperature': 0.8
})

# Create emotion
response = requests.post('http://localhost:8000/emotions', json={
    'name': 'Happy',
    'character': 'Alice',
    'exaggeration': 0.7
})
```

### JavaScript/Web
```javascript
// Generate speech
const response = await fetch('/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        text: 'Hello world!',
        emotion: 'nicole_expressive'
    })
});

const result = await response.json();
// Audio available at result.audio_url
```

This server provides a complete production-ready wrapper around Chatterbox TTS with significant performance improvements through voice precomputation and a user-friendly management interface.