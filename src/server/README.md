# Chatterbox TTS API Server

A high-performance HTTP API server for Chatterbox TTS with voice precomputation and emotion management.

## Features

- **FastAPI-based REST API** for TTS generation
- **Voice precomputation** - Pre-load voice conditionals for instant generation
- **Emotion profiles** - Manage different emotions/styles per character
- **Web interface** - Easy-to-use UI for emotion management and testing
- **Voice upload** - Add new voice samples through the web interface
- **Performance optimizations** - torch.compile and mixed precision support
- **Real-time testing** - Test emotion profiles with instant audio playback

## Quick Start

1. **Install dependencies:**
   ```bash
   # Install server dependencies
   pip install -r src/server/requirements.txt
   
   # Make sure chatterbox is installed
   pip install -e .
   ```

2. **Run the server:**
   ```bash
   python src/server/run.py
   ```

3. **Open web interface:**
   - Navigate to `http://localhost:8000` in your browser
   - The server will automatically set up example emotion profiles using Nicole's voice samples

## API Endpoints

### Generation
- `POST /generate` - Generate TTS audio using emotion profiles
- `POST /emotions/{id}/test` - Test an emotion with sample text

### Emotion Management  
- `GET /emotions` - List all emotion profiles
- `POST /emotions` - Create new emotion profile
- `GET /emotions/{id}` - Get specific emotion profile
- `PUT /emotions/{id}` - Update emotion profile
- `DELETE /emotions/{id}` - Delete emotion profile

### Voice Management
- `POST /emotions/{id}/voices` - Upload voice sample to emotion profile

### Status
- `GET /health` - Server health and statistics
- `GET /outputs/{filename}` - Serve generated audio files

## Configuration

The server can be configured via environment variables:

```bash
# Server settings
export CHATTERBOX_HOST=0.0.0.0
export CHATTERBOX_PORT=8000
export CHATTERBOX_DEVICE=cuda  # cuda, mps, or cpu

# Performance settings  
export CHATTERBOX_TORCH_COMPILE=true
export CHATTERBOX_COMPILE_MODE=reduce-overhead
export CHATTERBOX_MIXED_PRECISION=true

# File limits
export CHATTERBOX_MAX_VOICE_SIZE=52428800  # 50MB in bytes
```

## Usage Examples

### Generate Speech via API

```python
import requests

response = requests.post('http://localhost:8000/generate', json={
    'text': 'Hello, this is Nicole speaking with emotion!',
    'emotion': 'nicole_expressive',
    'temperature': 0.8,
    'cfg_weight': 0.5
})

if response.ok:
    data = response.json()
    print(f"Audio URL: {data['audio_url']}")
    print(f"Duration: {data['duration_seconds']}s")
```

### Create Emotion Profile

```python
import requests

response = requests.post('http://localhost:8000/emotions', json={
    'name': 'Happy',
    'character': 'Alice', 
    'exaggeration': 0.8,
    'description': 'Alice in a happy, cheerful mood'
})

emotion = response.json()
print(f"Created emotion: {emotion['id']}")
```

## Performance Notes

- **Precomputed conditionals** eliminate voice processing overhead (2-3x speedup)
- **torch.compile** provides additional 1.5-3x speedup on modern hardware
- **Mixed precision** reduces memory usage and improves GPU performance
- **Voice samples are cached** in memory for instant emotion switching

## File Structure

```
src/server/
├── run.py              # Main startup script
├── server.py           # FastAPI application
├── voice_manager.py    # Voice and emotion management
├── models.py           # Pydantic data models
├── config.py           # Configuration management
├── templates/          # Web interface templates
├── static/            # Static web assets
└── storage/           # Voice files and configurations
    ├── voices/        # Uploaded voice samples
    ├── configs/       # Emotion profiles config
    ├── cache/         # Cached voice conditionals
    └── outputs/       # Generated audio files
```

## Web Interface

The web interface provides three main tabs:

1. **Generate** - TTS generation with emotion selection and parameter tuning
2. **Manage Emotions** - Create, edit, and test emotion profiles
3. **Server Status** - Monitor server health and performance metrics

### Adding New Emotions

1. Go to the "Manage Emotions" tab
2. Click "Add New Emotion"
3. Fill in emotion details (name, character, exaggeration level)
4. Click "Add Voice" to upload audio samples
5. Test the emotion profile with sample text

The server will automatically precompute voice conditionals when you add voice samples, making generation fast for that emotion.

## Troubleshooting

**Server won't start:**
- Check that all dependencies are installed
- Verify CUDA/MPS is available if using GPU
- Check port availability

**Generation is slow:**
- Enable torch.compile in config
- Use GPU if available
- Ensure voice conditionals are precomputed

**Out of memory:**
- Reduce batch size or use CPU
- Enable mixed precision for GPU
- Monitor memory usage in status tab

**Audio quality issues:**
- Check voice sample quality and format
- Adjust exaggeration levels
- Try different generation parameters