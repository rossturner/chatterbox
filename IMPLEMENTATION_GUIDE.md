# Chatterbox Multilingual TTS - Implementation Guide

## Overview

The Chatterbox Multilingual TTS server provides high-performance, zero-shot voice cloning across 23 languages. This implementation guide shows you how to integrate the Docker container into your system and use the V2 API for multilingual text-to-speech generation.

### Key Features

- **Zero-shot Voice Cloning**: Generate speech in any voice using a reference audio sample
- **23 Languages Supported**: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Swahili, Turkish
- **GPU Accelerated**: NVIDIA CUDA with BFloat16 precision and CUDA graph optimizations
- **High Performance**: Real-time or faster generation (RTF typically 0.17-0.78)
- **Intelligent Caching**: Automatic caching of reference audio encodings for faster subsequent requests
- **Production Ready**: Thread-safe inference, proper error handling, health checks

### System Requirements

- Docker with Docker Compose
- NVIDIA GPU with CUDA support (minimum 8GB VRAM recommended)
- NVIDIA Container Toolkit (`nvidia-docker2`)
- Approximately 6GB disk space for Docker image and model cache

---

## Getting the Docker Image

This implementation guide is for **this specific repository's Docker image**, which includes:

- **V2 Zero-shot Voice Cloning API** (not in upstream Chatterbox)
- **Critical CUDA graph thread affinity fix** (prevents AssertionError crashes)
- **Reference audio conditional caching** (performance optimization)
- **Production-ready Docker configuration**

This is NOT the same as any official/upstream Chatterbox image. These features are specific to this implementation.

### Option 1: Build from Source (Recommended)

Clone this repository and build the image:

```bash
# Clone the repository
git clone https://github.com/rossturner/chatterbox.git
cd chatterbox

# Checkout the streaming branch (where V2 API lives)
git checkout streaming

# Build the Docker image
docker compose build

# Verify the image was created
docker images | grep chatterbox-tts
```

The build process takes ~5-10 minutes depending on your internet connection.

### Option 2: Use Pre-built Image

If a pre-built image is published to a container registry:

```bash
# Pull from Docker Hub (example - replace with actual registry)
docker pull rossturner/chatterbox-tts:latest

# Tag it locally
docker tag rossturner/chatterbox-tts:latest chatterbox-tts:latest
```

**Note:** Check the repository README for the current published image location.

---

## Docker Compose Integration

### Basic Configuration

Add this service to your `docker-compose.yml`:

```yaml
services:
  chatterbox-tts:
    image: chatterbox-tts:latest
    container_name: chatterbox-tts
    restart: unless-stopped

    ports:
      - "8091:8000"  # Map internal port 8000 to host port 8091

    volumes:
      # Persistent cache for HuggingFace models (~3GB)
      - huggingface-cache:/root/.cache/huggingface

      # Temporary audio files storage
      - temp-audio:/tmp

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1  # Use 1 GPU
              capabilities: [gpu]

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v2/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 90s  # Allow time for model loading

    environment:
      # Optional: Adjust CUDA memory allocation
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

volumes:
  huggingface-cache:
    driver: local
  temp-audio:
    driver: local
```

### Advanced Configuration

For production deployments with multiple services:

```yaml
version: '3.8'

services:
  chatterbox-tts:
    image: chatterbox-tts:latest
    container_name: chatterbox-tts
    restart: unless-stopped

    ports:
      - "8091:8000"

    volumes:
      - huggingface-cache:/root/.cache/huggingface
      - temp-audio:/tmp

      # Optional: Mount custom config
      # - ./configs/server_config.yaml:/app/configs/server_config.yaml:ro

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 12G  # Limit total memory usage

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v2/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 90s

    environment:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - LOG_LEVEL=INFO

    networks:
      - app-network

    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  huggingface-cache:
  temp-audio:

networks:
  app-network:
    driver: bridge
```

### Starting the Service

After obtaining the Docker image (see "Getting the Docker Image" section above):

```bash
# Start the service (builds image if needed)
docker compose up -d

# Check logs to monitor startup
docker compose logs -f chatterbox-tts

# Wait for service to be healthy (check logs for "Server initialization complete!")
# Then check health endpoint
curl http://localhost:8091/v2/health
```

If you need to rebuild the image after pulling updates:

```bash
# Rebuild and restart
docker compose down
docker compose build
docker compose up -d
```

### Expected Startup Time

- **Initial startup** (first time): ~60-90 seconds
  - Model download from HuggingFace Hub (~3GB)
  - Model compilation and warmup

- **Subsequent startups**: ~30-45 seconds
  - Models cached locally
  - Warmup and compilation only

---

## V2 API Reference

### Base URL

When running locally:
```
http://localhost:8091
```

When running in Docker network:
```
http://chatterbox-tts:8000
```

### Authentication

Currently no authentication required. For production, consider:
- Reverse proxy with authentication (nginx, traefik)
- API gateway with rate limiting
- Network isolation (internal Docker network)

---

## API Endpoints

### 1. Health Check

Check if the service is ready to accept requests.

**Endpoint:** `GET /v2/health`

**Response:**
```json
{
  "status": "healthy",
  "model": "multilingual",
  "model_path": null,
  "supported_languages": ["ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"],
  "processing": false,
  "requests_processed": 42,
  "cache_enabled": true
}
```

**Example:**
```bash
curl http://localhost:8091/v2/health
```

---

### 2. Generate Speech (Zero-shot Voice Cloning)

Generate speech in any language using a reference audio sample for voice cloning.

**Endpoint:** `POST /v2/generate`

**Content-Type:** `application/json`

#### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | ✅ | - | Text to synthesize (1-500 characters) |
| `language` | string | ✅ | - | Language code (see supported languages below) |
| `reference_audio_base64` | string | ✅ | - | Base64-encoded reference audio (WAV format) |
| `temperature` | float | ❌ | 0.8 | Sampling temperature (0.05-5.0) |
| `cfg_weight` | float | ❌ | 0.3 | CFG weight/pace (0.0-1.0) |
| `exaggeration` | float | ❌ | 0.5 | Voice exaggeration factor (0.1-2.0) |
| `min_p` | float | ❌ | 0.1 | Minimum probability threshold (0.0-1.0) |

#### Supported Languages

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `ar` | Arabic | `he` | Hebrew | `pl` | Polish |
| `da` | Danish | `hi` | Hindi | `pt` | Portuguese |
| `de` | German | `it` | Italian | `ru` | Russian |
| `el` | Greek | `ja` | Japanese | `sv` | Swedish |
| `en` | English | `ko` | Korean | `sw` | Swahili |
| `es` | Spanish | `ms` | Malay | `tr` | Turkish |
| `fi` | Finnish | `nl` | Dutch | `zh` | Chinese |
| `fr` | French | `no` | Norwegian | | |

#### Response

```json
{
  "audio": "UklGRiQBAgBXQVZFZm10IBAAAAABAAEA...",
  "duration": 3.52,
  "rtf": 0.234,
  "generation_time": 0.824,
  "queue_time": 0.002,
  "language_used": "en",
  "text_normalized": "Hello world, this is a test.",
  "cache_hit": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `audio` | string | Base64-encoded WAV audio (24kHz, mono) |
| `duration` | float | Audio duration in seconds |
| `rtf` | float | Real-Time Factor (generation_time / audio_duration). Values < 1.0 mean faster than real-time |
| `generation_time` | float | Time taken to generate audio in seconds |
| `queue_time` | float | Time spent waiting for server availability |
| `language_used` | string | Language code that was used |
| `text_normalized` | string | Normalized text that was actually synthesized |
| `cache_hit` | boolean | Whether reference audio was loaded from cache (subsequent requests with same reference audio) |

#### Example Request

```json
{
  "text": "Hello, this is a demonstration of zero-shot voice cloning.",
  "language": "en",
  "reference_audio_base64": "UklGRiQBAgBXQVZFZm10IBAAAAABA...",
  "temperature": 0.8,
  "cfg_weight": 0.3,
  "exaggeration": 0.5,
  "min_p": 0.1
}
```

---

## Usage Examples

### Python Example

```python
import base64
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8091"

def encode_audio_file(audio_path: str) -> str:
    """Read and base64-encode an audio file"""
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode('utf-8')

def decode_audio_response(audio_base64: str, output_path: str):
    """Decode base64 audio and save to file"""
    audio_bytes = base64.b64decode(audio_base64)
    with open(output_path, 'wb') as f:
        f.write(audio_bytes)

def generate_speech(text: str, language: str, reference_audio_path: str):
    """Generate speech using the V2 API"""

    # Encode reference audio
    reference_audio_base64 = encode_audio_file(reference_audio_path)

    # Prepare request
    request_data = {
        "text": text,
        "language": language,
        "reference_audio_base64": reference_audio_base64,
        "temperature": 0.8,
        "cfg_weight": 0.3,
        "exaggeration": 0.5,
        "min_p": 0.1
    }

    # Make request
    response = requests.post(
        f"{BASE_URL}/v2/generate",
        json=request_data,
        timeout=120  # Allow up to 2 minutes for generation
    )

    # Check response
    if response.status_code == 200:
        result = response.json()

        print(f"✓ Generation successful!")
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Generation time: {result['generation_time']:.2f}s")
        print(f"  RTF: {result['rtf']:.3f}")
        print(f"  Cache hit: {result['cache_hit']}")

        # Save audio
        output_path = "output.wav"
        decode_audio_response(result['audio'], output_path)
        print(f"  Audio saved to: {output_path}")

        return result
    else:
        print(f"✗ Generation failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return None

# Example usage
if __name__ == "__main__":
    # Check health
    health = requests.get(f"{BASE_URL}/v2/health").json()
    print(f"Server status: {health['status']}")
    print(f"Supported languages: {len(health['supported_languages'])}")

    # Generate speech
    generate_speech(
        text="Hello, this is a test of the multilingual TTS system.",
        language="en",
        reference_audio_path="reference_voice.wav"
    )
```

---

### JavaScript (Node.js) Example

```javascript
const fs = require('fs');
const axios = require('axios');

const BASE_URL = 'http://localhost:8091';

async function encodeAudioFile(audioPath) {
    const audioBuffer = fs.readFileSync(audioPath);
    return audioBuffer.toString('base64');
}

function decodeAudioResponse(audioBase64, outputPath) {
    const audioBuffer = Buffer.from(audioBase64, 'base64');
    fs.writeFileSync(outputPath, audioBuffer);
}

async function generateSpeech(text, language, referenceAudioPath) {
    try {
        // Encode reference audio
        const referenceAudioBase64 = await encodeAudioFile(referenceAudioPath);

        // Prepare request
        const requestData = {
            text: text,
            language: language,
            reference_audio_base64: referenceAudioBase64,
            temperature: 0.8,
            cfg_weight: 0.3,
            exaggeration: 0.5,
            min_p: 0.1
        };

        // Make request
        const response = await axios.post(
            `${BASE_URL}/v2/generate`,
            requestData,
            { timeout: 120000 } // 2 minutes timeout
        );

        console.log('✓ Generation successful!');
        console.log(`  Duration: ${response.data.duration.toFixed(2)}s`);
        console.log(`  Generation time: ${response.data.generation_time.toFixed(2)}s`);
        console.log(`  RTF: ${response.data.rtf.toFixed(3)}`);
        console.log(`  Cache hit: ${response.data.cache_hit}`);

        // Save audio
        const outputPath = 'output.wav';
        decodeAudioResponse(response.data.audio, outputPath);
        console.log(`  Audio saved to: ${outputPath}`);

        return response.data;

    } catch (error) {
        console.error('✗ Generation failed:', error.message);
        if (error.response) {
            console.error('  Status:', error.response.status);
            console.error('  Error:', error.response.data);
        }
        return null;
    }
}

// Example usage
async function main() {
    // Check health
    const healthResponse = await axios.get(`${BASE_URL}/v2/health`);
    console.log(`Server status: ${healthResponse.data.status}`);
    console.log(`Supported languages: ${healthResponse.data.supported_languages.length}`);

    // Generate speech
    await generateSpeech(
        'Hello, this is a test of the multilingual TTS system.',
        'en',
        'reference_voice.wav'
    );
}

main();
```

---

### cURL Example

```bash
#!/bin/bash

BASE_URL="http://localhost:8091"
REFERENCE_AUDIO="reference_voice.wav"
OUTPUT_AUDIO="output.wav"

# Encode reference audio to base64
REFERENCE_BASE64=$(base64 -w 0 "$REFERENCE_AUDIO")

# Prepare JSON request
REQUEST_JSON=$(cat <<EOF
{
  "text": "Hello, this is a test of the multilingual TTS system.",
  "language": "en",
  "reference_audio_base64": "$REFERENCE_BASE64",
  "temperature": 0.8,
  "cfg_weight": 0.3,
  "exaggeration": 0.5,
  "min_p": 0.1
}
EOF
)

# Make request
RESPONSE=$(curl -s -X POST "$BASE_URL/v2/generate" \
  -H "Content-Type: application/json" \
  -d "$REQUEST_JSON")

# Check if request succeeded
if echo "$RESPONSE" | jq -e '.audio' > /dev/null 2>&1; then
  echo "✓ Generation successful!"

  # Extract and decode audio
  echo "$RESPONSE" | jq -r '.audio' | base64 -d > "$OUTPUT_AUDIO"

  # Display metrics
  echo "  Duration: $(echo "$RESPONSE" | jq -r '.duration')s"
  echo "  Generation time: $(echo "$RESPONSE" | jq -r '.generation_time')s"
  echo "  RTF: $(echo "$RESPONSE" | jq -r '.rtf')"
  echo "  Cache hit: $(echo "$RESPONSE" | jq -r '.cache_hit')"
  echo "  Audio saved to: $OUTPUT_AUDIO"
else
  echo "✗ Generation failed"
  echo "$RESPONSE" | jq '.'
fi
```

---

## Best Practices

### Reference Audio Requirements

For optimal voice cloning quality:

- **Format**: WAV format (other formats may work but WAV is recommended)
- **Sample Rate**: 24kHz or higher
- **Duration**: 10-20 seconds is ideal
  - Minimum: 3 seconds
  - Maximum: No hard limit, but longer samples take more memory
- **Content**: Single speaker, clear speech, no background noise
- **Language**: Can be any language - voice characteristics are language-independent

### Parameter Tuning

- **temperature** (0.05-5.0, default: 0.8)
  - Lower = more consistent/monotone
  - Higher = more varied/expressive
  - Recommended: 0.7-0.9 for most cases

- **cfg_weight** (0.0-1.0, default: 0.3)
  - Controls adherence to reference voice
  - Lower = more freedom/variation
  - Higher = closer to reference
  - **Important**: Default is 0.3 to reduce babble/trailing artifacts

- **exaggeration** (0.1-2.0, default: 0.5)
  - Controls voice expressiveness
  - 0.5 = balanced
  - Higher = more dramatic/emphasized

- **min_p** (0.0-1.0, default: 0.1)
  - Minimum probability threshold
  - Helps reduce low-probability tokens
  - **Important**: Default is 0.1 to reduce babble/trailing artifacts

### Performance Optimization

1. **Reuse Reference Audio**: The system caches reference audio encodings. Using the same reference audio across multiple requests (even in different languages) will be significantly faster after the first request.

2. **Batch Requests**: If generating multiple outputs with the same voice, send them sequentially to take advantage of caching.

3. **Text Length**: Keep individual requests under 500 characters. For longer text, split into multiple requests.

4. **Concurrent Requests**: The server processes requests sequentially. Concurrent requests will queue automatically, but consider rate limiting on the client side.

### Error Handling

Always implement proper error handling:

```python
import requests
from requests.exceptions import Timeout, ConnectionError, HTTPError

def generate_with_retry(text, language, reference_audio_base64, max_retries=3):
    """Generate speech with automatic retry logic"""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{BASE_URL}/v2/generate",
                json={
                    "text": text,
                    "language": language,
                    "reference_audio_base64": reference_audio_base64
                },
                timeout=120
            )

            response.raise_for_status()
            return response.json()

        except Timeout:
            print(f"Attempt {attempt + 1}/{max_retries}: Request timed out")
            if attempt == max_retries - 1:
                raise

        except ConnectionError:
            print(f"Attempt {attempt + 1}/{max_retries}: Connection failed")
            if attempt == max_retries - 1:
                raise

        except HTTPError as e:
            if e.response.status_code == 503:
                print(f"Attempt {attempt + 1}/{max_retries}: Server busy")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

### Common HTTP Status Codes

- **200**: Success - audio generated
- **400**: Bad Request - invalid parameters (check text length, language code, audio encoding)
- **422**: Validation Error - invalid field values (check parameter ranges)
- **503**: Service Unavailable - server busy processing another request
- **500**: Internal Server Error - server-side issue (check logs)

---

## Production Considerations

### Monitoring

Monitor these metrics for production health:

1. **Response Time**: Track generation_time and rtf values
2. **Cache Hit Rate**: Monitor cache_hit field to optimize performance
3. **Error Rate**: Track 503 (busy) and 500 (error) responses
4. **Queue Time**: Monitor queue_time to detect congestion

### Scaling

For high-traffic scenarios:

1. **Vertical Scaling**: Use GPUs with more VRAM (12GB+ recommended)
2. **Horizontal Scaling**: Run multiple containers with load balancer
3. **Queue Management**: Implement external queue (Redis, RabbitMQ) for request buffering

### Resource Allocation

Typical resource usage:

- **VRAM**: 6-8GB during generation
- **RAM**: 8-12GB system memory
- **CPU**: Minimal (GPU-accelerated)
- **Disk**: ~6GB (3GB models + 3GB cache)

### Security

For production deployments:

1. **Authentication**: Add API key or OAuth2
2. **Rate Limiting**: Prevent abuse with request rate limits
3. **Input Validation**: Sanitize text input to prevent injection attacks
4. **Network Security**: Use internal Docker network, expose only necessary ports
5. **HTTPS**: Use reverse proxy (nginx, traefik) with TLS certificates

### Example Nginx Reverse Proxy

```nginx
server {
    listen 443 ssl http2;
    server_name tts.yourdomain.com;

    ssl_certificate /etc/ssl/certs/tts.crt;
    ssl_certificate_key /etc/ssl/private/tts.key;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=tts_limit:10m rate=10r/m;
    limit_req zone=tts_limit burst=5 nodelay;

    location / {
        proxy_pass http://chatterbox-tts:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Longer timeout for generation
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;
    }
}
```

---

## Troubleshooting

### Container Won't Start

**Issue**: Container exits immediately or fails health check

**Solutions**:
1. Check GPU availability: `nvidia-smi`
2. Verify NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`
3. Check logs: `docker compose logs chatterbox-tts`
4. Increase memory limits in docker-compose.yml
5. Ensure sufficient disk space for model download

### Slow First Request

**Issue**: First request takes 60-90 seconds

**Explanation**: This is expected behavior:
- Model download from HuggingFace Hub (~3GB)
- Model compilation with CUDA graphs
- Warmup runs

**Solutions**:
- Subsequent requests will be much faster (cache hit)
- Use persistent volume for HuggingFace cache
- Consider pre-warming during container initialization

### Server Returns 503 (Busy)

**Issue**: Getting "Server busy" errors

**Explanation**: Server processes one request at a time for thread safety

**Solutions**:
1. Implement retry logic with exponential backoff
2. Queue requests on client side
3. Scale horizontally with multiple containers
4. Monitor queue_time to detect congestion

### Poor Audio Quality

**Issue**: Generated audio sounds unnatural or has artifacts

**Solutions**:
1. Check reference audio quality:
   - Use high-quality WAV files (24kHz+)
   - Ensure single speaker, no background noise
   - Use 10-20 second samples
2. Adjust parameters:
   - Increase cfg_weight (closer to reference)
   - Adjust temperature (lower for consistency)
   - Try different exaggeration values
3. Keep text under 500 characters per request

### Memory Issues

**Issue**: Container OOM killed or GPU out of memory

**Solutions**:
1. Reduce text length (shorter requests)
2. Increase container memory limits
3. Use GPU with more VRAM (12GB+ recommended)
4. Check for memory leaks in logs
5. Restart container periodically in high-traffic scenarios

---

## API Migration from V1

If migrating from the emotion-based V1 API:

### Key Differences

| Feature | V1 (Emotion-based) | V2 (Zero-shot) |
|---------|-------------------|----------------|
| Voice Selection | Predefined emotions | Dynamic reference audio |
| Endpoint | `/generate` | `/v2/generate` |
| Voice Parameter | `emotion: "neutral"` | `reference_audio_base64: "..."` |
| Language Support | Single language per emotion | 23 languages with any voice |
| Voice Configuration | Server-side YAML config | Client-side audio upload |

### Migration Example

**Before (V1):**
```json
{
  "text": "Hello world",
  "emotion": "neutral",
  "temperature": 0.8
}
```

**After (V2):**
```json
{
  "text": "Hello world",
  "language": "en",
  "reference_audio_base64": "UklGRiQBAgBXQVZF...",
  "temperature": 0.8
}
```

---

## Support and Resources

- **Repository**: https://github.com/rossturner/chatterbox (streaming branch)
- **Docker Image**: `chatterbox-tts:latest` (build from source)
- **Internal Port**: 8000
- **Recommended Host Port**: 8091
- **Health Check**: `/v2/health`
- **API Documentation**: This guide

### Performance Benchmarks

Typical performance on NVIDIA RTX 4090:

- **First request** (cold start): 60-90 seconds
- **Subsequent requests** (cache hit): 0.5-3 seconds per request
- **RTF** (Real-Time Factor): 0.17-0.78 (faster than real-time)
- **Languages tested**: All 23 languages confirmed working

### Version Information

- **API Version**: V2
- **Model**: Chatterbox Multilingual
- **Optimization**: BFloat16, CUDA graphs, max-autotune mode
- **Default Parameters**: cfg_weight=0.3, min_p=0.1 (optimized to reduce babble)

---

## Complete Integration Example

Here's a complete example integrating the TTS service into a web application:

```python
from flask import Flask, request, jsonify, send_file
import base64
import requests
import tempfile
import os

app = Flask(__name__)
TTS_SERVICE_URL = "http://chatterbox-tts:8000"  # Internal Docker network

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """
    API endpoint that accepts text and reference audio,
    generates speech, and returns the audio file
    """
    try:
        # Parse request
        data = request.json
        text = data.get('text')
        language = data.get('language', 'en')
        reference_audio_base64 = data.get('reference_audio_base64')

        # Validate
        if not text or not reference_audio_base64:
            return jsonify({'error': 'Missing required fields'}), 400

        # Forward to TTS service
        tts_response = requests.post(
            f"{TTS_SERVICE_URL}/v2/generate",
            json={
                'text': text,
                'language': language,
                'reference_audio_base64': reference_audio_base64,
                'temperature': data.get('temperature', 0.8),
                'cfg_weight': data.get('cfg_weight', 0.3),
                'exaggeration': data.get('exaggeration', 0.5),
                'min_p': data.get('min_p', 0.1)
            },
            timeout=120
        )

        if tts_response.status_code != 200:
            return jsonify({
                'error': 'TTS generation failed',
                'details': tts_response.text
            }), tts_response.status_code

        result = tts_response.json()

        # Decode audio and save to temp file
        audio_bytes = base64.b64decode(result['audio'])

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Return audio file with metadata in headers
        response = send_file(
            tmp_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='generated_speech.wav'
        )

        response.headers['X-Audio-Duration'] = str(result['duration'])
        response.headers['X-Generation-Time'] = str(result['generation_time'])
        response.headers['X-RTF'] = str(result['rtf'])
        response.headers['X-Cache-Hit'] = str(result['cache_hit'])

        # Clean up temp file after sending
        @response.call_on_close
        def cleanup():
            try:
                os.unlink(tmp_path)
            except:
                pass

        return response

    except requests.Timeout:
        return jsonify({'error': 'TTS service timeout'}), 504
    except requests.ConnectionError:
        return jsonify({'error': 'TTS service unavailable'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts/health', methods=['GET'])
def health_check():
    """Check TTS service health"""
    try:
        response = requests.get(f"{TTS_SERVICE_URL}/v2/health", timeout=5)
        return jsonify(response.json()), response.status_code
    except:
        return jsonify({'error': 'TTS service unavailable'}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Corresponding docker-compose.yml:**

```yaml
services:
  chatterbox-tts:
    image: chatterbox-tts:latest
    container_name: chatterbox-tts
    restart: unless-stopped
    volumes:
      - huggingface-cache:/root/.cache/huggingface
      - temp-audio:/tmp
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - app-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v2/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 90s

  web-app:
    build: ./web-app
    container_name: web-app
    restart: unless-stopped
    ports:
      - "5000:5000"
    environment:
      - TTS_SERVICE_URL=http://chatterbox-tts:8000
    depends_on:
      chatterbox-tts:
        condition: service_healthy
    networks:
      - app-network

volumes:
  huggingface-cache:
  temp-audio:

networks:
  app-network:
    driver: bridge
```

---

## Conclusion

The Chatterbox Multilingual TTS V2 API provides a powerful, production-ready solution for zero-shot voice cloning across 23 languages. With proper Docker integration, caching optimization, and error handling, you can achieve reliable, high-performance speech synthesis in your applications.

For additional support or questions, refer to the Docker logs or health check endpoints for diagnostics.
