#!/usr/bin/env python3
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json

from .config import Config, load_emotions_config
from .models import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    EmotionsResponse,
    ModelInfoResponse,
    ServerStatusResponse,
    ErrorResponse,
    EmotionConfig,
    EmotionCreateRequest,
    EmotionUpdateRequest,
    EmotionTestRequest,
    EmotionListResponse,
    VoiceSample,
    WebSocketRequest
)
from .lock_manager import GenerationLock
from .tts_manager import TTSManager
from .emotion_manager import EmotionManager
from .websocket_handlers import WebSocketConnectionManager
from .utils import normalize_text, get_memory_usage, format_duration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chatterbox TTS Server",
    description="High-performance emotion-based text-to-speech synthesis server",
    version="1.0.0"
)

# Global instances
config: Optional[Config] = None
emotions_config: Optional[dict] = None
tts_manager: Optional[TTSManager] = None
emotion_manager: Optional[EmotionManager] = None
websocket_manager: Optional[WebSocketConnectionManager] = None
generation_lock = GenerationLock()
server_start_time = datetime.utcnow()
last_request_time: Optional[datetime] = None


@app.on_event("startup")
async def startup_event():
    """Initialize server on startup"""
    global config, emotions_config, tts_manager, emotion_manager, websocket_manager
    
    logger.info("Starting Chatterbox TTS Server...")
    
    try:
        # Load configurations
        config_path = Path("configs/server_config.yaml")
        emotions_path = Path("configs/emotions.yaml")
        
        if not config_path.exists():
            raise FileNotFoundError(f"Server config not found: {config_path}")
        if not emotions_path.exists():
            raise FileNotFoundError(f"Emotions config not found: {emotions_path}")
        
        # Load and validate config
        config = Config.from_yaml(config_path)
        config.validate()
        logger.info(f"Loaded server config: {config.model.type} model on {config.model.device}")
        
        # Load emotions config
        emotions_config = load_emotions_config(emotions_path)
        logger.info(f"Loaded {len(emotions_config)} emotions: {list(emotions_config.keys())}")
        
        # Initialize TTS manager
        tts_manager = TTSManager(config)
        tts_manager.initialize(emotions_config)
        
        # Initialize emotion manager
        emotion_manager = EmotionManager(config.server, tts_manager.conditionals_manager)
        
        # Initialize WebSocket connection manager
        websocket_manager = WebSocketConnectionManager(tts_manager, emotion_manager, generation_lock)
        
        logger.info("Server initialization complete!")
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down server...")
    
    if tts_manager:
        tts_manager.cleanup()
    
    logger.info("Server shutdown complete")


@app.get("/", response_class=HTMLResponse)
async def emotion_manager_ui():
    """Serve the emotion manager web UI"""
    template_path = Path(__file__).parent / "templates" / "emotion_manager.html"
    with open(template_path, 'r') as f:
        return HTMLResponse(content=f.read())


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model=config.model.type if config else "unknown",
        emotions=list(emotions_config.keys()) if emotions_config else [],
        processing=generation_lock.is_busy,
        requests_processed=tts_manager.total_requests if tts_manager else 0
    )


@app.get("/emotions", response_model=EmotionsResponse)
async def list_emotions():
    """List available emotions"""
    if not emotion_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    emotions = emotion_manager.list_emotions()
    return EmotionsResponse(emotions=list(emotions.keys()))


@app.post("/generate", response_model=GenerateResponse)
def generate_speech(request: GenerateRequest):
    """Generate speech with specified emotion (blocking)"""
    global last_request_time
    
    if not tts_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    # Validate emotion
    emotion = emotion_manager.get_emotion(request.emotion)
    if not emotion:
        available = list(emotion_manager.list_emotions().keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown emotion: {request.emotion}. Available: {available}"
        )
    
    # Try to acquire lock
    request_info = {
        "emotion": request.emotion,
        "text_length": len(request.text),
        "start_time": datetime.utcnow().isoformat()
    }
    
    acquired, queue_time = generation_lock.acquire_for_generation(
        timeout=config.server.busy_timeout,
        request_info=request_info
    )
    
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="Server busy processing another request. Please try again."
        )
    
    try:
        # Normalize text
        text_normalized = normalize_text(request.text)
        
        # Generate audio
        audio_tensor, duration, generation_time, emotion_used, voice_sample_used = tts_manager.generate(
            text=text_normalized,
            emotion=request.emotion,
            temperature=request.temperature,
            cfg_weight=request.cfg_weight,
            exaggeration=request.exaggeration
        )
        
        # Convert to base64
        audio_base64 = tts_manager.audio_to_base64(audio_tensor)
        
        # Calculate RTF
        rtf = generation_time / duration if duration > 0 else 0
        
        # Update last request time
        last_request_time = datetime.utcnow()
        
        return GenerateResponse(
            audio=audio_base64,
            duration=duration,
            rtf=rtf,
            generation_time=generation_time,
            queue_time=queue_time,
            emotion_used=emotion_used,
            text_normalized=text_normalized
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        generation_lock.release()


@app.post("/generate/raw")
def generate_speech_raw(request: GenerateRequest):
    """Generate speech and return raw WAV file (more efficient than base64)"""
    global last_request_time
    
    if not tts_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    # Validate emotion
    emotion = emotion_manager.get_emotion(request.emotion)
    if not emotion:
        available = list(emotion_manager.list_emotions().keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown emotion: {request.emotion}. Available: {available}"
        )
    
    # Try to acquire lock
    request_info = {
        "emotion": request.emotion,
        "text_length": len(request.text),
        "start_time": datetime.utcnow().isoformat()
    }
    
    acquired, queue_time = generation_lock.acquire_for_generation(
        timeout=config.server.busy_timeout,
        request_info=request_info
    )
    
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="Server busy processing another request. Please try again."
        )
    
    try:
        # Normalize text
        text_normalized = normalize_text(request.text)
        
        # Generate audio
        audio_tensor, duration, generation_time, emotion_used, voice_sample_used = tts_manager.generate(
            text=text_normalized,
            emotion=request.emotion,
            temperature=request.temperature,
            cfg_weight=request.cfg_weight,
            exaggeration=request.exaggeration
        )
        
        # Get WAV bytes
        audio_bytes = tts_manager.get_audio_bytes(audio_tensor)
        
        # Update last request time
        last_request_time = datetime.utcnow()
        
        # Return raw audio with appropriate headers
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Duration": str(duration),
                "X-RTF": str(generation_time / duration if duration > 0 else 0),
                "X-Generation-Time": str(generation_time),
                "X-Queue-Time": str(queue_time),
                "X-Emotion-Used": emotion_used,
                "X-Voice-Sample-Used": voice_sample_used,
                "X-Text-Normalized": text_normalized
            }
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        generation_lock.release()


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information"""
    if not tts_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    metrics = tts_manager.get_metrics()
    
    return ModelInfoResponse(
        model_type=metrics["model_type"],
        device=metrics["device"],
        vram_usage_mb=metrics.get("vram_usage_mb"),
        loaded_at=metrics["loaded_at"],
        processing=generation_lock.is_busy,
        total_requests=metrics["total_requests"],
        avg_rtf=metrics.get("avg_rtf"),
        avg_generation_time=metrics.get("avg_generation_time")
    )


@app.get("/status", response_model=ServerStatusResponse)
async def server_status():
    """Get server status"""
    uptime_seconds = (datetime.utcnow() - server_start_time).total_seconds()
    
    return ServerStatusResponse(
        busy=generation_lock.is_busy,
        current_request=generation_lock.current_request_info,
        uptime_seconds=uptime_seconds,
        last_request_time=last_request_time.isoformat() if last_request_time else None,
        memory_usage_mb=get_memory_usage()
    )


@app.get("/emotions/details", response_model=EmotionListResponse)
async def list_emotions_with_details():
    """List all emotions with their full configurations"""
    if not emotion_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    emotions = emotion_manager.list_emotions()
    return EmotionListResponse(
        emotions=emotions,
        total=len(emotions)
    )


@app.get("/emotions/{emotion_id}", response_model=EmotionConfig)
async def get_emotion(emotion_id: str):
    """Get details for a specific emotion"""
    if not emotion_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    emotion = emotion_manager.get_emotion(emotion_id)
    if not emotion:
        raise HTTPException(status_code=404, detail=f"Emotion '{emotion_id}' not found")
    
    return emotion


@app.post("/emotions/{emotion_id}", response_model=EmotionConfig)
async def create_emotion(emotion_id: str, request: EmotionCreateRequest):
    """Create a new emotion"""
    if not emotion_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        emotion = emotion_manager.create_emotion(emotion_id, request)
        
        # Reload emotions in TTS manager
        emotions_config = emotion_manager.list_emotions()
        tts_manager.conditionals_manager.prepare_conditionals(
            tts_manager.model, 
            emotions_config,
            batch_size=5
        )
        
        return emotion
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create emotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/emotions/{emotion_id}", response_model=EmotionConfig)
async def update_emotion(emotion_id: str, request: EmotionUpdateRequest):
    """Update an existing emotion"""
    if not emotion_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        emotion = emotion_manager.update_emotion(emotion_id, request)
        
        # Reload conditionals for all emotions to maintain cache consistency
        emotions_config = emotion_manager.list_emotions()
        tts_manager.conditionals_manager.prepare_conditionals(
            tts_manager.model,
            emotions_config,
            batch_size=5
        )
        
        return emotion
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update emotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/emotions/{emotion_id}")
async def delete_emotion(emotion_id: str):
    """Delete an emotion"""
    if not emotion_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        success = emotion_manager.delete_emotion(emotion_id)
        return {"success": success, "message": f"Emotion '{emotion_id}' deleted"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete emotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emotions/{emotion_id}/test")
async def test_emotion(emotion_id: str, request: EmotionTestRequest):
    """Test an emotion by generating sample audio"""
    if not emotion_manager or not tts_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    # Check emotion exists
    emotion = emotion_manager.get_emotion(emotion_id)
    if not emotion:
        raise HTTPException(status_code=404, detail=f"Emotion '{emotion_id}' not found")
    
    # Check if emotion has voice samples
    if not emotion.voice_samples:
        raise HTTPException(
            status_code=400, 
            detail=f"Emotion '{emotion_id}' has no voice samples configured"
        )
    
    # Acquire lock for generation
    request_info = {
        "emotion": emotion_id,
        "text_length": len(request.text),
        "test": True,
        "start_time": datetime.utcnow().isoformat()
    }
    
    acquired, queue_time = generation_lock.acquire_for_generation(
        timeout=30.0,
        request_info=request_info
    )
    
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="Server busy processing another request"
        )
    
    try:
        # Normalize text
        text_normalized = normalize_text(request.text)
        
        # Generate audio with test parameters
        audio_tensor, duration, generation_time, emotion_used, voice_sample_used = tts_manager.generate(
            text=text_normalized,
            emotion=emotion_id,
            temperature=request.temperature,
            cfg_weight=request.cfg_weight,
            exaggeration=request.exaggeration or emotion.exaggeration
        )
        
        # Get WAV bytes
        audio_bytes = tts_manager.get_audio_bytes(audio_tensor)
        
        # Extract just the filename from the path for display
        sample_filename = Path(voice_sample_used).name if voice_sample_used else "unknown"
        
        # Return audio with metadata
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Duration": str(duration),
                "X-RTF": str(generation_time / duration if duration > 0 else 0),
                "X-Generation-Time": str(generation_time),
                "X-Emotion": emotion_id,
                "X-Voice-Sample-Used": sample_filename,
                "X-Exaggeration": str(request.exaggeration or emotion.exaggeration)
            }
        )
        
    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        generation_lock.release()


@app.get("/emotions/{emotion_id}/voices", response_model=list[VoiceSample])
async def list_voice_samples(emotion_id: str):
    """List voice samples for an emotion"""
    if not emotion_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        samples = emotion_manager.list_voice_samples(emotion_id)
        return samples
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/emotions/{emotion_id}/voices/{sample_path:path}/audio")
async def get_voice_sample_audio(emotion_id: str, sample_path: str):
    """Get the audio file for a voice sample"""
    if not emotion_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    # Check emotion exists
    emotion = emotion_manager.get_emotion(emotion_id)
    if not emotion:
        raise HTTPException(status_code=404, detail=f"Emotion '{emotion_id}' not found")
    
    # Check if the sample belongs to this emotion
    if sample_path not in emotion.voice_samples:
        raise HTTPException(status_code=404, detail=f"Voice sample not found for this emotion")
    
    # Check if file exists - ensure we use absolute path for FileResponse
    file_path = Path(sample_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found")
    
    # Return the audio file
    return FileResponse(
        path=str(file_path.resolve()),
        media_type="audio/wav",
        filename=file_path.name
    )


@app.post("/emotions/{emotion_id}/voices/upload")
async def upload_voice_sample(
    emotion_id: str,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
):
    """Upload a new voice sample for an emotion"""
    if not emotion_manager or not tts_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    # Check emotion exists
    emotion = emotion_manager.get_emotion(emotion_id)
    if not emotion:
        raise HTTPException(status_code=404, detail=f"Emotion '{emotion_id}' not found")
    
    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported: WAV, MP3, FLAC, OGG"
        )
    
    try:
        # Read file content
        content = await file.read()
        logger.info(f"Read {len(content)} bytes from uploaded file {file.filename}")
        
        # Save the uploaded file
        saved_path = emotion_manager.save_uploaded_voice_sample(
            emotion_id, 
            file.filename,
            content
        )
        logger.info(f"Saved uploaded file to: {saved_path}")
        
        # Add to emotion's voice samples
        emotion = emotion_manager.add_voice_sample(emotion_id, saved_path)
        logger.info(f"Added voice sample to emotion {emotion_id}")
        
        # Try to regenerate conditionals for all emotions to maintain cache consistency
        try:
            logger.info(f"Regenerating conditionals after voice upload for emotion {emotion_id}")
            emotions_config = emotion_manager.list_emotions()
            tts_manager.conditionals_manager.prepare_conditionals(
                tts_manager.model,
                emotions_config,
                batch_size=5
            )
            logger.info(f"Successfully regenerated conditionals for all emotions")
        except Exception as conditional_error:
            logger.warning(f"Failed to regenerate conditionals (non-fatal): {conditional_error}")
        
        return {
            "success": True,
            "message": f"Voice sample uploaded successfully",
            "path": saved_path,
            "emotion": emotion_id
        }
        
    except Exception as e:
        logger.error(f"Failed to upload voice sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/emotions/{emotion_id}/voices/{sample_path:path}")
async def delete_voice_sample(emotion_id: str, sample_path: str):
    """Delete a voice sample from an emotion"""
    if not emotion_manager or not tts_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        emotion = emotion_manager.remove_voice_sample(emotion_id, sample_path)
        
        # Regenerate conditionals for all emotions to maintain cache consistency
        logger.info(f"Regenerating conditionals after voice deletion for emotion {emotion_id}")
        emotions_config = emotion_manager.list_emotions()
        tts_manager.conditionals_manager.prepare_conditionals(
            tts_manager.model,
            emotions_config,
            batch_size=5
        )
        
        return {
            "success": True,
            "message": f"Voice sample removed successfully",
            "emotion": emotion_id
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete voice sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/generate")
async def websocket_generate_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS generation"""
    if not websocket_manager:
        await websocket.close(code=1013)  # Service unavailable
        return
    
    connection_id = None
    
    try:
        # Accept connection and get connection ID
        connection_id = await websocket_manager.connect(websocket)
        
        # Handle messages
        while True:
            try:
                # Receive message from client
                raw_message = await websocket.receive_text()
                
                try:
                    message_data = json.loads(raw_message)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from connection {connection_id}: {e}")
                    await websocket_manager._send_error(
                        connection_id,
                        "Invalid JSON format",
                        str(e),
                        recoverable=True
                    )
                    continue
                
                # Process the message
                should_continue = await websocket_manager.handle_message(connection_id, message_data)
                
                if not should_continue:
                    break
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected: {connection_id}")
                break
                
            except Exception as e:
                logger.error(f"Error in WebSocket endpoint: {e}")
                if connection_id:
                    await websocket_manager._send_error(
                        connection_id,
                        "Server error",
                        str(e),
                        recoverable=False
                    )
                break
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        # Clean up connection
        if connection_id and websocket_manager:
            await websocket_manager.disconnect(connection_id)


@app.get("/ws/stats")
async def websocket_stats():
    """Get WebSocket connection statistics"""
    if not websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    
    return websocket_manager.get_connection_stats()


@app.get("/ws/metrics")
async def websocket_performance_metrics(limit: int = 50):
    """Get WebSocket performance metrics"""
    if not websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    
    return {
        "metrics": websocket_manager.get_performance_metrics(limit),
        "total_metrics": len(websocket_manager.performance_metrics)
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500
        }
    )


def main():
    """Main entry point"""
    # Load config to get server settings
    config_path = Path("configs/server_config.yaml")
    if config_path.exists():
        server_config = Config.from_yaml(config_path)
        host = server_config.server.host
        port = server_config.server.port
        workers = server_config.server.workers
    else:
        host = "0.0.0.0"
        port = 8000
        workers = 1
    
    logger.info(f"Starting server on {host}:{port} with {workers} worker(s)")
    
    uvicorn.run(
        "src.server.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False
    )


if __name__ == "__main__":
    main()