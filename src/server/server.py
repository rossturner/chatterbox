"""
FastAPI server for Chatterbox TTS with voice precomputation and emotion management.
"""

import asyncio
import logging
import time
import uuid
import base64
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import traceback

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request

from chatterbox.tts import ChatterboxTTS
from .models import (
    TTSGenerationRequest, TTSGenerationResponse, EmotionProfile, EmotionProfileCreate,
    EmotionProfileUpdate, VoiceUploadRequest, VoiceUploadResponse, EmotionListResponse,
    ServerStatus, ErrorResponse, TestGenerationRequest
)
from .voice_manager import VoiceManager
from .config import get_config

# Configure logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
config = get_config()
model: Optional[ChatterboxTTS] = None
voice_manager: Optional[VoiceManager] = None
server_start_time = time.time()


async def load_model():
    """Load the Chatterbox TTS model with optimizations."""
    global model
    
    logger.info(f"Loading Chatterbox TTS model on device: {config.model_device}")
    
    try:
        # Load base model
        model = ChatterboxTTS.from_pretrained(device=config.model_device)
        
        # Apply optimizations
        if config.enable_torch_compile and hasattr(torch, 'compile'):
            logger.info(f"Compiling model with mode: {config.compile_mode}")
            try:
                model.t3 = torch.compile(model.t3, mode=config.compile_mode)
                model.s3gen = torch.compile(model.s3gen, mode=config.compile_mode)
                logger.info("Model compilation successful")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        # Enable mixed precision if requested
        if config.enable_mixed_precision and config.model_device == "cuda":
            logger.info("Mixed precision enabled")
        
        logger.info("Model loaded successfully!")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global model, voice_manager
    
    # Startup
    logger.info("Starting Chatterbox TTS API server...")
    
    try:
        # Load model
        model = await load_model()
        
        # Initialize voice manager
        voice_manager = VoiceManager(model)
        
        # Warmup with a quick generation if we have any emotions loaded
        if voice_manager.emotions:
            first_emotion = list(voice_manager.emotions.keys())[0]
            if voice_manager.is_emotion_ready(first_emotion):
                logger.info("Warming up model with sample generation...")
                await voice_manager.use_emotion(first_emotion)
                try:
                    _ = model.generate("Warmup generation.")
                    logger.info("Model warmup completed")
                except Exception as e:
                    logger.warning(f"Model warmup failed: {e}")
        
        logger.info("Server startup completed successfully!")
        
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down server...")


# Create FastAPI app
app = FastAPI(
    title=config.api_title,
    version=config.api_version,
    description=config.api_description,
    docs_url="/docs" if config.enable_docs else None,
    redoc_url="/redoc" if config.enable_docs else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))


def get_voice_manager() -> VoiceManager:
    """Dependency to get voice manager."""
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="Voice manager not initialized")
    return voice_manager


def get_model() -> ChatterboxTTS:
    """Dependency to get model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


# Health check endpoint
@app.get("/health", response_model=ServerStatus)
async def health_check():
    """Get server health status."""
    try:
        memory_usage = 0
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        elif torch.backends.mps.is_available():
            memory_usage = torch.mps.current_allocated_memory() / (1024 ** 2)  # MB
        
        vm_stats = voice_manager.get_stats() if voice_manager else {}
        
        return ServerStatus(
            status="healthy",
            version=config.api_version,
            model_loaded=model is not None,
            emotions_loaded=vm_stats.get('ready_emotions', 0),
            memory_usage_mb=memory_usage,
            gpu_available=torch.cuda.is_available() or torch.backends.mps.is_available(),
            device=config.model_device,
            uptime_seconds=time.time() - server_start_time
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# TTS Generation endpoint
@app.post("/generate", response_model=TTSGenerationResponse)
async def generate_tts(
    request: TTSGenerationRequest,
    vm: VoiceManager = Depends(get_voice_manager),
    tts_model: ChatterboxTTS = Depends(get_model)
):
    """Generate TTS audio using precomputed emotion conditionals."""
    try:
        start_time = time.time()
        
        # Validate emotion exists and is ready
        if not vm.is_emotion_ready(request.emotion):
            raise HTTPException(
                status_code=400, 
                detail=f"Emotion '{request.emotion}' not found or not ready"
            )
        
        # Switch to the requested emotion
        if not await vm.use_emotion(request.emotion):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to switch to emotion '{request.emotion}'"
            )
        
        # Set random seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(request.seed)
        
        # Generate audio with mixed precision if enabled
        generation_start = time.time()
        
        if config.enable_mixed_precision and config.model_device == "cuda":
            with torch.cuda.amp.autocast():
                wav = tts_model.generate(
                    text=request.text,
                    temperature=request.temperature,
                    repetition_penalty=request.repetition_penalty,
                    min_p=request.min_p,
                    top_p=request.top_p,
                    cfg_weight=request.cfg_weight
                )
        else:
            wav = tts_model.generate(
                text=request.text,
                temperature=request.temperature,
                repetition_penalty=request.repetition_penalty,
                min_p=request.min_p,
                top_p=request.top_p,
                cfg_weight=request.cfg_weight
            )
        
        generation_time = time.time() - generation_start
        
        # Calculate duration
        duration_seconds = wav.shape[-1] / tts_model.sr
        
        # Save or encode audio
        if request.return_audio_data:
            # Return base64 encoded audio data as proper WAV file
            wav_numpy = wav.squeeze(0).cpu().numpy()
            
            # Create a proper WAV file in memory using torchaudio
            import io
            buffer = io.BytesIO()
            torchaudio.save(buffer, wav.cpu(), tts_model.sr, format="wav")
            audio_bytes = buffer.getvalue()
            audio_data = base64.b64encode(audio_bytes).decode('utf-8')
            
            return TTSGenerationResponse(
                success=True,
                message="Audio generated successfully",
                audio_data=audio_data,
                sample_rate=tts_model.sr,
                duration_seconds=duration_seconds,
                generation_time_seconds=generation_time,
                metadata={
                    "emotion": request.emotion,
                    "text_length": len(request.text),
                    "rtf": generation_time / duration_seconds,
                    "device": config.model_device
                }
            )
        else:
            # Save to file and return URL
            output_filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            output_path = config.get_output_path(output_filename)
            
            torchaudio.save(str(output_path), wav.cpu(), tts_model.sr)
            
            return TTSGenerationResponse(
                success=True,
                message="Audio generated successfully",
                audio_url=f"/outputs/{output_filename}",
                sample_rate=tts_model.sr,
                duration_seconds=duration_seconds,
                generation_time_seconds=generation_time,
                metadata={
                    "emotion": request.emotion,
                    "text_length": len(request.text),
                    "rtf": generation_time / duration_seconds,
                    "device": config.model_device
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# Emotion management endpoints
@app.get("/emotions", response_model=EmotionListResponse)
async def list_emotions(vm: VoiceManager = Depends(get_voice_manager)):
    """List all emotion profiles."""
    emotions = vm.list_emotions()
    characters = vm.list_characters()
    
    return EmotionListResponse(
        emotions=emotions,
        total_count=len(emotions),
        characters=characters
    )


@app.post("/emotions", response_model=EmotionProfile)
async def create_emotion(
    emotion_data: EmotionProfileCreate,
    vm: VoiceManager = Depends(get_voice_manager)
):
    """Create a new emotion profile."""
    try:
        emotion = await vm.create_emotion(emotion_data)
        return emotion
    except Exception as e:
        logger.error(f"Failed to create emotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emotions/{emotion_id}", response_model=EmotionProfile)
async def get_emotion(
    emotion_id: str,
    vm: VoiceManager = Depends(get_voice_manager)
):
    """Get a specific emotion profile."""
    emotion = vm.get_emotion(emotion_id)
    if not emotion:
        raise HTTPException(status_code=404, detail="Emotion not found")
    return emotion


@app.put("/emotions/{emotion_id}", response_model=EmotionProfile)
async def update_emotion(
    emotion_id: str,
    updates: EmotionProfileUpdate,
    vm: VoiceManager = Depends(get_voice_manager)
):
    """Update an emotion profile."""
    try:
        emotion = await vm.update_emotion(emotion_id, updates)
        if not emotion:
            raise HTTPException(status_code=404, detail="Emotion not found")
        return emotion
    except Exception as e:
        logger.error(f"Failed to update emotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/emotions/{emotion_id}")
async def delete_emotion(
    emotion_id: str,
    vm: VoiceManager = Depends(get_voice_manager)
):
    """Delete an emotion profile."""
    try:
        success = await vm.delete_emotion(emotion_id)
        if not success:
            raise HTTPException(status_code=404, detail="Emotion not found")
        return {"success": True, "message": "Emotion deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete emotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Voice upload endpoint
@app.post("/emotions/{emotion_id}/voices", response_model=VoiceUploadResponse)
async def upload_voice(
    emotion_id: str,
    file: UploadFile = File(...),
    description: Optional[str] = None,
    vm: VoiceManager = Depends(get_voice_manager)
):
    """Upload a voice sample to an emotion profile."""
    try:
        # Validate emotion exists
        if not vm.get_emotion(emotion_id):
            raise HTTPException(status_code=404, detail="Emotion not found")
        
        # Validate file type
        if not file.filename or not any(file.filename.lower().endswith(f".{fmt}") for fmt in config.allowed_audio_formats):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format. Allowed formats: {config.allowed_audio_formats}"
            )
        
        # Validate file size
        if file.size > config.max_voice_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {config.max_voice_file_size / (1024*1024):.1f}MB"
            )
        
        # Save uploaded file temporarily
        temp_path = config.cache_dir / f"temp_{uuid.uuid4().hex}{Path(file.filename).suffix}"
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Add to voice manager
            voice_sample = await vm.add_voice_sample(
                emotion_id, 
                temp_path, 
                file.filename,
                description
            )
            
            if not voice_sample:
                raise HTTPException(status_code=500, detail="Failed to add voice sample")
            
            return VoiceUploadResponse(
                success=True,
                message="Voice sample uploaded successfully",
                voice_id=voice_sample.id,
                file_path=voice_sample.file_path
            )
            
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Voice removal endpoint
@app.delete("/emotions/{emotion_id}/voices/remove")
async def remove_voice(
    emotion_id: str,
    voice_filename: str = Query(..., description="The voice filename to remove"),
    vm: VoiceManager = Depends(get_voice_manager)
):
    """Remove a voice sample from an emotion profile."""
    try:
        logger.info(f"Attempting to remove voice sample: '{voice_filename}' from emotion: '{emotion_id}'")
        
        # Debug: Check if emotion exists
        emotion = vm.get_emotion(emotion_id)
        if not emotion:
            logger.error(f"Emotion '{emotion_id}' not found")
            raise HTTPException(status_code=404, detail="Emotion not found")
            
        logger.info(f"Emotion found: '{emotion.name}' with voice samples: {emotion.voice_samples}")
        
        success = await vm.remove_voice_sample(emotion_id, voice_filename)
        if not success:
            raise HTTPException(status_code=404, detail="Voice sample not found")
        
        return {"success": True, "message": "Voice sample removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice removal failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Test generation endpoint
@app.post("/emotions/{emotion_id}/test")
async def test_emotion(
    emotion_id: str,
    test_request: TestGenerationRequest,
    vm: VoiceManager = Depends(get_voice_manager),
    tts_model: ChatterboxTTS = Depends(get_model)
):
    """Test an emotion profile with quick generation."""
    try:
        # Validate emotion exists and is ready
        if not vm.is_emotion_ready(emotion_id):
            raise HTTPException(
                status_code=400,
                detail=f"Emotion '{emotion_id}' not found or not ready"
            )
        
        # Use the emotion
        if not await vm.use_emotion(emotion_id):
            raise HTTPException(status_code=500, detail="Failed to switch to emotion")
        
        # Use custom settings if provided
        settings = test_request.settings or {}
        
        # Generate test audio
        wav = tts_model.generate(
            text=test_request.text,
            temperature=getattr(settings, 'temperature', 0.8),
            repetition_penalty=getattr(settings, 'repetition_penalty', 1.2),
            min_p=getattr(settings, 'min_p', 0.05),
            top_p=getattr(settings, 'top_p', 1.0),
            cfg_weight=getattr(settings, 'cfg_weight', 0.5)
        )
        
        # Save test audio
        test_filename = f"test_{emotion_id}_{uuid.uuid4().hex[:6]}.wav"
        test_path = config.get_output_path(test_filename)
        torchaudio.save(str(test_path), wav.cpu(), tts_model.sr)
        
        return {
            "success": True,
            "message": "Test generation completed",
            "audio_url": f"/outputs/{test_filename}",
            "duration_seconds": wav.shape[-1] / tts_model.sr
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# File serving endpoints
@app.get("/outputs/{filename}")
async def serve_output(filename: str):
    """Serve generated audio files."""
    file_path = config.get_output_path(filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.get("/")
async def root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details={"type": type(exc).__name__, "message": str(exc)}
        ).dict()
    )


def run_server():
    """Run the server with uvicorn."""
    uvicorn.run(
        "src.server.server:app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        reload=config.reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()