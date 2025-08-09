from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class GenerateRequest(BaseModel):
    """Request model for speech generation"""
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=500)
    emotion: str = Field(..., description="Emotion to use for synthesis")
    temperature: Optional[float] = Field(0.8, ge=0.05, le=5.0, description="Sampling temperature")
    cfg_weight: Optional[float] = Field(0.5, ge=0.2, le=1.0, description="CFG weight/pace")
    exaggeration: Optional[float] = Field(None, ge=0.1, le=2.0, description="Emotion exaggeration (overrides emotion default)")
    
    @validator('text')
    def validate_text(cls, v):
        # Remove excessive whitespace
        v = ' '.join(v.split())
        if len(v) == 0:
            raise ValueError("Text cannot be empty after normalization")
        return v


class GenerateResponse(BaseModel):
    """Response model for speech generation"""
    audio: str = Field(..., description="Base64 encoded WAV audio")
    duration: float = Field(..., description="Audio duration in seconds")
    rtf: float = Field(..., description="Real-time factor (generation_time / audio_duration)")
    generation_time: float = Field(..., description="Time taken to generate audio in seconds")
    queue_time: float = Field(..., description="Time spent waiting for lock in seconds")
    emotion_used: str = Field(..., description="Emotion that was used")
    text_normalized: str = Field(..., description="Normalized text that was synthesized")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Server status")
    model: str = Field(..., description="Model type loaded")
    emotions: List[str] = Field(..., description="Available emotions")
    processing: bool = Field(..., description="Whether currently processing a request")
    requests_processed: int = Field(..., description="Total requests processed")


class EmotionsResponse(BaseModel):
    """Response model for emotions list"""
    emotions: List[str] = Field(..., description="List of available emotion names")


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_type: str = Field(..., description="Type of model loaded")
    device: str = Field(..., description="Device model is running on")
    vram_usage_mb: Optional[float] = Field(None, description="VRAM usage in MB")
    loaded_at: str = Field(..., description="ISO timestamp when model was loaded")
    processing: bool = Field(..., description="Whether currently processing")
    total_requests: int = Field(..., description="Total requests processed")
    avg_rtf: Optional[float] = Field(None, description="Average RTF across all requests")
    avg_generation_time: Optional[float] = Field(None, description="Average generation time in seconds")


class ServerStatusResponse(BaseModel):
    """Response model for server status"""
    busy: bool = Field(..., description="Whether server is currently busy")
    current_request: Optional[dict] = Field(None, description="Current request being processed")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    last_request_time: Optional[str] = Field(None, description="ISO timestamp of last request")
    memory_usage_mb: Optional[float] = Field(None, description="Current memory usage in MB")


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")


class VoiceSample(BaseModel):
    """Model for a voice sample file"""
    filename: str = Field(..., description="Filename of the voice sample")
    path: str = Field(..., description="Relative path to the voice sample")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    uploaded_at: str = Field(..., description="ISO timestamp when uploaded")


class EmotionConfig(BaseModel):
    """Model for emotion configuration"""
    name: str = Field(..., description="Display name for the emotion")
    exaggeration: float = Field(0.5, ge=0.1, le=2.0, description="Default exaggeration level")
    voice_samples: List[str] = Field(..., description="List of voice sample paths")
    created_at: str = Field(..., description="ISO timestamp when created")
    updated_at: str = Field(..., description="ISO timestamp when last updated")


class EmotionCreateRequest(BaseModel):
    """Request model for creating an emotion"""
    name: str = Field(..., description="Display name for the emotion", min_length=1, max_length=50)
    exaggeration: float = Field(0.5, ge=0.1, le=2.0, description="Default exaggeration level")
    voice_samples: Optional[List[str]] = Field([], description="Initial voice sample paths")


class EmotionUpdateRequest(BaseModel):
    """Request model for updating an emotion"""
    name: Optional[str] = Field(None, description="New display name", min_length=1, max_length=50)
    exaggeration: Optional[float] = Field(None, ge=0.1, le=2.0, description="New exaggeration level")


class EmotionTestRequest(BaseModel):
    """Request model for testing an emotion"""
    text: str = Field("Hello! This is a test of the emotion voice.", description="Text to synthesize")
    temperature: float = Field(0.8, ge=0.05, le=5.0, description="Sampling temperature")
    cfg_weight: float = Field(0.5, ge=0.2, le=1.0, description="CFG weight")
    exaggeration: Optional[float] = Field(None, ge=0.1, le=2.0, description="Override exaggeration")


class EmotionListResponse(BaseModel):
    """Response model for listing emotions with details"""
    emotions: dict = Field(..., description="Dictionary of emotion IDs to configs")
    total: int = Field(..., description="Total number of emotions")