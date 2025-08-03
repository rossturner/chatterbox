"""
Pydantic models for the Chatterbox TTS API server.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import base64


class TTSGenerationRequest(BaseModel):
    """Request model for TTS generation."""
    text: str = Field(..., min_length=1, max_length=2000, description="Text to synthesize")
    emotion: str = Field(..., description="Emotion/character name to use")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")
    repetition_penalty: float = Field(1.2, ge=1.0, le=2.0, description="Repetition penalty")
    min_p: float = Field(0.05, ge=0.01, le=1.0, description="Min-P sampling parameter")
    top_p: float = Field(1.0, ge=0.1, le=1.0, description="Top-P sampling parameter") 
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0, description="Classifier-free guidance weight")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    return_audio_data: bool = Field(False, description="Return base64 encoded audio data instead of file")


class TTSGenerationResponse(BaseModel):
    """Response model for TTS generation."""
    success: bool
    message: str
    audio_url: Optional[str] = None
    audio_data: Optional[str] = None  # base64 encoded audio
    sample_rate: int
    duration_seconds: float
    generation_time_seconds: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmotionProfile(BaseModel):
    """Model for emotion profile configuration."""
    id: str = Field(..., description="Unique emotion identifier")
    name: str = Field(..., description="Display name for the emotion")
    character: str = Field(..., description="Character name this emotion belongs to")
    voice_samples: List[str] = Field(..., description="List of voice sample file paths")
    exaggeration: float = Field(0.5, ge=0.0, le=1.0, description="Emotion exaggeration level")
    description: Optional[str] = Field(None, description="Description of the emotion")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class EmotionProfileCreate(BaseModel):
    """Model for creating new emotion profiles."""
    name: str = Field(..., min_length=1, max_length=100)
    character: str = Field(..., min_length=1, max_length=100)
    exaggeration: float = Field(0.5, ge=0.0, le=1.0)
    description: Optional[str] = Field(None, max_length=500)


class EmotionProfileUpdate(BaseModel):
    """Model for updating emotion profiles."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    character: Optional[str] = Field(None, min_length=1, max_length=100) 
    exaggeration: Optional[float] = Field(None, ge=0.0, le=1.0)
    description: Optional[str] = Field(None, max_length=500)


class VoiceUploadRequest(BaseModel):
    """Model for voice sample upload."""
    emotion_id: str = Field(..., description="Emotion profile ID to add voice to")
    filename: str = Field(..., description="Original filename")
    description: Optional[str] = Field(None, description="Description of the voice sample")


class VoiceUploadResponse(BaseModel):
    """Response model for voice upload."""
    success: bool
    message: str
    voice_id: Optional[str] = None
    file_path: Optional[str] = None


class EmotionListResponse(BaseModel):
    """Response model for listing emotions."""
    emotions: List[EmotionProfile]
    total_count: int
    characters: List[str]  # List of unique character names


class ServerStatus(BaseModel):
    """Server status and health information."""
    status: str
    version: str
    model_loaded: bool
    emotions_loaded: int
    memory_usage_mb: float
    gpu_available: bool
    device: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Standard error response model."""
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class GenerationSettings(BaseModel):
    """Default generation settings."""
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    min_p: float = 0.05
    top_p: float = 1.0
    cfg_weight: float = 0.5


class VoiceSample(BaseModel):
    """Model for individual voice sample."""
    id: str
    filename: str
    file_path: str
    description: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)


class TestGenerationRequest(BaseModel):
    """Request model for testing emotion profiles."""
    emotion_id: str = Field(..., description="Emotion profile to test")
    text: str = Field("Hello, this is a test of the emotion profile.", 
                     description="Test text to generate")
    settings: Optional[GenerationSettings] = Field(None, 
                                                   description="Custom generation settings")