from pydantic import BaseModel, Field, validator
from typing import Optional, List, Union
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


# WebSocket Models

class WebSocketMessageType(str, Enum):
    """Types of WebSocket messages"""
    REQUEST = "request"
    AUDIO = "audio"
    PROGRESS = "progress"
    ERROR = "error"
    COMPLETE = "complete"
    STATUS = "status"


class WebSocketRequest(BaseModel):
    """WebSocket request model for streaming TTS generation"""
    type: WebSocketMessageType = Field(WebSocketMessageType.REQUEST, description="Message type")
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=1000)
    emotion: str = Field(..., description="Emotion to use for synthesis")
    temperature: Optional[float] = Field(0.8, ge=0.05, le=5.0, description="Sampling temperature")
    cfg_weight: Optional[float] = Field(0.5, ge=0.2, le=1.0, description="CFG weight/pace")
    exaggeration: Optional[float] = Field(None, ge=0.1, le=2.0, description="Emotion exaggeration")
    
    # Audio conditioning
    reference_audio: Optional[str] = Field(None, description="Path to reference audio file for voice conditioning")
    
    # Streaming-specific options
    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")
    include_progress: bool = Field(True, description="Whether to send progress updates")
    
    @validator('text')
    def validate_text(cls, v):
        # Remove excessive whitespace
        v = ' '.join(v.split())
        if len(v) == 0:
            raise ValueError("Text cannot be empty after normalization")
        return v


class WebSocketAudioResponse(BaseModel):
    """WebSocket audio chunk response"""
    type: WebSocketMessageType = Field(WebSocketMessageType.AUDIO, description="Message type")
    request_id: Optional[str] = Field(None, description="Request ID if provided")
    sentence_index: int = Field(..., description="Index of the sentence (0-based)")
    sentence_text: str = Field(..., description="Text of this sentence")
    audio_base64: str = Field(..., description="Base64 encoded WAV audio chunk")
    duration: float = Field(..., description="Audio duration in seconds")
    generation_time: float = Field(..., description="Time taken to generate this chunk")
    rtf: float = Field(..., description="Real-time factor for this chunk")
    is_final: bool = Field(False, description="Whether this is the final chunk")
    cumulative_duration: float = Field(..., description="Total audio duration so far")


class WebSocketProgressResponse(BaseModel):
    """WebSocket progress update response"""
    type: WebSocketMessageType = Field(WebSocketMessageType.PROGRESS, description="Message type")
    request_id: Optional[str] = Field(None, description="Request ID if provided")
    sentence_index: int = Field(..., description="Current sentence being processed")
    sentence_text: str = Field(..., description="Text of current sentence")
    total_sentences: int = Field(..., description="Total number of sentences")
    progress_percent: float = Field(..., description="Progress percentage (0-100)")
    estimated_remaining_time: Optional[float] = Field(None, description="Estimated remaining time in seconds")


class WebSocketCompleteResponse(BaseModel):
    """WebSocket completion response"""
    type: WebSocketMessageType = Field(WebSocketMessageType.COMPLETE, description="Message type")
    request_id: Optional[str] = Field(None, description="Request ID if provided")
    total_sentences: int = Field(..., description="Total sentences processed")
    total_duration: float = Field(..., description="Total audio duration in seconds")
    total_generation_time: float = Field(..., description="Total generation time in seconds")
    overall_rtf: float = Field(..., description="Overall real-time factor")
    time_to_first_sentence: float = Field(..., description="Time to first sentence in seconds")
    emotion_used: str = Field(..., description="Emotion that was used")
    text_normalized: str = Field(..., description="Final normalized text")


class WebSocketErrorResponse(BaseModel):
    """WebSocket error response"""
    type: WebSocketMessageType = Field(WebSocketMessageType.ERROR, description="Message type")
    request_id: Optional[str] = Field(None, description="Request ID if provided")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    sentence_index: Optional[int] = Field(None, description="Sentence where error occurred")
    recoverable: bool = Field(False, description="Whether the error is recoverable")


class WebSocketStatusResponse(BaseModel):
    """WebSocket status response"""
    type: WebSocketMessageType = Field(WebSocketMessageType.STATUS, description="Message type")
    connected: bool = Field(..., description="Connection status")
    server_busy: bool = Field(..., description="Whether server is busy")
    queue_position: Optional[int] = Field(None, description="Position in queue if applicable")
    estimated_wait_time: Optional[float] = Field(None, description="Estimated wait time in seconds")


# Union type for all WebSocket messages
WebSocketMessage = Union[
    WebSocketRequest,
    WebSocketAudioResponse,
    WebSocketProgressResponse,
    WebSocketCompleteResponse,
    WebSocketErrorResponse,
    WebSocketStatusResponse
]


class StreamingPerformanceMetrics(BaseModel):
    """Performance metrics for streaming TTS generation"""
    request_id: Optional[str] = Field(None, description="Request ID")
    text_length: int = Field(..., description="Length of input text")
    sentence_count: int = Field(..., description="Number of sentences")
    time_to_first_sentence: float = Field(..., description="Time to first sentence (TTFS)")
    total_generation_time: float = Field(..., description="Total generation time")
    total_audio_duration: float = Field(..., description="Total audio duration")
    overall_rtf: float = Field(..., description="Overall RTF")
    average_sentence_generation_time: float = Field(..., description="Average time per sentence")
    sentence_metrics: List[dict] = Field(..., description="Per-sentence metrics")
    emotion_used: str = Field(..., description="Emotion used")
    timestamp: str = Field(..., description="ISO timestamp of request")