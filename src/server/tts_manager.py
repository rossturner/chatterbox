import time
import base64
import logging
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

import torch
import torchaudio
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.chatterbox.tts import ChatterboxTTS
from src.chatterbox.models.s3gen import S3GEN_SR
from .config import Config
from .conditionals_manager import ConditionalsManager

logger = logging.getLogger(__name__)


class TTSManager:
    """Manager for TTS model and generation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[ChatterboxTTS] = None
        self.conditionals_manager = ConditionalsManager(Path(config.caching.conditionals_dir))
        self.device = config.model.device
        self.sample_rate = S3GEN_SR
        
        # Metrics tracking
        self.loaded_at = None
        self.total_requests = 0
        self.total_generation_time = 0.0
        self.total_rtf = 0.0
        
    def initialize(self, emotions_config: dict) -> None:
        """Initialize TTS model and prepare conditionals"""
        logger.info("Initializing TTS Manager...")
        
        # Load the model
        self._load_model()
        
        # Prepare and cache conditionals
        if self.config.caching.precompute_on_startup:
            logger.info("Pre-computing conditionals for all emotions...")
            prepared, loaded = self.conditionals_manager.prepare_conditionals(
                self.model,
                emotions_config,
                batch_size=self.config.caching.batch_size
            )
            logger.info(f"Conditionals ready: {prepared} prepared, {loaded} loaded from cache")
        else:
            logger.info("Loading existing conditionals from cache...")
            self.conditionals_manager.load_all_to_ram()
        
        self.loaded_at = datetime.utcnow()
        logger.info("TTS Manager initialized successfully")
    
    def _load_model(self) -> None:
        """Load the TTS model based on configuration"""
        model_type = self.config.model.type
        model_path = self.config.model.path
        
        logger.info(f"Loading {model_type} model on {self.device}...")
        start_time = time.time()
        
        try:
            if model_type == "base":
                # Load from HuggingFace
                self.model = ChatterboxTTS.from_pretrained(self.device)
                logger.info("Loaded base model from HuggingFace")
                
            elif model_type in ["grpo", "quantized"]:
                # Load from local path
                if not model_path:
                    raise ValueError(f"Model path required for {model_type} model")
                
                model_path = Path(model_path)
                if not model_path.exists():
                    raise FileNotFoundError(f"Model not found: {model_path}")
                
                self.model = ChatterboxTTS.from_local(model_path, self.device)
                logger.info(f"Loaded {model_type} model from {model_path}")
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")
            
            # Report memory usage
            if torch.cuda.is_available() and self.device == "cuda":
                torch.cuda.synchronize()
                memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                logger.info(f"GPU memory usage: {memory_mb:.1f} MB")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(
        self,
        text: str,
        emotion: str,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        exaggeration: Optional[float] = None
    ) -> Tuple[torch.Tensor, float, float, str]:
        """
        Generate speech for text with specified emotion.
        
        Args:
            text: Text to synthesize
            emotion: Emotion to use
            temperature: Sampling temperature
            cfg_weight: CFG weight
            exaggeration: Optional exaggeration override
            
        Returns:
            Tuple of (audio_tensor, duration, generation_time, emotion_used)
        """
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        # Get random conditionals from RAM
        conditionals_cpu, voice_sample_used = self.conditionals_manager.get_random_conditionals(emotion)
        
        # Transfer to GPU (fast, ~0.11 MB)
        conditionals_gpu = conditionals_cpu.to(device=self.device)
        
        # Get emotion config for default exaggeration if not specified
        if exaggeration is None:
            emotion_config = self.conditionals_manager.get_emotion_config(emotion)
            if emotion_config:
                if hasattr(emotion_config, 'exaggeration'):
                    exaggeration = emotion_config.exaggeration
                else:
                    exaggeration = emotion_config.get('exaggeration', 0.5)
            else:
                exaggeration = 0.5
        
        # Update exaggeration in conditionals if different
        if hasattr(conditionals_gpu.t3, 'emotion_adv'):
            current_exag = conditionals_gpu.t3.emotion_adv[0, 0, 0].item()
            if abs(current_exag - exaggeration) > 0.01:  # Only update if significantly different
                from src.chatterbox.models.t3.modules.cond_enc import T3Cond
                conditionals_gpu.t3 = T3Cond(
                    speaker_emb=conditionals_gpu.t3.speaker_emb,
                    cond_prompt_speech_tokens=conditionals_gpu.t3.cond_prompt_speech_tokens,
                    emotion_adv=exaggeration * torch.ones(1, 1, 1),
                ).to(device=self.device)
        
        try:
            # Set pre-computed conditionals
            self.model.conds = conditionals_gpu
            
            # Generate audio
            start_time = time.time()
            audio_tensor = self.model.generate(
                text=text,
                temperature=temperature,
                cfg_weight=cfg_weight
            )
            generation_time = time.time() - start_time
            
            # Calculate duration
            duration = len(audio_tensor.squeeze()) / self.sample_rate
            
            # Update metrics
            self.total_requests += 1
            self.total_generation_time += generation_time
            rtf = generation_time / duration if duration > 0 else 0
            self.total_rtf += rtf
            
            logger.info(f"Generated {duration:.2f}s audio in {generation_time:.2f}s (RTF: {rtf:.3f}) using {voice_sample_used}")
            
            return audio_tensor, duration, generation_time, emotion, voice_sample_used
            
        finally:
            # Clean up GPU memory if configured
            if not self.config.model.keep_warm:
                del conditionals_gpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def audio_to_base64(self, audio_tensor: torch.Tensor) -> str:
        """Convert audio tensor to base64 encoded WAV"""
        # Save to temporary bytes buffer
        import io
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, self.sample_rate, format='wav')
        buffer.seek(0)
        
        # Encode to base64
        audio_bytes = buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return audio_base64
    
    def get_audio_bytes(self, audio_tensor: torch.Tensor) -> bytes:
        """Convert audio tensor to WAV bytes"""
        import io
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, self.sample_rate, format='wav')
        buffer.seek(0)
        return buffer.read()
    
    def get_metrics(self) -> dict:
        """Get current metrics"""
        metrics = {
            "model_type": self.config.model.type,
            "device": self.device,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "total_requests": self.total_requests,
        }
        
        if self.total_requests > 0:
            metrics["avg_generation_time"] = self.total_generation_time / self.total_requests
            metrics["avg_rtf"] = self.total_rtf / self.total_requests
        else:
            metrics["avg_generation_time"] = None
            metrics["avg_rtf"] = None
        
        # Add memory usage if available
        if torch.cuda.is_available() and self.device == "cuda":
            metrics["vram_usage_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            metrics["vram_usage_mb"] = None
        
        return metrics
    
    def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
            self.model = None
        
        self.conditionals_manager.clear_cache()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("TTS Manager cleaned up")