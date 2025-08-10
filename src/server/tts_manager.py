import time
import base64
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Generator
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
    ) -> Tuple[torch.Tensor, float, float, str, str]:
        """
        Generate speech for text with specified emotion.
        
        Args:
            text: Text to synthesize
            emotion: Emotion to use
            temperature: Sampling temperature
            cfg_weight: CFG weight
            exaggeration: Optional exaggeration override
            
        Returns:
            Tuple of (audio_tensor, duration, generation_time, emotion_used, voice_sample_used)
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
        import numpy as np
        import struct
        
        # Ensure tensor is properly shaped and on CPU
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()
        audio_tensor = audio_tensor.cpu().numpy()
        
        # Normalize audio to [-1, 1] range
        if audio_tensor.max() > 1.0 or audio_tensor.min() < -1.0:
            audio_tensor = audio_tensor / np.max(np.abs(audio_tensor))
        
        # Convert to 16-bit integers
        audio_int16 = (audio_tensor * 32767).astype(np.int16)
        
        # Create WAV file manually to avoid torchaudio channel layout issues
        buffer = io.BytesIO()
        
        # WAV header
        num_channels = 1
        sample_width = 2  # 16-bit = 2 bytes
        num_frames = len(audio_int16)
        
        # Write WAV header
        buffer.write(b'RIFF')
        buffer.write(struct.pack('<L', 36 + num_frames * num_channels * sample_width))
        buffer.write(b'WAVE')
        buffer.write(b'fmt ')
        buffer.write(struct.pack('<L', 16))  # PCM format chunk size
        buffer.write(struct.pack('<H', 1))   # PCM format
        buffer.write(struct.pack('<H', num_channels))
        buffer.write(struct.pack('<L', self.sample_rate))
        buffer.write(struct.pack('<L', self.sample_rate * num_channels * sample_width))
        buffer.write(struct.pack('<H', num_channels * sample_width))
        buffer.write(struct.pack('<H', 16))  # bits per sample
        buffer.write(b'data')
        buffer.write(struct.pack('<L', num_frames * num_channels * sample_width))
        
        # Write audio data
        buffer.write(audio_int16.tobytes())
        
        buffer.seek(0)
        return buffer.read()
    
    def trim_silence(self, audio_tensor: torch.Tensor, silence_threshold: float = 0.01) -> torch.Tensor:
        """
        Trim leading and trailing silence from audio tensor.
        
        Args:
            audio_tensor: Audio tensor to trim
            silence_threshold: Amplitude threshold below which audio is considered silence
            
        Returns:
            Trimmed audio tensor with preserved format
        """
        # Preserve original tensor format
        original_shape = audio_tensor.shape
        
        # Work with squeezed version for processing
        if audio_tensor.dim() > 1:
            audio_1d = audio_tensor.squeeze()
        else:
            audio_1d = audio_tensor
        
        # Find first and last non-silent samples
        non_silent = torch.abs(audio_1d) > silence_threshold
        if not non_silent.any():
            # If entire audio is silent, return a short silence with original format
            silence = torch.zeros(int(self.sample_rate * 0.1), dtype=audio_tensor.dtype, device=audio_tensor.device)
            if len(original_shape) > 1:
                silence = silence.unsqueeze(0)
            return silence
        
        first_sound = torch.argmax(non_silent.float())
        last_sound = len(audio_1d) - torch.argmax(non_silent.flip(0).float()) - 1
        
        # Keep small padding to avoid harsh cuts
        padding = int(self.sample_rate * 0.02)  # 20ms padding
        start = max(0, first_sound - padding)
        end = min(len(audio_1d), last_sound + padding + 1)
        
        # Trim the audio
        trimmed = audio_1d[start:end]
        
        # Restore original format if needed
        if len(original_shape) > 1:
            trimmed = trimmed.unsqueeze(0)
        
        return trimmed
    
    def prepare_conditionals(self, reference_audio_path: str):
        """
        Prepare voice conditioning from reference audio file.
        This will be used for all subsequent generations until changed.
        
        Args:
            reference_audio_path: Path to reference audio file
        """
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        try:
            from pathlib import Path
            audio_path = Path(reference_audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
            
            logger.info(f"Preparing conditionals from reference audio: {audio_path.name}")
            
            # Use the model's prepare_conditionals method
            self.model.prepare_conditionals(str(audio_path))
            
            logger.info(f"Voice conditioning prepared from {audio_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to prepare conditionals from {reference_audio_path}: {e}")
            raise
    
    def generate_sentences_stream(
        self,
        sentences_data: List[dict],
        emotion: str,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        exaggeration: Optional[float] = None
    ) -> Generator[Tuple[int, str, torch.Tensor, float, float, str], None, None]:
        """
        Generate audio for a list of sentences sequentially, maintaining voice consistency.
        
        This method is designed for streaming TTS where each sentence is generated
        independently but with shared voice conditioning for consistency.
        
        Args:
            sentences_data: List of sentence dictionaries with 'index', 'text', etc.
            emotion: Emotion to use for all sentences
            temperature: Sampling temperature
            cfg_weight: CFG weight
            exaggeration: Optional exaggeration override
            
        Yields:
            Tuple of (sentence_index, sentence_text, audio_tensor, duration, generation_time, voice_sample_used)
        """
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        if not sentences_data:
            return
        
        logger.info(f"Starting sentence stream generation for {len(sentences_data)} sentences with emotion '{emotion}'")
        
        # Check if voice conditioning is already set from reference audio
        voice_conditionals_preset = hasattr(self.model, 'conds') and self.model.conds is not None
        
        if voice_conditionals_preset:
            logger.info("Using pre-set voice conditioning from reference audio")
            # Keep existing conditionals but update emotion if needed
            base_voice_sample = "reference_audio"
            conditionals_cpu = None  # Will use existing conditionals
        else:
            # Get random conditionals once for consistency across all sentences
            conditionals_cpu, base_voice_sample = self.conditionals_manager.get_random_conditionals(emotion)
        
        # Handle conditionals setup
        if conditionals_cpu is not None:
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
            
            # Set pre-computed conditionals (shared across all sentences)
            self.model.conds = conditionals_gpu
            conditionals_to_cleanup = conditionals_gpu
        else:
            # Using pre-set conditionals from reference audio
            conditionals_to_cleanup = None
            if exaggeration is None:
                exaggeration = 0.5  # Default exaggeration
        
        try:
            
            # Generate each sentence
            for sentence_data in sentences_data:
                sentence_index = sentence_data['index']
                sentence_text = sentence_data['text']
                
                logger.debug(f"Generating sentence {sentence_index}: '{sentence_text[:50]}{'...' if len(sentence_text) > 50 else ''}'")
                
                # Generate audio for this sentence
                start_time = time.time()
                audio_tensor = self.model.generate(
                    text=sentence_text,
                    temperature=temperature,
                    cfg_weight=cfg_weight
                )
                generation_time = time.time() - start_time
                
                # Trim silence for better concatenation quality
                audio_tensor = self.trim_silence(audio_tensor)
                
                # Calculate duration (after trimming)
                duration = len(audio_tensor.squeeze()) / self.sample_rate
                
                # Update metrics (but don't double-count since this is part of a streaming request)
                rtf = generation_time / duration if duration > 0 else 0
                
                logger.debug(f"Sentence {sentence_index} completed: {duration:.2f}s audio, {generation_time:.2f}s gen, RTF: {rtf:.3f}")
                
                yield sentence_index, sentence_text, audio_tensor, duration, generation_time, base_voice_sample
                
        finally:
            # Clean up GPU memory if configured (only if we created new conditionals)
            if not self.config.model.keep_warm and conditionals_to_cleanup is not None:
                del conditionals_to_cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
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