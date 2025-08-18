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
from .optimized_model_loader import OptimizedModelLoader
from .audio_trimmer import AudioTrimmer

logger = logging.getLogger(__name__)


class TTSManager:
    """Manager for TTS model and generation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[ChatterboxTTS] = None
        self.conditionals_manager = ConditionalsManager(Path(config.caching.conditionals_dir))
        self.device = config.model.device
        self.sample_rate = S3GEN_SR
        
        # Initialize audio trimmer if enabled
        self.audio_trimmer: Optional[AudioTrimmer] = None
        self.enable_audio_trimming = getattr(config.model, 'enable_audio_trimming', True)
        
        if self.enable_audio_trimming:
            self.audio_trimmer = AudioTrimmer(model_size="tiny.en", device="cpu")
            logger.info("Audio trimmer initialized (Faster-Whisper tiny.en on CPU)")
        else:
            logger.info("Audio trimming disabled in configuration")
        
        # Metrics tracking
        self.loaded_at = None
        self.total_requests = 0
        self.total_generation_time = 0.0
        self.total_rtf = 0.0
        
        # Optimization info
        self.optimization_info = {}
        
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
        
        # Warm up audio trimmer if enabled
        if self.enable_audio_trimming and self.audio_trimmer:
            try:
                warmup_time = self.audio_trimmer.warmup()
                logger.info(f"Audio trimmer warmed up successfully in {warmup_time:.2f}s")
            except Exception as e:
                logger.warning(f"Audio trimmer warmup failed (non-fatal): {e}")
        
        self.loaded_at = datetime.utcnow()
        logger.info("TTS Manager initialized successfully")
    
    def _load_model(self) -> None:
        """Load and optimize the TTS model based on configuration"""
        model_type = self.config.model.type
        model_path = self.config.model.path
        
        # Check if optimizations are enabled (can be configured)
        use_optimizations = getattr(self.config.model, 'use_optimizations', True)
        
        if use_optimizations:
            logger.info(f"Loading {model_type} model with optimizations on {self.device}...")
            
            # Use the optimized model loader
            loader = OptimizedModelLoader(self.device)
            
            # Convert model_path to Path if needed
            if model_path:
                model_path = Path(model_path)
            
            try:
                # Load and optimize the model
                self.model, self.optimization_info = loader.load_and_optimize(
                    model_type=model_type,
                    model_path=model_path
                )
                
                # Perform warmup compilation if torch.compile was applied
                if self.optimization_info.get('torch_compile'):
                    logger.info("Performing warmup to trigger torch.compile compilation...")
                    
                    # Try to find an audio file for warmup
                    warmup_audio = None
                    import glob
                    search_patterns = [
                        "configs/voice_samples/**/*.wav",
                        "configs/voice_samples/*.wav",
                        "audio_data/*.wav",
                        "audio_data_v2/*.wav"
                    ]
                    
                    for pattern in search_patterns:
                        audio_files = glob.glob(pattern, recursive=True)
                        if audio_files:
                            warmup_audio = audio_files[0]
                            break
                    
                    if warmup_audio:
                        compilation_time = loader.warmup_model(
                            self.model,
                            warmup_text="This is a warmup run to trigger model compilation.",
                            warmup_audio_path=warmup_audio
                        )
                        self.optimization_info['compilation_time'] = compilation_time
                        logger.info(f"Model compilation complete! Ready for fast inference.")
                    else:
                        logger.warning("No audio files found for warmup. Model may not be fully optimized.")
                
                # Log optimization summary
                logger.info(loader.get_optimization_summary(self.optimization_info))
                
            except Exception as e:
                logger.error(f"Failed to load optimized model: {e}")
                raise
        else:
            # Fall back to original loading method without optimizations
            logger.info(f"Loading {model_type} model without optimizations on {self.device}...")
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
                
                self.optimization_info = {
                    'load_time': load_time,
                    'optimizations_applied': False
                }
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
        
        # Report memory usage
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.synchronize()
            memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            logger.info(f"GPU memory usage: {memory_mb:.1f} MB")
    
    def generate(
        self,
        text: str,
        emotion: str,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        exaggeration: Optional[float] = None
    ) -> Tuple[torch.Tensor, float, float, str, str, Optional[dict]]:
        """
        Generate speech for text with specified emotion.
        
        Args:
            text: Text to synthesize
            emotion: Emotion to use
            temperature: Sampling temperature
            cfg_weight: CFG weight
            exaggeration: Optional exaggeration override
            
        Returns:
            Tuple of (audio_tensor, duration, generation_time, emotion_used, voice_sample_used, trim_metrics)
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
            
            # Calculate original duration
            original_duration = len(audio_tensor.squeeze()) / self.sample_rate
            
            # Apply audio trimming if enabled
            trim_metrics = None
            if self.enable_audio_trimming and self.audio_trimmer:
                try:
                    audio_tensor, alignment_time, trim_metrics = self.audio_trimmer.trim_audio(
                        audio_tensor, self.sample_rate, text
                    )
                    logger.debug(f"Audio trimming completed in {alignment_time:.3f}s")
                except Exception as e:
                    logger.warning(f"Audio trimming failed, using original audio: {e}")
                    trim_metrics = {'error': str(e), 'alignment_time': 0.0}
            
            # Calculate final duration (after trimming)
            final_duration = len(audio_tensor.squeeze()) / self.sample_rate
            
            # Update metrics
            self.total_requests += 1
            self.total_generation_time += generation_time
            rtf = generation_time / final_duration if final_duration > 0 else 0
            self.total_rtf += rtf
            
            # Log generation results
            if trim_metrics and trim_metrics.get('amount_trimmed', 0) > 0:
                logger.info(f"Generated {final_duration:.2f}s audio (trimmed from {original_duration:.2f}s, "
                           f"removed {trim_metrics['amount_trimmed']:.2f}s) in {generation_time:.2f}s "
                           f"(RTF: {rtf:.3f}) using {voice_sample_used}")
            else:
                logger.info(f"Generated {final_duration:.2f}s audio in {generation_time:.2f}s "
                           f"(RTF: {rtf:.3f}) using {voice_sample_used}")
            
            return audio_tensor, final_duration, generation_time, emotion, voice_sample_used, trim_metrics
            
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
        
        # Add optimization info if available
        if self.optimization_info:
            metrics["optimizations"] = {
                "bfloat16": self.optimization_info.get('bfloat16', False),
                "torch_compile": self.optimization_info.get('torch_compile', False),
                "reduced_cache": self.optimization_info.get('reduced_cache', False),
                "compile_mode": self.optimization_info.get('compile_mode'),
                "compilation_time": self.optimization_info.get('compilation_time'),
                "load_time": self.optimization_info.get('load_time')
            }
        
        return metrics
    
    def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
            self.model = None
        
        if self.audio_trimmer:
            self.audio_trimmer.cleanup()
        
        self.conditionals_manager.clear_cache()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("TTS Manager cleaned up")