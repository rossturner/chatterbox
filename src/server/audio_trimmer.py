import time
import logging
import re
import subprocess
import shlex
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple
import threading

import torch
import torchaudio
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class AudioTrimmer:
    """
    Audio trimming utility using Faster-Whisper to align generated audio
    with intended text and remove babble tails.
    """
    
    def __init__(self, model_size: str = "tiny.en", device: str = "cpu"):
        """
        Initialize the audio trimmer.
        
        Args:
            model_size: Whisper model size ("tiny.en", "small.en", etc.)
            device: Device to run on ("cpu" recommended for fastest inference)
        """
        self.model_size = model_size
        self.device = device
        self.model: Optional[WhisperModel] = None
        self._model_lock = threading.Lock()
        
        # Performance configuration
        self.beam_size = int(os.environ.get('WHISPER_BEAM_SIZE', '2'))  # Optimized for speed/quality
        self.cpu_threads = int(os.environ.get('WHISPER_CPU_THREADS', '0'))  # Auto-detect
        self.use_vad = os.environ.get('WHISPER_USE_VAD', 'false').lower() == 'true'
        
        # Pre-generated warmup audio for performance
        self._warmup_audio: Optional[np.ndarray] = None
        
        # Performance tracking
        self.total_trims = 0
        self.total_trim_time = 0.0
        self.total_transcribe_time = 0.0
        
    def _ensure_model_loaded(self) -> WhisperModel:
        """Ensure the Whisper model is loaded (thread-safe)"""
        if self.model is None:
            with self._model_lock:
                if self.model is None:
                    logger.info(f"Loading Faster-Whisper model: {self.model_size} on {self.device}")
                    start_time = time.time()
                    
                    try:
                        # CPU optimization parameters
                        model_kwargs = {
                            "device": self.device,
                            "compute_type": "int8",  # Fastest inference
                        }
                        
                        # Add CPU thread configuration if specified
                        if self.cpu_threads > 0:
                            model_kwargs["cpu_threads"] = self.cpu_threads
                        
                        self.model = WhisperModel(self.model_size, **model_kwargs)
                        load_time = time.time() - start_time
                        logger.info(f"Faster-Whisper model loaded in {load_time:.2f}s (beam_size={self.beam_size}, cpu_threads={self.cpu_threads}, vad={self.use_vad})")
                        
                    except Exception as e:
                        logger.error(f"Failed to load Faster-Whisper model: {e}")
                        raise
        
        return self.model
    
    def _normalize_tokens(self, text: str) -> list[str]:
        """Normalize text for token matching"""
        # Remove punctuation except apostrophes, lowercase, split on whitespace
        normalized = re.sub(r"[^a-zA-Z0-9'\s]+", " ", text).lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized.split()
    
    def _calculate_similarity(self, tokens1: list[str], tokens2: list[str]) -> float:
        """Calculate similarity score between two token sequences (0.0 to 1.0)"""
        if not tokens1 or not tokens2:
            return 0.0
        
        # Use a simple token-based similarity
        matches = 0
        min_len = min(len(tokens1), len(tokens2))
        
        for i in range(min_len):
            # Exact match
            if tokens1[i] == tokens2[i]:
                matches += 1
            # Handle common variations (plurals, etc.)
            elif (tokens1[i].rstrip('s') == tokens2[i].rstrip('s') or 
                  abs(len(tokens1[i]) - len(tokens2[i])) <= 1):
                matches += 0.8  # Partial credit for similar words
        
        return matches / max(len(tokens1), len(tokens2))
    
    def _find_best_sequence_match(self, rec_tokens: list[str], rec_ends: list[float], 
                                target_tokens: list[str]) -> Optional[float]:
        """
        Find the best full sequence match of the intended text in recognized tokens.
        Returns the end time of the FIRST good match to avoid repeated text in babble tails.
        """
        if len(target_tokens) < 4:  # Too short for reliable full matching
            return None
        
        target_len = len(target_tokens)
        best_end_time = None
        best_similarity = 0.7  # Minimum similarity threshold
        
        # Slide through the recognized tokens, looking for good matches
        for start_idx in range(len(rec_tokens) - target_len + 1):
            candidate_tokens = rec_tokens[start_idx:start_idx + target_len]
            
            # Calculate similarity
            similarity = self._calculate_similarity(candidate_tokens, target_tokens)
            
            if similarity > best_similarity:
                # Found a good match - return immediately to get FIRST occurrence
                end_idx = start_idx + target_len - 1
                if end_idx < len(rec_ends):
                    return rec_ends[end_idx]
        
        # Try with some flexibility - allow the sequence to be slightly longer or shorter
        for target_flex in [target_len + 1, target_len + 2, target_len - 1]:
            if target_flex <= 0 or target_flex > len(rec_tokens):
                continue
                
            for start_idx in range(len(rec_tokens) - target_flex + 1):
                candidate_tokens = rec_tokens[start_idx:start_idx + target_flex]
                
                # Compare against the target with some flexibility
                similarity = self._calculate_similarity(candidate_tokens, target_tokens)
                
                if similarity > best_similarity:
                    # Found a good flexible match
                    end_idx = start_idx + min(target_len, target_flex) - 1
                    if end_idx < len(rec_ends):
                        return rec_ends[end_idx]
        
        return None
    
    def _find_end_time(self, recognized_words: list[tuple], intended_text: str) -> Optional[float]:
        """
        Find the timestamp where the intended text ends in the recognized words.
        Uses full sequence matching to find the FIRST occurrence and avoid repeated text in babble tails.
        
        Args:
            recognized_words: List of (word, end_time) tuples from Whisper
            intended_text: The original intended text
            
        Returns:
            End timestamp in seconds, or None if not found
        """
        if not recognized_words:
            return None
        
        # Extract tokens and timestamps
        rec_tokens = [self._normalize_tokens(word)[0] if self._normalize_tokens(word) else "" 
                     for word, _ in recognized_words]
        rec_ends = [end_time for _, end_time in recognized_words]
        
        # Normalize intended text
        target_tokens = self._normalize_tokens(intended_text)
        if not target_tokens:
            return rec_ends[-1] if rec_ends else None
        
        # Strategy 1: Try to find the best full sequence match (first occurrence)
        end_time = self._find_best_sequence_match(rec_tokens, rec_ends, target_tokens)
        if end_time is not None:
            return end_time
        
        # Strategy 2: Try progressive suffix matching from the beginning (forward search)
        max_k = min(8, len(target_tokens))  # Try longer suffixes for better accuracy
        
        for k in range(max_k, 3, -1):  # Start with longer sequences, minimum 4 words
            suffix = target_tokens[-k:]
            
            # Scan from the BEGINNING for the first occurrence of this suffix
            for i in range(len(rec_tokens) - k + 1):
                if rec_tokens[i:i+k] == suffix:
                    return rec_ends[i+k-1]
        
        # Strategy 3: Fallback to shorter suffix matching
        for k in range(min(3, len(target_tokens)), 0, -1):
            suffix = target_tokens[-k:]
            
            # Scan from beginning for first occurrence
            for i in range(len(rec_tokens) - k + 1):
                if rec_tokens[i:i+k] == suffix:
                    return rec_ends[i+k-1]
        
        # Final fallback: use the last recognized word (only if nothing else works)
        return rec_ends[-1] if rec_ends else None
    
    def trim_audio(
        self, 
        audio_tensor: torch.Tensor, 
        sample_rate: int, 
        intended_text: str,
        margin_ms: float = 0.0,
        fade_ms: float = 15.0
    ) -> Tuple[torch.Tensor, float, dict]:
        """
        Trim audio to remove content beyond the intended text.
        
        Args:
            audio_tensor: Audio tensor to trim (any shape)
            sample_rate: Audio sample rate
            intended_text: The original intended text
            margin_ms: Additional margin to keep after last intended word (milliseconds)
            fade_ms: Fade-out duration (milliseconds)
            
        Returns:
            Tuple of (trimmed_audio, trim_time, metrics_dict)
        """
        start_time = time.time()
        
        # Store original format
        original_shape = audio_tensor.shape
        original_duration = len(audio_tensor.squeeze()) / sample_rate
        
        # Ensure we have a 1D audio tensor for processing
        if audio_tensor.dim() > 1:
            audio_1d = audio_tensor.squeeze()
        else:
            audio_1d = audio_tensor
        
        try:
            # Load the Whisper model
            model = self._ensure_model_loaded()
            
            # Prepare audio for Whisper (avoid file I/O by using numpy arrays directly)
            transcribe_start = time.time()
            
            # Resample to 16kHz if needed (Whisper's native sample rate)
            if sample_rate != 16000:
                resampled = torchaudio.functional.resample(
                    audio_1d.unsqueeze(0), sample_rate, 16000
                ).squeeze(0)
                working_sr = 16000
            else:
                resampled = audio_1d
                working_sr = sample_rate
            
            # Convert to numpy array for direct processing (eliminates file I/O)
            audio_np = resampled.cpu().numpy().astype(np.float32)
            
            # Transcribe with word timestamps using optimized settings
            segments, _ = model.transcribe(
                audio_np,  # Direct numpy array input (major performance boost)
                word_timestamps=True,
                beam_size=self.beam_size,  # Optimized beam size (default: 2)
                temperature=0.0,
                vad_filter=self.use_vad,  # Configurable VAD
                condition_on_previous_text=False
            )
            transcribe_time = time.time() - transcribe_start
            
            # Extract recognized words with timestamps and full transcript
            recognized_words = []
            recognized_transcript_parts = []
            
            for segment in segments:
                if segment.text:
                    recognized_transcript_parts.append(segment.text.strip())
                
                if segment.words:
                    for word in segment.words:
                        if word.word and word.end:
                            recognized_words.append((word.word.strip(), float(word.end)))
            
            # Combine transcript parts
            recognized_transcript = " ".join(recognized_transcript_parts).strip()
            
            # Find the end time for trimming
            end_time = self._find_end_time(recognized_words, intended_text)
            
            if end_time is not None:
                # Whisper returns timestamps in seconds, use directly
                # No need to scale by sample rate - time is time regardless of sample rate
                end_time_original = end_time
                
                # Apply margin and calculate trim point
                margin_seconds = margin_ms / 1000.0
                fade_seconds = fade_ms / 1000.0
                
                cut_at = max(0.0, end_time_original + margin_seconds)
                fade_start = max(0.0, cut_at - fade_seconds)
                
                # Convert to sample indices
                cut_sample = min(len(audio_1d), int(cut_at * sample_rate))
                fade_start_sample = max(0, int(fade_start * sample_rate))
                
                # Trim the audio
                if cut_sample < len(audio_1d):
                    trimmed_audio = audio_1d[:cut_sample]
                    
                    # Apply fade-out
                    if fade_start_sample < cut_sample:
                        fade_length = cut_sample - fade_start_sample
                        if fade_length > 0:
                            fade_curve = torch.linspace(1.0, 0.0, fade_length)
                            trimmed_audio[fade_start_sample:cut_sample] *= fade_curve
                    
                    amount_trimmed = (len(audio_1d) - cut_sample) / sample_rate
                else:
                    trimmed_audio = audio_1d
                    amount_trimmed = 0.0
                
            else:
                # No end time found, return original
                trimmed_audio = audio_1d
                amount_trimmed = 0.0
            
            # Restore original tensor format if needed
            if len(original_shape) > 1:
                trimmed_audio = trimmed_audio.unsqueeze(0)
            
            trim_time = time.time() - start_time
            trimmed_duration = len(trimmed_audio.squeeze()) / sample_rate
            
            # Calculate total alignment time (transcription + alignment + trimming)
            # This is what the user expects to see as "alignment & trim time"
            total_alignment_time = time.time() - transcribe_start
            
            # Update statistics
            self.total_trims += 1
            self.total_trim_time += total_alignment_time
            self.total_transcribe_time += transcribe_time
            
            # Create metrics dictionary
            metrics = {
                'original_duration': original_duration,
                'trimmed_duration': trimmed_duration,
                'amount_trimmed': amount_trimmed,
                'trim_time': total_alignment_time,  # Total time for transcription + alignment + trimming
                'transcribe_time': transcribe_time,
                'recognized_words': len(recognized_words),
                'end_time_found': end_time is not None,
                'recognized_transcript': recognized_transcript,
                'intended_text': intended_text
            }
            
            logger.debug(
                f"Audio trimmed: {original_duration:.2f}s -> {trimmed_duration:.2f}s "
                f"(removed {amount_trimmed:.2f}s) in {total_alignment_time:.3f}s"
            )
            
            return trimmed_audio, total_alignment_time, metrics
            
        except Exception as e:
            logger.error(f"Audio trimming failed: {e}")
            # Return original audio on error
            total_time_elapsed = time.time() - start_time
            metrics = {
                'original_duration': original_duration,
                'trimmed_duration': original_duration,
                'amount_trimmed': 0.0,
                'trim_time': total_time_elapsed,
                'transcribe_time': 0.0,
                'recognized_words': 0,
                'end_time_found': False,
                'recognized_transcript': '',
                'intended_text': intended_text,
                'error': str(e)
            }
            return audio_tensor, total_time_elapsed, metrics
    
    def get_stats(self) -> dict:
        """Get trimmer performance statistics"""
        return {
            'total_trims': self.total_trims,
            'total_trim_time': self.total_trim_time,
            'total_transcribe_time': self.total_transcribe_time,
            'avg_trim_time': self.total_trim_time / max(1, self.total_trims),
            'avg_transcribe_time': self.total_transcribe_time / max(1, self.total_trims),
            'model_size': self.model_size,
            'device': self.device
        }
    
    def _get_warmup_audio(self) -> np.ndarray:
        """Get cached warmup audio, generating it once if needed."""
        if self._warmup_audio is None:
            # Create 1 second of 440Hz sine wave at 16kHz (standard warmup audio)
            sample_rate = 16000
            duration = 1.0
            frequency = 440.0
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            self._warmup_audio = (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
            
        return self._warmup_audio
    
    def warmup(self) -> float:
        """
        Warm up the Whisper model to eliminate first-request latency.
        
        Returns:
            Warmup time in seconds
        """
        start_time = time.time()
        logger.info("Warming up Faster-Whisper model...")
        
        try:
            # Force model loading
            model = self._ensure_model_loaded()
            
            # Use cached warmup audio (eliminates file I/O during warmup)
            synthetic_audio = self._get_warmup_audio()
            
            # Perform warmup transcription using the same optimized settings as production
            segments, _ = model.transcribe(
                synthetic_audio,  # Direct numpy array (no file I/O)
                word_timestamps=True,
                beam_size=self.beam_size,  # Same as production
                temperature=0.0,
                vad_filter=self.use_vad,  # Same as production
                condition_on_previous_text=False
            )
            
            # Consume the generator to complete the transcription
            list(segments)
            
            warmup_time = time.time() - start_time
            logger.info(f"Faster-Whisper model warmed up in {warmup_time:.2f}s")
            return warmup_time
            
        except Exception as e:
            warmup_time = time.time() - start_time
            logger.warning(f"Whisper warmup failed after {warmup_time:.2f}s: {e}")
            return warmup_time
    
    def cleanup(self):
        """Clean up resources"""
        if self.model is not None:
            try:
                del self.model
                self.model = None
                logger.info("Audio trimmer cleaned up")
            except Exception as e:
                logger.warning(f"Error during audio trimmer cleanup: {e}")