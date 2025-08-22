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
    
    Uses adaptive margin calculation and phonetic awareness to prevent
    cutting off words while still removing unwanted babble.
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
        
        # Phonetic word ending patterns that need extra margin
        self.plosive_endings = {'p', 't', 'k', 'b', 'd', 'g', 'ed', 'ing', 's'}
        self.fricative_endings = {'f', 'v', 's', 'z', 'sh', 'th', 'ch'}
        self.common_suffixes = {'ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness'}
        
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
        """Normalize text for token matching with suffix preservation"""
        # Remove punctuation except apostrophes, lowercase, split on whitespace
        normalized = re.sub(r"[^a-zA-Z0-9'\s]+", " ", text).lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized.split()
    
    def _get_word_ending_margin(self, word: str) -> float:
        """Calculate minimal adaptive margin based on word ending characteristics"""
        if not word:
            return 0.02  # Default 20ms - much smaller
        
        word_lower = word.lower().strip()
        
        # Check for common suffixes that often get cut off
        for suffix in self.common_suffixes:
            if word_lower.endswith(suffix):
                return 0.04  # 40ms for suffix words (reduced from 100ms)
        
        # Check last 2 characters for phonetic patterns
        ending = word_lower[-2:] if len(word_lower) >= 2 else word_lower
        
        # Plosive endings need more margin (sudden cutoff)
        if any(ending.endswith(p) for p in self.plosive_endings):
            return 0.03  # 30ms for plosives (reduced from 80ms)
        
        # Fricative endings need moderate margin (gradual decay)
        if any(ending.endswith(f) for f in self.fricative_endings):
            return 0.025  # 25ms for fricatives (reduced from 60ms)
        
        # Vowel endings need less margin (natural decay)
        if ending[-1] in 'aeiou':
            return 0.015  # 15ms for vowels (reduced from 40ms)
        
        return 0.02  # Default 20ms (reduced from 50ms)
    
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
            # Handle common variations (plurals, suffixes, etc.)
            elif self._tokens_are_similar(tokens1[i], tokens2[i]):
                matches += 0.8  # Partial credit for similar words
        
        return matches / max(len(tokens1), len(tokens2))
    
    def _tokens_are_similar(self, token1: str, token2: str) -> bool:
        """Check if two tokens are similar considering common variations"""
        if not token1 or not token2:
            return False
        
        # Check for plural variations
        if token1.rstrip('s') == token2.rstrip('s'):
            return True
        
        # Check for common suffix variations (ed, ing, er, ly)
        suffixes = ['ed', 'ing', 'er', 'ly', 's', 'es']
        for suffix in suffixes:
            if (token1.endswith(suffix) and token1[:-len(suffix)] == token2) or \
               (token2.endswith(suffix) and token2[:-len(suffix)] == token1):
                return True
        
        # Check for similar length and character overlap
        if abs(len(token1) - len(token2)) <= 1:
            # Allow one character difference for transcription errors
            if len(token1) == len(token2):
                diff_count = sum(c1 != c2 for c1, c2 in zip(token1, token2))
                return diff_count <= 1
            else:
                # One insertion/deletion
                shorter, longer = (token1, token2) if len(token1) < len(token2) else (token2, token1)
                return shorter in longer
        
        return False
    
    def _ends_with_intended_text(self, recognized_transcript: str, intended_text: str) -> bool:
        """
        Check if the recognized transcript ends with the intended text (allowing for minor variations).
        
        Args:
            recognized_transcript: The full transcript from Whisper
            intended_text: The original intended text
            
        Returns:
            True if the recognized text ends with the intended text, False otherwise
        """
        if not recognized_transcript or not intended_text:
            return False
        
        # Normalize both texts for comparison
        rec_tokens = self._normalize_tokens(recognized_transcript)
        intended_tokens = self._normalize_tokens(intended_text)
        
        if not rec_tokens or not intended_tokens:
            return False
        
        # Check if the recognized text ends with the intended text
        if len(rec_tokens) < len(intended_tokens):
            return False
        
        # Extract the ending portion of recognized text that matches intended length
        rec_ending = rec_tokens[-len(intended_tokens):]
        
        # Calculate similarity between the ending and intended text
        similarity = self._calculate_similarity(rec_ending, intended_tokens)
        
        # Use a high threshold (0.9) to ensure we only skip trimming when very confident
        return similarity >= 0.9
    
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
    
    def _find_end_time(self, recognized_words: list[tuple], intended_text: str, adaptive_margin: bool = True) -> Optional[float]:
        """
        Find the timestamp where the intended text ends in the recognized words.
        Uses smart word boundary detection to prevent cutting off words.
        
        Args:
            recognized_words: List of (word, end_time) tuples from Whisper
            intended_text: The original intended text
            
        Returns:
            End timestamp in seconds with adaptive margin, or None if not found
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
            # Apply adaptive margin based on the last word characteristics if enabled
            if adaptive_margin and recognized_words:
                last_word = recognized_words[-1][0] if len(recognized_words) > 0 else ""
                word_margin = self._get_word_ending_margin(last_word)
                return end_time + word_margin
            return end_time
        
        # Strategy 2: Try progressive suffix matching with similarity check
        max_k = min(8, len(target_tokens))  # Try longer suffixes for better accuracy
        
        for k in range(max_k, 3, -1):  # Start with longer sequences, minimum 4 words
            suffix = target_tokens[-k:]
            
            # Scan from the BEGINNING for the first occurrence of this suffix
            for i in range(len(rec_tokens) - k + 1):
                candidate = rec_tokens[i:i+k]
                if candidate == suffix or self._calculate_similarity(candidate, suffix) >= 0.8:
                    # Get the corresponding recognized word for margin calculation
                    if adaptive_margin:
                        word_idx = min(i+k-1, len(recognized_words)-1)
                        last_word = recognized_words[word_idx][0] if word_idx < len(recognized_words) else ""
                        word_margin = self._get_word_ending_margin(last_word)
                        return rec_ends[i+k-1] + word_margin
                    else:
                        return rec_ends[i+k-1]
        
        # Strategy 3: Fallback to shorter suffix matching with adaptive margin
        for k in range(min(3, len(target_tokens)), 0, -1):
            suffix = target_tokens[-k:]
            
            # Scan from beginning for first occurrence
            for i in range(len(rec_tokens) - k + 1):
                candidate = rec_tokens[i:i+k]
                if candidate == suffix or self._calculate_similarity(candidate, suffix) >= 0.7:
                    # Apply adaptive margin for the matched word
                    if adaptive_margin:
                        word_idx = min(i+k-1, len(recognized_words)-1)
                        last_word = recognized_words[word_idx][0] if word_idx < len(recognized_words) else ""
                        word_margin = self._get_word_ending_margin(last_word)
                        return rec_ends[i+k-1] + word_margin
                    else:
                        return rec_ends[i+k-1]
        
        # Final fallback: use the last recognized word with adaptive margin
        if rec_ends and recognized_words:
            if adaptive_margin:
                last_word = recognized_words[-1][0] if recognized_words else ""
                word_margin = self._get_word_ending_margin(last_word)
                return rec_ends[-1] + word_margin
            else:
                return rec_ends[-1]
        
        return None
    
    def _find_extended_end_time(self, recognized_words: list[tuple], initial_end_time: float, 
                               intended_tokens: list[str]) -> Optional[float]:
        """
        Find an extended end time when the initial trim was too aggressive.
        Only extends by a small amount to capture the cut-off word.
        """
        if not recognized_words or not initial_end_time:
            return None
        
        # Look for intended content in a small window after the initial cut
        intended_set = set(intended_tokens)
        last_match_time = initial_end_time
        
        for word, word_end_time in recognized_words:
            # Only look in a small window after initial trim (max 0.2s extension)
            if initial_end_time < word_end_time <= initial_end_time + 0.2:
                normalized_word_tokens = self._normalize_tokens(word.strip())
                # Only extend for exact matches
                if any(token in intended_set for token in normalized_word_tokens):
                    # Found intended content, update with minimal margin
                    adaptive_margin = self._get_word_ending_margin(word.strip())
                    last_match_time = word_end_time + adaptive_margin
                    break  # Take the first match, don't keep extending
        
        return last_match_time if last_match_time > initial_end_time else None
    
    def _verify_post_trim_content(self, recognized_words: list[tuple], end_time: float, 
                                 intended_tokens: list[str]) -> bool:
        """
        Verify that content after the trim point contains only babble/garbage.
        Returns True if safe to trim, False if clear legitimate content detected.
        Now much more selective - only prevents trimming if very confident.
        """
        if not recognized_words or not end_time:
            return True  # Safe to trim if no words or no end time
        
        # Find words immediately after the proposed trim point (smaller window)
        immediate_post_words = []
        for word, word_end_time in recognized_words:
            # Only look at words very close to the trim point (0.05s = 50ms window)
            if end_time < word_end_time <= end_time + 0.05:
                immediate_post_words.append(word.strip().lower())
        
        if not immediate_post_words:
            return True  # No words immediately after trim point, safe to trim
        
        # Only check for EXACT matches of intended words (be very selective)
        intended_set = set(intended_tokens)
        exact_matches = 0
        
        for word in immediate_post_words:
            normalized_word_tokens = self._normalize_tokens(word)
            # Only count exact token matches, not partial similarities
            for token in normalized_word_tokens:
                if token in intended_set:
                    exact_matches += 1
                    logger.debug(f"Found exact intended token '{token}' immediately after trim point")
        
        # Only extend if we find multiple exact matches (high confidence)
        if exact_matches >= 2:
            logger.debug(f"Found {exact_matches} exact intended tokens after trim, extending trim")
            return False
        
        return True  # Safe to trim - most babble will be trimmed
    
    def trim_audio(
        self, 
        audio_tensor: torch.Tensor, 
        sample_rate: int, 
        intended_text: str,
        margin_ms: float = 25.0,  # Minimal default margin - adaptive will add what's needed
        fade_ms: float = 15.0,
        adaptive_margin: bool = True,  # Enable smart margin calculation
        min_margin_ms: float = 15.0,  # Minimum margin to prevent cutting off words
        max_margin_ms: float = 60.0  # Maximum margin to avoid keeping too much babble
    ) -> Tuple[torch.Tensor, float, dict]:
        """
        Trim audio to remove content beyond the intended text.
        
        Args:
            audio_tensor: Audio tensor to trim (any shape)
            sample_rate: Audio sample rate
            intended_text: The original intended text
            margin_ms: Base margin to keep after last intended word (milliseconds)
            fade_ms: Fade-out duration (milliseconds)
            adaptive_margin: Whether to use smart margin calculation based on word characteristics
            min_margin_ms: Minimum margin to prevent cutting off words (milliseconds)
            max_margin_ms: Maximum margin to avoid keeping too much babble (milliseconds)
            
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
            
            # Check if the recognized transcript ends with the intended text
            # If it does, skip trimming and return the original audio
            if self._ends_with_intended_text(recognized_transcript, intended_text):
                logger.debug(f"Audio ends correctly with intended text, skipping trim. Recognized: '{recognized_transcript}', Intended: '{intended_text}'")
                
                # Update statistics for this non-trim
                self.total_trims += 1
                self.total_trim_time += time.time() - transcribe_start
                self.total_transcribe_time += transcribe_time
                
                # Return original audio with metrics
                metrics = {
                    'original_duration': original_duration,
                    'trimmed_duration': original_duration,
                    'amount_trimmed': 0.0,
                    'trim_time': time.time() - transcribe_start,
                    'transcribe_time': transcribe_time,
                    'recognized_words': len(recognized_words),
                    'end_time_found': True,
                    'recognized_transcript': recognized_transcript,
                    'intended_text': intended_text,
                    'adaptive_margin_used': adaptive_margin,
                    'initial_end_time': None,
                    'final_end_time': None,
                    'margin_applied_ms': 0,
                    'cut_at_seconds': 0,
                    'fade_start_seconds': 0,
                    'trimming_skipped': True,
                    'skip_reason': 'ends_with_intended_text'
                }
                
                # Restore original tensor format if needed
                if len(original_shape) > 1:
                    audio_1d = audio_1d.unsqueeze(0)
                
                return audio_1d if len(original_shape) == 1 else audio_1d.unsqueeze(0), time.time() - transcribe_start, metrics
            
            # Find the end time for trimming with two-pass verification
            initial_end_time = self._find_end_time(recognized_words, intended_text, adaptive_margin)
            
            # Two-pass verification: check if content after trim point is legitimate
            if initial_end_time is not None:
                target_tokens = self._normalize_tokens(intended_text)
                if not self._verify_post_trim_content(recognized_words, initial_end_time, target_tokens):
                    # Extend trim point if legitimate content detected after initial cut
                    extended_end_time = self._find_extended_end_time(recognized_words, initial_end_time, target_tokens)
                    end_time = extended_end_time if extended_end_time else initial_end_time
                    logger.debug(f"Extended trim time from {initial_end_time:.3f}s to {end_time:.3f}s")
                else:
                    end_time = initial_end_time
            else:
                end_time = None
            
            if end_time is not None:
                # Whisper returns timestamps in seconds, use directly
                # No need to scale by sample rate - time is time regardless of sample rate
                end_time_original = end_time
                
                # Apply additional margin and calculate trim point
                base_margin_seconds = margin_ms / 1000.0
                min_margin_seconds = min_margin_ms / 1000.0
                max_margin_seconds = max_margin_ms / 1000.0
                fade_seconds = fade_ms / 1000.0
                
                # If adaptive margin is enabled, word-specific margin is already applied in end_time
                # Apply minimal additional base margin to avoid double-counting
                if adaptive_margin:
                    # Very small additional margin since word-specific margin already applied
                    additional_margin = base_margin_seconds * 0.1  # Just 10% of base margin
                else:
                    # Use base margin when adaptive is disabled (but still respect limits)
                    additional_margin = min(base_margin_seconds, max_margin_seconds)
                
                # Ensure we don't exceed reasonable limits
                additional_margin = max(0.0, min(additional_margin, max_margin_seconds - min_margin_seconds))
                
                cut_at = max(0.0, end_time_original + additional_margin)
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
            
            # Create comprehensive metrics dictionary with debugging info
            metrics = {
                'original_duration': original_duration,
                'trimmed_duration': trimmed_duration,
                'amount_trimmed': amount_trimmed,
                'trim_time': total_alignment_time,  # Total time for transcription + alignment + trimming
                'transcribe_time': transcribe_time,
                'recognized_words': len(recognized_words),
                'end_time_found': end_time is not None,
                'recognized_transcript': recognized_transcript,
                'intended_text': intended_text,
                'adaptive_margin_used': adaptive_margin,
                'initial_end_time': initial_end_time,
                'final_end_time': end_time,
                'margin_applied_ms': (cut_at - end_time_original) * 1000 if end_time is not None else 0,
                'cut_at_seconds': cut_at if end_time is not None else 0,
                'fade_start_seconds': fade_start if end_time is not None else 0,
                'trimming_skipped': False,
                'skip_reason': None
            }
            
            # Enhanced logging with more detailed information
            if amount_trimmed > 0:
                logger.debug(
                    f"Audio trimmed: {original_duration:.2f}s -> {trimmed_duration:.2f}s "
                    f"(removed {amount_trimmed:.2f}s) in {total_alignment_time:.3f}s"
                )
            else:
                logger.debug(
                    f"Audio processed: {original_duration:.2f}s (no trimming needed) in {total_alignment_time:.3f}s"
                )
            
            if end_time is not None:
                margin_applied = (cut_at - end_time_original) * 1000
                logger.debug(
                    f"Trim details: end_time={end_time:.3f}s, margin={margin_applied:.1f}ms, "
                    f"adaptive={adaptive_margin}, words_recognized={len(recognized_words)}"
                )
                
                # Log potential issues
                if amount_trimmed < 0.1 and amount_trimmed > 0:
                    logger.debug("Very little audio trimmed - may indicate babble detection issues")
                elif amount_trimmed > original_duration * 0.5:
                    logger.warning(f"Trimmed over 50% of audio ({amount_trimmed:.2f}s) - check alignment accuracy")
            else:
                logger.warning("No trim end time found - returning original audio")
            
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
                'error': str(e),
                'adaptive_margin_used': adaptive_margin,
                'initial_end_time': None,
                'final_end_time': None,
                'margin_applied_ms': 0,
                'cut_at_seconds': 0,
                'fade_start_seconds': 0,
                'trimming_skipped': False,
                'skip_reason': None
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
            'device': self.device,
            'beam_size': self.beam_size,
            'use_vad': self.use_vad,
            'cpu_threads': self.cpu_threads
        }
    
    def get_config(self) -> dict:
        """Get trimmer configuration for debugging"""
        return {
            'model_size': self.model_size,
            'device': self.device,
            'beam_size': self.beam_size,
            'cpu_threads': self.cpu_threads,
            'use_vad': self.use_vad,
            'plosive_endings': self.plosive_endings,
            'fricative_endings': self.fricative_endings,
            'common_suffixes': self.common_suffixes
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