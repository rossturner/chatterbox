"""
Single-slot conditional cache for zero-shot voice cloning.

Caches the most recently used reference audio file path to avoid
redundant decoding and file I/O when the same reference audio is used
across multiple requests (regardless of target language).
"""

import hashlib
import threading
from typing import Optional, Tuple
from dataclasses import dataclass
import torch


@dataclass
class CacheEntry:
    """Single cache entry storing audio file path for a reference audio"""
    audio_hash: str
    audio_file_path: str  # Path to temporary audio file


class ConditionalCache:
    """
    Thread-safe single-slot cache for reference audio file paths.

    The cache stores the temporary file path for the most recently used
    reference audio, keyed by a hash of the base64-encoded audio string.
    This avoids redundant base64 decoding and file I/O on cache hits.

    Language is NOT part of the cache key since reference audio encoding
    is language-independent - the same voice can speak any language.
    """

    def __init__(self):
        self._cache_entry: Optional[CacheEntry] = None
        self._lock = threading.Lock()

    def _hash_audio(self, audio_base64: str) -> str:
        """
        Generate SHA256 hash of base64 audio string.

        Args:
            audio_base64: Base64-encoded audio string

        Returns:
            Hex digest of SHA256 hash
        """
        return hashlib.sha256(audio_base64.encode('utf-8')).hexdigest()

    def get(self, audio_base64: str) -> Optional[str]:
        """
        Retrieve audio file path from cache if available.

        Cache hit requires audio hash to match cached audio hash.
        Language is not part of the cache key since the reference audio
        encoding is language-independent.

        Args:
            audio_base64: Base64-encoded reference audio string

        Returns:
            Path to cached audio file if cache hit, None otherwise
        """
        with self._lock:
            if self._cache_entry is None:
                return None

            audio_hash = self._hash_audio(audio_base64)

            if self._cache_entry.audio_hash == audio_hash:
                return self._cache_entry.audio_file_path

            return None

    def set(self, audio_base64: str, audio_file_path: str) -> None:
        """
        Store audio file path in cache, overwriting previous entry.

        Args:
            audio_base64: Base64-encoded reference audio string
            audio_file_path: Path to temporary audio file
        """
        with self._lock:
            audio_hash = self._hash_audio(audio_base64)
            self._cache_entry = CacheEntry(
                audio_hash=audio_hash,
                audio_file_path=audio_file_path
            )

    def clear(self) -> None:
        """Clear the cache entry"""
        with self._lock:
            self._cache_entry = None

    def has_entry(self) -> bool:
        """Check if cache has an entry (regardless of whether it matches current request)"""
        with self._lock:
            return self._cache_entry is not None
