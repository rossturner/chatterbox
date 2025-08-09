import threading
import time
from typing import Optional, Tuple


class GenerationLock:
    """Thread lock manager for single-request processing"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._processing = False
        self._start_time: Optional[float] = None
        self._current_request_info: Optional[dict] = None
        
    def acquire_for_generation(self, timeout: float = 1.0, request_info: Optional[dict] = None) -> Tuple[bool, float]:
        """
        Try to acquire lock for generation.
        
        Args:
            timeout: Maximum time to wait for lock (seconds)
            request_info: Optional info about the request
            
        Returns:
            Tuple of (success, wait_time_seconds)
        """
        start = time.time()
        acquired = self._lock.acquire(timeout=timeout)
        wait_time = time.time() - start
        
        if acquired:
            self._processing = True
            self._start_time = time.time()
            self._current_request_info = request_info
            
        return acquired, wait_time
    
    def release(self):
        """Release the generation lock"""
        self._processing = False
        self._start_time = None
        self._current_request_info = None
        self._lock.release()
    
    @property
    def is_busy(self) -> bool:
        """Check if currently processing a request"""
        return self._processing
    
    @property
    def processing_time(self) -> Optional[float]:
        """Get how long the current request has been processing"""
        if self._processing and self._start_time:
            return time.time() - self._start_time
        return None
    
    @property
    def current_request_info(self) -> Optional[dict]:
        """Get info about the current request being processed"""
        return self._current_request_info
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        if self._processing:
            self.release()