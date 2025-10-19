"""
Thread-safe inference executor for CUDA graph compiled models.

CUDA graphs require execution in the same thread context where compilation occurred.
This module provides a dedicated inference thread that processes all model calls,
ensuring thread-local storage compatibility.
"""

import threading
import queue
import logging
from typing import Any, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InferenceTask:
    """Task to be executed in the inference thread"""
    func: Callable
    args: tuple
    kwargs: dict
    result_queue: queue.Queue
    task_id: str


class InferenceThread:
    """
    Dedicated thread for model inference to ensure CUDA graph thread affinity.

    All model.generate() calls are queued and executed in a single dedicated thread,
    ensuring they run in the same thread context where torch.compile created CUDA graphs.
    """

    def __init__(self):
        self._task_queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._compilation_thread_id: Optional[int] = None

    def start(self):
        """Start the inference worker thread"""
        if self._running:
            logger.warning("Inference thread already running")
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="InferenceThread",
            daemon=True
        )
        self._worker_thread.start()
        self._compilation_thread_id = self._worker_thread.ident
        logger.info(f"Inference thread started (TID: {self._compilation_thread_id})")

    def stop(self, timeout: float = 5.0):
        """Stop the inference worker thread"""
        if not self._running:
            return

        self._running = False
        # Send sentinel to wake up worker
        self._task_queue.put(None)

        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)
            if self._worker_thread.is_alive():
                logger.warning("Inference thread did not stop gracefully")
            else:
                logger.info("Inference thread stopped")

    def execute(self, func: Callable, *args, task_id: str = "unknown", timeout: float = 300.0, **kwargs) -> Any:
        """
        Execute a function in the inference thread and wait for result.

        Args:
            func: Function to execute
            *args: Positional arguments
            task_id: Identifier for logging
            timeout: Maximum wait time for result (seconds)
            **kwargs: Keyword arguments

        Returns:
            Result from the function

        Raises:
            RuntimeError: If inference thread is not running
            TimeoutError: If execution exceeds timeout
            Exception: Any exception raised by the function
        """
        if not self._running:
            raise RuntimeError("Inference thread is not running")

        result_queue = queue.Queue()
        task = InferenceTask(
            func=func,
            args=args,
            kwargs=kwargs,
            result_queue=result_queue,
            task_id=task_id
        )

        # Queue the task
        self._task_queue.put(task)

        # Wait for result
        try:
            result = result_queue.get(timeout=timeout)

            # Check if result is an exception
            if isinstance(result, Exception):
                raise result

            return result

        except queue.Empty:
            raise TimeoutError(f"Inference task '{task_id}' timed out after {timeout}s")

    def _worker_loop(self):
        """Worker thread main loop"""
        logger.info("Inference worker thread started")

        while self._running:
            try:
                # Get next task (blocking)
                task = self._task_queue.get(timeout=1.0)

                # Check for sentinel (shutdown signal)
                if task is None:
                    break

                # Execute the task
                try:
                    result = task.func(*task.args, **task.kwargs)
                    task.result_queue.put(result)
                except Exception as e:
                    logger.error(f"Error executing inference task '{task.task_id}': {e}", exc_info=True)
                    task.result_queue.put(e)

            except queue.Empty:
                # Timeout is normal, just continue
                continue
            except Exception as e:
                logger.error(f"Unexpected error in inference worker loop: {e}", exc_info=True)

        logger.info("Inference worker thread stopped")

    @property
    def is_running(self) -> bool:
        """Check if the inference thread is running"""
        return self._running and self._worker_thread and self._worker_thread.is_alive()

    @property
    def thread_id(self) -> Optional[int]:
        """Get the thread ID of the inference worker"""
        return self._compilation_thread_id
