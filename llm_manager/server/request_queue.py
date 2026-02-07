"""Request queue with backpressure for API rate limiting.

Provides bounded queues to prevent system overload under high load.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from time import time
from typing import Any, TypeVar

from fastapi import HTTPException

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class QueuedRequest:
    """A request waiting in the queue."""
    
    id: str
    timestamp: float = field(default_factory=time)
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    
    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = time()


class RequestQueue:
    """Bounded request queue with backpressure.
    
    Prevents system overload by limiting concurrent and queued requests.
    When limits are exceeded, returns 503 Service Unavailable.
    
    Args:
        max_concurrent: Maximum number of requests being processed concurrently
        max_queued: Maximum number of requests waiting in queue
        queue_timeout: Maximum time to wait in queue before rejection (seconds)
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        max_queued: int = 100,
        queue_timeout: float = 30.0,
    ):
        self.max_concurrent = max_concurrent
        self.max_queued = max_queued
        self.queue_timeout = queue_timeout
        
        # Semaphore for concurrent request limiting
        self._concurrent_sem = asyncio.Semaphore(max_concurrent)
        
        # Queue for waiting requests
        self._queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(maxsize=max_queued)
        
        # Metrics
        self._total_processed = 0
        self._total_rejected = 0
        self._total_timeout = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self, request_id: str) -> None:
        """Acquire a slot for request processing.
        
        Args:
            request_id: Unique identifier for the request
            
        Raises:
            HTTPException: If queue is full or timeout occurs
        """
        # Fast path: try to acquire without queuing
        if self._concurrent_sem.locked():
            # Need to queue
            await self._enqueue_and_wait(request_id)
        else:
            await self._concurrent_sem.acquire()
    
    async def _enqueue_and_wait(self, request_id: str) -> None:
        """Enqueue request and wait for processing slot."""
        # Check if queue is full
        if self._queue.qsize() >= self.max_queued:
            async with self._lock:
                self._total_rejected += 1
            logger.warning(
                f"Request {request_id} rejected: queue full "
                f"({self._queue.qsize()}/{self.max_queued})"
            )
            raise HTTPException(
                status_code=503,
                detail=f"Server overloaded. Queue full ({self._queue.qsize()} waiting). "
                       f"Try again later."
            )
        
        # Create queued request
        queued = QueuedRequest(id=request_id)
        
        try:
            # Add to queue with timeout
            await asyncio.wait_for(
                self._queue.put(queued),
                timeout=5.0  # Should be quick, just adding to queue
            )
        except asyncio.TimeoutError:
            async with self._lock:
                self._total_rejected += 1
            raise HTTPException(status_code=503, detail="Failed to queue request")
        
        # Wait for our turn with overall timeout
        try:
            await asyncio.wait_for(queued.future, timeout=self.queue_timeout)
            # Now we have the semaphore
        except asyncio.TimeoutError:
            # Remove from queue if still there
            # Note: In practice, we'd need a way to cancel
            async with self._lock:
                self._total_timeout += 1
            logger.warning(f"Request {request_id} timed out in queue")
            raise HTTPException(
                status_code=503,
                detail=f"Request timed out waiting in queue after {self.queue_timeout}s"
            )
    
    def release(self) -> None:
        """Release a processing slot."""
        self._concurrent_sem.release()
        
        # Process next queued request if any
        if not self._queue.empty():
            asyncio.create_task(self._process_queue())
    
    async def _process_queue(self) -> None:
        """Process the next request in queue."""
        try:
            queued = self._queue.get_nowait()
            # Acquire semaphore for the queued request
            await self._concurrent_sem.acquire()
            # Signal that they can proceed
            queued.future.set_result(None)
        except asyncio.QueueEmpty:
            pass
        except Exception as e:
            logger.error(f"Error processing queue: {e}")
    
    async def __aenter__(self) -> "RequestQueue":
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
    
    @property
    def stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            "queue_size": self._queue.qsize(),
            "max_queued": self.max_queued,
            "max_concurrent": self.max_concurrent,
            "available_slots": self.max_concurrent - self._concurrent_sem._value,
            "total_processed": self._total_processed,
            "total_rejected": self._total_rejected,
            "total_timeout": self._total_timeout,
        }


# Global request queue instance
_request_queue: RequestQueue | None = None


def get_request_queue(
    max_concurrent: int = 10,
    max_queued: int = 100,
    queue_timeout: float = 30.0,
) -> RequestQueue:
    """Get or create the global request queue."""
    global _request_queue
    if _request_queue is None:
        _request_queue = RequestQueue(
            max_concurrent=max_concurrent,
            max_queued=max_queued,
            queue_timeout=queue_timeout,
        )
    return _request_queue


def reset_request_queue() -> None:
    """Reset the global request queue (for testing)."""
    global _request_queue
    _request_queue = None
