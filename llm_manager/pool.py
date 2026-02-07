"""
Worker process pool for concurrent model operations.

Provides a pool of WorkerProcess instances to handle multiple requests in parallel.
"""

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from queue import Queue
from typing import Any

from .workers import AsyncWorkerProcess, WorkerProcess

logger = logging.getLogger(__name__)

DEFAULT_POOL_SIZE = 8


class WorkerPool:
    """
    Pool of WorkerProcess instances.

    Manages worker lifecycle and provides safe acquisition/release of workers.
    """

    def __init__(self, size: int = 4, idle_timeout: int = 3600):
        """
        Initialize worker pool.

        Args:
            size: Maximum number of workers in the pool
            idle_timeout: Idle timeout for each worker in seconds
        """
        self.size = min(size, DEFAULT_POOL_SIZE)
        self.idle_timeout = idle_timeout
        self._workers: list[WorkerProcess] = []
        self._available_queue: Queue[WorkerProcess] = Queue()
        self._lock = threading.Lock()
        self._started = False

        logger.debug(f"Initialized WorkerPool with size={self.size}")

    def start(self) -> None:
        """Start all workers in the pool."""
        with self._lock:
            if self._started:
                return

            for _ in range(self.size):
                worker = WorkerProcess(idle_timeout=self.idle_timeout)
                worker.start()  # Start the subprocess
                self._workers.append(worker)
                self._available_queue.put(worker)

            self._started = True
            logger.info(f"Started WorkerPool with {self.size} workers")

    @contextmanager
    def acquire(self) -> Iterator[WorkerProcess]:
        """
        Acquire a worker from the pool.

        Yields:
            A WorkerProcess instance
        """
        if not self._started:
            self.start()

        worker = self._available_queue.get()
        try:
            yield worker
        finally:
            self._available_queue.put(worker)

    def shutdown(self) -> None:
        """Shutdown all workers in the pool."""
        with self._lock:
            if not self._started:
                return

            for worker in self._workers:
                worker.stop()

            self._workers.clear()
            self._available_queue = Queue()
            self._started = False
            logger.info("Shutdown WorkerPool")

    def __enter__(self) -> "WorkerPool":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.shutdown()


class AsyncWorkerPool:
    """
    Async pool of AsyncWorkerProcess instances.

    Provides non-blocking acquisition of workers for async operations.
    """

    def __init__(self, size: int = 4, idle_timeout: int = 3600):
        """
        Initialize async worker pool.

        Args:
            size: Maximum number of workers
            idle_timeout: Idle timeout in seconds
        """
        self.size = min(size, DEFAULT_POOL_SIZE)
        self.idle_timeout = idle_timeout
        self._workers: list[AsyncWorkerProcess] = []
        self._available_queue: asyncio.Queue[AsyncWorkerProcess] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._started = False

        logger.debug(f"Initialized AsyncWorkerPool with size={self.size}")

    async def start(self) -> None:
        """Start all async workers in the pool."""
        async with self._lock:
            if self._started:
                return

            for _ in range(self.size):
                worker = AsyncWorkerProcess(idle_timeout=self.idle_timeout)
                await worker.start()  # Start the subprocess
                self._workers.append(worker)
                self._available_queue.put_nowait(worker)

            self._started = True
            logger.info(f"Started AsyncWorkerPool with {self.size} workers")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[AsyncWorkerProcess]:
        """
        Acquire an async worker from the pool.

        Yields:
            An AsyncWorkerProcess instance
        """
        if not self._started:
            await self.start()

        worker = await self._available_queue.get()
        try:
            yield worker
        finally:
            self._available_queue.put_nowait(worker)

    async def shutdown(self) -> None:
        """Shutdown all async workers in the pool."""
        async with self._lock:
            if not self._started:
                return

            for worker in self._workers:
                await worker.stop()

            self._workers.clear()
            self._available_queue = asyncio.Queue()
            self._started = False
            logger.info("Shutdown AsyncWorkerPool")

    async def __aenter__(self) -> "AsyncWorkerPool":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.shutdown()
