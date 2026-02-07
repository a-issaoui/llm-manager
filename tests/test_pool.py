"""
Tests for llm_manager/pool.py - Worker process pool.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_manager.pool import DEFAULT_POOL_SIZE, AsyncWorkerPool, WorkerPool


class TestWorkerPoolInit:
    """Tests for WorkerPool initialization."""

    def test_default_init(self):
        """Test default initialization."""
        pool = WorkerPool()
        assert pool.size == 4
        assert pool.idle_timeout == 3600
        assert pool._started is False

    def test_custom_size(self):
        """Test initialization with custom size."""
        pool = WorkerPool(size=8)
        assert pool.size == 8

    def test_custom_idle_timeout(self):
        """Test initialization with custom idle timeout."""
        pool = WorkerPool(idle_timeout=1800)
        assert pool.idle_timeout == 1800

    def test_size_capped_at_max(self):
        """Test size is capped at DEFAULT_POOL_SIZE."""
        pool = WorkerPool(size=100)
        assert pool.size == DEFAULT_POOL_SIZE


class TestWorkerPoolStart:
    """Tests for WorkerPool.start()."""

    @patch("llm_manager.pool.WorkerProcess")
    def test_start_creates_workers(self, mock_worker_class):
        """Test start creates worker processes."""
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker

        pool = WorkerPool(size=3)
        pool.start()

        assert pool._started is True
        assert mock_worker_class.call_count == 3

    @patch("llm_manager.pool.WorkerProcess")
    def test_start_idempotent(self, mock_worker_class):
        """Test start is idempotent."""
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker

        pool = WorkerPool(size=2)
        pool.start()
        pool.start()

        assert mock_worker_class.call_count == 2


class TestWorkerPoolAcquire:
    """Tests for WorkerPool.acquire()."""

    @patch("llm_manager.pool.WorkerProcess")
    def test_acquire_gets_worker(self, mock_worker_class):
        """Test acquire returns a worker."""
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker

        pool = WorkerPool(size=2)

        with pool.acquire() as worker:
            assert worker is mock_worker

    @patch("llm_manager.pool.WorkerProcess")
    def test_acquire_auto_starts(self, mock_worker_class):
        """Test acquire auto-starts pool if not started."""
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker

        pool = WorkerPool(size=2)

        with pool.acquire() as worker:
            pass

        assert pool._started is True


class TestWorkerPoolShutdown:
    """Tests for WorkerPool.shutdown()."""

    @patch("llm_manager.pool.WorkerProcess")
    def test_shutdown_stops_workers(self, mock_worker_class):
        """Test shutdown stops all workers."""
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker

        pool = WorkerPool(size=3)
        pool.start()
        pool.shutdown()

        assert mock_worker.stop.call_count == 3
        assert pool._started is False

    def test_shutdown_not_started(self):
        """Test shutdown when not started."""
        pool = WorkerPool()
        pool.shutdown()


class TestWorkerPoolContextManager:
    """Tests for WorkerPool context manager."""

    @patch("llm_manager.pool.WorkerProcess")
    def test_context_manager(self, mock_worker_class):
        """Test context manager starts and shuts down."""
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker

        with WorkerPool(size=2) as pool:
            assert pool._started is True

        assert pool._started is False


class TestAsyncWorkerPoolInit:
    """Tests for AsyncWorkerPool initialization."""

    def test_default_init(self):
        """Test default initialization."""
        pool = AsyncWorkerPool()
        assert pool.size == 4
        assert pool.idle_timeout == 3600
        assert pool._started is False

    def test_size_capped_at_max(self):
        """Test size is capped at DEFAULT_POOL_SIZE."""
        pool = AsyncWorkerPool(size=100)
        assert pool.size == DEFAULT_POOL_SIZE


class TestAsyncWorkerPoolStart:
    """Tests for AsyncWorkerPool.start()."""

    @pytest.mark.asyncio
    @patch("llm_manager.pool.AsyncWorkerProcess")
    async def test_start_creates_workers(self, mock_worker_class):
        """Test start creates async workers."""
        mock_worker = Mock()
        mock_worker.start = AsyncMock()  # Support await worker.start()
        mock_worker_class.return_value = mock_worker

        pool = AsyncWorkerPool(size=3)
        await pool.start()

        assert pool._started is True
        assert mock_worker_class.call_count == 3
        assert mock_worker.start.call_count == 3  # Verify start was called

    @pytest.mark.asyncio
    @patch("llm_manager.pool.AsyncWorkerProcess")
    async def test_start_idempotent(self, mock_worker_class):
        """Test start is idempotent."""
        mock_worker = Mock()
        mock_worker.start = AsyncMock()  # Support await worker.start()
        mock_worker_class.return_value = mock_worker

        pool = AsyncWorkerPool(size=2)
        await pool.start()
        await pool.start()

        assert mock_worker_class.call_count == 2


class TestAsyncWorkerPoolAcquire:
    """Tests for AsyncWorkerPool.acquire()."""

    @pytest.mark.asyncio
    @patch("llm_manager.pool.AsyncWorkerProcess")
    async def test_acquire_gets_worker(self, mock_worker_class):
        """Test acquire returns a worker."""
        mock_worker = Mock()
        mock_worker.start = AsyncMock()  # Support await worker.start()
        mock_worker_class.return_value = mock_worker

        pool = AsyncWorkerPool(size=2)
        await pool.start()

        async with pool.acquire() as worker:
            assert worker is mock_worker

    @pytest.mark.asyncio
    @patch("llm_manager.pool.AsyncWorkerProcess")
    async def test_acquire_auto_starts(self, mock_worker_class):
        """Test acquire auto-starts pool if not started."""
        mock_worker = Mock()
        mock_worker.start = AsyncMock()  # Support await worker.start()
        mock_worker_class.return_value = mock_worker

        pool = AsyncWorkerPool(size=2)

        async with pool.acquire() as worker:
            pass

        assert pool._started is True


class TestAsyncWorkerPoolShutdown:
    """Tests for AsyncWorkerPool.shutdown()."""

    @pytest.mark.asyncio
    @patch("llm_manager.pool.AsyncWorkerProcess")
    async def test_shutdown_stops_workers(self, mock_worker_class):
        """Test shutdown stops all workers."""
        mock_worker = Mock()
        mock_worker.start = AsyncMock()  # Support await worker.start()
        mock_worker.stop = AsyncMock()
        mock_worker_class.return_value = mock_worker

        pool = AsyncWorkerPool(size=3)
        await pool.start()
        await pool.shutdown()

        assert pool._started is False

    @pytest.mark.asyncio
    async def test_shutdown_not_started(self):
        """Test shutdown when not started."""
        pool = AsyncWorkerPool()
        await pool.shutdown()


class TestAsyncWorkerPoolContextManager:
    """Tests for AsyncWorkerPool context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager starts and shuts down."""
        pool = AsyncWorkerPool(size=2)

        # Mock the start and stop methods
        pool.start = AsyncMock()
        pool.shutdown = AsyncMock()

        async with pool:
            pass


class TestPoolConcurrency:
    """Tests for concurrent pool usage."""

    @patch("llm_manager.pool.WorkerProcess")
    def test_multiple_acquires(self, mock_worker_class):
        """Test multiple concurrent acquisitions."""
        workers = [Mock() for _ in range(3)]
        mock_worker_class.side_effect = workers

        pool = WorkerPool(size=3)
        pool.start()

        with pool.acquire() as w1:
            with pool.acquire() as w2:
                with pool.acquire() as w3:
                    assert pool._available_queue.qsize() == 0


class TestPoolEdgeCases:
    """Edge case tests."""

    def test_worker_pool_size_zero(self):
        """Test pool with size 0."""
        pool = WorkerPool(size=0)
        pool.start()
        assert len(pool._workers) == 0

    @pytest.mark.asyncio
    async def test_async_worker_pool_size_zero(self):
        """Test async pool with size 0."""
        pool = AsyncWorkerPool(size=0)
        await pool.start()
        assert len(pool._workers) == 0


class TestPoolLogging:
    """Tests for pool logging."""

    @patch("llm_manager.pool.logger")
    @patch("llm_manager.pool.WorkerProcess")
    def test_start_logging(self, mock_worker_class, mock_logger):
        """Test logging on pool start."""
        mock_worker_class.return_value = Mock()

        pool = WorkerPool(size=2)
        pool.start()

        mock_logger.info.assert_called()

    @patch("llm_manager.pool.logger")
    def test_debug_logging_on_init(self, mock_logger):
        """Test debug logging on initialization."""
        pool = WorkerPool(size=4)
        mock_logger.debug.assert_called()
