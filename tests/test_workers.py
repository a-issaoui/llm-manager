"""
Tests for llm_manager/workers.py - Worker process management.
"""

import asyncio
import json
import selectors
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call

import pytest

from llm_manager.workers import (
    WorkerProcess,
    AsyncWorkerProcess,
    _emergency_cleanup,
    _cleanup_temp_files,
    _WORKER_PROCESSES,
    _TEMP_FILES,
    _WORKER_PROCESSES_LOCK,
    _TEMP_FILES_LOCK,
    WORKER_SCRIPT,
)
from llm_manager.exceptions import WorkerError, WorkerTimeoutError


class TestEmergencyCleanup:
    """Tests for emergency cleanup functions."""

    def test_emergency_cleanup_kills_processes(self):
        mock_proc = Mock()
        mock_proc.poll.return_value = None

        with patch('llm_manager.workers._WORKER_PROCESSES', [mock_proc]):
            _emergency_cleanup()
            mock_proc.kill.assert_called_once()

    def test_emergency_cleanup_timeout(self):
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("cmd", 2)

        with patch('llm_manager.workers._WORKER_PROCESSES', [mock_proc]):
            _emergency_cleanup()
            mock_proc.kill.assert_called()

    def test_emergency_cleanup_empty(self):
        with patch('llm_manager.workers._WORKER_PROCESSES', []):
            _emergency_cleanup()

    def test_cleanup_temp_files(self, tmp_path):
        temp_file = tmp_path / "test_worker.py"
        temp_file.write_text("test")

        with patch('llm_manager.workers._TEMP_FILES', [temp_file]):
            _cleanup_temp_files()
            assert not temp_file.exists()

    def test_cleanup_temp_files_handles_missing(self, tmp_path):
        temp_file = tmp_path / "nonexistent.py"

        with patch('llm_manager.workers._TEMP_FILES', [temp_file]):
            _cleanup_temp_files()

    def test_cleanup_temp_files_error(self, tmp_path):
        """Test cleanup handles OSErrors."""
        temp_file = tmp_path / "denied.py"
        temp_file.write_text("test")

        with patch('llm_manager.workers._TEMP_FILES', [temp_file]):
            with patch.object(Path, 'unlink', side_effect=OSError("Permission denied")):
                _cleanup_temp_files()

        # Should clear the list regardless
        from llm_manager.workers import _TEMP_FILES, _TEMP_FILES_LOCK
        with _TEMP_FILES_LOCK:
            assert len(_TEMP_FILES) == 0


class TestWorkerScript:
    """Tests for the worker script template."""

    def test_worker_script_contains_idle_timeout_placeholder(self):
        assert '__IDLE_TIMEOUT__' in WORKER_SCRIPT

    def test_worker_script_contains_main_operations(self):
        assert 'operation' in WORKER_SCRIPT
        assert 'load' in WORKER_SCRIPT
        assert 'generate' in WORKER_SCRIPT
        assert 'unload' in WORKER_SCRIPT
        assert 'ping' in WORKER_SCRIPT
        assert 'exit' in WORKER_SCRIPT

    def test_worker_script_contains_streaming_support(self):
        assert 'stream' in WORKER_SCRIPT
        assert 'chunk' in WORKER_SCRIPT


class TestWorkerProcessInit:
    """Tests for WorkerProcess initialization."""

    def test_default_init(self):
        worker = WorkerProcess()
        assert worker.process is None
        assert worker.worker_file is None
        assert worker._request_count == 0

    def test_custom_idle_timeout(self):
        worker = WorkerProcess(idle_timeout=1800)
        assert worker.idle_timeout == 1800


class TestWorkerProcessStart:
    """Tests for WorkerProcess.start()."""

    @patch('subprocess.Popen')
    @patch('selectors.DefaultSelector')
    def test_start_creates_process(self, mock_selector_class, mock_popen, tmp_path):
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.readline.return_value = '{"message": "pong"}'
        mock_popen.return_value = mock_proc

        mock_selector = Mock()
        mock_selector.select.return_value = [(Mock(), selectors.EVENT_READ)]
        mock_selector_class.return_value = mock_selector

        worker = WorkerProcess()
        with patch.object(worker, '_create_worker_file'):
            worker.worker_file = tmp_path / "worker.py"
            worker.start()

        mock_popen.assert_called_once()

    @patch('subprocess.Popen')
    def test_start_already_running(self, mock_popen, tmp_path):
        mock_proc = Mock()
        mock_proc.poll.return_value = None

        worker = WorkerProcess()
        worker.process = mock_proc
        worker.worker_file = tmp_path / "worker.py"

        worker.start()
        mock_popen.assert_not_called()

    @patch('subprocess.Popen')
    def test_start_failure(self, mock_popen, tmp_path):
        """Test start failure when Popen raises error."""
        mock_popen.side_effect = Exception("Failed to start")
        worker = WorkerProcess()
        with patch.object(worker, '_create_worker_file'):
            worker.worker_file = tmp_path / "worker.py"
            with pytest.raises(WorkerError) as exc_info:
                worker.start()
        assert "Failed to start" in str(exc_info.value)

    @patch('subprocess.Popen')
    @patch('selectors.DefaultSelector')
    def test_start_timeout(self, mock_selector_class, mock_popen, tmp_path):
        """Test start timeout when ready signal not received."""
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        # Mock selector to simulate timeout (empty list)
        mock_selector = Mock()
        mock_selector.select.return_value = []
        mock_selector_class.return_value = mock_selector

        worker = WorkerProcess()
        with patch.object(worker, '_create_worker_file'):
            worker.worker_file = tmp_path / "worker.py"
            with pytest.raises(WorkerError) as exc_info:
                worker.start()
        assert "within" in str(exc_info.value).lower()

    @patch('subprocess.Popen')
    @patch('selectors.DefaultSelector')
    def test_wait_for_ready_read_error(self, mock_selector_class, mock_popen, tmp_path):
        """Test handling of read error in _wait_for_ready."""
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.readline.side_effect = Exception("Read error")
        mock_popen.return_value = mock_proc

        mock_selector = Mock()
        mock_selector.select.return_value = [(Mock(), selectors.EVENT_READ)]
        mock_selector_class.return_value = mock_selector

        worker = WorkerProcess()
        with patch.object(worker, '_create_worker_file'):
            worker.worker_file = tmp_path / "worker.py"
            with pytest.raises(WorkerError) as exc_info: # _wait_for_ready returns False, start() raises WorkerError wrapping WorkerTimeoutError
                worker.start()
        assert "within" in str(exc_info.value).lower()
        mock_popen.assert_called_once()

    def test_sync_worker_wait_for_ready_process_died(self):
        """Cover process.poll() is not None in _wait_for_ready."""
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.poll.side_effect = [None, 0] # First alive, then dead
        worker.process = mock_proc

        with patch('llm_manager.workers.selectors.DefaultSelector') as mock_sel:
            mock_selector = Mock()
            mock_selector.select.return_value = [] # Timeout in select
            mock_sel.return_value = mock_selector

            # Mock time to not timeout too fast or wait too long
            with patch('time.time', side_effect=[0, 0.05, 0.1, 0.15]):
                result = worker._wait_for_ready()

        assert result is False


class TestWorkerProcessSendCommand:
    """Tests for WorkerProcess.send_command()."""

    def test_send_command_success(self):
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.poll.return_value = None

        with patch.object(worker, '_send_command_once') as mock_send:
            mock_send.return_value = {"success": True, "data": "result"}
            result = worker.send_command({"operation": "test"})

        assert result["success"] is True

    def test_send_command_retry_on_error(self):
        worker = WorkerProcess()

        with patch.object(worker, '_send_command_once') as mock_send:
            mock_send.side_effect = [WorkerError("Error"), {"success": True}]
            result = worker.send_command({"operation": "test"})

        assert result["success"] is True
        assert mock_send.call_count == 2

    def test_send_command_max_retries_exceeded(self):
        worker = WorkerProcess()
        with patch.object(worker, '_send_command_once') as mock_send:
            mock_send.side_effect = WorkerError("Persistent Error")
            with pytest.raises(WorkerError) as exc_info:
                worker.send_command({"operation": "test"}, max_retries=1)
        assert "Persistent Error" in str(exc_info.value)
        assert mock_send.call_count == 2

    def test_send_command_no_retry_on_timeout(self):
        worker = WorkerProcess()
        with patch.object(worker, '_send_command_once') as mock_send:
            mock_send.side_effect = WorkerTimeoutError("Timeout")
            with pytest.raises(WorkerTimeoutError):
                worker.send_command({"operation": "test"})
        assert mock_send.call_count == 1

    def test_send_command_once_reuse_limit(self):
        """Test restart when reuse limit reached."""
        from llm_manager.workers import WORKER_REUSE_LIMIT
        worker = WorkerProcess()
        worker._request_count = WORKER_REUSE_LIMIT

        mock_proc = Mock()
        mock_proc.poll.return_value = None
        worker.process = mock_proc

        with patch.object(worker, '_kill_process') as mock_kill, \
             patch.object(worker, 'start'), \
             patch.object(worker, '_read_response', return_value={"success": True}):
            worker._send_command_once({"op": "test"}, timeout=1.0)

        mock_kill.assert_called_once()
        assert worker._request_count == 1

    def test_read_response_id_mismatch(self):
        """Test handling of ID mismatch in response."""
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        # First returns wrong ID, then correct ID
        mock_proc.stdout.readline.side_effect = [
            b'{"id": "wrong_id"}\n',
            b'{"id": "correct_id", "success": true}\n'
        ]
        mock_proc.stdout.fileno = Mock(return_value=1)
        worker.process = mock_proc

        with patch('selectors.DefaultSelector') as mock_sel:
            mock_selector = Mock()
            mock_selector.select.return_value = [(Mock(), selectors.EVENT_READ)]
            mock_sel.return_value = mock_selector

            response = worker._read_response("correct_id", timeout=1.0)

        assert response["success"] is True

    def test_read_response_json_error(self):
        """Test handling of invalid JSON in response."""
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.readline.side_effect = [
            b'invalid json\n',
            b'{"id": "req_id", "success": true}\n'
        ]
        mock_proc.stdout.fileno = Mock(return_value=1)
        worker.process = mock_proc

        with patch('selectors.DefaultSelector') as mock_sel:
            mock_selector = Mock()
            mock_selector.select.return_value = [(Mock(), selectors.EVENT_READ)]
            mock_sel.return_value = mock_selector

            response = worker._read_response("req_id", timeout=1.0)

        assert response["success"] is True

    def test_read_response_process_died(self):
        """Test handling when process dies during read."""
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.poll.return_value = 0 # Died
        mock_proc.stdout.fileno = Mock(return_value=1)
        worker.process = mock_proc

        with patch('selectors.DefaultSelector') as mock_sel:
            mock_selector = Mock()
            mock_selector.select.return_value = [(Mock(), selectors.EVENT_READ)]
            mock_sel.return_value = mock_selector

            with pytest.raises(WorkerError) as exc_info:
                worker._read_response("req_id", timeout=1.0)
        assert "died unexpectedly" in str(exc_info.value)

    def test_read_response_generic_exception(self):
        """Test handling of generic exception during read."""
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.readline.side_effect = [Exception("Read error")] * 10
        mock_proc.stdout.fileno = Mock(return_value=1)
        worker.process = mock_proc

        # We need it to eventually timeout so it doesn't loop forever
        with patch('selectors.DefaultSelector') as mock_sel:
            mock_selector = Mock()
            mock_selector.select.return_value = [(Mock(), selectors.EVENT_READ)]
            mock_sel.return_value = mock_selector

            with patch('time.time', side_effect=[0, 0, 0.5, 0.5, 2.0, 2.0]):
                with pytest.raises(WorkerTimeoutError):
                    worker._read_response("req_id", timeout=1.0)

    def test_send_command_no_retry_on_timeout(self):
        worker = WorkerProcess()

        with patch.object(worker, '_send_command_once') as mock_send:
            mock_send.side_effect = WorkerTimeoutError("Timeout")
            with pytest.raises(WorkerTimeoutError):
                worker.send_command({"operation": "test"})

    def test_sync_worker_send_command_once_start_if_not_running(self):
        """Cover self.start() in _send_command_once if not running."""
        worker = WorkerProcess()
        worker.process = Mock()
        worker.process.poll.return_value = 0 # Not running

        with patch.object(worker, 'start') as mock_start, \
             patch.object(worker, '_read_response', return_value={"success": True}):
            # Mock stdin
            worker.process.stdin = Mock()

            worker._send_command_once({"op": "test"}, timeout=1.0)

        mock_start.assert_called_once()


class TestWorkerProcessStats:
    """Tests for get_stats()."""

    def test_stats(self):
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        worker.process = mock_proc
        worker._request_count = 10

        stats = worker.get_stats()

        assert stats["alive"] is True
        assert stats["request_count"] == 10
        assert stats["pid"] == 12345

    def test_stats_no_process(self):
        worker = WorkerProcess()

        stats = worker.get_stats()

        assert stats["alive"] is False
        assert stats["pid"] is None


class TestWorkerProcessContextManager:
    """Tests for context manager protocol."""

    @patch.object(WorkerProcess, 'start')
    @patch.object(WorkerProcess, 'stop')
    def test_context_manager(self, mock_stop, mock_start):
        with WorkerProcess() as worker:
            pass

        mock_start.assert_called_once()
        mock_stop.assert_called()


class TestAsyncWorkerProcess:
    """Tests for AsyncWorkerProcess."""

    @pytest.mark.asyncio
    async def test_async_init(self):
        worker = AsyncWorkerProcess()
        assert worker.process is None
        assert worker._started is False

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    async def test_async_start(self, mock_create_subproc):
        mock_proc = Mock()
        mock_proc.returncode = None
        mock_proc.pid = 12345
        mock_proc.stdout.readline = AsyncMock(return_value=b'{"message": "pong"}')
        mock_proc.stdin.drain = AsyncMock()
        mock_create_subproc.return_value = mock_proc

        worker = AsyncWorkerProcess()
        with patch.object(worker, '_create_worker_file'):
            worker.worker_file = Path("/tmp/worker.py")
            await worker.start()

        assert worker._started is True

    @pytest.mark.asyncio
    async def test_async_send_command(self):
        worker = AsyncWorkerProcess()
        worker._request_count = 0

        # Calculate what the req_id will be
        import os
        expected_id = f"async_{os.getpid()}_{id(worker)}_0"

        mock_proc = Mock()
        mock_proc.returncode = None
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(return_value=f'{{"id": "{expected_id}", "success": true}}'.encode())
        worker.process = mock_proc

        result = await worker.send_command({"operation": "test"})

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_async_is_alive(self):
        worker = AsyncWorkerProcess()
        assert worker.is_alive() is False

        mock_proc = Mock()
        mock_proc.returncode = None
        worker.process = mock_proc

        assert worker.is_alive() is True

    @pytest.mark.asyncio
    async def test_async_stop(self):
        worker = AsyncWorkerProcess()
        mock_proc = Mock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        worker.process = mock_proc
        worker._started = True

        await worker.stop()

        assert worker.process is None
        assert worker._started is False

    @pytest.mark.asyncio
    async def test_async_worker_start_ready_false(self):
        """Cover ready is False in AsyncWorkerProcess.start."""
        worker = AsyncWorkerProcess()
        with patch('llm_manager.workers.asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_exec.return_value = mock_proc

            # Mock _wait_for_ready to return False
            with patch.object(worker, '_wait_for_ready', return_value=False):
                with patch.object(worker, 'stop', new_callable=AsyncMock) as mock_stop:
                    with pytest.raises(WorkerError) as exc_info:
                        await worker.start()
                    assert "failed to start" in str(exc_info.value).lower()
                    mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_worker_start_exception(self):
        """Cover exception in AsyncWorkerProcess.start."""
        worker = AsyncWorkerProcess()
        with patch('llm_manager.workers.asyncio.create_subprocess_exec', side_effect=Exception("Exec failed")):
            with patch.object(worker, 'stop', new_callable=AsyncMock) as mock_stop:
                with pytest.raises(WorkerError) as exc_info:
                    await worker.start()
                assert "failed to start" in str(exc_info.value).lower()
                mock_stop.assert_called_once()


class TestAsyncWorkerProcessStreamGenerate:
    """Tests for AsyncWorkerProcess.send_streaming_command_gen."""

    @pytest.mark.asyncio
    async def test_stream_generate_success(self):
        """Test successful streaming generation."""
        worker = AsyncWorkerProcess()

        import os
        expected_id = f"async_stream_{os.getpid()}_{id(worker)}_0"

        mock_proc = Mock()
        mock_proc.returncode = None
        mock_proc.stdin.drain = AsyncMock()
        # Return streaming chunks with "type" field
        mock_proc.stdout.readline = AsyncMock(
            side_effect=[
                f'{{"id": "{expected_id}", "type": "chunk", "chunk": "Hello"}}'.encode(),
                f'{{"id": "{expected_id}", "type": "chunk", "chunk": " world"}}'.encode(),
                f'{{"id": "{expected_id}", "type": "done"}}'.encode(),
            ]
        )
        mock_proc.wait = AsyncMock()
        worker.process = mock_proc
        worker._request_count = 0

        chunks = []
        async for chunk in worker.send_streaming_command_gen({"prompt": "test"}):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_generate_error_response(self):
        """Test streaming with error response."""
        worker = AsyncWorkerProcess()

        import os
        expected_id = f"async_stream_{os.getpid()}_{id(worker)}_0"

        mock_proc = Mock()
        mock_proc.returncode = None
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.wait = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(
            side_effect=[
                Exception("Read error"),
            ]
        )
        worker.process = mock_proc
        worker._request_count = 0

        with pytest.raises(WorkerError) as exc_info:
            async for chunk in worker.send_streaming_command_gen({"prompt": "test"}):
                pass

        assert "Read error" in str(exc_info.value)


    @pytest.mark.asyncio
    async def test_async_worker_auto_start_streaming(self):
        """Cover async worker auto-start in streaming."""
        worker = AsyncWorkerProcess()

        mock_proc = Mock()
        mock_proc.stdin.write.side_effect = None
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(return_value=b"")

        async def mock_start():
            worker.process = mock_proc

        worker.start = AsyncMock(side_effect=mock_start)
        worker.process = None # Not started

        gen = worker.send_streaming_command_gen({})

        async for _ in gen: pass

        worker.start.assert_awaited()

class TestWorkerFileCreation:
    """Tests for worker file creation."""

    def test_creates_file_with_script(self, tmp_path):
        worker = WorkerProcess(idle_timeout=1800)

        with patch('tempfile.gettempdir', return_value=str(tmp_path)):
            worker._create_worker_file()

        assert worker.worker_file.exists()
        content = worker.worker_file.read_text()
        assert '1800' in content

    def test_file_permissions(self, tmp_path):
        worker = WorkerProcess()

        with patch('tempfile.gettempdir', return_value=str(tmp_path)):
            worker._create_worker_file()

        mode = worker.worker_file.stat().st_mode
        assert mode & 0o777 == 0o600


class TestWorkerIsAlive:
    """Tests for is_alive()."""

    def test_alive_when_running(self):
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        worker.process = mock_proc

        assert worker.is_alive() is True

    def test_not_alive_when_none(self):
        worker = WorkerProcess()
        assert worker.is_alive() is False

    def test_not_alive_when_dead(self):
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.poll.return_value = 0
        worker.process = mock_proc

        assert worker.is_alive() is False


class TestWorkerProcessStop:
    """Tests for WorkerProcess.stop()."""

    def test_stop_sends_exit_command(self):
        """Test stop sends exit command via stdin."""
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.stdin.write.return_value = None
        mock_proc.stdin.flush.return_value = None
        mock_proc.wait.return_value = 0
        worker.process = mock_proc
        worker.worker_file = Path("/tmp/worker.py")

        worker.stop()

        # Should write exit command to stdin
        mock_proc.stdin.write.assert_called_once()
        assert 'exit' in mock_proc.stdin.write.call_args[0][0]
        assert worker.process is None

    def test_stop_error_handling(self):
        """Test stop handles errors gracefully."""
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.stdin.write.side_effect = Exception("Write failed")
        mock_proc.kill.return_value = None
        mock_proc.wait.return_value = 0
        worker.process = mock_proc
        worker.worker_file = Path("/tmp/worker.py")

        worker.stop()  # Should not raise

        assert worker.process is None

    def test_stop_not_running(self):
        """Test stop when process not running."""
        worker = WorkerProcess()
        worker.process = None

        worker.stop()  # Should not raise
        assert worker.process is None


class TestAsyncWorkerProcessSendCommand:
    """Tests for AsyncWorkerProcess.send_command error handling."""

    @pytest.mark.asyncio
    async def test_async_send_command_auto_start(self):
        """Test send_command auto-starts when not running."""
        worker = AsyncWorkerProcess()
        worker.process = None

        # The implementation tries to auto-start, so we should get a start failure
        with patch.object(worker, 'start', new_callable=AsyncMock) as mock_start:
            mock_start.side_effect = WorkerError("Failed to start")

            with pytest.raises(WorkerError) as exc_info:
                await worker.send_command({"operation": "test"})

            assert "Failed to start" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_send_command_error_handling(self):
        """Test send_command handles errors."""
        worker = AsyncWorkerProcess()

        import os
        expected_id = f"async_{os.getpid()}_{id(worker)}_0"

        mock_proc = Mock()
        mock_proc.returncode = None
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.wait = AsyncMock()
        # Simulate read error
        mock_proc.stdout.readline = AsyncMock(side_effect=Exception("Read error"))
        worker.process = mock_proc
        worker._started = True
        worker._request_count = 0

        with pytest.raises(WorkerError) as exc_info:
            await worker.send_command({"operation": "test"})

        assert "Read error" in str(exc_info.value)


class TestAsyncWorkerContextManager:
    """Tests for AsyncWorkerProcess async context manager."""

    @pytest.mark.asyncio
    @patch.object(AsyncWorkerProcess, 'start')
    @patch.object(AsyncWorkerProcess, 'stop')
    async def test_async_context_manager(self, mock_stop, mock_start):
        """Test async context manager."""
        async with AsyncWorkerProcess() as worker:
            pass

        mock_start.assert_called_once()
        mock_stop.assert_called_once()


class TestEmergencyCleanupErrorHandling:
    """Tests for emergency cleanup error handling."""

    def test_emergency_cleanup_kill_error(self):
        """Test emergency cleanup handles kill errors."""
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.kill.side_effect = Exception("Kill failed")
        mock_proc.wait.return_value = None

        with patch('llm_manager.workers._WORKER_PROCESSES', [mock_proc]):
            _emergency_cleanup()  # Should not raise

    def test_emergency_cleanup_wait_error(self):
        """Test emergency cleanup handles wait errors."""
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = Exception("Wait failed")

        with patch('llm_manager.workers._WORKER_PROCESSES', [mock_proc]):
            _emergency_cleanup()  # Should not raise


class TestAsyncWorkerProcessStopErrors:
    """Tests for AsyncWorkerProcess.stop error handling."""

    @pytest.mark.asyncio
    async def test_async_stop_send_error(self):
        """Test stop handles send error."""
        worker = AsyncWorkerProcess()
        mock_proc = Mock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        mock_proc.stdin.drain = AsyncMock(side_effect=Exception("Drain failed"))
        mock_proc.terminate = Mock()
        worker.process = mock_proc
        worker._started = True

        await worker.stop()  # Should not raise
        assert worker.process is None

    @pytest.mark.asyncio
    async def test_async_worker_stop_errors(self):
        """Cover AsyncWorkerProcess stop errors."""
        worker = AsyncWorkerProcess()
        worker.process = Mock()
        worker.process.terminate.side_effect = Exception("Term Fail")
        worker.process.kill = Mock()
        worker.process.wait = AsyncMock()
        worker.worker_file = Mock()
        worker.worker_file.exists.return_value = True
        worker.worker_file.unlink.side_effect = OSError("Unlink Fail")

        # Check process.kill is called
        mock_process = worker.process

        # Should rely on kill and ignore unlink error
        await worker.stop()

        mock_process.kill.assert_called_once()
        worker.worker_file.unlink.assert_called_once()


class TestAsyncWorkerProcessRestart:
    """Tests for AsyncWorkerProcess restart."""

    @pytest.mark.asyncio
    @patch.object(AsyncWorkerProcess, 'stop')
    @patch.object(AsyncWorkerProcess, 'start')
    async def test_async_restart_success(self, mock_start, mock_stop):
        """Test successful async restart via internal method."""
        worker = AsyncWorkerProcess()
        await worker._restart()

        mock_stop.assert_called_once()
        mock_start.assert_called_once()
        assert worker._request_count == 0


class TestAsyncWorkerProcessSpecialCases:
    """Specialized tests for AsyncWorkerProcess error paths."""

    @pytest.mark.asyncio
    async def test_async_wait_for_ready_failure(self):
        """Test async wait_for_ready failure scenario."""
        worker = AsyncWorkerProcess()
        mock_proc = Mock()
        mock_proc.stdout.readline = AsyncMock(return_value=b'invalid\n')
        worker.process = mock_proc

        # Should return False on timeout
        with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
            result = await worker._wait_for_ready(timeout=0.01)
        assert result is False

    @pytest.mark.asyncio
    async def test_async_send_command_stdout_closed(self):
        """Test send_command when stdout is closed."""
        worker = AsyncWorkerProcess()
        mock_proc = Mock()
        mock_proc.returncode = None
        mock_proc.stdout.readline = AsyncMock(return_value=b'') # EOF
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.wait = AsyncMock()
        worker.process = mock_proc
        worker._started = True

        with pytest.raises(WorkerError) as exc_info:
            await worker.send_command({"op": "test"})
        assert "stdout closed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_send_streaming_gen_stdout_closed(self):
        """Test streaming gen when stdout is closed."""
        worker = AsyncWorkerProcess()
        mock_proc = Mock()
        mock_proc.returncode = None
        mock_proc.stdout.readline = AsyncMock(return_value=b'') # EOF
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.wait = AsyncMock()
        worker.process = mock_proc
        worker._started = True

        # Should just exit the generator
        chunks = []
        async for chunk in worker.send_streaming_command_gen({"op": "test"}):
            chunks.append(chunk)
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_async_send_streaming_not_implemented(self):
        """Test send_streaming_command raises NotImplementedError."""
        worker = AsyncWorkerProcess()
        with pytest.raises(NotImplementedError):
            await worker.send_streaming_command({"op": "test"})


class TestWorkerProcessStreaming:
    """Tests for WorkerProcess.send_streaming_command()."""

    def test_send_streaming_auto_start(self, tmp_path):
        """Test streaming command starts worker if not running."""
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.fileno = Mock(return_value=1)

        # Consistent request ID
        import os, threading
        req_id = f"{os.getpid()}_{threading.get_ident()}_1"

        mock_proc.stdout.readline.side_effect = [
            f'{{"id": "{req_id}", "type": "chunk", "chunk": "hello"}}\n'.encode(),
            f'{{"id": "{req_id}", "type": "done"}}\n'.encode(),
            b'' # EOF
        ]

        with patch.object(worker, 'start') as mock_start, \
             patch('selectors.DefaultSelector') as mock_sel:
            mock_selector = Mock()
            mock_selector.select.return_value = [(Mock(), selectors.EVENT_READ)]
            mock_sel.return_value = mock_selector

            worker.process = mock_proc
            worker._request_count = 1
            chunks = list(worker.send_streaming_command({"prompt": "test"}))

            assert chunks == ["hello"]
            mock_start.assert_not_called()

    def test_send_streaming_json_error(self):
        """Test streaming handles invalid JSON."""
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.readline.side_effect = [
            b'invalid\n',
            b'{"id": "req_0", "type": "done"}\n'
        ]
        worker.process = mock_proc

        with patch('selectors.DefaultSelector') as mock_sel:
            mock_selector = Mock()
            mock_selector.select.return_value = [(Mock(), selectors.EVENT_READ)]
            mock_sel.return_value = mock_selector

            # Use os.getpid and threading.get_ident and id(worker) to match req_id
            import os, threading
            req_id = f"{os.getpid()}_{threading.get_ident()}_0"
            mock_proc.stdout.readline.side_effect = [
                b'invalid\n',
                f'{{"id": "{req_id}", "type": "done"}}\n'.encode()
            ]

            list(worker.send_streaming_command({"prompt": "test"}))
            # Should just continue and reach "done"

    def test_send_streaming_exception(self):
        """Test streaming handles generic exception."""
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.readline.side_effect = [Exception("Read error")] * 5
        mock_proc.stdout.fileno = Mock(return_value=1)
        worker.process = mock_proc

        with pytest.raises(WorkerError):
            list(worker.send_streaming_command({"prompt": "test"}))

        mock_proc.kill.assert_called()

    def test_sync_worker_streaming_start_if_not_running(self):
        """Cover self.start() in send_streaming_command if not running."""
        worker = WorkerProcess()
        worker.process = Mock()
        worker.process.poll.return_value = 0 # Not running

        import os, threading, selectors
        req_id = f"{os.getpid()}_{threading.get_ident()}_0"

        with patch.object(worker, 'start') as mock_start, \
             patch('llm_manager.workers.selectors.DefaultSelector') as mock_sel:
            mock_selector = Mock()
            mock_selector.select.return_value = [(Mock(), selectors.EVENT_READ)]
            mock_sel.return_value = mock_selector

            worker.process.stdout.fileno = Mock(return_value=1)
            worker.process.stdout.readline.return_value = f'{{"id": "{req_id}", "type": "done"}}\n'.encode()
            worker.process.stdin = Mock()

            list(worker.send_streaming_command({"op": "test"}))

        mock_start.assert_called_once()

    def test_worker_process_streaming_error_handling(self):
        """Cover streaming error handling in workers.py."""
        worker = WorkerProcess(idle_timeout=10)
        worker.process = Mock()
        worker.process.stdin.write = Mock()
        worker.process.stdout.readline = Mock(side_effect=Exception("Stream Error"))
        worker.process.poll.return_value = None

        command = {"operation": "generate", "stream": True}

        gen = worker.send_streaming_command(command, timeout=1.0)

        with pytest.raises(Exception):
             next(gen)


class TestWorkerFailures:
    """Tests for miscellaneous worker failures."""

    def test_kill_process_error_in_reuse(self):
        """Cover _kill_process exception in send_command reuse check."""
        worker = WorkerProcess()
        worker.process = Mock()
        worker.process.poll.return_value = None
        worker._request_count = 1000 # Trigger reuse limit (assuming 100)

        # We need to mock WORKER_REUSE_LIMIT to be lower or set count high
        with patch('llm_manager.workers.WORKER_REUSE_LIMIT', 10):
            with patch.object(worker, '_kill_process', side_effect=Exception("Kill failed")) as mock_kill:
                 # Call _send_command_once directly to verify reuse logic
                 # It should catch kill exception and continue
                with patch('llm_manager.workers.time.sleep'): # skip sleep
                    try:
                        worker._send_command_once({"op": "test"}, timeout=1)
                    except Exception:
                         # It might fail later in write if we don't mock stdin
                         pass

                mock_kill.assert_called()
                # count should NOT be reset if kill failed
                assert worker._request_count >= 1000

    def test_send_command_once_write_error(self):
        """Cover stdin write exception in _send_command_once."""
        worker = WorkerProcess()
        worker.process = Mock()
        worker.process.poll.return_value = None
        worker.process.stdin.write.side_effect = Exception("Write failed")

        # Should catch exception, log error, kill process, and raise WorkerError
        with patch.object(worker, '_kill_process') as mock_kill:
            with pytest.raises(WorkerError) as exc_info:
                worker._send_command_once({"op": "test"}, timeout=1)

            assert "Write failed" in str(exc_info.value)
            mock_kill.assert_called_once()

    def test_create_worker_file_exists(self, tmp_path):
        """Cover early return in _create_worker_file if exists."""
        worker = WorkerProcess()
        worker.worker_file = tmp_path / "worker.py"
        worker.worker_file.touch()

        with patch('pathlib.Path.exists', return_value=True):
            # Should return immediately coverage line 591
            with patch('tempfile.gettempdir') as mock_tmp:
                worker._create_worker_file()
                mock_tmp.assert_not_called()

    def test_workers_exception_paths(self, tmp_path):
        """Cover worker process exception paths (workers.py)."""
        # 1. Start failure (line 431)
        # We need to make popen raise generic Exception
        wp = WorkerProcess(idle_timeout=10)
        with patch('subprocess.Popen', side_effect=Exception("Popen fail")):
             with pytest.raises(WorkerError, match="Failed to start worker: Popen fail"):
                 wp.start()

        # 2. send_command generic exception (line 564)
        # Patch init to prevent real subprocess start on retry
        with patch('subprocess.Popen') as mock_popen:
            # Setup mock process that always fails write
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_proc.stdin.write.side_effect = Exception("Write fail")
            mock_popen.return_value = mock_proc

            wp.process = mock_proc # Initial process

            # Mock _wait_for_ready to avoid selector errors on mock stdout
            wp._wait_for_ready = Mock(return_value=True)

            # This should hit the generic Exception catch in send_command
            # Note: retry logic will try to restart (calling mocked Popen), which also fails write
            with pytest.raises(WorkerError, match="Worker communication failed: Write fail"):
                 wp.send_command({})

            # 3. send_streaming_command exception (lines 666-669)
            # Reuse failing writer
            # Note: process was killed in Part 2, so this will try to start() again
            # We need mock_popen active to fail the write in new process (or reused mock)
            # Match "Streaming failed" because send_streaming_command wraps the error
            with pytest.raises(WorkerError, match="Streaming failed: Write fail"):
                 list(wp.send_streaming_command({}))


class TestWorkerDefensivePaths:
    """Tests for defensive error handling paths."""
    
    def test_kill_process_oskill_error(self):
        """Cover OSError in _kill_process when os.kill fails (lines 631-632, 639-640)."""
        from unittest.mock import patch
        
        worker = WorkerProcess()
        mock_proc = Mock()
        mock_proc.poll.return_value = None  # Process is running
        mock_proc.stdin = Mock()
        mock_proc.stdout = Mock()
        mock_proc.stderr = Mock()
        worker.process = mock_proc
        
        # Make os.kill raise OSError (simulating permission denied)
        with patch('os.kill', side_effect=OSError("No such process")):
            # Should not raise - should handle gracefully
            worker._kill_process()
    
    def test_send_command_all_retries_fail(self):
        """Cover line 431: raise last_error or WorkerError when all retries fail."""
        from llm_manager.exceptions import WorkerError, WorkerTimeoutError
        
        worker = WorkerProcess()
        
        # Mock to simulate continuous failures
        worker.start = Mock()
        worker.is_alive = Mock(return_value=True)
        
        # _send_command_once will fail
        with patch.object(worker, '_send_command_once', side_effect=WorkerError("Connection failed")):
            with pytest.raises(WorkerError) as exc_info:
                worker.send_command({"operation": "test"}, max_retries=2)
            
            # Should re-raise the last error (from the final retry)
            assert "Connection failed" in str(exc_info.value)
    
    def test_send_command_timeout_not_retried(self):
        """Cover line 419-420: Timeout errors are not retried."""
        from llm_manager.exceptions import WorkerTimeoutError
        
        worker = WorkerProcess()
        worker.start = Mock()
        worker.is_alive = Mock(return_value=True)
        
        # Timeout should be raised immediately without retry
        with patch.object(worker, '_send_command_once', side_effect=WorkerTimeoutError("Timeout")):
            with pytest.raises(WorkerTimeoutError):
                worker.send_command({"operation": "test"})


class TestWorkerPermissionErrors:
    """Tests for permission error handling in workers."""

    def test_create_worker_file_permission_error(self, tmp_path):
        """Cover PermissionError handling (line 610-611)."""
        from llm_manager.exceptions import WorkerError
        from unittest.mock import patch
        
        worker = WorkerProcess()
        
        with patch('tempfile.gettempdir', return_value=str(tmp_path)):
            with patch('pathlib.Path.write_text', side_effect=PermissionError("No permission")):
                with pytest.raises(WorkerError) as exc_info:
                    worker._create_worker_file()
                
                assert "Failed to create worker file" in str(exc_info.value)

    def test_create_worker_file_os_error(self, tmp_path):
        """Cover OSError handling (line 610-611)."""
        from llm_manager.exceptions import WorkerError
        from unittest.mock import patch
        
        worker = WorkerProcess()
        
        with patch('tempfile.gettempdir', return_value=str(tmp_path)):
            with patch('pathlib.Path.chmod', side_effect=OSError("OS error")):
                with pytest.raises(WorkerError) as exc_info:
                    worker._create_worker_file()
                
                assert "Failed to create worker file" in str(exc_info.value)
