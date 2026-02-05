"""
Worker process management for subprocess isolation.

Provides safe process isolation to prevent freezes and memory leaks.
"""

import asyncio
import atexit
import json
import logging
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator

from .exceptions import WorkerError, WorkerTimeoutError

logger = logging.getLogger(__name__)

# Default timeout and limit constants
WORKER_START_TIMEOUT_SECONDS = 10.0
CRITICAL_TIMEOUT_SECONDS = 30.0
WORKER_IDLE_TIMEOUT_SECONDS = 3600
WORKER_REUSE_LIMIT = 100

# Global tracking for cleanup
_WORKER_PROCESSES: List[subprocess.Popen] = []
_WORKER_PROCESSES_LOCK = threading.Lock()
_TEMP_FILES: List[Path] = []
_TEMP_FILES_LOCK = threading.Lock()


def _emergency_cleanup() -> None:
    """Emergency cleanup of all worker processes on exit."""
    with _WORKER_PROCESSES_LOCK:
        for proc in _WORKER_PROCESSES[:]:
            try:
                if proc and proc.poll() is None:
                    proc.kill()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            except Exception as e:
                logger.debug(f"Cleanup error: {e}")

        _WORKER_PROCESSES.clear()


def _cleanup_temp_files() -> None:
    """Cleanup temporary worker files."""
    with _TEMP_FILES_LOCK:
        for temp_file in _TEMP_FILES[:]:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except OSError as e:
                logger.debug(f"Failed to cleanup {temp_file}: {e}")
        _TEMP_FILES.clear()


# Register cleanup handlers
atexit.register(_emergency_cleanup)
atexit.register(_cleanup_temp_files)


# Worker script that runs in subprocess
WORKER_SCRIPT = '''
"""
Worker process for LLM operations.
Isolated subprocess that handles model loading and generation.
"""

import sys
import json
import gc
import time
import signal
import selectors
import logging

logger = logging.getLogger("llm_worker")
logger.setLevel(logging.WARNING)

# Signal handlers for clean shutdown
def handle_signal(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

# Set line buffering
if hasattr(sys.stdin, 'reconfigure'):
    sys.stdin.reconfigure(line_buffering=True)
    sys.stdout.reconfigure(line_buffering=True)

# Global model state
current_model = None
current_model_path = None
current_model_config = {}


def load_model(path: str, config: dict):
    """Load llama-cpp model."""
    global current_model, current_model_path, current_model_config

    # Check if already loaded
    if (current_model is not None and
        path == current_model_path and
        config == current_model_config):
        return current_model

    # Unload existing
    if current_model is not None:
        del current_model
        gc.collect()
        current_model = None

    # Load new model
    try:
        from llama_cpp import Llama
    except ImportError:
        raise RuntimeError("llama-cpp-python not installed")

    config = config.copy()
    config.pop("model_path", None)

    current_model = Llama(model_path=path, **config)
    current_model_path = path
    current_model_config = config.copy()

    return current_model


def unload_model():
    """Unload current model."""
    global current_model, current_model_path, current_model_config

    if current_model:
        del current_model
        gc.collect()
        current_model = None
        current_model_path = None
        current_model_config = {}


def process_request(request: dict) -> dict:
    """Process a single request."""
    operation = request.get("operation")
    req_id = request.get("id")

    def make_response(data: dict) -> dict:
        if req_id:
            data["id"] = req_id
        return data

    try:
        if operation == "ping":
            return make_response({"success": True, "message": "pong"})

        elif operation == "exit":
            sys.exit(0)

        elif operation == "unload":
            unload_model()
            return make_response({"success": True})

        elif operation == "load":
            model_path = request.get("model_path")
            config = request.get("config", {})
            load_model(model_path, config)
            return make_response({"success": True})

        elif operation == "generate":
            if current_model is None:
                # Auto-load if needed
                model_path = request.get("model_path")
                config = request.get("config", {})
                if model_path:
                    load_model(model_path, config)
                else:
                    return make_response({
                        "success": False,
                        "error": "No model loaded"
                    })

            messages = request.get("messages", [])
            max_tokens = request.get("max_tokens", 256)
            temperature = request.get("temperature", 0.7)
            stream = request.get("stream", False)

            if stream:
                # Streaming mode - yield chunks directly to stdout
                for chunk in current_model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                ):
                    # Send each chunk as JSON line
                    chunk_response = make_response({
                        "type": "chunk",
                        "chunk": chunk
                    })
                    print(json.dumps(chunk_response), flush=True)

                # Signal end of stream
                return make_response({"type": "done", "success": True})
            else:
                response = current_model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False
                )
                return make_response({"success": True, "response": response})

        else:
            return make_response({
                "success": False,
                "error": f"Unknown operation: {operation}"
            })

    except Exception as e:
        import traceback
        return make_response({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        })


def main():
    """Main worker loop."""
    last_activity = time.time()

    # Use selectors for cross-platform I/O
    sel = selectors.DefaultSelector()
    sel.register(sys.stdin, selectors.EVENT_READ)

    # Send ready signal
    print(json.dumps({"success": True, "message": "pong"}))
    sys.stdout.flush()

    while True:
        try:
            # Check for input with timeout
            events = sel.select(timeout=1.0)

            if not events:
                # Check idle timeout
                if time.time() - last_activity > __IDLE_TIMEOUT__:
                    logger.warning("Idle timeout, exiting")
                    break
                continue

            # Read request
            line = sys.stdin.readline()
            if not line:
                break

            last_activity = time.time()

            # Parse and process
            request = json.loads(line.strip())
            response = process_request(request)

            # Send response
            print(json.dumps(response))
            sys.stdout.flush()

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Worker error: {{e}}")
            error_response = {{
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }}
            print(json.dumps(error_response))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
'''


class WorkerProcess:
    """
    Manages a worker subprocess for model operations.

    Provides safe process isolation with timeout protection and
    automatic cleanup.
    """

    def __init__(self, idle_timeout: int = WORKER_IDLE_TIMEOUT_SECONDS):
        """
        Initialize worker process manager.

        Args:
            idle_timeout: Seconds before idle worker self-terminates
        """
        self.idle_timeout = idle_timeout
        self.process: Optional[subprocess.Popen] = None
        self.worker_file: Optional[Path] = None
        self._lock = threading.RLock()
        self._request_count = 0

    def start(self) -> None:
        """
        Start worker process.

        Raises:
            WorkerError: If worker fails to start
            WorkerTimeoutError: If worker doesn't respond in time
        """
        with self._lock:
            if self.process and self.process.poll() is None:
                return  # Already running

            # Create worker script file
            self._create_worker_file()

            try:
                # Start process
                self.process = subprocess.Popen(
                    [sys.executable, str(self.worker_file)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,  # Prevent stderr blocking
                    text=True,
                    bufsize=1,
                    start_new_session=True  # Detach from parent
                )

                # Register for cleanup
                with _WORKER_PROCESSES_LOCK:
                    _WORKER_PROCESSES.append(self.process)

                # Wait for ready signal with timeout
                ready = self._wait_for_ready()

                if not ready:
                    self._kill_process()
                    raise WorkerTimeoutError(
                        f"Worker failed to start within {WORKER_START_TIMEOUT_SECONDS}s"
                    )

                logger.info("Worker process started successfully")

            except Exception as e:
                self._kill_process()
                raise WorkerError(f"Failed to start worker: {e}")

    def _wait_for_ready(self) -> bool:
        """Wait for worker ready signal."""
        start_time = time.time()
        sel = selectors.DefaultSelector()
        sel.register(self.process.stdout, selectors.EVENT_READ)

        try:
            while time.time() - start_time < WORKER_START_TIMEOUT_SECONDS:
                if self.process.poll() is not None:
                    return False  # Process died

                # Use selectors for cross-platform compatibility
                events = sel.select(timeout=0.1)

                if events:
                    try:
                        line = self.process.stdout.readline()
                        if line and "pong" in line.lower():
                            return True
                    except Exception as e:
                        logger.error(f"Error reading ready signal: {e}")
                        return False

            return False
        finally:
            sel.unregister(self.process.stdout)
            sel.close()

    def send_command(
        self,
        command: Dict[str, Any],
        timeout: float = CRITICAL_TIMEOUT_SECONDS,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Send command to worker and wait for response.

        Args:
            command: Command dict with 'operation' and other params
            timeout: Timeout in seconds
            max_retries: Maximum retry attempts for transient failures

        Returns:
            Response dict

        Raises:
            WorkerError: If communication fails
            WorkerTimeoutError: If response times out
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return self._send_command_once(command, timeout)
            except WorkerTimeoutError:
                # Don't retry timeouts
                raise
            except WorkerError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"Worker communication failed, retrying ({attempt + 1}/{max_retries})")
                    # Restart worker for next attempt
                    self._kill_process()
                    time.sleep(0.1)
                else:
                    raise

        raise last_error or WorkerError("Worker communication failed")

    def _send_command_once(
        self,
        command: Dict[str, Any],
        timeout: float
    ) -> Dict[str, Any]:
        """Send a single command to worker (internal)."""
        with self._lock:
            # Check reuse limit
            if self._request_count >= WORKER_REUSE_LIMIT:
                logger.info(f"Worker reuse limit ({WORKER_REUSE_LIMIT}) reached, restarting")
                try:
                    self._kill_process()
                    self._request_count = 0  # Only reset after successful kill
                except Exception as e:
                    logger.warning(f"Failed to kill worker, will retry: {e}")
                    # Don't reset count if kill failed

            # Ensure worker is running
            if not self.process or self.process.poll() is not None:
                self.start()

            # Add request ID - use fast counter instead of UUID
            req_id = f"{os.getpid()}_{threading.get_ident()}_{self._request_count}"
            command = command.copy()  # Don't mutate original
            command["id"] = req_id

            try:
                # Send command
                self.process.stdin.write(json.dumps(command) + "\n")
                self.process.stdin.flush()

                # Wait for response
                response = self._read_response(req_id, timeout)

                self._request_count += 1
                return response

            except Exception as e:
                logger.error(f"Worker communication error: {e}")
                self._kill_process()
                raise WorkerError(f"Worker communication failed: {e}")

    def send_streaming_command(
        self,
        command: Dict[str, Any],
        timeout: float = CRITICAL_TIMEOUT_SECONDS
    ) -> Iterator[Dict[str, Any]]:
        """
        Send command and yield streaming response chunks.

        Args:
            command: Command dict with stream=True
            timeout: Timeout for entire stream

        Yields:
            Response chunks from worker
        """
        with self._lock:
            # Ensure worker is running
            if not self.process or self.process.poll() is not None:
                self.start()

            # Add request ID
            req_id = f"{os.getpid()}_{threading.get_ident()}_{self._request_count}"
            command = command.copy()
            command["id"] = req_id
            command["stream"] = True

            try:
                # Send command
                self.process.stdin.write(json.dumps(command) + "\n")
                self.process.stdin.flush()

                # Read streaming responses
                sel = selectors.DefaultSelector()
                sel.register(self.process.stdout, selectors.EVENT_READ)

                start = time.time()
                try:
                    while time.time() - start < timeout:
                        if self.process.poll() is not None:
                            break

                        events = sel.select(timeout=0.1)
                        if events:
                            line = self.process.stdout.readline()
                            if not line:
                                continue

                            try:
                                response = json.loads(line.strip())
                                if response.get("id") == req_id:
                                    if response.get("type") == "chunk":
                                        yield response.get("chunk", {})
                                    elif response.get("type") == "done":
                                        self._request_count += 1
                                        return
                            except json.JSONDecodeError:
                                continue
                finally:
                    sel.unregister(self.process.stdout)
                    sel.close()

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                self._kill_process()
                raise WorkerError(f"Streaming failed: {e}") from e

    def _read_response(
        self,
        req_id: str,
        timeout: float
    ) -> Dict[str, Any]:
        """Read response from worker."""
        start_time = time.time()
        sel = selectors.DefaultSelector()
        sel.register(self.process.stdout, selectors.EVENT_READ)

        try:
            while time.time() - start_time < timeout:
                # Check if process died
                if self.process.poll() is not None:
                    raise WorkerError("Worker process died unexpectedly")

                # Check for output using selectors (cross-platform)
                events = sel.select(timeout=0.1)

                if events:
                    try:
                        line = self.process.stdout.readline()
                        if not line:
                            continue

                        response = json.loads(line.strip())

                        # Check if this is our response
                        if response.get("id") == req_id:
                            return response
                        else:
                            logger.warning(
                                f"Response ID mismatch: {response.get('id')} != {req_id}"
                            )

                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON from worker: {e}")
                    except Exception as e:
                        logger.error(f"Error reading response: {e}")

            raise WorkerTimeoutError(
                f"Worker response timeout after {timeout}s"
            )
        finally:
            sel.unregister(self.process.stdout)
            sel.close()

    def _create_worker_file(self) -> None:
        """Create temporary worker script file."""
        if self.worker_file and self.worker_file.exists():
            return

        try:
            # Generate unique filename
            temp_dir = Path(tempfile.gettempdir())
            unique_id = f"{os.getpid()}_{threading.get_ident()}_{time.time_ns()}"
            self.worker_file = temp_dir / f"llm_worker_{unique_id}.py"

            # Write script with timeout filled in
            script = WORKER_SCRIPT.replace('__IDLE_TIMEOUT__', str(self.idle_timeout))
            self.worker_file.write_text(script, encoding="utf-8")

            # Set restrictive permissions
            self.worker_file.chmod(0o600)

            # Register for cleanup
            with _TEMP_FILES_LOCK:
                _TEMP_FILES.append(self.worker_file)

        except (PermissionError, OSError) as e:
            raise WorkerError(f"Failed to create worker file: {e}") from e

    def _kill_process(self) -> None:
        """Kill worker process."""
        if not self.process:
            return

        try:
            if self.process.poll() is None:
                # Try graceful shutdown - write directly to stdin to avoid deadlock
                # (we may already be holding self._lock from stop())
                try:
                    self.process.stdin.write('{"operation": "exit"}\n')
                    self.process.stdin.flush()
                except Exception:
                    pass

                # Wait a bit for graceful exit
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass

                # Force kill if still alive
                if self.process.poll() is None:
                    self.process.kill()
                    try:
                        self.process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass

        except Exception as e:
            logger.debug(f"Error killing process: {e}")

        finally:
            if self.process:
                if self.process.stdin:
                    self.process.stdin.close()
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()

            with _WORKER_PROCESSES_LOCK:
                if self.process in _WORKER_PROCESSES:
                    _WORKER_PROCESSES.remove(self.process)
            self.process = None

    def stop(self) -> None:
        """Stop worker process."""
        with self._lock:
            self._kill_process()

            # Cleanup worker file
            if self.worker_file and self.worker_file.exists():
                try:
                    self.worker_file.unlink()
                except OSError as e:
                    logger.debug(f"Failed to delete worker file: {e}")

    def is_alive(self) -> bool:
        """Check if worker is alive."""
        return self.process is not None and self.process.poll() is None

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "alive": self.is_alive(),
            "request_count": self._request_count,
            "pid": self.process.pid if self.process else None,
        }

    def __enter__(self) -> "WorkerProcess":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            if hasattr(self, 'process') and self.process:
                self.stop()
        except Exception:
            pass


class AsyncWorkerProcess:
    """
    Async version of WorkerProcess using asyncio.subprocess.

    Provides non-blocking communication for LLM operations.
    """

    def __init__(
        self,
        idle_timeout: int = WORKER_IDLE_TIMEOUT_SECONDS
    ):
        self.idle_timeout = idle_timeout
        self.process: Optional[asyncio.subprocess.Process] = None
        self.worker_file: Optional[Path] = None
        self._request_count = 0
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        """Start worker process asynchronously."""
        async with self._lock:
            if self.process and self.process.returncode is None:
                return

            self._create_worker_file()

            try:
                self.process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(self.worker_file),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )

                # Register for cleanup
                with _WORKER_PROCESSES_LOCK:
                    # We store the underlying popen object for emergency cleanup
                    _WORKER_PROCESSES.append(self.process._transport.get_extra_info('subprocess'))

                # Wait for ready signal (ping -> pong)
                if not await self._wait_for_ready():
                    await self.stop()
                    raise WorkerError("Worker failed to start or signal ready")
            except WorkerError:
                raise
            except Exception as e:
                await self.stop()
                raise WorkerError(f"Failed to start worker: {e}")

            self._started = True
            logger.debug(f"Async worker started (pid={self.process.pid})")

    async def _wait_for_ready(self, timeout: float = WORKER_START_TIMEOUT_SECONDS) -> bool:
        """Wait for worker to send ready signal."""
        try:
            # Send ping
            ping = json.dumps({"operation": "ping", "id": "init"}) + "\n"
            self.process.stdin.write(ping.encode())
            await self.process.stdin.drain()

            # Read response
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=timeout)
            if line and b"pong" in line.lower():
                return True
        except Exception as e:
            logger.error(f"Async worker ready signal error: {e}")
        return False

    async def send_command(
        self,
        command: Dict[str, Any],
        timeout: float = CRITICAL_TIMEOUT_SECONDS
    ) -> Dict[str, Any]:
        """Send command and wait for response asynchronously."""
        async with self._lock:
            if not self.process or self.process.returncode is not None:
                await self.start()

            req_id = f"async_{os.getpid()}_{id(self)}_{self._request_count}"
            command = command.copy()
            command["id"] = req_id

            try:
                self.process.stdin.write((json.dumps(command) + "\n").encode())
                await self.process.stdin.drain()

                # Read response with timeout
                while True:
                    line = await asyncio.wait_for(self.process.stdout.readline(), timeout=timeout)
                    if not line:
                        raise WorkerError("Worker stdout closed")

                    response = json.loads(line.decode().strip())
                    if response.get("id") == req_id:
                        self._request_count += 1

                        # Handle reuse limit
                        if self._request_count >= WORKER_REUSE_LIMIT:
                            await self._restart()

                        return response
            except Exception as e:
                logger.error(f"Async worker communication error: {e}")
                await self.stop()
                raise WorkerError(f"Async worker failed: {e}") from e

    async def send_streaming_command(
        self,
        command: Dict[str, Any],
        timeout: float = CRITICAL_TIMEOUT_SECONDS
    ) -> asyncio.Queue:
        """
        Send command and return an async queue for chunks.

        Args:
            command: Command dict
            timeout: Overall timeout

        Returns:
            Queue that will receive chunks: {"type": "chunk", "chunk": ...}
            or {"type": "done"} or {"type": "error"}
        """
        # This is a bit complex for a simple queue, let's use a generator instead
        raise NotImplementedError("Use send_streaming_command_gen")

    async def send_streaming_command_gen(
        self,
        command: Dict[str, Any],
        timeout: float = CRITICAL_TIMEOUT_SECONDS
    ):
        """Async generator for streaming responses."""
        async with self._lock:
            if not self.process or self.process.returncode is not None:
                await self.start()

            req_id = f"async_stream_{os.getpid()}_{id(self)}_{self._request_count}"
            command = command.copy()
            command["id"] = req_id
            command["stream"] = True

            try:
                self.process.stdin.write((json.dumps(command) + "\n").encode())
                await self.process.stdin.drain()

                start_time = asyncio.get_event_loop().time()
                while asyncio.get_event_loop().time() - start_time < timeout:
                    line = await asyncio.wait_for(self.process.stdout.readline(), timeout=timeout)
                    if not line:
                        break

                    response = json.loads(line.decode().strip())
                    if response.get("id") == req_id:
                        if response.get("type") == "chunk":
                            yield response.get("chunk")
                        elif response.get("type") == "done":
                            self._request_count += 1
                            return
            except Exception as e:
                logger.error(f"Async streaming error: {e}")
                await self.stop()
                raise WorkerError(f"Async streaming failed: {e}") from e

    def _create_worker_file(self) -> None:
        """Create temporary worker script file (reused from sync version logic)."""
        temp_dir = Path(tempfile.gettempdir())
        unique_id = f"async_{os.getpid()}_{time.time_ns()}"
        self.worker_file = temp_dir / f"llm_worker_{unique_id}.py"
        script = WORKER_SCRIPT.replace('__IDLE_TIMEOUT__', str(self.idle_timeout))
        self.worker_file.write_text(script, encoding="utf-8")
        self.worker_file.chmod(0o600)
        with _TEMP_FILES_LOCK:
            _TEMP_FILES.append(self.worker_file)

    async def _restart(self) -> None:
        """Restart worker process."""
        logger.info("Restarting async worker (reuse limit reached)")
        await self.stop()
        await self.start()
        self._request_count = 0

    async def stop(self) -> None:
        """Stop worker process asynchronously."""
        if self.process:
            try:
                if self.process:
                    if self.process.stdin:
                        self.process.stdin.close()
                    # stdout/stderr are managed by asyncio and usually don't need manual close
                    # but setting process to None helps.
                    self.process.terminate()
                    await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except Exception:
                if self.process:
                    self.process.kill()
                    await self.process.wait()
            finally:
                self.process = None
                self._started = False

        if self.worker_file and self.worker_file.exists():
            try:
                self.worker_file.unlink()
            except OSError:
                pass

    def is_alive(self) -> bool:
        """Check if worker is alive."""
        return self.process is not None and self.process.returncode is None

    async def __aenter__(self) -> "AsyncWorkerProcess":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
