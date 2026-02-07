"""
Tests for P0/P1 security features:
- Path traversal protection
- Request queue with backpressure
- Input validation (message limits, injection prevention)
"""

import asyncio
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from llm_manager.exceptions import ModelNotFoundError
from llm_manager.schemas.openai import ChatMessage
from llm_manager.server.request_queue import (
    RequestQueue,
    get_request_queue,
    reset_request_queue,
)
from llm_manager.server.routes.chat import validate_messages


# =============================================================================
# Path Traversal Security Tests
# =============================================================================


class TestPathTraversalSecurity:
    """Test path traversal protection in _resolve_model_path."""

    def test_absolute_path_outside_models_dir_blocked(self, tmp_path):
        """P0: Absolute paths outside models_dir should be blocked."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        external_file = tmp_path / "external.gguf"
        external_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            from llm_manager.core import LLMManager

            manager = LLMManager(models_dir=str(models_dir))

            with pytest.raises(ModelNotFoundError) as exc_info:
                manager._resolve_model_path(str(external_file))

        assert "access denied" in str(exc_info.value).lower()
        assert "outside models directory" in str(exc_info.value).lower()

    def test_path_traversal_basic_blocked(self, tmp_path):
        """P0: Basic path traversal like ../ should be blocked."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            from llm_manager.core import LLMManager

            manager = LLMManager(models_dir=str(tmp_path))

            with pytest.raises(ModelNotFoundError) as exc_info:
                manager._resolve_model_path("../outside.gguf")

            assert "traversal" in str(exc_info.value).lower()

    def test_path_traversal_nested_blocked(self, tmp_path):
        """P0: Nested path traversal like foo/../../../etc/passwd blocked."""
        nested = tmp_path / "nested" / "deep"
        nested.mkdir(parents=True)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            from llm_manager.core import LLMManager

            manager = LLMManager(models_dir=str(tmp_path))

            with pytest.raises(ModelNotFoundError) as exc_info:
                manager._resolve_model_path("nested/deep/../../../../etc/passwd")

            assert "traversal" in str(exc_info.value).lower()

    def test_path_traversal_with_null_bytes_blocked(self, tmp_path):
        """P0: Path with null bytes should be handled safely."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            from llm_manager.core import LLMManager

            manager = LLMManager(models_dir=str(tmp_path))

            # Null bytes in path should not bypass security
            with pytest.raises((ModelNotFoundError, ValueError)):
                manager._resolve_model_path("model\x00.gguf")

    def test_non_gguf_extension_blocked(self, tmp_path):
        """P0: Only .gguf files should be allowed."""
        malicious = tmp_path / "model.exe"
        malicious.write_bytes(b"MZ" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            from llm_manager.core import LLMManager

            manager = LLMManager(models_dir=str(tmp_path))

            with pytest.raises(ModelNotFoundError) as exc_info:
                manager._resolve_model_path("model.exe")

            assert ".gguf" in str(exc_info.value).lower()

    def test_case_insensitive_gguf_check(self, tmp_path):
        """P0: .GGUF extension should be allowed (case insensitive)."""
        model = tmp_path / "model.GGUF"
        model.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            from llm_manager.core import LLMManager

            manager = LLMManager(models_dir=str(tmp_path))
            result = manager._resolve_model_path("model.GGUF")

            assert result.name == "model.GGUF"

    def test_directory_traversal_with_unicode_blocked(self, tmp_path):
        """P0: Unicode path traversal attempts should be blocked."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            from llm_manager.core import LLMManager

            manager = LLMManager(models_dir=str(tmp_path))

            # Unicode dots and slashes
            with pytest.raises(ModelNotFoundError):
                manager._resolve_model_path("..\u2215..\u2215etc\u2215passwd.gguf")


# =============================================================================
# Request Queue Backpressure Tests
# =============================================================================


class TestRequestQueue:
    """Test RequestQueue backpressure system."""

    def setup_method(self):
        """Reset queue before each test."""
        reset_request_queue()

    def teardown_method(self):
        """Reset queue after each test."""
        reset_request_queue()

    @pytest.mark.asyncio
    async def test_request_queue_initialization(self):
        """P1: RequestQueue should initialize with correct parameters."""
        queue = RequestQueue(max_concurrent=5, max_queued=50, queue_timeout=10.0)
        assert queue.max_concurrent == 5
        assert queue.max_queued == 50
        assert queue.queue_timeout == 10.0

    @pytest.mark.asyncio
    async def test_request_queue_acquire_release(self):
        """P1: Should be able to acquire and release slots."""
        queue = RequestQueue(max_concurrent=2)

        # Acquire first slot
        await queue.acquire("req-1")
        assert queue.stats["available_slots"] == 1

        # Acquire second slot
        await queue.acquire("req-2")
        assert queue.stats["available_slots"] == 2

        # Release slots
        queue.release()
        queue.release()

    @pytest.mark.asyncio
    async def test_request_queue_context_manager(self):
        """P1: Context manager should handle release on exit."""
        queue = RequestQueue(max_concurrent=1)

        # Acquire first, then use context manager for cleanup
        await queue.acquire("req-1")
        assert queue._concurrent_sem._value == 0  # Semaphore acquired

        async with queue:
            pass  # Context manager just ensures release on exit

        # After context exit, should be released
        assert queue._concurrent_sem._value == 1

    @pytest.mark.asyncio
    async def test_request_queue_rejects_when_full(self):
        """P1: Should reject when max concurrent reached and queue full."""
        queue = RequestQueue(max_concurrent=1, max_queued=0, queue_timeout=0.01)

        # Acquire the only slot
        await queue.acquire("req-1")

        # Second acquire should be rejected (queue is full)
        with pytest.raises(HTTPException) as exc_info:
            await queue.acquire("req-2")

        assert exc_info.value.status_code == 503
        assert "queue full" in str(exc_info.value.detail).lower()

        queue.release()

    @pytest.mark.asyncio
    async def test_request_queue_multiple_concurrent(self):
        """P1: Should allow up to max_concurrent requests."""
        queue = RequestQueue(max_concurrent=3)

        # Acquire 3 slots
        await queue.acquire("req-1")
        await queue.acquire("req-2")
        await queue.acquire("req-3")

        assert queue.stats["available_slots"] == 3

        # Release all
        for _ in range(3):
            queue.release()

    @pytest.mark.asyncio
    async def test_request_queue_stats_tracking(self):
        """P1: Stats should track queue state."""
        queue = RequestQueue(max_concurrent=2, max_queued=10)

        stats = queue.stats
        assert "queue_size" in stats
        assert "max_queued" in stats
        assert "max_concurrent" in stats
        assert "available_slots" in stats
        assert stats["max_concurrent"] == 2
        assert stats["max_queued"] == 10


class TestRequestQueueSingleton:
    """Test RequestQueue singleton behavior."""

    def setup_method(self):
        """Reset queue before each test."""
        reset_request_queue()

    def teardown_method(self):
        """Reset queue after each test."""
        reset_request_queue()

    def test_get_request_queue_creates_singleton(self):
        """P1: get_request_queue should return same instance."""
        queue1 = get_request_queue(max_concurrent=5)
        queue2 = get_request_queue(max_concurrent=10)  # Different param ignored

        assert queue1 is queue2
        assert queue1.max_concurrent == 5  # First value wins

    def test_get_request_queue_default_params(self):
        """P1: Default params should be reasonable."""
        queue = get_request_queue()

        assert queue.max_concurrent == 10  # Default
        assert queue.max_queued == 100  # Default
        assert queue.queue_timeout == 30.0  # Default


# =============================================================================
# Input Validation Security Tests
# =============================================================================


class TestInputValidation:
    """Test input validation for security."""

    def test_validate_messages_too_many_messages(self):
        """P1: Should reject requests with too many messages."""
        messages = [
            ChatMessage(role="user", content=f"Message {i}")
            for i in range(101)  # MAX_MESSAGES + 1
        ]

        with pytest.raises(HTTPException) as exc_info:
            validate_messages(messages)

        assert exc_info.value.status_code == 400
        assert "too many messages" in str(exc_info.value.detail).lower()

    def test_validate_messages_max_messages_allowed(self):
        """P1: Should allow exactly MAX_MESSAGES."""
        messages = [
            ChatMessage(role="user", content=f"Message {i}")
            for i in range(100)  # Exactly MAX_MESSAGES
        ]

        # Should not raise
        validate_messages(messages)

    def test_validate_messages_content_too_long(self):
        """P1: Should reject messages exceeding max length."""
        long_content = "x" * 100_001  # MAX_MESSAGE_LENGTH + 1
        messages = [ChatMessage(role="user", content=long_content)]

        with pytest.raises(HTTPException) as exc_info:
            validate_messages(messages)

        assert exc_info.value.status_code == 400
        assert "maximum length" in str(exc_info.value.detail).lower()

    def test_validate_messages_max_length_allowed(self):
        """P1: Should allow exactly MAX_MESSAGE_LENGTH."""
        content = "x" * 100_000  # Exactly MAX_MESSAGE_LENGTH
        messages = [ChatMessage(role="user", content=content)]

        # Should not raise
        validate_messages(messages)

    def test_validate_messages_script_injection_blocked(self):
        """P1: Should block script tags."""
        messages = [
            ChatMessage(role="user", content='<script>alert("xss")</script>')
        ]

        with pytest.raises(HTTPException) as exc_info:
            validate_messages(messages)

        assert exc_info.value.status_code == 400
        assert "unsafe content" in str(exc_info.value.detail).lower()

    def test_validate_messages_javascript_protocol_blocked(self):
        """P1: Should block javascript: protocol."""
        messages = [
            ChatMessage(role="user", content='javascript:alert("xss")')
        ]

        with pytest.raises(HTTPException) as exc_info:
            validate_messages(messages)

        assert exc_info.value.status_code == 400
        assert "unsafe content" in str(exc_info.value.detail).lower()

    def test_validate_messages_case_insensitive_injection(self):
        """P1: Should block injection patterns case-insensitively."""
        messages = [
            ChatMessage(role="user", content='<SCRIPT>alert(1)</SCRIPT>')
        ]

        with pytest.raises(HTTPException) as exc_info:
            validate_messages(messages)

        assert exc_info.value.status_code == 400

    def test_validate_messages_total_content_limit(self):
        """P1: Should reject if total content exceeds limit."""
        # Create messages with total content > 1MB (each under max length)
        messages = [
            ChatMessage(role="user", content="x" * 50_000)  # 50KB each, under 100KB limit
            for _ in range(25)  # 1.25MB total, under 100 message limit
        ]

        with pytest.raises(HTTPException) as exc_info:
            validate_messages(messages)

        assert exc_info.value.status_code == 400
        assert "total content length" in str(exc_info.value.detail).lower()

    def test_validate_messages_valid_content_allowed(self):
        """P1: Should allow valid content."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello, how are you?"),
            ChatMessage(role="assistant", content="I'm doing well!"),
        ]

        # Should not raise
        validate_messages(messages)

    def test_validate_messages_code_content_allowed(self):
        """P1: Code examples should be allowed if no script tags."""
        messages = [
            ChatMessage(
                role="user",
                content='Here is Python code: `print("hello")`'
            )
        ]

        # Should not raise (no actual script tag)
        validate_messages(messages)

    def test_validate_messages_code_with_script_blocked(self):
        """P1: Code containing actual script tags should be blocked."""
        messages = [
            ChatMessage(
                role="user",
                content='HTML example: <script>alert(1)</script>'
            )
        ]

        with pytest.raises(HTTPException) as exc_info:
            validate_messages(messages)

        assert exc_info.value.status_code == 400

    def test_validate_messages_empty_content_allowed(self):
        """P1: Should allow empty content."""
        messages = [ChatMessage(role="user", content="")]

        # Should not raise
        validate_messages(messages)

    def test_validate_messages_unicode_allowed(self):
        """P1: Should allow unicode content."""
        messages = [
            ChatMessage(role="user", content="Hello 世界 🌍 مرحبا")
        ]

        # Should not raise
        validate_messages(messages)

    def test_validate_messages_none_content_allowed(self):
        """P1: Should allow None content."""
        messages = [ChatMessage(role="user", content=None)]

        # Should not raise
        validate_messages(messages)

    def test_validate_messages_long_but_valid_content(self):
        """P1: Should allow content at boundary of max length."""
        # 100KB exactly
        messages = [ChatMessage(role="user", content="x" * 100_000)]

        # Should not raise
        validate_messages(messages)

    def test_validate_messages_multiple_checks_fail_early(self):
        """P1: Should fail on first validation error."""
        # Too many messages AND too long content - should fail on count
        messages = [
            ChatMessage(role="user", content="x" * 200_000)
            for _ in range(200)
        ]

        with pytest.raises(HTTPException) as exc_info:
            validate_messages(messages)

        # Should fail on message count, not length
        assert "too many messages" in str(exc_info.value.detail).lower()
