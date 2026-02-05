"""
Tests for llm_manager/exceptions.py - Custom exceptions.
"""

import pytest

from llm_manager.exceptions import (
    LLMManagerError,
    ModelLoadError,
    ModelNotFoundError,
    GenerationError,
    WorkerError,
    WorkerTimeoutError,
    ContextError,
    ValidationError,
    ResourceError,
)


class TestLLMManagerError:
    """Test base LLMManagerError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = LLMManagerError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == {}

    def test_error_with_details(self):
        """Test error with details."""
        error = LLMManagerError("Test error", {"key": "value"})
        assert error.message == "Test error"
        assert error.details == {"key": "value"}

    def test_error_inheritance(self):
        """Test that all errors inherit from LLMManagerError."""
        exceptions = [
            ModelLoadError("test"),
            ModelNotFoundError("test"),
            GenerationError("test"),
            WorkerError("test"),
            WorkerTimeoutError("test"),
            ContextError("test"),
            ValidationError("test"),
            ResourceError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, LLMManagerError)
            assert str(exc) == "test"


class TestModelLoadError:
    """Test ModelLoadError."""

    def test_model_load_error(self):
        """Test ModelLoadError creation."""
        error = ModelLoadError("Failed to load model")
        assert isinstance(error, LLMManagerError)
        assert "Failed to load model" in str(error)


class TestModelNotFoundError:
    """Test ModelNotFoundError."""

    def test_model_not_found_error(self):
        """Test ModelNotFoundError creation."""
        error = ModelNotFoundError("Model not found", {"path": "/tmp/model.gguf"})
        assert isinstance(error, LLMManagerError)
        assert isinstance(error, ModelLoadError)
        assert error.details["path"] == "/tmp/model.gguf"


class TestGenerationError:
    """Test GenerationError."""

    def test_generation_error(self):
        """Test GenerationError creation."""
        error = GenerationError("Generation failed")
        assert isinstance(error, LLMManagerError)
        assert "Generation failed" in str(error)


class TestWorkerError:
    """Test WorkerError."""

    def test_worker_error(self):
        """Test WorkerError creation."""
        error = WorkerError("Worker communication failed")
        assert isinstance(error, LLMManagerError)
        assert "Worker communication failed" in str(error)


class TestWorkerTimeoutError:
    """Test WorkerTimeoutError."""

    def test_worker_timeout_error(self):
        """Test WorkerTimeoutError creation."""
        error = WorkerTimeoutError("Worker timeout after 30s")
        assert isinstance(error, LLMManagerError)
        assert isinstance(error, WorkerError)
        assert "Worker timeout" in str(error)


class TestContextError:
    """Test ContextError."""

    def test_context_error(self):
        """Test ContextError creation."""
        error = ContextError("Context size too small")
        assert isinstance(error, LLMManagerError)
        assert "Context size too small" in str(error)


class TestValidationError:
    """Test ValidationError."""

    def test_validation_error(self):
        """Test ValidationError creation."""
        error = ValidationError("Invalid temperature", {"temperature": 5.0})
        assert isinstance(error, LLMManagerError)
        assert error.details["temperature"] == 5.0


class TestResourceError:
    """Test ResourceError."""

    def test_resource_error(self):
        """Test ResourceError creation."""
        error = ResourceError("Insufficient disk space")
        assert isinstance(error, LLMManagerError)
        assert "Insufficient disk space" in str(error)
