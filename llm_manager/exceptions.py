"""
Custom exceptions for the LLM Manager system.

All exceptions inherit from LLMManagerError for easy catching of all
manager-related errors.
"""

__all__ = [
    "ContextError",
    "GenerationError",
    "LLMManagerError",
    "ModelLoadError",
    "ModelNotFoundError",
    "ValidationError",
    "WorkerError",
]



from typing import Any


class LLMManagerError(Exception):
    """Base exception for all LLM Manager errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelLoadError(LLMManagerError):
    """Raised when model loading fails."""

    pass


class ModelNotFoundError(ModelLoadError):
    """Raised when a model file cannot be found."""

    pass


class GenerationError(LLMManagerError):
    """Raised when text generation fails."""

    pass


class WorkerError(LLMManagerError):
    """Raised when worker process operations fail."""

    pass


class WorkerTimeoutError(WorkerError):
    """Raised when worker operations time out."""

    pass


class ContextError(LLMManagerError):
    """Raised when context management operations fail."""

    pass


class ValidationError(LLMManagerError):
    """Raised when input validation fails."""

    pass


class ResourceError(LLMManagerError):
    """Raised when system resources are insufficient."""

    pass
