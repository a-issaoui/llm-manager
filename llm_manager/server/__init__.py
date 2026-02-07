"""OpenAI-compatible REST API server for llm_manager."""

from .app import LLMServer, create_app
from .dependencies import get_llm_manager, get_or_load_model

__all__ = [
    "LLMServer",
    "create_app",
    "get_llm_manager",
    "get_or_load_model",
]
