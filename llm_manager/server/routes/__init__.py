"""API route handlers."""

from .chat import router as chat_router
from .completions import router as completions_router
from .health import router as health_router
from .metrics import router as metrics_router
from .models import router as models_router

__all__ = [
    "chat_router",
    "completions_router",
    "health_router",
    "metrics_router",
    "models_router",
]
