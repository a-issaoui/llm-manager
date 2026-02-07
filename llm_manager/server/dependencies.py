"""FastAPI dependencies for LLMManager injection."""

import logging
import os
import secrets
from typing import Any

from fastapi import Header, HTTPException
from fastapi.params import Header as HeaderClass

from ..core import LLMManager

logger = logging.getLogger(__name__)

# Server configuration (set during startup)
_server_config: dict[str, Any] = {}

# Global instance cache (for backward compatibility and direct usage)
_manager_instance: LLMManager | None = None


def configure_server(
    models_dir: str | None = None,
    api_key: str | None = None,
    default_model: str | None = None,
    **kwargs: Any,
) -> None:
    """Configure the server before starting.

    Args:
        models_dir: Directory containing GGUF models
        api_key: Optional API key for authentication
        default_model: Default model to load on startup
        **kwargs: Additional config passed to LLMManager
    """
    new_config = {
        "models_dir": models_dir or os.getenv("LLM_MODELS_DIR", "./models"),
        "api_key": api_key or os.getenv("LLM_API_KEY"),
        "default_model": default_model or os.getenv("LLM_DEFAULT_MODEL"),
        "manager_kwargs": kwargs,
    }
    _server_config.clear()
    _server_config.update(new_config)


def get_server_config() -> dict[str, Any]:
    """Get the server configuration."""
    return _server_config


def _get_or_create_manager() -> LLMManager:
    """Internal function to get or create the manager instance."""
    global _manager_instance

    if _manager_instance is not None:
        return _manager_instance

    models_dir = _server_config.get("models_dir", "./models")
    manager_kwargs = _server_config.get("manager_kwargs", {})

    logger.info(f"Initializing LLMManager with models_dir: {models_dir}")

    _manager_instance = LLMManager(models_dir=models_dir, use_subprocess=False, **manager_kwargs)

    return _manager_instance


def get_llm_manager() -> LLMManager:
    """Get LLMManager singleton instance.

    Uses global singleton pattern for simplicity and backward compatibility.
    The instance is created on first call and reused thereafter.
    """
    return _get_or_create_manager()


def get_current_model_info(manager: LLMManager) -> dict[str, Any]:
    """Get information about currently loaded model."""
    if not manager.is_loaded():
        return {"loaded": False, "name": None, "path": None}

    return {
        "loaded": True,
        "name": manager.current_model_name if hasattr(manager, "current_model_name") else None,
        "path": str(manager.model_path) if manager.model_path else None,
    }


async def get_or_load_model(manager: LLMManager, model_name: str) -> LLMManager:
    """Get or load a model by name.

    Args:
        manager: LLMManager instance
        model_name: Name of the model to load

    Returns:
        The LLMManager instance (for method chaining)

    Raises:
        HTTPException: If model not found or load fails
    """
    from ..exceptions import ModelLoadError, ModelNotFoundError

    try:
        await manager.get_or_load_async(model_name)
        return manager
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ModelLoadError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error loading model: {e}") from e


def shutdown_manager() -> None:
    """Cleanup resources on shutdown."""
    global _manager_instance
    logger.info("Shutting down LLM Manager...")

    # Clean up global instance if exists
    if _manager_instance is not None:
        try:
            _manager_instance.unload_model()
        except Exception as e:
            logger.warning(f"Error unloading model during shutdown: {e}")
        _manager_instance = None

    # Note: App state cleanup is handled by lifespan context


async def verify_api_key(
    authorization: str | None = Header(None), x_api_key: str | None = Header(None)
) -> str | None:
    """Verify API key if configured.

    Args:
        authorization: Authorization header (Bearer token)
        x_api_key: X-API-Key header

    Returns:
        The validated API key if auth required, None if no auth required

    Raises:
        HTTPException: If key is invalid
    """
    expected_key = _server_config.get("api_key")

    if not expected_key:
        return None

    # Check if x_api_key is actually provided (not a Header default object)
    x_api_key_provided = (
        x_api_key is not None
        and not isinstance(x_api_key, HeaderClass)
        and str(x_api_key).strip() not in ("", "None")
    )

    # Check if authorization is actually provided
    auth_provided = (
        authorization is not None
        and not isinstance(authorization, HeaderClass)
        and str(authorization).strip() not in ("", "None")
    )

    # Try X-API-Key header first
    if x_api_key_provided:
        provided_key = str(x_api_key).strip()
    # Then try Authorization header
    elif auth_provided:
        auth_str = str(authorization)
        # Handle Bearer prefix (case-insensitive)
        lower_auth = auth_str.lower()
        if lower_auth.startswith("bearer "):
            provided_key = auth_str[7:].strip()
        else:
            provided_key = auth_str.strip()
    else:
        provided_key = None

    if not provided_key:
        raise HTTPException(
            status_code=401, detail="API key required", headers={"WWW-Authenticate": "Bearer"}
        )

    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(provided_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    return provided_key
