"""FastAPI dependencies for LLMManager injection."""

import os
import logging
import secrets
from typing import Optional

from fastapi import HTTPException, Header

from ..core import LLMManager
from ..exceptions import ModelLoadError, ModelNotFoundError, GenerationError

logger = logging.getLogger(__name__)

# Global instance cache
_manager_instance: Optional[LLMManager] = None
_server_config: dict = {}


def configure_server(models_dir: Optional[str] = None, 
                     api_key: Optional[str] = None,
                     default_model: Optional[str] = None,
                     **kwargs):
    """Configure the server before starting.
    
    Args:
        models_dir: Directory containing GGUF models
        api_key: Optional API key for authentication
        default_model: Default model to load on startup
        **kwargs: Additional config passed to LLMManager
    """
    global _server_config
    _server_config = {
        "models_dir": models_dir or os.getenv("LLM_MODELS_DIR", "./models"),
        "api_key": api_key or os.getenv("LLM_API_KEY"),
        "default_model": default_model or os.getenv("LLM_DEFAULT_MODEL"),
        "manager_kwargs": kwargs
    }


def get_llm_manager() -> LLMManager:
    """Get or create singleton LLMManager instance.
    
    This is cached to ensure only one manager exists across requests.
    """
    global _manager_instance, _server_config
    
    if _manager_instance is not None:
        return _manager_instance
    
    models_dir = _server_config.get("models_dir", "./models")
    manager_kwargs = _server_config.get("manager_kwargs", {})
    
    logger.info(f"Initializing LLMManager with models_dir: {models_dir}")
    
    _manager_instance = LLMManager(
        models_dir=models_dir,
        use_subprocess=False,  # Use in-process for server (faster)
        **manager_kwargs
    )
    
    return _manager_instance


def get_current_model_info(manager: LLMManager) -> dict:
    """Get information about currently loaded model."""
    if not manager.is_loaded():
        return {
            "loaded": False,
            "name": None,
            "path": None
        }
    
    return {
        "loaded": True,
        "name": manager.current_model_name if hasattr(manager, 'current_model_name') else None,
        "path": str(manager.model_path) if manager.model_path else None
    }


async def get_or_load_model(manager: LLMManager, model_name: str) -> LLMManager:
    """Ensure requested model is loaded, auto-switching if necessary.
    
    Args:
        manager: LLMManager instance
        model_name: Requested model name
        
    Returns:
        Manager with requested model loaded
        
    Raises:
        HTTPException: If model not found or failed to load
    """
    # Check if already loaded
    current = get_current_model_info(manager)
    if current["loaded"] and current["name"] == model_name:
        return manager
    
    # Try to find and load the model
    try:
        # Check if model exists in registry
        metadata = manager.registry.get(model_name)
        if not metadata:
            # Try partial match
            available = manager.registry.list_models()
            matches = [m for m in available if model_name.lower() in m.lower()]
            if len(matches) == 1:
                model_name = matches[0]
                metadata = manager.registry.get(model_name)
            elif len(matches) > 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Ambiguous model name '{model_name}'. Matches: {matches}"
                )
        
        if not metadata:
            available = manager.registry.list_models()
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available: {available}"
            )
        
        # Load or switch to the model
        logger.info(f"Loading model: {model_name}")
        
        model_path = manager.models_dir / model_name
        
        if manager.is_loaded():
            # Use hot-swap for faster switching
            success = await manager.switch_model_async(str(model_path))
        else:
            success = await manager.load_model_async(str(model_path))
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {model_name}"
            )
        
        logger.info(f"Model loaded successfully: {model_name}")
        return manager
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model loading error: {str(e)}"
        )


async def verify_api_key(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Verify API key if configured.
    
    For local use, this can be disabled by not setting LLM_API_KEY.
    Expects "Bearer <token>" format in Authorization header.
    """
    global _server_config
    expected_key = _server_config.get("api_key")
    
    if not expected_key:
        # No auth required
        return None
    
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Parse Bearer token
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format. Use 'Bearer <token>'"
        )
    
    token = parts[1]
    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(token, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return token


def shutdown_manager():
    """Cleanup function to call on server shutdown."""
    global _manager_instance
    if _manager_instance:
        logger.info("Shutting down LLMManager")
        try:
            _manager_instance.unload_model()
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
        _manager_instance = None
