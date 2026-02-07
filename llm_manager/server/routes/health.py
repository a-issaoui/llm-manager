"""Health and status endpoints."""

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ...core import LLMManager
from ...schemas.openai import HealthStatus, ServerInfo
from ..dependencies import _server_config, get_llm_manager, verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

# Track server start time for uptime
_start_time = time.time()


@router.get("/health", response_model=HealthStatus)
async def health_check(manager: LLMManager = Depends(get_llm_manager)) -> HealthStatus:
    """Health check endpoint.

    Returns server health status and current model information.
    """
    is_loaded = manager.is_loaded()
    current_model = None

    if is_loaded and manager.model_path:
        current_model = manager.model_path.stem

    return HealthStatus(
        status="healthy",
        model_loaded=is_loaded,
        current_model=current_model,
        uptime_seconds=time.time() - _start_time,
    )


@router.get("/v1/health", response_model=HealthStatus)
async def health_check_openai(manager: LLMManager = Depends(get_llm_manager)) -> HealthStatus:
    """Health check at OpenAI-style path."""
    return await health_check(manager)


@router.get("/info", response_model=ServerInfo)
async def server_info() -> ServerInfo:
    """Get server information and capabilities."""
    return ServerInfo(
        models_dir=_server_config.get("models_dir", "./models"),
        default_model=_server_config.get("default_model"),
        supports_streaming=True,
        supports_function_calling=False,
        max_context_length=32768,
    )


@router.get("/ready")
async def readiness_check(manager: LLMManager = Depends(get_llm_manager)) -> dict[str, str]:
    """Kubernetes-style readiness check.

    Returns 200 when server is ready to accept requests.
    """
    # Server is ready if it can respond (model loading is on-demand)
    return {"status": "ready"}


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """Kubernetes-style liveness check.

    Returns 200 if server is alive.
    """
    return {"status": "alive"}


@router.post("/admin/reload", response_model=None)
async def reload_models(
    manager: LLMManager = Depends(get_llm_manager), _: str = Depends(verify_api_key)
) -> dict[str, Any] | JSONResponse:
    """Reload model registry.

    Rescans the models directory for new models.
    """
    try:
        if not manager.registry:
            raise HTTPException(status_code=400, detail="Registry not enabled")

        manager.registry.refresh()
        models = manager.registry.list_models()
        return {"status": "success", "models_found": len(models), "models": models}
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@router.get("/admin/stats")
async def server_stats(
    manager: LLMManager = Depends(get_llm_manager), _: str = Depends(verify_api_key)
) -> dict[str, Any]:
    """Get server statistics."""
    stats = {
        "uptime_seconds": time.time() - _start_time,
        "model_loaded": manager.is_loaded(),
        "current_model": None,
        "models_dir": _server_config.get("models_dir"),
    }

    if manager.is_loaded() and manager.model_path:
        stats["current_model"] = {"name": manager.model_path.stem, "path": str(manager.model_path)}

    # Add metrics if available
    if hasattr(manager, "metrics"):
        metrics_stats = manager.metrics.get_stats()
        stats["metrics"] = {
            "total_requests": metrics_stats.total_requests,
            "tokens_per_second": metrics_stats.tokens_per_second,
            "success_rate": metrics_stats.success_rate,
        }

    return stats
