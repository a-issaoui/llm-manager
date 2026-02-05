"""Models endpoint (/v1/models)."""

import logging
from fastapi import APIRouter, Depends, HTTPException

from ...schemas.openai import ModelInfo, ModelList
from ...core import LLMManager
from ..dependencies import get_llm_manager, verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"])


@router.get("/v1/models", response_model=ModelList)
async def list_models(
    manager: LLMManager = Depends(get_llm_manager),
    _: str = Depends(verify_api_key)
):
    """List available models.
    
    Returns all models found in the models directory.
    Models are scanned on startup and cached.
    """
    try:
        model_names = manager.registry.list_models()
        
        model_infos = []
        for model_name in model_names:
            # Get metadata if available
            metadata = manager.registry.get(model_name)
            family = None
            if metadata and metadata.specs:
                family = metadata.specs.architecture
            
            model_info = ModelInfo(
                id=model_name,
                owned_by=family or "unknown",
                root=str(manager.models_dir / model_name)
            )
            model_infos.append(model_info)
        
        return ModelList(data=model_infos)
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    manager: LLMManager = Depends(get_llm_manager),
    _: str = Depends(verify_api_key)
):
    """Get information about a specific model."""
    try:
        # Try exact match first
        metadata = manager.registry.get(model_id)
        
        if not metadata:
            # Try partial match
            all_models = manager.registry.list_models()
            matches = [m for m in all_models if model_id.lower() in m.lower()]
            if len(matches) == 1:
                model_id = matches[0]
                metadata = manager.registry.get(model_id)
            elif len(matches) > 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Ambiguous model ID. Matches: {matches}"
                )
        
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        family = metadata.specs.architecture if metadata and metadata.specs else None
        
        return ModelInfo(
            id=model_id,
            owned_by=family or "unknown",
            root=str(manager.models_dir / model_id)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model: {str(e)}")
