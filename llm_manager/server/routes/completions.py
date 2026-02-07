"""Legacy completions endpoint (/v1/completions)."""

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...core import LLMManager
from ...estimation import TokenEstimator
from ...schemas.openai import CompletionChoice, CompletionRequest, CompletionResponse, UsageInfo
from ..dependencies import get_llm_manager, get_or_load_model, verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["completions"])

# Global estimator instance
_token_estimator = TokenEstimator()

# Security: Input validation limits
MAX_PROMPT_LENGTH = 100_000  # 100KB


def validate_prompt(prompt: str | list[str]) -> None:
    """Validate prompt for security and sanity.
    
    Raises:
        HTTPException: If validation fails
    """
    if isinstance(prompt, list):
        if len(prompt) > 100:
            raise HTTPException(
                status_code=400,
                detail=f"Too many prompts: {len(prompt)}. Maximum is 100."
            )
        for i, p in enumerate(prompt):
            if len(p) > MAX_PROMPT_LENGTH:
                raise HTTPException(
                    status_code=400,
                    detail=f"Prompt {i} exceeds maximum length of {MAX_PROMPT_LENGTH} characters."
                )
    else:
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters."
            )


def convert_prompt_to_messages(prompt: str | list[str]) -> list[dict[str, Any]]:
    """Convert legacy prompt format to chat messages."""
    if isinstance(prompt, list):
        # Multiple prompts - use first one
        text = prompt[0] if prompt else ""
    else:
        text = prompt

    return [{"role": "user", "content": text}]


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate token count using TokenEstimator."""
    return _token_estimator.estimate_heuristic(messages).total_tokens


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    manager: LLMManager = Depends(get_llm_manager),
    _: str = Depends(verify_api_key),
) -> CompletionResponse:
    """Create a completion (legacy endpoint).

    This is the legacy completion endpoint for compatibility with older clients.
    Maps to chat completions internally.
    """
    request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    
    # Security: Validate input
    validate_prompt(request.prompt)

    # Ensure model is loaded
    manager = await get_or_load_model(manager, request.model)

    # Convert prompt to messages
    messages = convert_prompt_to_messages(request.prompt)

    # Add suffix if provided
    if request.suffix:
        messages[0]["content"] += request.suffix

    # Estimate tokens using proper estimator
    input_tokens = estimate_tokens(messages)

    # Build generation parameters
    gen_kwargs: dict[str, Any] = {
        "temperature": request.temperature,
        "top_p": request.top_p,
    }

    if request.max_tokens:
        gen_kwargs["max_tokens"] = request.max_tokens
    if request.stop:
        gen_kwargs["stop"] = request.stop
    if request.seed is not None:
        gen_kwargs["seed"] = request.seed
    if request.presence_penalty != 0:
        gen_kwargs["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty != 0:
        gen_kwargs["frequency_penalty"] = request.frequency_penalty
    if request.logit_bias:
        gen_kwargs["logit_bias"] = request.logit_bias

    try:
        # Generate response with timeout
        response = await asyncio.wait_for(
            manager.generate_async(messages, **gen_kwargs),
            timeout=120.0,  # 2 minute timeout
        )

        if not response or "choices" not in response or not response["choices"]:
            raise HTTPException(status_code=500, detail="Empty response from model")

        content = response["choices"][0]["message"]["content"]
        output_tokens = estimate_tokens([{"role": "assistant", "content": content}])

        # Handle echo mode
        if request.echo:
            if isinstance(request.prompt, str):
                prompt_text = request.prompt
            elif request.prompt:
                prompt_text = request.prompt[0]
            else:
                prompt_text = ""
            content = prompt_text + content

        # Build response
        return CompletionResponse(
            id=request_id,
            model=request.model,
            choices=[CompletionChoice(text=content, index=0, finish_reason="stop")],
            usage=UsageInfo(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )

    except asyncio.TimeoutError:
        logger.error("Completion timeout")
        raise HTTPException(status_code=408, detail="Completion timeout after 120s") from None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail=f"Completion failed: {e!s}") from e
