"""Chat completions endpoint (/v1/chat/completions)."""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...core import LLMManager
from ...estimation import TokenEstimator
from ...schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChoiceChunk,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)
from ...tool_parser import has_tool_calls, parse_tool_calls
from ..dependencies import get_llm_manager, get_or_load_model, verify_api_key
from ..request_queue import get_request_queue

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

# Security: Input validation limits
MAX_MESSAGE_LENGTH = 100_000  # 100KB per message
MAX_MESSAGES = 100  # Maximum number of messages in a conversation
MAX_TOTAL_CONTENT = 1_000_000  # 1MB total content


def validate_messages(messages: list[ChatMessage]) -> None:
    """Validate messages for security and sanity.

    Raises:
        HTTPException: If validation fails
    """
    if len(messages) > MAX_MESSAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages: {len(messages)}. Maximum is {MAX_MESSAGES}."
        )

    total_length = 0
    for i, msg in enumerate(messages):
        # Check content length
        if msg.content and len(msg.content) > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} exceeds maximum length of {MAX_MESSAGE_LENGTH} characters."
            )

        # Check for potential injection patterns
        if msg.content:
            content_lower = msg.content.lower()
            # Block obvious script/content injection attempts
            if "<script>" in content_lower or "javascript:" in content_lower:
                raise HTTPException(
                    status_code=400,
                    detail=f"Message {i} contains potentially unsafe content."
                )
            total_length += len(msg.content)

    if total_length > MAX_TOTAL_CONTENT:
        raise HTTPException(
            status_code=400,
            detail=f"Total content length exceeds maximum of {MAX_TOTAL_CONTENT} characters."
        )


# Token estimator created per-call for thread safety
def convert_messages_to_prompt(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert Pydantic messages to dict format for LLMManager."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        msg_dict: dict[str, Any] = {"role": msg.role}
        if msg.content is not None:
            msg_dict["content"] = msg.content
        if msg.name:
            msg_dict["name"] = msg.name
        if msg.tool_calls:
            msg_dict["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id
        result.append(msg_dict)
    return result


# Module-level estimator singleton (thread-safe)
_token_estimator = TokenEstimator()


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate token count using TokenEstimator."""
    return _token_estimator.estimate_heuristic(messages).total_tokens


def create_usage_info(input_tokens: int, output_tokens: int) -> UsageInfo:
    """Create usage statistics."""
    return UsageInfo(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


async def stream_chat_completion(
    manager: LLMManager, request: ChatCompletionRequest, request_id: str
) -> AsyncGenerator[str, None]:
    """Generate SSE stream for chat completion with real token streaming.

    Yields Server-Sent Events in OpenAI format.
    """
    # Security: Validate input
    validate_messages(request.messages)

    prompt = convert_messages_to_prompt(request.messages)

    # Estimate input tokens using proper estimator
    input_tokens = estimate_tokens(prompt)

    # Generation parameters
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
    if request.tools:
        gen_kwargs["tools"] = [t.model_dump() for t in request.tools]
    if request.tool_choice and request.tool_choice != "auto":
        gen_kwargs["tool_choice"] = (
            request.tool_choice
            if isinstance(request.tool_choice, str)
            else request.tool_choice.model_dump()
        )

    # Send initial role message
    initial_chunk = ChatCompletionChunk(
        id=request_id,
        model=request.model,
        choices=[
            ChatCompletionChoiceChunk(
                index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Track generated content
    full_content = ""
    output_tokens = 0

    try:
        # Use real streaming if available (subprocess mode)
        if manager.use_subprocess:
            stream_gen: AsyncGenerator[Any, None] = await manager.generate_async(
                prompt, stream=True, **gen_kwargs  # type: ignore[assignment]
            )
            async for chunk in stream_gen:
                if isinstance(chunk, dict) and "choices" in chunk:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_content += content
                        # Use proper token estimator for accurate count
                        output_tokens = estimate_tokens(
                            [{"role": "assistant", "content": full_content}]
                        )

                        stream_chunk = ChatCompletionChunk(
                            id=request_id,
                            model=request.model,
                            choices=[
                                ChatCompletionChoiceChunk(
                                    index=0, delta=DeltaMessage(content=content), finish_reason=None
                                )
                            ],
                        )
                        yield f"data: {stream_chunk.model_dump_json()}\n\n"
        else:
            # Direct mode: fall back to sentence-level streaming for responsiveness
            response = await manager.generate_async(prompt, **gen_kwargs)

            if response and "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]

                # Stream sentence by sentence for better UX than word-by-word
                # but not wait for full generation
                sentences = []
                current = ""
                for char in content:
                    current += char
                    if char in ".!?\n" and len(current) > 10:
                        sentences.append(current)
                        current = ""
                if current:
                    sentences.append(current)

                for sentence in sentences:
                    full_content += sentence
                    # Use proper token estimator for accurate count
                    output_tokens = estimate_tokens(
                        [{"role": "assistant", "content": full_content}]
                    )

                    chunk = ChatCompletionChunk(
                        id=request_id,
                        model=request.model,
                        choices=[
                            ChatCompletionChoiceChunk(
                                index=0, delta=DeltaMessage(content=sentence), finish_reason=None
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                    # Small delay for natural feeling
                    await asyncio.sleep(0.01)

        # Send final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id=request_id,
            model=request.model,
            choices=[
                ChatCompletionChoiceChunk(index=0, delta=DeltaMessage(), finish_reason="stop")
            ],
            usage=create_usage_info(input_tokens, output_tokens)
            if request.stream_options and request.stream_options.include_usage
            else None,
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {"error": {"message": str(e), "type": "generation_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"

    # End of stream
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    manager: LLMManager = Depends(get_llm_manager),
    _: str = Depends(verify_api_key),
) -> ChatCompletionResponse | StreamingResponse:
    """Create a chat completion.

    OpenAI-compatible endpoint for chat completions.
    Supports streaming and non-streaming responses.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    # Security: Validate input
    validate_messages(request.messages)

    # Backpressure: Acquire queue slot
    queue = get_request_queue()
    await queue.acquire(request_id)

    try:
        # Ensure model is loaded
        manager = await get_or_load_model(manager, request.model)

        # Handle streaming request
        if request.stream:
            # Release queue slot when streaming completes
            async def stream_with_cleanup() -> AsyncGenerator[str, None]:
                try:
                    async for chunk in stream_chat_completion(manager, request, request_id):
                        yield chunk
                finally:
                    queue.release()

            return StreamingResponse(
                stream_with_cleanup(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Non-streaming: process and release
        return await _handle_non_streaming(manager, request, request_id, queue)
    except Exception:
        # Release on error
        queue.release()
        raise


async def _handle_non_streaming(
    manager: LLMManager,
    request: ChatCompletionRequest,
    request_id: str,
    queue: Any,
) -> ChatCompletionResponse:
    """Handle non-streaming chat completion."""
    try:
        # Non-streaming request
        prompt = convert_messages_to_prompt(request.messages)

        # Estimate input tokens using proper estimator
        input_tokens = estimate_tokens(prompt)

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
        if request.tools:
            gen_kwargs["tools"] = [t.model_dump() for t in request.tools]
        if request.tool_choice and request.tool_choice != "auto":
            gen_kwargs["tool_choice"] = (
                request.tool_choice
                if isinstance(request.tool_choice, str)
                else request.tool_choice.model_dump()
            )

        # Generate response with timeout
        response = await asyncio.wait_for(
            manager.generate_async(prompt, **gen_kwargs),
            timeout=120.0,  # 2 minute timeout
        )

        if not response or "choices" not in response or not response["choices"]:
            raise HTTPException(status_code=500, detail="Empty response from model")

        content = response["choices"][0]["message"]["content"] or ""

        # Parse tool calls from content if tools were requested
        tool_calls = None
        finish_reason = "stop"

        if request.tools and has_tool_calls(content):
            cleaned_content, tool_calls = parse_tool_calls(content)
            content = cleaned_content if cleaned_content else None
            finish_reason = "tool_calls"

        output_tokens = estimate_tokens([{"role": "assistant", "content": content or ""}])

        # Build response
        return ChatCompletionResponse(
            id=request_id,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls),
                    finish_reason=finish_reason,
                )
            ],
            usage=create_usage_info(input_tokens, output_tokens),
        )

    except asyncio.TimeoutError:
        logger.error("Generation timeout")
        raise HTTPException(status_code=408, detail="Generation timeout after 120s") from None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e!s}") from e
    finally:
        queue.release()
