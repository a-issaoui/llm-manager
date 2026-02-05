"""Chat completions endpoint (/v1/chat/completions)."""

import json
import uuid
import logging
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    ChatCompletionChunk,
    ChatCompletionChoiceChunk,
    DeltaMessage,
    UsageInfo
)
from ...core import LLMManager
from ...estimation import TokenEstimator
from ..dependencies import get_llm_manager, get_or_load_model, verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

# Global estimator instance
_token_estimator = TokenEstimator()


def convert_messages_to_prompt(messages: list[ChatMessage]) -> list[dict]:
    """Convert Pydantic messages to dict format for LLMManager."""
    result = []
    for msg in messages:
        msg_dict = {"role": msg.role}
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


def estimate_tokens(messages: list[dict]) -> int:
    """Estimate token count using TokenEstimator."""
    return _token_estimator.estimate_heuristic(messages).total_tokens


def create_usage_info(input_tokens: int, output_tokens: int) -> UsageInfo:
    """Create usage statistics."""
    return UsageInfo(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens
    )


async def stream_chat_completion(
    manager: LLMManager,
    request: ChatCompletionRequest,
    request_id: str
) -> AsyncGenerator[str, None]:
    """Generate SSE stream for chat completion with real token streaming.
    
    Yields Server-Sent Events in OpenAI format.
    """
    prompt = convert_messages_to_prompt(request.messages)
    
    # Estimate input tokens using proper estimator
    input_tokens = estimate_tokens(prompt)
    
    # Generation parameters
    gen_kwargs = {
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
    
    # Send initial role message
    initial_chunk = ChatCompletionChunk(
        id=request_id,
        model=request.model,
        choices=[ChatCompletionChoiceChunk(
            index=0,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None
        )]
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"
    
    # Track generated content
    full_content = ""
    output_tokens = 0
    
    try:
        # Use real streaming if available (subprocess mode)
        if manager.use_subprocess:
            stream_gen = await manager.generate_async(prompt, stream=True, **gen_kwargs)
            async for chunk in stream_gen:
                if isinstance(chunk, dict) and "choices" in chunk:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_content += content
                        output_tokens += len(content) // 4
                        
                        stream_chunk = ChatCompletionChunk(
                            id=request_id,
                            model=request.model,
                            choices=[ChatCompletionChoiceChunk(
                                index=0,
                                delta=DeltaMessage(content=content),
                                finish_reason=None
                            )]
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
                    output_tokens += len(sentence) // 4
                    
                    chunk = ChatCompletionChunk(
                        id=request_id,
                        model=request.model,
                        choices=[ChatCompletionChoiceChunk(
                            index=0,
                            delta=DeltaMessage(content=sentence),
                            finish_reason=None
                        )]
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    
                    # Small delay for natural feeling
                    await asyncio.sleep(0.01)
        
        # Send final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id=request_id,
            model=request.model,
            choices=[ChatCompletionChoiceChunk(
                index=0,
                delta=DeltaMessage(),
                finish_reason="stop"
            )],
            usage=create_usage_info(input_tokens, output_tokens) if request.stream_options and request.stream_options.include_usage else None
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {
            "error": {"message": str(e), "type": "generation_error"}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    
    # End of stream
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    manager: LLMManager = Depends(get_llm_manager),
    _: str = Depends(verify_api_key)
):
    """Create a chat completion.
    
    OpenAI-compatible endpoint for chat completions.
    Supports streaming and non-streaming responses.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    
    # Ensure model is loaded
    manager = await get_or_load_model(manager, request.model)
    
    # Handle streaming request
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(manager, request, request_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    # Non-streaming request
    prompt = convert_messages_to_prompt(request.messages)
    
    # Estimate input tokens using proper estimator
    input_tokens = estimate_tokens(prompt)
    
    # Build generation parameters
    gen_kwargs = {
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
    
    try:
        # Generate response with timeout
        response = await asyncio.wait_for(
            manager.generate_async(prompt, **gen_kwargs),
            timeout=120.0  # 2 minute timeout
        )
        
        if not response or "choices" not in response or not response["choices"]:
            raise HTTPException(status_code=500, detail="Empty response from model")
        
        content = response["choices"][0]["message"]["content"]
        output_tokens = estimate_tokens([{"role": "assistant", "content": content}])
        
        # Build response
        return ChatCompletionResponse(
            id=request_id,
            model=request.model,
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=content
                ),
                finish_reason="stop"
            )],
            usage=create_usage_info(input_tokens, output_tokens)
        )
        
    except asyncio.TimeoutError:
        logger.error("Generation timeout")
        raise HTTPException(status_code=408, detail="Generation timeout after 120s")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
