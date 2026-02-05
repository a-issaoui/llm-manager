"""OpenAI-compatible request/response schemas.

Matches OpenAI API specification for drop-in compatibility.
Reference: https://platform.openai.com/docs/api-reference
"""

from typing import List, Dict, Optional, Union, Literal, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
import time


# =============================================================================
# Common Types
# =============================================================================

class StreamOptions(BaseModel):
    """Options for streaming responses."""
    include_usage: bool = False


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ErrorResponse(BaseModel):
    """Error response format."""
    error: Dict[str, Any]


# =============================================================================
# Chat Completions
# =============================================================================

class ChatMessage(BaseModel):
    """A chat message in the conversation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "user",
                "content": "Hello, how are you?"
            }
        }
    )
    
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ResponseFormat(BaseModel):
    """Response format specification."""
    type: Literal["text", "json_object"] = "text"


class ToolFunction(BaseModel):
    """Function definition for tool calling."""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """Tool definition for function calling."""
    type: Literal["function"] = "function"
    function: ToolFunction


class ToolChoice(BaseModel):
    """Tool choice specification."""
    type: Literal["function", "auto", "none"] = "auto"
    function: Optional[Dict[str, str]] = None


class ChatCompletionRequest(BaseModel):
    """Request body for chat completions."""
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1, le=128)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(None, ge=1)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    response_format: Optional[ResponseFormat] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, ToolChoice]] = "auto"
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "qwen2.5-7b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "temperature": 0.7,
                "max_tokens": 512
            }
        }
    )


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    """Response from chat completions endpoint."""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = None


# =============================================================================
# Streaming Chat Completions
# =============================================================================

class DeltaMessage(BaseModel):
    """Delta message for streaming responses."""
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionChoiceChunk(BaseModel):
    """A single chunk choice for streaming."""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk response."""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoiceChunk]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None


# =============================================================================
# Legacy Completions (for older clients)
# =============================================================================

class CompletionRequest(BaseModel):
    """Request body for legacy text completions."""
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None


class CompletionChoice(BaseModel):
    """A single completion choice (legacy)."""
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None


class CompletionResponse(BaseModel):
    """Response from legacy completions endpoint."""
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


# =============================================================================
# Models
# =============================================================================

class ModelInfo(BaseModel):
    """Information about a model."""
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "llm-manager"
    permission: List[Dict[str, Any]] = Field(default_factory=list)
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelList(BaseModel):
    """List of available models."""
    object: Literal["list"] = "list"
    data: List[ModelInfo]


# =============================================================================
# Embeddings
# =============================================================================

class EmbeddingRequest(BaseModel):
    """Request body for embeddings."""
    model: str
    input: Union[str, List[str]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    """Single embedding result."""
    object: Literal["embedding"] = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Response from embeddings endpoint."""
    object: Literal["list"] = "list"
    data: List[EmbeddingData]
    model: str
    usage: UsageInfo


# =============================================================================
# Health & Status
# =============================================================================

class HealthStatus(BaseModel):
    """Health check response."""
    status: Literal["healthy", "unhealthy"] = "healthy"
    model_loaded: bool = False
    current_model: Optional[str] = None
    version: str = "1.0.0"
    uptime_seconds: float = 0.0


class ServerInfo(BaseModel):
    """Server information."""
    name: str = "llm-manager"
    version: str = "1.0.0"
    models_dir: str
    default_model: Optional[str] = None
    max_context_length: int = 32768
    supports_streaming: bool = True
    supports_function_calling: bool = False
