"""OpenAI-compatible request/response schemas.

Matches OpenAI API specification for drop-in compatibility.
Reference: https://platform.openai.com/docs/api-reference
"""

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Constants for validation
MAX_MESSAGES = 256
MAX_TOTAL_CHARS = 500_000
MAX_MESSAGE_LENGTH = 100_000


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

    error: dict[str, Any]


# =============================================================================
# Chat Completions
# =============================================================================


class ChatMessage(BaseModel):
    """A chat message in the conversation."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"role": "user", "content": "Hello, how are you?"}}
    )

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ResponseFormat(BaseModel):
    """Response format specification."""

    type: Literal["text", "json_object"] = "text"


class ToolFunction(BaseModel):
    """Function definition for tool calling."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class Tool(BaseModel):
    """Tool definition for function calling."""

    type: Literal["function"] = "function"
    function: ToolFunction


class ToolChoice(BaseModel):
    """Tool choice specification."""

    type: Literal["function", "auto", "none"] = "auto"
    function: dict[str, str] | None = None


class ChatCompletionRequest(BaseModel):
    """Request body for chat completions."""

    model: str = Field(..., description="ID of the model to use")
    messages: list[ChatMessage] = Field(..., description="Conversation history")
    temperature: float | None = Field(0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(1.0, ge=0.0, le=1.0)
    n: int | None = Field(1, ge=1, le=128)
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(None, ge=1)
    presence_penalty: float | None = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(0.0, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    seed: int | None = None
    response_format: ResponseFormat | None = None
    tools: list[Tool] | None = None
    tool_choice: str | ToolChoice | None = "auto"

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Validate message count and total size."""
        if len(messages) > MAX_MESSAGES:
            raise ValueError(f"Too many messages: {len(messages)} > {MAX_MESSAGES}")

        total_chars = sum(len(m.content or "") for m in messages)
        if total_chars > MAX_TOTAL_CHARS:
            raise ValueError(f"Total message content too large: {total_chars} chars")

        for i, msg in enumerate(messages):
            if msg.content and len(msg.content) > MAX_MESSAGE_LENGTH:
                raise ValueError(f"Message {i} too long: {len(msg.content)} > {MAX_MESSAGE_LENGTH}")

        return messages

    @field_validator("n")
    @classmethod
    def validate_n(cls, n: int | None) -> int:
        """Validate n parameter - only n=1 is supported."""
        if n is not None and n != 1:
            raise ValueError("Only n=1 is currently supported")
        return n or 1

    @field_validator("logit_bias")
    @classmethod
    def validate_logit_bias(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        """Validate logit_bias format if provided."""
        if v:
            # Validate that keys are strings (token IDs) and values are floats
            for token_id, bias in v.items():
                try:
                    int(token_id)  # Token ID should be parseable as int
                except ValueError:
                    raise ValueError(f"Invalid token ID in logit_bias: {token_id}") from None
                if not isinstance(bias, (int, float)):
                    raise ValueError(f"logit_bias value must be numeric: {bias}")
                if not -100.0 <= bias <= 100.0:
                    raise ValueError(f"logit_bias value must be between -100 and 100: {bias}")
        return v

    @field_validator("tools", "tool_choice")
    @classmethod
    def validate_tools(cls, v: Any, _info: Any) -> Any:
        """Validate tools - schema exists but not fully functional."""
        # Tools/tool_choice schema is accepted but not wired to llama.cpp
        # This would require significant implementation for function calling
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "qwen2.5-7b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "temperature": 0.7,
                "max_tokens": 512,
            }
        }
    )


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = None
    logprobs: dict[str, Any] | None = None


class ChatCompletionResponse(BaseModel):
    """Response from chat completions endpoint."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo
    system_fingerprint: str | None = None


# =============================================================================
# Streaming Chat Completions
# =============================================================================


class DeltaMessage(BaseModel):
    """Delta message for streaming responses."""

    role: Literal["system", "user", "assistant", "tool"] | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionChoiceChunk(BaseModel):
    """A single chunk choice for streaming."""

    index: int
    delta: DeltaMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = None
    logprobs: dict[str, Any] | None = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk response."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoiceChunk]
    usage: UsageInfo | None = None
    system_fingerprint: str | None = None


# =============================================================================
# Legacy Completions (for older clients)
# =============================================================================


class CompletionRequest(BaseModel):
    """Request body for legacy text completions."""

    model: str
    prompt: str | list[str]
    suffix: str | None = None
    max_tokens: int | None = 16
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    logprobs: int | None = None
    echo: bool | None = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    best_of: int | None = 1
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    seed: int | None = None


class CompletionChoice(BaseModel):
    """A single completion choice (legacy)."""

    text: str
    index: int
    logprobs: dict[str, Any] | None = None
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class CompletionResponse(BaseModel):
    """Response from legacy completions endpoint."""

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
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
    permission: list[dict[str, Any]] = Field(default_factory=list)
    root: str | None = None
    parent: str | None = None


class ModelList(BaseModel):
    """List of available models."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


# =============================================================================
# Embeddings
# =============================================================================


class EmbeddingRequest(BaseModel):
    """Request body for embeddings."""

    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None


class EmbeddingData(BaseModel):
    """Single embedding result."""

    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Response from embeddings endpoint."""

    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: UsageInfo


# =============================================================================
# Health & Status
# =============================================================================


class HealthStatus(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"] = "healthy"
    model_loaded: bool = False
    current_model: str | None = None
    version: str = "1.0.0"
    uptime_seconds: float = 0.0


class ServerInfo(BaseModel):
    """Server information."""

    name: str = "llm-manager"
    version: str = "1.0.0"
    models_dir: str
    default_model: str | None = None
    max_context_length: int = 32768
    supports_streaming: bool = True
    supports_function_calling: bool = True
