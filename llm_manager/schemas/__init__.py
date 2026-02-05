"""OpenAI-compatible API schemas."""

from .openai import (
    # Chat completions
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionChoiceChunk,
    DeltaMessage,
    UsageInfo,
    # Legacy completions
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    # Models
    ModelInfo,
    ModelList,
    # Embeddings
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    # Errors
    ErrorResponse,
    # Common
    StreamOptions,
)

__all__ = [
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "ChatCompletionChoice",
    "ChatCompletionChoiceChunk",
    "DeltaMessage",
    "UsageInfo",
    "CompletionRequest",
    "CompletionResponse",
    "CompletionChoice",
    "ModelInfo",
    "ModelList",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingData",
    "ErrorResponse",
    "StreamOptions",
]
