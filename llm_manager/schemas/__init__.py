"""OpenAI-compatible API schemas."""

from .openai import (
    ChatCompletionChoice,
    ChatCompletionChoiceChunk,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    # Chat completions
    ChatMessage,
    CompletionChoice,
    # Legacy completions
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    EmbeddingData,
    # Embeddings
    EmbeddingRequest,
    EmbeddingResponse,
    # Errors
    ErrorResponse,
    # Models
    ModelInfo,
    ModelList,
    # Common
    StreamOptions,
    UsageInfo,
)

__all__ = [
    "ChatCompletionChoice",
    "ChatCompletionChoiceChunk",
    "ChatCompletionChunk",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "CompletionChoice",
    "CompletionRequest",
    "CompletionResponse",
    "DeltaMessage",
    "EmbeddingData",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ErrorResponse",
    "ModelInfo",
    "ModelList",
    "StreamOptions",
    "UsageInfo",
]
