"""
Token estimation for LLM inputs.

Provides both fast heuristic estimation and accurate template-based counting.
"""

import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

from .exceptions import ValidationError
from .utils import LRUCache, is_base64_content, is_cjk, is_code

logger = logging.getLogger(__name__)

# Token estimation heuristics
TOKENS_PER_WORD_TEXT = 1.3
TOKENS_PER_WORD_CODE = 3.5
TOKENS_PER_CHAR_CJK = 1.5
TEMPLATE_OVERHEAD_PER_MESSAGE = 30
SPECIAL_TOKENS_BASE = 50
IMAGE_TOKEN_ESTIMATE = 1000


class ContentType(Enum):
    """Types of content for token estimation."""

    TEXT = "text"
    CODE = "code"
    CJK = "cjk"
    IMAGE = "image"
    DOCUMENT = "document"


class ConversationType(Enum):
    """Types of conversations for context optimization."""

    CHAT = "chat"
    CODING = "coding"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"
    TOOL = "tool"  # Tool/function calling


# Module-level constants for performance
_REASONING_KEYWORDS = frozenset(
    [
        "think",
        "analyze",
        "reason",
        "proof",
        "demonstrate",
        "explain",
        "derive",
        "chain of thought",
        "step by step",
    ]
)


@dataclass(slots=True)
class TokenEstimate:
    """
    Result of token estimation.

    Attributes:
        total_tokens: Total estimated tokens
        content_tokens: Tokens from message content
        template_tokens: Tokens from chat template
        special_tokens: Special tokens (BOS, EOS, etc)
        content_type: Detected content type
        is_accurate: Whether this is an accurate count or heuristic
        breakdown: Optional per-message breakdown
    """

    total_tokens: int
    content_tokens: int
    template_tokens: int
    special_tokens: int
    content_type: ContentType
    is_accurate: bool
    breakdown: dict[str, Any] | None = None

    def __repr__(self) -> str:
        accuracy = "accurate" if self.is_accurate else "estimated"
        return (
            f"TokenEstimate(total={self.total_tokens}, type={self.content_type.value}, {accuracy})"
        )


class TokenEstimator:
    """
    Estimates token counts for messages.

    Provides both fast heuristic estimation and accurate template-based
    counting when a tokenizer is available.
    """

    def __init__(self, cache_size: int = 1000, disk_cache_path: Path | None = None):
        """
        Initialize token estimator.

        Args:
            cache_size: Maximum size of in-memory cache
            disk_cache_path: Optional path to SQLite cache file for persistence
        """
        self._cache = LRUCache(maxsize=cache_size)
        self._disk_cache = None

        if disk_cache_path:
            from .cache import DiskCache

            self._disk_cache = DiskCache(disk_cache_path)

        self._stats = {
            "hits": 0,
            "misses": 0,
            "disk_hits": 0,
            "heuristic_calls": 0,
            "accurate_calls": 0,
        }

    def estimate_tokens(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> TokenEstimate:
        """Estimate tokens for a list of messages."""
        total_tokens = 0

        # ... logic ...

        return TokenEstimate(
            total_tokens=total_tokens,
            content_tokens=0, # Placeholder
            template_tokens=0, # Placeholder
            special_tokens=0, # Placeholder
            content_type=ContentType.TEXT, # Placeholder
            is_accurate=False # Placeholder
        )
    def estimate_heuristic(self, messages: list[dict[str, Any]]) -> TokenEstimate:
        """
        Fast heuristic token estimation.

        Uses word/character counts and content type detection to estimate
        token count without needing a tokenizer.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            TokenEstimate with is_accurate=False

        Examples:
            >>> estimator = TokenEstimator()
            >>> messages = [{"role": "user", "content": "Hello world"}]
            >>> estimate = estimator.estimate_heuristic(messages)
            >>> estimate.total_tokens
            82  # ~2 content + 30 template + 50 special
        """
        self._stats["heuristic_calls"] += 1

        # Generate cache key
        cache_key = self._generate_cache_key(messages, "heuristic")

        # Check memory cache first
        if cache_key in self._cache:
            self._stats["hits"] += 1
            return cast(TokenEstimate, self._cache[cache_key])

        # Check disk cache if available
        if self._disk_cache:
            disk_result = self._disk_cache.get(cache_key)
            if disk_result:
                # Reconstruct TokenEstimate from dict
                estimate = TokenEstimate(
                    total_tokens=disk_result["total_tokens"],
                    content_tokens=disk_result["content_tokens"],
                    template_tokens=disk_result["template_tokens"],
                    special_tokens=disk_result["special_tokens"],
                    content_type=ContentType(disk_result["content_type"]),
                    is_accurate=disk_result["is_accurate"],
                )
                # Promote to memory cache
                self._cache[cache_key] = estimate
                self._stats["disk_hits"] += 1
                return estimate

        self._stats["misses"] += 1

        content_tokens = 0
        detected_type = ContentType.TEXT

        for msg in messages:
            content = msg.get("content", "")

            # Handle multimodal content
            if not isinstance(content, str):
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") in ("image", "image_url"):
                                content_tokens += IMAGE_TOKEN_ESTIMATE
                                detected_type = ContentType.IMAGE
                            elif "text" in item:
                                content_tokens += self._estimate_text_tokens(item["text"])
                continue

            # Detect content type
            if is_base64_content(content):
                if "image" in content[:100].lower():
                    detected_type = ContentType.IMAGE
                    content_tokens += len(content) // 4
                else:
                    detected_type = ContentType.DOCUMENT
                    content_tokens += len(content) // 4
            elif is_code(content):
                detected_type = ContentType.CODE
                words = len(content.split())
                content_tokens += int(words * TOKENS_PER_WORD_CODE)
            elif is_cjk(content):
                detected_type = ContentType.CJK
                content_tokens += int(len(content) * TOKENS_PER_CHAR_CJK)
            else:
                words = len(content.split())
                content_tokens += int(words * TOKENS_PER_WORD_TEXT)

        # Add template overhead
        template_tokens = len(messages) * TEMPLATE_OVERHEAD_PER_MESSAGE
        special_tokens = SPECIAL_TOKENS_BASE

        total = content_tokens + template_tokens + special_tokens

        estimate = TokenEstimate(
            total_tokens=total,
            content_tokens=content_tokens,
            template_tokens=template_tokens,
            special_tokens=special_tokens,
            content_type=detected_type,
            is_accurate=False,
        )

        # Save to memory cache
        self._cache[cache_key] = estimate

        # Save to disk cache for persistence
        if self._disk_cache:
            self._disk_cache.set(
                cache_key,
                {
                    "total_tokens": estimate.total_tokens,
                    "content_tokens": estimate.content_tokens,
                    "template_tokens": estimate.template_tokens,
                    "special_tokens": estimate.special_tokens,
                    "content_type": estimate.content_type.value,
                    "is_accurate": estimate.is_accurate,
                },
            )

        return estimate

    def estimate_accurate(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Any,
        template: str | None = None,
    ) -> TokenEstimate:
        """
        Accurate token counting using actual tokenizer.

        Args:
            messages: List of message dicts
            tokenizer: Tokenizer instance with encode() method
            template: Optional chat template string

        Returns:
            TokenEstimate with is_accurate=True

        Raises:
            ValidationError: If tokenizer is invalid
        """
        self._stats["accurate_calls"] += 1

        if not hasattr(tokenizer, "encode"):
            raise ValidationError(
                "Tokenizer must have encode() method", {"tokenizer_type": type(tokenizer).__name__}
            )

        # Generate cache key
        cache_key = self._generate_cache_key(messages, f"accurate_{id(tokenizer)}")

        if cache_key in self._cache:
            self._stats["hits"] += 1
            return cast(TokenEstimate, self._cache[cache_key])

        self._stats["misses"] += 1

        try:
            # Format messages with template if provided
            if template:
                # This would use the actual template rendering
                # For now, just concatenate
                text = self._format_with_template(messages, template)
            else:
                # Simple concatenation
                text = " ".join(
                    msg.get("content", "")
                    for msg in messages
                    if isinstance(msg.get("content"), str)
                )

            # Tokenize
            tokens = tokenizer.encode(text)
            total_tokens = len(tokens)

            # Estimate breakdown
            content_tokens = int(total_tokens * 0.85)  # Rough estimate
            template_tokens = int(total_tokens * 0.10)
            special_tokens = total_tokens - content_tokens - template_tokens

            detected_type = self._detect_content_type(messages)

            estimate = TokenEstimate(
                total_tokens=total_tokens,
                content_tokens=content_tokens,
                template_tokens=template_tokens,
                special_tokens=special_tokens,
                content_type=detected_type,
                is_accurate=True,
            )

            self._cache[cache_key] = estimate
            return estimate

        except Exception as e:
            logger.warning(f"Accurate estimation failed: {e}, falling back to heuristic")
            return self.estimate_heuristic(messages)

    def _estimate_text_tokens(self, text: str) -> int:
        """Estimate tokens for a single text string."""
        if is_code(text):
            words = len(text.split())
            return int(words * TOKENS_PER_WORD_CODE)
        elif is_cjk(text):
            return int(len(text) * TOKENS_PER_CHAR_CJK)
        else:
            words = len(text.split())
            return int(words * TOKENS_PER_WORD_TEXT)

    def _detect_content_type(self, messages: list[dict[str, Any]]) -> ContentType:
        """Detect overall content type from messages."""
        code_count = 0
        cjk_count = 0
        image_count = 0

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                if is_base64_content(content):
                    image_count += 1
                elif is_code(content):
                    code_count += 1
                elif is_cjk(content):
                    cjk_count += 1

        total = len(messages)
        if image_count > 0:
            return ContentType.IMAGE
        if code_count > total * 0.5:
            return ContentType.CODE
        if cjk_count > total * 0.3:
            return ContentType.CJK

        return ContentType.TEXT

    def _format_with_template(self, messages: list[dict[str, Any]], template: str) -> str:
        """Format messages with chat template (simplified)."""
        # Simplified template rendering
        # In production, use proper Jinja2 rendering
        result = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            result += f"{role}: {content}\n"
        return result

    def _generate_cache_key(self, messages: list[dict[str, Any]], prefix: str) -> str:
        """Generate cache key from messages using fast hash."""
        # Create stable representation - sample for performance
        parts = []
        for msg in messages[:10]:  # Limit to first 10 messages
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"{role}:{content[:100]}")
            else:
                parts.append(f"{role}:multimodal")

        # Use stable hashlib.md5 for cross-process consistency
        msg_repr = "|".join(parts)
        hash_digest = hashlib.md5(msg_repr.encode(), usedforsecurity=False).hexdigest()[:8]
        return f"{prefix}_{hash_digest}"

    def get_stats(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with hits, misses, and call counts
        """
        return self._stats.copy()

    def clear_cache(self) -> None:
        """Clear estimation cache."""
        self._cache.clear()


def detect_conversation_type(messages: list[dict[str, Any]]) -> ConversationType:
    """
    Detect conversation type from messages.

    Args:
        messages: List of message dicts

    Returns:
        Detected conversation type

    Examples:
        >>> messages = [{"role": "user", "content": "def hello(): pass"}]
        >>> detect_conversation_type(messages)
        <ConversationType.CODING: 'coding'>
    """
    if not messages:
        return ConversationType.CHAT

    # Check for multimodal
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in ("image", "image_url"):
                    return ConversationType.MULTIMODAL
        if isinstance(content, str) and is_base64_content(content):
            return ConversationType.MULTIMODAL

    # Check for tool/function calling
    for msg in messages:
        role = msg.get("role", "")
        if role in ("tool", "function", "tool_result"):
            return ConversationType.TOOL
        # Also check for tool_calls in assistant messages
        if msg.get("tool_calls") or msg.get("function_call"):
            return ConversationType.TOOL

    # Check for coding
    code_messages = sum(
        1 for msg in messages if isinstance(msg.get("content"), str) and is_code(msg["content"])
    )

    if code_messages / len(messages) > 0.5:
        return ConversationType.CODING

    # Check for reasoning keywords (using module-level constant)
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in _REASONING_KEYWORDS):
                return ConversationType.REASONING

    return ConversationType.CHAT
