"""
Utility functions for the LLM Manager.
"""

import hashlib
import logging
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .exceptions import ValidationError

logger = logging.getLogger(__name__)

# Default minimum disk space (100 MB)
MIN_DISK_SPACE_MB = 100

# ==============================================================================
# Compiled Patterns for Performance
# ==============================================================================

_CODE_PATTERN = re.compile(
    r"\b(def|class|import|function|return|for|while|if|const|let|var|public|private)\b"
    r"|[{}\[\];()]|//|/\*|=>|\+=|-=|\*=|/="
)

_BASE64_PREFIX_PATTERN = re.compile(r"^data:(image|application|video)/")

# Constants for base64 detection heuristics
_BASE64_MIN_LENGTH = 5000
_BASE64_MAX_SPACE_RATIO = 0.05

# ==============================================================================
# Content Type Detection
# ==============================================================================


def is_code(text: str) -> bool:
    """
    Detect if text contains code.

    Args:
        text: Text to analyze

    Returns:
        True if text appears to contain code

    Examples:
        >>> is_code("def hello(): pass")
        True
        >>> is_code("Hello world")
        False
    """
    if not isinstance(text, str) or not text:
        return False
    return bool(_CODE_PATTERN.search(text))


def is_cjk(text: str) -> bool:
    """
    Detect if text contains CJK (Chinese, Japanese, Korean) characters.

    Uses early exit for performance - returns True on first CJK character found.
    Only samples first 100 characters to avoid scanning huge documents.

    Args:
        text: Text to analyze

    Returns:
        True if CJK characters detected

    Examples:
        >>> is_cjk("こんにちは")
        True
        >>> is_cjk("Hello")
        False
    """
    if not isinstance(text, str) or not text:
        return False

    # Sample first 100 chars for performance
    sample = text[:100]

    for char in sample:
        # Check for CJK Unified Ideographs, Hiragana, Katakana, and Hangul
        if "\u4e00" <= char <= "\u9fff":
            return True  # CJK Unified Ideographs
        if "\u3040" <= char <= "\u30ff":
            return True  # Hiragana & Katakana
        if "\uac00" <= char <= "\ud7af":
            return True  # Hangul

    return False


def is_base64_content(text: str) -> bool:
    """
    Detect if text is base64-encoded content (images, documents, etc).

    Args:
        text: Text to analyze

    Returns:
        True if text appears to be base64 content

    Examples:
        >>> is_base64_content("data:image/png;base64,iVBORw...")
        True
        >>> is_base64_content("Regular text")
        False
    """
    if not isinstance(text, str) or not text:
        return False

    # Check for data URL prefix
    if _BASE64_PREFIX_PATTERN.match(text):
        return True

    # Check for base64 indicator
    if "base64," in text and len(text) > 1000:
        return True

    # Very long strings with few spaces are likely base64
    if len(text) > _BASE64_MIN_LENGTH:
        space_ratio = text.count(" ") / len(text)
        if space_ratio < _BASE64_MAX_SPACE_RATIO:
            return True

    return False


# ==============================================================================
# File System Utilities
# ==============================================================================


def check_disk_space(path: str, required_mb: int = MIN_DISK_SPACE_MB) -> tuple[bool, float]:
    """
    Check if sufficient disk space is available.

    Args:
        path: Path to check (or its parent directory)
        required_mb: Minimum required space in MB

    Returns:
        Tuple of (has_space, available_mb)

    Raises:
        ResourceError: If disk space check fails
    """
    try:
        dir_path = Path(path)
        if not dir_path.exists():
            dir_path = dir_path.parent

        stat = os.statvfs(str(dir_path))
        available_mb = (stat.f_frsize * stat.f_bavail) / (1024 * 1024)

        return available_mb >= required_mb, available_mb

    except (OSError, AttributeError) as e:
        logger.warning(f"Disk space check failed: {e}")
        # Assume space is available if check fails
        return True, -1


def validate_model_path(path: Path) -> Path:
    """
    Validate that a model file exists and is a GGUF file.

    Args:
        path: Path to model file

    Returns:
        Resolved absolute path

    Raises:
        ValidationError: If path is invalid or file doesn't exist
    """
    if not isinstance(path, Path):
        path = Path(path)

    # Check existence
    if not path.exists():
        raise ValidationError(f"Model file not found: {path}", {"path": str(path)})

    # Check it's a file
    if not path.is_file():
        raise ValidationError(f"Path is not a file: {path}", {"path": str(path)})

    # Check GGUF magic bytes
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValidationError(
                    f"File is not a valid GGUF model: {path}",
                    {"path": str(path), "magic": magic.hex()},
                )
    except OSError as e:
        raise ValidationError(
            f"Cannot read model file: {path}", {"path": str(path), "error": str(e)}
        ) from e

    return path.resolve()


def compute_file_hash(path: Path, algorithm: str = "md5") -> str:
    """
    Compute hash of file for integrity checking.

    Args:
        path: Path to file
        algorithm: Hash algorithm (md5, sha256, etc)

    Returns:
        Hex digest of file hash

    Raises:
        OSError: If file cannot be read
    """
    hasher = hashlib.new(algorithm)

    with open(path, "rb") as f:
        # Read in chunks for memory efficiency
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


# ==============================================================================
# Validation Utilities
# ==============================================================================


def validate_temperature(temperature: float) -> float:
    """
    Validate temperature parameter.

    Args:
        temperature: Temperature value

    Returns:
        Validated temperature

    Raises:
        ValidationError: If temperature is invalid
    """
    if not isinstance(temperature, (int, float)):
        raise ValidationError(
            f"Temperature must be numeric, got {type(temperature).__name__}",
            {"temperature": temperature},
        )

    if not 0.0 <= temperature <= 2.0:
        raise ValidationError(
            f"Temperature must be between 0.0 and 2.0, got {temperature}",
            {"temperature": temperature},
        )

    return float(temperature)


def validate_max_tokens(max_tokens: int) -> int:
    """
    Validate max_tokens parameter.

    Args:
        max_tokens: Maximum tokens to generate

    Returns:
        Validated max_tokens

    Raises:
        ValidationError: If max_tokens is invalid
    """
    if not isinstance(max_tokens, int):
        raise ValidationError(
            f"max_tokens must be integer, got {type(max_tokens).__name__}",
            {"max_tokens": max_tokens},
        )

    if max_tokens < 1:
        raise ValidationError(
            f"max_tokens must be positive, got {max_tokens}", {"max_tokens": max_tokens}
        )

    if max_tokens > 100000:
        logger.warning(f"Very large max_tokens: {max_tokens}")

    return max_tokens


def validate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Validate messages list structure.

    Args:
        messages: List of message dicts

    Returns:
        Validated messages

    Raises:
        ValidationError: If messages structure is invalid
    """
    if not isinstance(messages, list):
        raise ValidationError(
            f"messages must be list, got {type(messages).__name__}",
            {"messages_type": type(messages).__name__},
        )

    if not messages:
        raise ValidationError("messages list cannot be empty")

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValidationError(
                f"Message {i} must be dict, got {type(msg).__name__}",
                {"index": i, "type": type(msg).__name__},
            )

        if "role" not in msg:
            raise ValidationError(
                f"Message {i} missing required 'role' field", {"index": i, "message": msg}
            )

        if "content" not in msg:
            raise ValidationError(
                f"Message {i} missing required 'content' field", {"index": i, "message": msg}
            )

        valid_roles = {"system", "user", "assistant", "tool"}
        if msg["role"] not in valid_roles:
            raise ValidationError(
                f"Message {i} has invalid role '{msg['role']}'",
                {"index": i, "role": msg["role"], "valid_roles": list(valid_roles)},
            )

    return messages


# ==============================================================================
# Performance Utilities
# ==============================================================================


class Timer:
    """
    Context manager for timing operations.

    Only logs if debug logging is enabled for performance.

    Examples:
        >>> with Timer("my_operation"):
        ...     do_something()
        [DEBUG] [PERF] my_operation: 123.45ms
    """

    def __init__(self, name: str):
        self.name = name
        self.start_time: float | None = None
        self.elapsed_ms: float | None = None

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self.start_time is not None:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000

            # Use lazy % formatting to avoid string formatting when disabled
            logger.debug("[PERF] %s: %.2fms", self.name, self.elapsed_ms)


class LRUCache(OrderedDict[Any, Any]):
    """
    O(1) LRU cache using OrderedDict.

    When max size is reached, oldest items are evicted.

    Examples:
        >>> cache = LRUCache(maxsize=2)
        >>> cache['a'] = 1
        >>> cache['b'] = 2
        >>> cache['c'] = 3  # Evicts 'a'
        >>> 'a' in cache
        False
    """

    def __init__(self, maxsize: int = 128):
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key: Any) -> Any:
        # Move to end (most recently used) - O(1)
        self.move_to_end(key)
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        # If key exists, move to end
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)

        # Evict oldest if over size - O(1)
        while len(self) > self.maxsize:
            self.popitem(last=False)

    def __contains__(self, key: Any) -> bool:
        # Don't move on contains check
        return OrderedDict.__contains__(self, key)
