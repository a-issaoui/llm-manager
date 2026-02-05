"""
Context window management and dynamic resizing.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from .estimation import ConversationType, detect_conversation_type, TokenEstimator
from .exceptions import ContextError

logger = logging.getLogger(__name__)

# Context window configuration constants
CONTEXT_TIERS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
DEFAULT_RESPONSE_BUFFER_TOKENS = 2048
REASONING_RESPONSE_BUFFER_TOKENS = 4096
TOOL_RESPONSE_BUFFER_TOKENS = 1024
SAFETY_MARGIN_TOKENS = 512
UPSIZE_THRESHOLD = 0.9
DOWNSIZE_THRESHOLD = 0.5
MIN_DOWNSIZE_CONTEXT = 4096
MIN_REDUCTION_FACTOR = 0.75
HIGH_CONTEXT_WARNING_THRESHOLD = 0.8
BATCH_SIZE_SMALL_CTX = 1024
BATCH_SIZE_MEDIUM_CTX = 512
BATCH_SIZE_LARGE_CTX = 256


@dataclass(slots=True)
class ContextStats:
    """
    Context window statistics.

    Attributes:
        loaded: Whether model is loaded
        model_name: Name of loaded model
        allocated_context: Currently allocated context size
        used_tokens: Tokens currently used
        max_context: Maximum context supported by model
        utilization_percent: Percentage of allocated context used
        allocated_percent: Percentage of max context allocated
        can_grow_to: Additional tokens available
        conversation_type: Detected conversation type
        n_batch: Batch size setting
        n_ubatch: Micro-batch size setting
        flash_attn: Whether flash attention is enabled
    """
    loaded: bool
    model_name: Optional[str]
    allocated_context: int
    used_tokens: int
    max_context: int
    utilization_percent: float
    allocated_percent: float
    can_grow_to: int
    conversation_type: Optional[ConversationType]
    n_batch: Optional[int]
    n_ubatch: Optional[int]
    flash_attn: Optional[bool]

    def is_high_usage(self) -> bool:
        """Check if context usage is high."""
        return self.utilization_percent >= HIGH_CONTEXT_WARNING_THRESHOLD * 100

    def is_near_full(self) -> bool:
        """Check if context is near full."""
        return self.utilization_percent >= UPSIZE_THRESHOLD * 100

    def is_underutilized(self) -> bool:
        """Check if context is underutilized."""
        return self.utilization_percent <= DOWNSIZE_THRESHOLD * 100

    def __str__(self) -> str:
        """Human-readable string representation."""
        if not self.loaded:
            return "Context: No model loaded"

        return (
            f"Context: {self.used_tokens:,}/{self.allocated_context:,} tokens "
            f"({self.utilization_percent:.1f}% used, "
            f"{self.allocated_percent:.1f}% of max {self.max_context:,})"
        )


class ContextManager:
    """
    Manages context window sizing and optimization.

    Handles:
    - Initial context calculation based on message length
    - Dynamic resizing when context fills up or is underutilized
    - Conversation type detection for optimizations
    - Batch size calculation based on context size
    """

    def __init__(
        self,
        estimator: Optional[TokenEstimator] = None,
        resize_cooldown_seconds: float = 60.0
    ):
        """
        Initialize context manager.

        Args:
            estimator: Token estimator instance
            resize_cooldown_seconds: Minimum time between resizes
        """
        self.estimator = estimator or TokenEstimator()
        self.resize_cooldown = resize_cooldown_seconds
        self._last_resize_time = 0.0
        self._resize_count = 0

    def calculate_context_size(
        self,
        messages: List[Dict[str, Any]],
        max_context: int,
        max_tokens: int = 256,
        use_heuristic: bool = True
    ) -> int:
        """
        Calculate optimal context size for messages.

        Args:
            messages: List of message dicts
            max_context: Maximum context supported by model
            max_tokens: Tokens to reserve for response
            use_heuristic: Use fast heuristic (True) or accurate count (False)

        Returns:
            Recommended context size

        Examples:
            >>> manager = ContextManager()
            >>> messages = [{"role": "user", "content": "Hello" * 100}]
            >>> context = manager.calculate_context_size(messages, 32768)
            >>> context >= 2048
            True
        """
        # Estimate input tokens
        if use_heuristic:
            estimate = self.estimator.estimate_heuristic(messages)
        else:
            # Accurate estimation would require a tokenizer
            # Log this and fall back to heuristic
            logger.debug(
                "Accurate token estimation requested but no tokenizer available, "
                "using heuristic"
            )
            estimate = self.estimator.estimate_heuristic(messages)

        input_tokens = estimate.total_tokens

        # Detect conversation type
        conv_type = detect_conversation_type(messages)

        # Determine response buffer based on conversation type
        if conv_type == ConversationType.REASONING:
            response_buffer = REASONING_RESPONSE_BUFFER_TOKENS
        elif conv_type == ConversationType.CODING:
            response_buffer = max(max_tokens, DEFAULT_RESPONSE_BUFFER_TOKENS)
        elif conv_type == ConversationType.MULTIMODAL:
            # Multimodal may have image outputs requiring more buffer
            response_buffer = max(max_tokens, DEFAULT_RESPONSE_BUFFER_TOKENS * 2)
        elif conv_type == ConversationType.TOOL:
            # Tool calling needs extra buffer for JSON responses
            response_buffer = max(max_tokens, TOOL_RESPONSE_BUFFER_TOKENS)
        else:
            response_buffer = max(max_tokens, DEFAULT_RESPONSE_BUFFER_TOKENS)

        # Calculate total needed
        total_needed = input_tokens + response_buffer + SAFETY_MARGIN_TOKENS

        # Find appropriate tier
        context_size = self._find_tier(total_needed, max_context)

        logger.debug(
            f"Context calculation: input={input_tokens}, "
            f"buffer={response_buffer}, total={total_needed}, "
            f"selected={context_size}, type={conv_type.value}"
        )

        return context_size

    def _find_tier(self, required: int, max_context: int) -> int:
        """
        Find appropriate context tier for required tokens.

        Args:
            required: Required token count
            max_context: Maximum allowed context

        Returns:
            Context size from CONTEXT_TIERS
        """
        # Find smallest tier that fits
        for tier in CONTEXT_TIERS:
            if tier >= required and tier <= max_context:
                return tier

        # If all tiers too small, use max
        return min(max_context, CONTEXT_TIERS[-1])

    def should_resize(
        self,
        current_used: int,
        current_allocated: int,
        max_context: int
    ) -> Tuple[bool, Optional[int]]:
        """
        Determine if context should be resized.

        Args:
            current_used: Currently used tokens
            current_allocated: Currently allocated context
            max_context: Maximum context supported

        Returns:
            Tuple of (should_resize, new_size or None)

        Examples:
            >>> manager = ContextManager()
            >>> should, new_size = manager.should_resize(3800, 4096, 32768)
            >>> should
            True
            >>> new_size > 4096
            True
        """
        # Check cooldown
        if time.time() - self._last_resize_time < self.resize_cooldown:
            return False, None

        if current_allocated == 0:
            return False, None

        utilization = current_used / current_allocated

        # Should upsize?
        if utilization >= UPSIZE_THRESHOLD:
            new_size = self._calculate_upsize(
                current_allocated,
                current_used,
                max_context
            )
            if new_size > current_allocated:
                logger.info(
                    f"Context upsize recommended: {current_allocated} -> {new_size} "
                    f"(utilization: {utilization:.1%})"
                )
                return True, new_size

        # Should downsize?
        elif utilization <= DOWNSIZE_THRESHOLD:
            new_size = self._calculate_downsize(
                current_allocated,
                current_used
            )
            if new_size < current_allocated and new_size >= MIN_DOWNSIZE_CONTEXT:
                logger.info(
                    f"Context downsize recommended: {current_allocated} -> {new_size} "
                    f"(utilization: {utilization:.1%})"
                )
                return True, new_size

        return False, None

    def _calculate_upsize(
        self,
        current: int,
        used: int,
        max_context: int
    ) -> int:
        """Calculate new size when upsizing."""
        # Double the current size
        target = current * 2

        # Find next tier
        new_size = self._find_tier(target, max_context)

        # Ensure it's actually larger
        if new_size <= current:
            # Try next tier up
            for tier in CONTEXT_TIERS:
                if tier > current and tier <= max_context:
                    new_size = tier
                    break

        return min(new_size, max_context)

    def _calculate_downsize(
        self,
        current: int,
        used: int
    ) -> int:
        """Calculate new size when downsizing."""
        # Target size with headroom
        target = int(used * (1.0 + (1.0 - DOWNSIZE_THRESHOLD)))

        # Apply minimum reduction factor
        min_size = int(current * MIN_REDUCTION_FACTOR)
        target = max(target, min_size)

        # Find appropriate tier
        new_size = self._find_tier(target, current)

        # Don't downsize below minimum
        return max(new_size, MIN_DOWNSIZE_CONTEXT)

    def mark_resized(self) -> None:
        """Mark that a resize occurred (for cooldown tracking)."""
        self._last_resize_time = time.time()
        self._resize_count += 1

    def calculate_batch_size(
        self,
        context_size: int,
        vram_gb: float = 0.0
    ) -> Tuple[int, int]:
        """
        Calculate optimal batch sizes.

        Args:
            context_size: Context window size
            vram_gb: Available VRAM in GB (0 for CPU)

        Returns:
            Tuple of (n_batch, n_ubatch)

        Examples:
            >>> manager = ContextManager()
            >>> n_batch, n_ubatch = manager.calculate_batch_size(4096)
            >>> n_batch >= n_ubatch
            True
        """
        # Determine base batch size by context tier
        if context_size <= 8192:
            base_batch = BATCH_SIZE_SMALL_CTX
        elif context_size <= 32768:
            base_batch = BATCH_SIZE_MEDIUM_CTX
        else:
            base_batch = BATCH_SIZE_LARGE_CTX

        # Adjust for VRAM
        if vram_gb > 0:
            # More VRAM allows larger batches
            if vram_gb >= 24:
                base_batch = min(base_batch * 2, 2048)
            elif vram_gb >= 12:
                base_batch = min(int(base_batch * 1.5), 1536)
        else:
            # CPU mode - use smaller batches
            base_batch = min(base_batch, 512)

        # n_ubatch is typically n_batch / 2 or n_batch
        n_batch = base_batch
        n_ubatch = base_batch // 2 if base_batch > 512 else base_batch

        logger.debug(
            f"Batch sizes: n_batch={n_batch}, n_ubatch={n_ubatch} "
            f"(ctx={context_size}, vram={vram_gb:.1f}GB)"
        )

        return n_batch, n_ubatch

    def get_stats(self) -> Dict[str, Any]:
        """
        Get context manager statistics.

        Returns:
            Dict with stats
        """
        return {
            "resize_count": self._resize_count,
            "last_resize_time": self._last_resize_time,
            "cooldown_remaining": max(
                0,
                self.resize_cooldown - (time.time() - self._last_resize_time)
            ),
        }

    def validate_context_size(
        self,
        requested: int,
        max_context: int
    ) -> int:
        """
        Validate and adjust context size request.

        Args:
            requested: Requested context size
            max_context: Maximum allowed context

        Returns:
            Validated context size

        Raises:
            ContextError: If context size is invalid
        """
        if requested < 512:
            raise ContextError(
                f"Context size too small: {requested} (min 512)",
                {"requested": requested, "minimum": 512}
            )

        if requested > max_context:
            logger.warning(
                f"Requested context {requested} exceeds max {max_context}, "
                f"using max"
            )
            return max_context

        # Round to nearest tier
        for tier in CONTEXT_TIERS:
            if tier >= requested:
                if tier != requested:
                    logger.debug(
                        f"Rounded context from {requested} to tier {tier}"
                    )
                return tier

        return max_context
