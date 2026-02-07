"""Simple in-memory rate limiter for API endpoints."""

import logging
import time
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class RateLimitEntry:
    """Tracks request count and window start for a client."""

    count: int = 0
    window_start: float = field(default_factory=time.time)


class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window.

    Args:
        requests_per_minute: Maximum requests allowed per window
        window_seconds: Time window in seconds (default: 60)
    """

    def __init__(self, requests_per_minute: int = 60, window_seconds: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self._storage: dict[str, RateLimitEntry] = {}
        self._lock = Lock()
        self._cleanup_interval = 300  # Cleanup every 5 minutes
        self._last_cleanup = time.time()

    def is_allowed(self, client_id: str) -> tuple[bool, int | None]:
        """
        Check if request is allowed for client.

        Args:
            client_id: Unique identifier for the client (e.g., IP address)

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        now = time.time()

        with self._lock:
            # Periodic cleanup of old entries
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup_old_entries(now)

            entry = self._storage.get(client_id)

            if entry is None:
                # First request from this client
                self._storage[client_id] = RateLimitEntry(count=1, window_start=now)
                return True, None

            # Check if window has expired
            if now - entry.window_start > self.window_seconds:
                # Reset window
                entry.count = 1
                entry.window_start = now
                return True, None

            # Check if under limit
            if entry.count < self.requests_per_minute:
                entry.count += 1
                return True, None

            # Rate limit exceeded
            retry_after = int(self.window_seconds - (now - entry.window_start))
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return False, max(1, retry_after)

    def _cleanup_old_entries(self, now: float) -> None:
        """Remove expired entries to prevent memory growth."""
        expired = [
            key
            for key, entry in self._storage.items()
            if now - entry.window_start > self.window_seconds
        ]
        for key in expired:
            del self._storage[key]
        self._last_cleanup = now
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired rate limit entries")


# Global rate limiter instance (can be configured per-app)
_default_limiter: RateLimiter | None = None


def get_rate_limiter(requests_per_minute: int = 60) -> RateLimiter:
    """Get or create the default rate limiter."""
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = RateLimiter(requests_per_minute=requests_per_minute)
    return _default_limiter


def reset_rate_limiter() -> None:
    """Reset the default rate limiter (useful for testing)."""
    global _default_limiter
    _default_limiter = None
