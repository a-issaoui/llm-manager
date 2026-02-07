#!/usr/bin/env python3
"""Tests for rate limiter."""

import time

from llm_manager.server.rate_limiter import RateLimiter, get_rate_limiter, reset_rate_limiter


class TestRateLimiter:
    """Test RateLimiter functionality."""

    def setup_method(self):
        """Reset rate limiter before each test."""
        reset_rate_limiter()

    def test_is_allowed_first_request(self):
        """Test first request is always allowed."""
        limiter = RateLimiter(requests_per_minute=10)

        allowed, retry_after = limiter.is_allowed("client1")

        assert allowed is True
        assert retry_after is None

    def test_is_allowed_within_limit(self):
        """Test requests within limit are allowed."""
        limiter = RateLimiter(requests_per_minute=5)

        for i in range(5):
            allowed, _ = limiter.is_allowed("client1")
            assert allowed is True

    def test_is_allowed_over_limit(self):
        """Test requests over limit are rejected."""
        limiter = RateLimiter(requests_per_minute=2)

        # Make 2 allowed requests
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        # Third request should be blocked
        allowed, retry_after = limiter.is_allowed("client1")

        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0

    def test_is_allowed_different_clients(self):
        """Test rate limiting is per-client."""
        limiter = RateLimiter(requests_per_minute=2)

        # Client 1 makes 2 requests
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        # Client 2 should still be allowed
        allowed, _ = limiter.is_allowed("client2")
        assert allowed is True

    def test_window_expiration(self):
        """Test rate limit resets after window expires."""
        limiter = RateLimiter(requests_per_minute=2, window_seconds=1)

        # Make 2 requests
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        # Third request blocked
        allowed, _ = limiter.is_allowed("client1")
        assert allowed is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        allowed, _ = limiter.is_allowed("client1")
        assert allowed is True

    def test_cleanup_old_entries(self):
        """Test cleanup of expired entries."""
        limiter = RateLimiter(requests_per_minute=10, window_seconds=1)
        # Set short cleanup interval for testing
        limiter._cleanup_interval = 0.1

        # Add some entries
        limiter.is_allowed("client1")
        limiter.is_allowed("client2")

        assert len(limiter._storage) == 2

        # Wait for expiration and cleanup interval
        time.sleep(1.2)

        # Trigger cleanup by making a new request
        limiter.is_allowed("client3")

        # Old entries should be cleaned up
        assert "client1" not in limiter._storage
        assert "client2" not in limiter._storage


class TestGlobalRateLimiter:
    """Test global rate limiter instance."""

    def setup_method(self):
        """Reset before each test."""
        reset_rate_limiter()

    def test_get_rate_limiter_singleton(self):
        """Test singleton pattern."""
        r1 = get_rate_limiter()
        r2 = get_rate_limiter()

        assert r1 is r2

    def test_reset_rate_limiter(self):
        """Test resetting the limiter."""
        r1 = get_rate_limiter()
        r1.is_allowed("test")

        reset_rate_limiter()

        r2 = get_rate_limiter()
        assert r1 is not r2
        assert len(r2._storage) == 0

    def test_get_rate_limiter_with_custom_rate(self):
        """Test creating limiter with custom rate."""
        reset_rate_limiter()
        limiter = get_rate_limiter(requests_per_minute=30)

        assert limiter.requests_per_minute == 30
