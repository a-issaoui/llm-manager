"""
Metrics and telemetry for LLM operations.

Provides performance monitoring, cost tracking, and operational statistics
for agent workflows.
"""

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    timestamp: float
    duration_ms: float
    input_tokens: int
    output_tokens: int
    model_name: str
    success: bool
    error_type: str | None = None


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""

    # Throughput
    tokens_per_second: float = 0.0
    requests_per_minute: float = 0.0

    # Latency (milliseconds)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Token counts
    total_tokens_generated: int = 0
    total_tokens_input: int = 0
    total_requests: int = 0

    # Success rate
    success_rate: float = 1.0
    error_count: int = 0

    # VRAM (if available)
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None


class MetricsCollector:
    """
    Collects and aggregates metrics for LLM operations.

    Example:
        >>> metrics = MetricsCollector()
        >>>
        >>> # Record a request
        >>> with metrics.record_request("model.gguf", input_tokens=100):
        ...     response = manager.generate(messages)
        ...     return len(response["choices"][0]["message"]["content"])
        >>>
        >>> # Get stats
        >>> stats = metrics.get_stats()
        >>> print(f"Tokens/sec: {stats.tokens_per_second}")
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.

        Args:
            max_history: Maximum number of requests to keep in history
        """
        self.max_history = max_history
        self._requests: deque[RequestMetrics] = deque(maxlen=max_history)
        self._start_time: float | None = None

        # Callbacks for real-time monitoring
        self._callbacks: list[Callable[[RequestMetrics], None]] = []

    def record_request(
        self,
        model_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: float | None = None,
        success: bool = True,
        error_type: str | None = None,
    ) -> None:
        """
        Record metrics for a completed request.

        Args:
            model_name: Name of the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            duration_ms: Request duration in milliseconds
            success: Whether request succeeded
            error_type: Type of error if failed
        """
        metric = RequestMetrics(
            timestamp=time.time(),
            duration_ms=duration_ms or 0.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=model_name,
            success=success,
            error_type=error_type,
        )

        self._requests.append(metric)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.warning(f"Metrics callback failed: {e}")

    def record_callback(self, model_name: str, input_tokens: int = 0) -> Any:
        """
        Context manager/decorator for recording requests.

        Args:
            model_name: Name of the model
            input_tokens: Input token count

        Returns:
            Context manager that records duration
        """

        class MetricsContext:
            # pylint: disable=no-self-argument
            def __init__(ctx_self, collector: "MetricsCollector", model: str, tokens: int) -> None:  # noqa: N805
                ctx_self.collector = collector
                ctx_self.model = model
                ctx_self.input_tokens = tokens
                ctx_self.output_tokens = 0
                ctx_self.start_time: float = 0.0
                ctx_self.success = True
                ctx_self.error_type: str | None = None

            def __enter__(ctx_self) -> "MetricsContext":  # noqa: N805
                ctx_self.start_time = time.time()
                return ctx_self

            def __exit__(ctx_self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: N805
                duration_ms = (time.time() - ctx_self.start_time) * 1000
                ctx_self.success = exc_type is None
                ctx_self.error_type = exc_type.__name__ if exc_type else None

                ctx_self.collector.record_request(
                    model_name=ctx_self.model,
                    input_tokens=ctx_self.input_tokens,
                    output_tokens=ctx_self.output_tokens,
                    duration_ms=duration_ms,
                    success=ctx_self.success,
                    error_type=ctx_self.error_type,
                )

            def set_output_tokens(ctx_self, tokens: int) -> None:  # noqa: N805
                """Set output token count."""
                ctx_self.output_tokens = tokens

        return MetricsContext(self, model_name, input_tokens)

    def get_stats(self, window_seconds: float | None = None) -> PerformanceStats:
        """
        Get aggregated performance statistics.

        Args:
            window_seconds: Optional time window for stats (default: all history)

        Returns:
            PerformanceStats with aggregated metrics
        """
        if not self._requests:
            return PerformanceStats()

        # Filter by time window if specified
        requests = list(self._requests)
        if window_seconds:
            cutoff = time.time() - window_seconds
            requests = [r for r in requests if r.timestamp >= cutoff]

        if not requests:
            return PerformanceStats()

        # Calculate statistics
        durations = [r.duration_ms for r in requests]
        total_input = sum(r.input_tokens for r in requests)
        total_output = sum(r.output_tokens for r in requests)
        total_duration_sec = sum(durations) / 1000

        success_count = sum(1 for r in requests if r.success)
        error_count = len(requests) - success_count

        # Calculate throughput
        tokens_per_sec = total_output / total_duration_sec if total_duration_sec > 0 else 0

        # Time window for requests/minute
        if len(requests) > 1:
            time_span_min = (requests[-1].timestamp - requests[0].timestamp) / 60
            requests_per_min = len(requests) / time_span_min if time_span_min > 0 else 0
        else:
            requests_per_min = 0

        # Percentiles
        sorted_durations = sorted(durations)
        n = len(sorted_durations)

        def percentile(p: float) -> float:
            idx = int(n * p / 100)
            return sorted_durations[max(0, min(idx, n - 1))]

        # Try to get VRAM info
        vram_used = None
        vram_total = None
        try:
            import torch

            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1e9
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        except ImportError:
            pass

        return PerformanceStats(
            tokens_per_second=round(tokens_per_sec, 2),
            requests_per_minute=round(requests_per_min, 2),
            latency_p50_ms=round(percentile(50), 2),
            latency_p95_ms=round(percentile(95), 2),
            latency_p99_ms=round(percentile(99), 2),
            total_tokens_generated=total_output,
            total_tokens_input=total_input,
            total_requests=len(requests),
            success_rate=round(success_count / len(requests), 4),
            error_count=error_count,
            vram_used_gb=round(vram_used, 2) if vram_used else None,
            vram_total_gb=round(vram_total, 2) if vram_total else None,
        )

    def get_model_breakdown(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics broken down by model.

        Returns:
            Dict mapping model names to their stats
        """
        by_model: dict[str, list[RequestMetrics]] = {}

        for req in self._requests:
            if req.model_name not in by_model:
                by_model[req.model_name] = []
            by_model[req.model_name].append(req)

        breakdown = {}
        for model, requests in by_model.items():
            total_output = sum(r.output_tokens for r in requests)
            total_duration = sum(r.duration_ms for r in requests) / 1000

            breakdown[model] = {
                "request_count": len(requests),
                "total_tokens": total_output,
                "tokens_per_second": round(total_output / total_duration, 2)
                if total_duration > 0
                else 0,
                "avg_latency_ms": round(mean(r.duration_ms for r in requests), 2),
                "success_rate": round(sum(1 for r in requests if r.success) / len(requests), 4),
            }

        return breakdown

    def add_callback(self, callback: Callable[[RequestMetrics], None]) -> None:
        """
        Add a callback for real-time metrics monitoring.

        Args:
            callback: Function to call with each new metric
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[RequestMetrics], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def clear(self) -> None:
        """Clear all metrics history."""
        self._requests.clear()
        logger.debug("Metrics history cleared")

    def export(self) -> dict[str, Any]:
        """
        Export metrics to dictionary.

        Returns:
            Dictionary with all request data
        """
        return {
            "requests": [
                {
                    "timestamp": r.timestamp,
                    "duration_ms": r.duration_ms,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "model_name": r.model_name,
                    "success": r.success,
                    "error_type": r.error_type,
                }
                for r in self._requests
            ]
        }

    def get_slow_requests(self, threshold_ms: float = 1000.0) -> list[RequestMetrics]:
        """
        Get requests that exceeded latency threshold.

        Args:
            threshold_ms: Latency threshold in milliseconds

        Returns:
            List of slow requests
        """
        return [r for r in self._requests if r.duration_ms > threshold_ms]

    def get_error_summary(self) -> dict[str, int]:
        """
        Get summary of errors by type.

        Returns:
            Dict mapping error types to counts
        """
        errors: dict[str, int] = {}
        for r in self._requests:
            if not r.success and r.error_type:
                errors[r.error_type] = errors.get(r.error_type, 0) + 1
        return errors


# Global metrics instance for convenience
_global_metrics: MetricsCollector | None = None


def get_global_metrics() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics
