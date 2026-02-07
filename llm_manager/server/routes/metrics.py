"""Prometheus metrics endpoint for monitoring."""

import logging
import time
from dataclasses import dataclass, field
from threading import Lock

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from ..dependencies import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["metrics"])


@dataclass
class MetricsCollector:
    """Simple Prometheus-style metrics collector."""

    # Counters
    requests_total: dict[str, int] = field(default_factory=lambda: {})
    tokens_total: dict[str, int] = field(default_factory=lambda: {})

    # Gauges
    model_loaded: int = 0
    model_load_time: float = 0.0

    # Histograms (simplified as counters per bucket)
    request_duration: dict[str, float] = field(default_factory=lambda: {})

    _lock: Lock = field(default_factory=Lock)

    def increment_request(self, model: str, status: str = "success") -> None:
        """Increment request counter."""
        with self._lock:
            key = f"{model}:{status}"
            self.requests_total[key] = self.requests_total.get(key, 0) + 1

    def increment_tokens(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Increment token counters."""
        with self._lock:
            self.tokens_total[f"{model}:prompt"] = (
                self.tokens_total.get(f"{model}:prompt", 0) + prompt_tokens
            )
            self.tokens_total[f"{model}:completion"] = (
                self.tokens_total.get(f"{model}:completion", 0) + completion_tokens
            )

    def record_duration(self, model: str, duration: float) -> None:
        """Record request duration."""
        with self._lock:
            key = f"{model}"
            # Store as simple average for now
            current = self.request_duration.get(key, 0.0)
            count = self.requests_total.get(f"{key}:success", 0)
            if count > 0:
                self.request_duration[key] = (current * (count - 1) + duration) / count
            else:
                self.request_duration[key] = duration

    def set_model_loaded(self, loaded: bool, load_time: float = 0.0) -> None:
        """Set model loaded state."""
        with self._lock:
            self.model_loaded = 1 if loaded else 0
            self.model_load_time = load_time

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        # Header
        lines.append("# LLM Manager Metrics")
        lines.append(f"# Generated at {int(time.time())}")
        lines.append("")

        # Request counter
        lines.append("# HELP llm_requests_total Total requests by model and status")
        lines.append("# TYPE llm_requests_total counter")
        with self._lock:
            for key, value in self.requests_total.items():
                model, status = key.split(":")
                lines.append(f'llm_requests_total{{model="{model}",status="{status}"}} {value}')
        lines.append("")

        # Token counter
        lines.append("# HELP llm_tokens_total Total tokens by model and type")
        lines.append("# TYPE llm_tokens_total counter")
        with self._lock:
            for key, value in self.tokens_total.items():
                model, token_type = key.split(":")
                lines.append(f'llm_tokens_total{{model="{model}",type="{token_type}"}} {value}')
        lines.append("")

        # Model loaded gauge
        lines.append("# HELP llm_model_loaded Whether a model is currently loaded")
        lines.append("# TYPE llm_model_loaded gauge")
        with self._lock:
            lines.append(f"llm_model_loaded {self.model_loaded}")
        lines.append("")

        # Model load time gauge
        lines.append("# HELP llm_model_load_seconds Time to load the model")
        lines.append("# TYPE llm_model_load_seconds gauge")
        with self._lock:
            lines.append(f"llm_model_load_seconds {self.model_load_time:.3f}")
        lines.append("")

        # Average request duration
        lines.append("# HELP llm_request_duration_seconds Average request duration")
        lines.append("# TYPE llm_request_duration_seconds gauge")
        with self._lock:
            for model, duration in self.request_duration.items():
                lines.append(f'llm_request_duration_seconds{{model="{model}"}} {duration:.3f}')

        return "\n".join(lines)


# Global metrics collector
_metrics_collector: MetricsCollector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics_collector


def reset_metrics_collector() -> None:
    """Reset the global metrics collector (useful for testing)."""
    global _metrics_collector
    _metrics_collector = MetricsCollector()


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics(_: str = Depends(verify_api_key)) -> str:
    """Prometheus metrics endpoint."""
    collector = get_metrics_collector()
    return collector.export_prometheus()
