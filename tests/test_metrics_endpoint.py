#!/usr/bin/env python3
"""Tests for Prometheus metrics endpoint."""

import pytest

pytest.importorskip("fastapi")

from llm_manager.server.routes.metrics import (
    MetricsCollector,
    get_metrics_collector,
    reset_metrics_collector,
)


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    def setup_method(self):
        """Reset collector before each test."""
        reset_metrics_collector()

    def test_increment_request(self):
        """Test incrementing request counter."""
        collector = MetricsCollector()
        collector.increment_request("test-model", "success")
        collector.increment_request("test-model", "success")
        collector.increment_request("test-model", "error")

        assert collector.requests_total.get("test-model:success") == 2
        assert collector.requests_total.get("test-model:error") == 1

    def test_increment_tokens(self):
        """Test incrementing token counters."""
        collector = MetricsCollector()
        collector.increment_tokens("test-model", 100, 50)
        collector.increment_tokens("test-model", 200, 100)

        assert collector.tokens_total.get("test-model:prompt") == 300
        assert collector.tokens_total.get("test-model:completion") == 150

    def test_record_duration(self):
        """Test recording request duration."""
        collector = MetricsCollector()
        collector.increment_request("test-model", "success")
        collector.record_duration("test-model", 1.5)

        assert "test-model" in collector.request_duration
        assert collector.request_duration["test-model"] == 1.5

    def test_set_model_loaded(self):
        """Test setting model loaded state."""
        collector = MetricsCollector()
        collector.set_model_loaded(True, 5.5)

        assert collector.model_loaded == 1
        assert collector.model_load_time == 5.5

    def test_export_prometheus_format(self):
        """Test Prometheus export format."""
        collector = MetricsCollector()
        collector.increment_request("qwen", "success")
        collector.increment_tokens("qwen", 100, 50)
        collector.set_model_loaded(True, 3.0)

        output = collector.export_prometheus()

        assert "llm_requests_total" in output
        assert "llm_tokens_total" in output
        assert "llm_model_loaded" in output
        assert 'model="qwen"' in output
        assert "llm_model_load_seconds" in output

    def test_multiple_models(self):
        """Test metrics with multiple models."""
        collector = MetricsCollector()
        collector.increment_request("model-a", "success")
        collector.increment_request("model-b", "success")
        collector.increment_request("model-a", "error")

        output = collector.export_prometheus()

        assert 'model="model-a"' in output
        assert 'model="model-b"' in output


class TestGlobalCollector:
    """Test global metrics collector instance."""

    def setup_method(self):
        """Reset before each test."""
        reset_metrics_collector()

    def test_get_metrics_collector_singleton(self):
        """Test that get_metrics_collector returns singleton."""
        c1 = get_metrics_collector()
        c2 = get_metrics_collector()

        assert c1 is c2

    def test_reset_metrics_collector(self):
        """Test resetting the collector."""
        c1 = get_metrics_collector()
        c1.increment_request("test", "success")

        reset_metrics_collector()

        c2 = get_metrics_collector()
        assert c1 is not c2
        assert len(c2.requests_total) == 0


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    """Test /metrics endpoint returns Prometheus format."""
    from llm_manager.server.routes.metrics import get_metrics_collector

    # Add some test data
    collector = get_metrics_collector()
    collector.increment_request("test-model", "success")
    collector.increment_tokens("test-model", 100, 50)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "llm_requests_total" in response.text
    assert "llm_tokens_total" in response.text
    assert "text/plain" in response.headers["content-type"]
