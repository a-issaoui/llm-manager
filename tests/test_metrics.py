"""
Tests for llm_manager/metrics.py - Metrics & Telemetry
"""

import time
from unittest.mock import MagicMock, Mock, patch

from llm_manager.metrics import (
    MetricsCollector,
    PerformanceStats,
    RequestMetrics,
    get_global_metrics,
)


class TestRequestMetrics:
    """Tests for RequestMetrics dataclass."""

    def test_creation(self):
        """Test creating RequestMetrics."""
        metric = RequestMetrics(
            timestamp=time.time(),
            duration_ms=500.0,
            input_tokens=100,
            output_tokens=200,
            model_name="test.gguf",
            success=True,
        )
        assert metric.model_name == "test.gguf"
        assert metric.input_tokens == 100
        assert metric.output_tokens == 200
        assert metric.success is True
        assert metric.error_type is None

    def test_creation_with_error(self):
        """Test creating RequestMetrics with error."""
        metric = RequestMetrics(
            timestamp=time.time(),
            duration_ms=100.0,
            input_tokens=50,
            output_tokens=0,
            model_name="test.gguf",
            success=False,
            error_type="TimeoutError",
        )
        assert metric.success is False
        assert metric.error_type == "TimeoutError"


class TestPerformanceStats:
    """Tests for PerformanceStats dataclass."""

    def test_defaults(self):
        """Test default values."""
        stats = PerformanceStats()
        assert stats.tokens_per_second == 0.0
        assert stats.total_requests == 0
        assert stats.success_rate == 1.0
        assert stats.vram_used_gb is None


class TestMetricsCollectorInit:
    """Tests for MetricsCollector initialization."""

    def test_default_init(self):
        """Test default initialization."""
        collector = MetricsCollector()
        assert collector.max_history == 1000
        assert len(collector._requests) == 0

    def test_custom_max_history(self):
        """Test initialization with custom max history."""
        collector = MetricsCollector(max_history=100)
        assert collector.max_history == 100


class TestMetricsCollectorRecord:
    """Tests for recording metrics."""

    def test_record_request(self):
        """Test recording a request."""
        collector = MetricsCollector()

        collector.record_request(
            model_name="test.gguf",
            input_tokens=100,
            output_tokens=200,
            duration_ms=500.0,
            success=True,
        )

        assert len(collector._requests) == 1
        assert collector._requests[0].model_name == "test.gguf"

    def test_record_multiple_requests(self):
        """Test recording multiple requests."""
        collector = MetricsCollector()

        for i in range(10):
            collector.record_request(
                model_name=f"model{i}.gguf",
                input_tokens=100 + i,
                output_tokens=200 + i,
                duration_ms=500.0 + i * 10,
            )

        assert len(collector._requests) == 10

    def test_record_with_callback(self):
        """Test that callbacks are called on record."""
        collector = MetricsCollector()
        callback_mock = Mock()

        collector.add_callback(callback_mock)
        collector.record_request(model_name="test.gguf")

        callback_mock.assert_called_once()
        assert isinstance(callback_mock.call_args[0][0], RequestMetrics)

    def test_callback_exception_handled(self):
        """Test that callback exceptions are handled gracefully."""
        collector = MetricsCollector()
        bad_callback = Mock(side_effect=Exception("Callback error"))

        collector.add_callback(bad_callback)

        # Should not raise
        collector.record_request(model_name="test.gguf")


class TestMetricsCollectorRecordCallback:
    """Tests for record_callback context manager."""

    def test_record_callback_success(self):
        """Test successful context manager usage."""
        collector = MetricsCollector()

        with collector.record_callback("test.gguf", input_tokens=100) as cb:
            # Simulate work
            time.sleep(0.01)
            cb.set_output_tokens(200)

        assert len(collector._requests) == 1
        assert collector._requests[0].input_tokens == 100
        assert collector._requests[0].output_tokens == 200
        assert collector._requests[0].success is True

    def test_record_callback_with_exception(self):
        """Test context manager with exception."""
        collector = MetricsCollector()

        try:
            with collector.record_callback("test.gguf", input_tokens=100) as cb:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert len(collector._requests) == 1
        assert collector._requests[0].success is False
        assert collector._requests[0].error_type == "ValueError"

    def test_record_callback_without_set_output(self):
        """Test context manager without setting output tokens."""
        collector = MetricsCollector()

        with collector.record_callback("test.gguf", input_tokens=100):
            pass

        assert collector._requests[0].output_tokens == 0


class TestMetricsCollectorGetStats:
    """Tests for get_stats method."""

    def test_empty_stats(self):
        """Test stats with no requests."""
        collector = MetricsCollector()
        stats = collector.get_stats()

        assert isinstance(stats, PerformanceStats)
        assert stats.total_requests == 0

    def test_basic_stats(self):
        """Test basic statistics calculation."""
        collector = MetricsCollector()

        collector.record_request(
            model_name="test.gguf", input_tokens=100, output_tokens=200, duration_ms=1000.0
        )

        stats = collector.get_stats()
        assert stats.total_requests == 1
        assert stats.total_tokens_generated == 200
        assert stats.total_tokens_input == 100

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        collector = MetricsCollector()

        # 8 successes, 2 failures
        for i in range(8):
            collector.record_request(model_name="test.gguf", success=True)
        for i in range(2):
            collector.record_request(model_name="test.gguf", success=False, error_type="Error")

        stats = collector.get_stats()
        assert stats.success_rate == 0.8
        assert stats.error_count == 2

    def test_latency_percentiles(self):
        """Test latency percentile calculations."""
        collector = MetricsCollector()

        # Add requests with varying durations
        for i in range(100):
            collector.record_request(
                model_name="test.gguf",
                duration_ms=float(i * 10),  # 0, 10, 20, ..., 990
            )

        stats = collector.get_stats()
        assert stats.latency_p50_ms == 500.0  # Median
        assert stats.latency_p95_ms == 950.0
        assert stats.latency_p99_ms == 990.0

    def test_window_filtering(self):
        """Test stats with time window filtering."""
        collector = MetricsCollector()

        # Add old request
        old_metric = RequestMetrics(
            timestamp=time.time() - 100,  # 100 seconds ago
            duration_ms=100.0,
            input_tokens=10,
            output_tokens=20,
            model_name="test.gguf",
            success=True,
        )
        collector._requests.append(old_metric)

        # Add recent requests
        for i in range(5):
            collector.record_request(model_name="test.gguf")

        # Get stats for last 10 seconds
        stats = collector.get_stats(window_seconds=10)
        assert stats.total_requests == 5  # Old one filtered out


class TestMetricsCollectorModelBreakdown:
    """Tests for model breakdown stats."""

    def test_empty_breakdown(self):
        """Test breakdown with no requests."""
        collector = MetricsCollector()
        breakdown = collector.get_model_breakdown()

        assert breakdown == {}

    def test_single_model_breakdown(self):
        """Test breakdown with single model."""
        collector = MetricsCollector()

        collector.record_request(model_name="test.gguf", output_tokens=100, duration_ms=1000.0)

        breakdown = collector.get_model_breakdown()
        assert "test.gguf" in breakdown
        assert breakdown["test.gguf"]["request_count"] == 1
        assert breakdown["test.gguf"]["total_tokens"] == 100

    def test_multiple_models_breakdown(self):
        """Test breakdown with multiple models."""
        collector = MetricsCollector()

        collector.record_request(model_name="model-a.gguf", output_tokens=100)
        collector.record_request(model_name="model-b.gguf", output_tokens=200)
        collector.record_request(model_name="model-a.gguf", output_tokens=150)

        breakdown = collector.get_model_breakdown()

        assert breakdown["model-a.gguf"]["request_count"] == 2
        assert breakdown["model-a.gguf"]["total_tokens"] == 250
        assert breakdown["model-b.gguf"]["request_count"] == 1
        assert breakdown["model-b.gguf"]["total_tokens"] == 200


class TestMetricsCollectorCallbacks:
    """Tests for callback management."""

    def test_add_callback(self):
        """Test adding callback."""
        collector = MetricsCollector()
        callback = Mock()

        collector.add_callback(callback)
        assert callback in collector._callbacks

    def test_remove_callback(self):
        """Test removing callback."""
        collector = MetricsCollector()
        callback = Mock()

        collector.add_callback(callback)
        collector.remove_callback(callback)

        assert callback not in collector._callbacks

    def test_remove_nonexistent_callback(self):
        """Test removing callback that wasn't added."""
        collector = MetricsCollector()
        callback = Mock()

        # Should not raise
        collector.remove_callback(callback)


class TestMetricsCollectorClear:
    """Tests for clearing metrics."""

    def test_clear(self):
        """Test clearing all metrics."""
        collector = MetricsCollector()

        for i in range(10):
            collector.record_request(model_name="test.gguf")

        collector.clear()

        assert len(collector._requests) == 0
        stats = collector.get_stats()
        assert stats.total_requests == 0


class TestMetricsCollectorExport:
    """Tests for export functionality."""

    def test_export(self):
        """Test exporting metrics."""
        collector = MetricsCollector()

        collector.record_request(
            model_name="test.gguf",
            input_tokens=100,
            output_tokens=200,
            duration_ms=500.0,
            success=True,
        )

        exported = collector.export()

        assert "requests" in exported
        assert len(exported["requests"]) == 1
        assert exported["requests"][0]["model_name"] == "test.gguf"
        assert exported["requests"][0]["input_tokens"] == 100


class TestMetricsCollectorSlowRequests:
    """Tests for slow request detection."""

    def test_get_slow_requests(self):
        """Test getting slow requests."""
        collector = MetricsCollector()

        # Fast request
        collector.record_request(model_name="test.gguf", duration_ms=100.0)
        # Slow request
        collector.record_request(model_name="test.gguf", duration_ms=2000.0)
        # Another fast request
        collector.record_request(model_name="test.gguf", duration_ms=150.0)

        slow = collector.get_slow_requests(threshold_ms=1000.0)

        assert len(slow) == 1
        assert slow[0].duration_ms == 2000.0


class TestMetricsCollectorErrorSummary:
    """Tests for error summary."""

    def test_error_summary(self):
        """Test error summary."""
        collector = MetricsCollector()

        collector.record_request(model_name="test.gguf", success=True)
        collector.record_request(model_name="test.gguf", success=False, error_type="TimeoutError")
        collector.record_request(model_name="test.gguf", success=False, error_type="TimeoutError")
        collector.record_request(
            model_name="test.gguf", success=False, error_type="ConnectionError"
        )

        summary = collector.get_error_summary()

        assert summary["TimeoutError"] == 2
        assert summary["ConnectionError"] == 1

    def test_empty_error_summary(self):
        """Test error summary with no errors."""
        collector = MetricsCollector()

        collector.record_request(model_name="test.gguf", success=True)

        summary = collector.get_error_summary()

        assert summary == {}


class TestMetricsCollectorThroughput:
    """Tests for throughput calculations."""

    def test_tokens_per_second(self):
        """Test tokens per second calculation."""
        collector = MetricsCollector()

        # Add request with known throughput
        collector.record_request(
            model_name="test.gguf",
            output_tokens=1000,
            duration_ms=10000.0,  # 10 seconds
        )

        stats = collector.get_stats()
        assert stats.tokens_per_second == 100.0  # 1000 tokens / 10 seconds

    def test_requests_per_minute(self):
        """Test requests per minute calculation."""
        collector = MetricsCollector()

        # Add requests quickly
        for i in range(10):
            collector.record_request(model_name="test.gguf")

        stats = collector.get_stats()
        # Should have some RPM value
        assert stats.requests_per_minute >= 0


class TestGlobalMetrics:
    """Tests for global metrics instance."""

    def test_get_global_metrics(self):
        """Test getting global metrics."""
        metrics1 = get_global_metrics()
        metrics2 = get_global_metrics()

        # Should return same instance
        assert metrics1 is metrics2

    def test_global_metrics_is_collector(self):
        """Test that global metrics is MetricsCollector."""
        metrics = get_global_metrics()
        assert isinstance(metrics, MetricsCollector)


class TestMetricsWithVRAM:
    """Tests for VRAM tracking (when torch available)."""

    def test_vram_stats_when_torch_available(self):
        """Test VRAM stats when torch is available."""
        collector = MetricsCollector()
        collector.record_request(model_name="test.gguf")

        # Mock torch.cuda
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 8_000_000_000  # 8GB
        mock_torch.cuda.get_device_properties.return_value.total_memory = 16_000_000_000  # 16GB

        with patch.dict("sys.modules", {"torch": mock_torch}):
            stats = collector.get_stats()
            assert stats.vram_used_gb == 8.0
            assert stats.vram_total_gb == 16.0

    def test_vram_none_when_torch_unavailable(self):
        """Test VRAM stats are None when torch unavailable."""
        collector = MetricsCollector()
        collector.record_request(model_name="test.gguf")

        # Ensure torch not available
        with patch.dict("sys.modules", {"torch": None}):
            stats = collector.get_stats()
            assert stats.vram_used_gb is None
            assert stats.vram_total_gb is None


class TestMetricsCollectorMaxHistory:
    """Tests for max history limit."""

    def test_max_history_respected(self):
        """Test that max_history limit is respected."""
        collector = MetricsCollector(max_history=5)

        # Add more than max_history requests
        for i in range(10):
            collector.record_request(model_name="test.gguf")

        # Should only keep last 5
        assert len(collector._requests) == 5
