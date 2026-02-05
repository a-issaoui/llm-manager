#!/usr/bin/env python3
"""
Example 07: Metrics & Monitoring

Demonstrates:
- MetricsCollector for tracking performance
- Request recording
- Statistics and reporting
- Performance monitoring

No external dependencies required.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_manager.metrics import MetricsCollector, get_global_metrics
import time


def get_first_model():
    """Get first available model from registry."""
    from llm_manager import get_config
    models_file = get_config().models.get_registry_path()
    if models_file.exists():
        with open(models_file) as f:
            models = json.load(f)
        if models:
            return list(models.keys())[0]
    return "qwen2.5-7b"


def example_basic_metrics():
    """Basic metrics collection."""
    print("=" * 60)
    print("Metrics: Basic Collection")
    print("=" * 60)
    
    model_name = get_first_model()
    
    # Create metrics collector
    metrics = MetricsCollector()
    
    # Record some requests
    for i in range(5):
        metrics.record_request(
            model_name=model_name,
            input_tokens=100,
            output_tokens=50 + i * 10,
            duration_ms=100 + i * 20,
            success=True
        )
    
    # Get statistics
    stats = metrics.get_stats()
    
    print(f"Total requests: {stats.total_requests}")
    print(f"Input tokens: {stats.total_tokens_input}")
    print(f"Output tokens: {stats.total_tokens_generated}")
    print(f"Tokens/sec: {stats.tokens_per_second:.2f}")
    print(f"Success rate: {stats.success_rate:.1f}%")
    print(f"Latency P50: {stats.latency_p50_ms:.2f}ms")
    print(f"Latency P95: {stats.latency_p95_ms:.2f}ms")
    print(f"Latency P99: {stats.latency_p99_ms:.2f}ms\n")


def example_global_metrics():
    """Global singleton metrics."""
    print("=" * 60)
    print("Metrics: Global Singleton")
    print("=" * 60)
    
    # Get global instance
    metrics1 = get_global_metrics()
    metrics2 = get_global_metrics()
    
    # Same instance
    print(f"Same instance: {metrics1 is metrics2}")
    
    model_name = get_first_model()
    
    # Record on global
    metrics1.record_request(
        model_name=model_name,
        input_tokens=50,
        output_tokens=100,
        duration_ms=200,
        success=True
    )
    
    # Visible from both
    print(f"Requests via ref1: {metrics1.get_stats().total_requests}")
    print(f"Requests via ref2: {metrics2.get_stats().total_requests}")
    print(f"Same instance: {metrics1 is metrics2}\n")


def example_callback_timing():
    """Using callback for automatic timing."""
    print("=" * 60)
    print("Metrics: Callback Timing")
    print("=" * 60)
    
    model_name = get_first_model()
    metrics = MetricsCollector()
    
    # Simulate generation with automatic timing
    with metrics.record_callback(model_name, input_tokens=50) as cb:
        # Simulate work
        time.sleep(0.05)
        cb.set_output_tokens(150)
    
    stats = metrics.get_stats()
    print(f"Recorded request")
    print(f"Tokens/sec: {stats.tokens_per_second:.2f}\n")


def example_error_tracking():
    """Track errors and failures."""
    print("=" * 60)
    print("Metrics: Error Tracking")
    print("=" * 60)
    
    model_name = get_first_model()
    metrics = MetricsCollector()
    
    # Record successful requests
    for _ in range(8):
        metrics.record_request(
            model_name=model_name,
            input_tokens=100,
            output_tokens=50,
            duration_ms=100,
            success=True
        )
    
    # Record failed requests
    for _ in range(2):
        metrics.record_request(
            model_name=model_name,
            input_tokens=100,
            output_tokens=0,
            duration_ms=50,
            success=False
        )
    
    stats = metrics.get_stats()
    print(f"Success rate: {stats.success_rate:.1f}%")
    print(f"Expected: 80.0% (8 success / 10 total)\n")


def example_windowed_stats():
    """Windowed statistics (recent requests only)."""
    print("=" * 60)
    print("Metrics: Windowed Statistics")
    print("=" * 60)
    
    model_name = get_first_model()
    metrics = MetricsCollector()
    
    # Record 100 requests
    for i in range(100):
        metrics.record_request(
            model_name=model_name,
            input_tokens=100,
            output_tokens=50,
            duration_ms=100 + i,  # Increasing latency
            success=True
        )
    
    # Get stats for last 60 seconds only
    recent_stats = metrics.get_stats(window_seconds=60)
    all_stats = metrics.get_stats()
    
    print(f"All requests: {all_stats.total_requests}")
    print(f"Recent latency P50: {recent_stats.latency_p50_ms:.2f}ms")
    print(f"All latency P50: {all_stats.latency_p50_ms:.2f}ms")
    print("(Recent shows higher latency as we increased over time)")
    print("(Recent shows higher latency as we increased over time)\n")


def example_model_breakdown():
    """Breakdown by model."""
    print("=" * 60)
    print("Metrics: Model Breakdown")
    print("=" * 60)
    
    metrics = MetricsCollector()
    
    # Record for different models
    models = [get_first_model(), get_first_model(), "llama2-7b", "llama2-7b", "llama2-7b"]
    
    for model in models:
        metrics.record_request(
            model_name=model,
            input_tokens=100,
            output_tokens=50,
            duration_ms=100,
            success=True
        )
    
    breakdown = metrics.get_model_breakdown()
    
    print("Requests per model:")
    for model, count in breakdown.items():
        print(f"  {model}: {count}")
    print()


def example_slow_requests():
    """Identify slow requests."""
    print("=" * 60)
    print("Metrics: Slow Request Detection")
    print("=" * 60)
    
    model_name = get_first_model()
    metrics = MetricsCollector()
    
    # Mix of fast and slow requests
    durations = [50, 60, 2000, 70, 5000, 80, 3000, 90]  # ms
    
    for duration in durations:
        metrics.record_request(
            model_name=model_name,
            input_tokens=100,
            output_tokens=50,
            duration_ms=duration,
            success=True
        )
    
    # Get slow requests (threshold 1000ms)
    slow = metrics.get_slow_requests(threshold_ms=1000)
    
    print(f"Total requests: {metrics.get_stats().total_requests}")
    print(f"Slow requests (>1000ms): {len(slow)}")
    for req in slow:
        print(f"  - {req.duration_ms:.0f}ms")
    print()


def example_throughput():
    """Calculate throughput metrics."""
    print("=" * 60)
    print("Metrics: Throughput")
    print("=" * 60)
    
    model_name = get_first_model()
    metrics = MetricsCollector()
    
    # Simulate requests over time
    import random
    for _ in range(60):  # 60 requests
        metrics.record_request(
            model_name=model_name,
            input_tokens=random.randint(50, 200),
            output_tokens=random.randint(50, 200),
            duration_ms=random.randint(50, 200),
            success=True
        )
    
    stats = metrics.get_stats()
    
    print(f"Tokens per second: {stats.tokens_per_second:.2f}")
    print(f"Requests per minute: {stats.requests_per_minute:.2f}")
    print(f"Input tokens: {stats.total_tokens_input}")
    print(f"Output tokens: {stats.total_tokens_generated}\n")


def example_export():
    """Export metrics data."""
    print("=" * 60)
    print("Metrics: Export")
    print("=" * 60)
    
    model_name = get_first_model()
    metrics = MetricsCollector()
    
    # Record some data
    for i in range(5):
        metrics.record_request(
            model_name=model_name,
            input_tokens=100,
            output_tokens=50,
            duration_ms=100,
            success=True
        )
    
    # Export to dict
    data = metrics.export()
    
    print(f"Exported fields: {list(data.keys())}")
    print(f"Stats included: {'stats' in data}\n")


def example_callbacks():
    """Custom callbacks for metrics."""
    print("=" * 60)
    print("Metrics: Custom Callbacks")
    print("=" * 60)
    
    model_name = get_first_model()
    metrics = MetricsCollector()
    
    # Custom callback function
    def on_metric(recorded):
        print(f"Callback: Request recorded - {recorded['total_requests']} total")
    
    # Add callback
    metrics.add_callback(on_metric)
    
    # Record requests (callbacks will be called)
    metrics.record_request(
        model_name=model_name,
        input_tokens=100,
        output_tokens=50,
        duration_ms=100,
        success=True
    )
    
    metrics.record_request(
        model_name=model_name,
        input_tokens=100,
        output_tokens=50,
        duration_ms=100,
        success=True
    )
    
    # Remove callback
    metrics.remove_callback(on_metric)
    print("Callback removed\n")


def main():
    """Run metrics examples."""
    print("\n" + "=" * 60)
    print("LLM Manager - Metrics & Monitoring Examples")
    print("=" * 60 + "\n")
    
    example_basic_metrics()
    example_global_metrics()
    example_callback_timing()
    example_error_tracking()
    example_windowed_stats()
    example_model_breakdown()
    example_slow_requests()
    example_throughput()
    example_export()
    example_callbacks()


if __name__ == "__main__":
    main()
