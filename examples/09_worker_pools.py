#!/usr/bin/env python3
"""
Example 09: Worker Pools

Demonstrates:
- Subprocess worker management
- Worker pools for concurrent processing
- Async workers
- Error handling and recovery
"""

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_manager import LLMManager, get_config


def get_first_model():
    """Get first available model from registry."""
    models_file = get_config().models.get_registry_path()
    if models_file.exists():
        with open(models_file) as f:
            models = json.load(f)
        if models:
            return list(models.keys())[0]
    return None


def example_subprocess_mode():
    """Subprocess mode for crash safety."""
    print("=" * 60)
    print("Worker Pools: Subprocess Mode")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    try:
        # Subprocess mode isolates model in separate process
        # If model crashes, main process survives
        manager = LLMManager(
            models_dir=get_config().models.dir,
            use_subprocess=True,  # Enable subprocess isolation
            pool_size=4           # Number of worker processes
        )
        
        # Load and use model
        manager.load_model(model_name, n_gpu_layers=-1)
        response = manager.generate(
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=50
        )
        content = response["choices"][0]["message"]["content"]
        print(f"Response: {content[:60]}...")
        
        # Process is automatically managed
        manager.unload_model()
        print("Subprocess mode example complete\n")
        
    except Exception as e:
        print(f"Note: {e}\n")


def example_pool_configuration():
    """Worker pool configuration."""
    print("=" * 60)
    print("Worker Pools: Configuration")
    print("=" * 60)
    
    # Different pool configurations
    
    # Small pool - low memory, sequential-like
    manager_small = LLMManager(
        models_dir=get_config().models.dir,
        use_subprocess=True,
        pool_size=1  # Single worker
    )
    print("Small pool: pool_size=1")
    manager_small.cleanup()
    
    # Medium pool - balanced
    manager_medium = LLMManager(
        models_dir=get_config().models.dir,
        use_subprocess=True,
        pool_size=4  # 4 concurrent workers
    )
    print("Medium pool: pool_size=4")
    manager_medium.cleanup()
    
    # Large pool - high throughput
    manager_large = LLMManager(
        models_dir=get_config().models.dir,
        use_subprocess=True,
        pool_size=8  # 8 concurrent workers
    )
    print("Large pool: pool_size=8")
    manager_large.cleanup()
    
    print()


async def example_concurrent_requests():
    """Handle concurrent requests."""
    print("=" * 60)
    print("Worker Pools: Concurrent Requests")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    try:
        manager = LLMManager(
            models_dir=get_config().models.dir,
            use_subprocess=True,
            pool_size=4
        )
        
        await manager.load_model_async(model_name, n_gpu_layers=-1)
        
        user_requests = [
            "What is Python?",
            "What is AI?",
            "What is ML?",
            "What is DL?"
        ]
        
        async def process_one(request):
            return await manager.generate_async(
                messages=[{"role": "user", "content": request}],
                max_tokens=50
            )
        
        # Process all concurrently (up to pool_size at a time)
        results = await asyncio.gather(*[
            process_one(req) for req in user_requests
        ])
        
        for req, result in zip(user_requests, results):
            content = result["choices"][0]["message"]["content"][:40] if "choices" in result else "Error"
            print(f"  '{req}' -> {content}...")
        
        manager.unload_model()
        print()
        
    except Exception as e:
        print(f"Note: {e}\n")


def example_worker_lifecycle():
    """Worker lifecycle management."""
    print("=" * 60)
    print("Worker Pools: Lifecycle Management")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    try:
        manager = LLMManager(
            models_dir=get_config().models.dir,
            use_subprocess=True,
            pool_size=2
        )
        
        # 1. Workers start automatically with first request
        manager.load_model(model_name, n_gpu_layers=-1)
        print("Workers initialized")
        
        # 2. Workers process requests
        for i in range(3):
            response = manager.generate(
                messages=[{"role": "user", "content": f"Question {i}"}],
                max_tokens=30
            )
            content = response["choices"][0]["message"]["content"][:30]
            print(f"  Request {i+1}: {content}...")
        
        # 3. Workers can be restarted if needed
        manager.restart_workers()
        print("Workers restarted")
        
        # 4. Graceful shutdown
        manager.unload_model()  # Workers terminated cleanly
        print("Workers stopped")
        print()
        
    except Exception as e:
        print(f"Note: {e}\n")


def example_error_recovery():
    """Error handling and recovery."""
    print("=" * 60)
    print("Worker Pools: Error Recovery")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    try:
        manager = LLMManager(
            models_dir=get_config().models.dir,
            use_subprocess=True,
            pool_size=2
        )
        
        manager.load_model(model_name, n_gpu_layers=-1)
        
        print("Testing error recovery...")
        
        # Normal request
        try:
            response = manager.generate(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=30
            )
            content = response["choices"][0]["message"]["content"][:40]
            print(f"Normal request: {content}...")
        except Exception as e:
            print(f"Worker failed: {e}")
        
        print("Error recovery example complete\n")
        manager.unload_model()
        
    except Exception as e:
        print(f"Note: {e}\n")


def example_performance_tuning():
    """Performance tuning."""
    print("=" * 60)
    print("Worker Pools: Performance Tuning")
    print("=" * 60)
    
    print("Tips for optimal performance:")
    print()
    print("1. Pool Size:")
    print("   - CPU-bound: 1-2 workers per core")
    print("   - GPU-bound: 1 worker per GPU")
    print("   - Mixed: Balance based on workload")
    print()
    print("2. Memory:")
    print("   - Each worker loads model separately")
    print("   - Ensure enough RAM/VRAM for pool_size")
    print()
    print("3. Latency vs Throughput:")
    print("   - Small pool: Lower latency per request")
    print("   - Large pool: Higher total throughput")
    print()
    print("4. Context Switching:")
    print("   - Avoid frequent model switches")
    print("   - Batch same-model requests")
    print()


def example_monitoring():
    """Monitor worker health."""
    print("=" * 60)
    print("Worker Pools: Monitoring")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    try:
        from llm_manager import get_global_metrics
        
        manager = LLMManager(
            models_dir=get_config().models.dir,
            use_subprocess=True,
            pool_size=4
        )
        
        metrics = get_global_metrics()
        
        # Do a request to generate some metrics
        manager.load_model(model_name, n_gpu_layers=-1)
        manager.generate(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=20
        )
        
        # Check worker health
        stats = metrics.get_stats()
        
        print(f"Active requests: {stats.total_requests}")
        print(f"Success rate: {stats.success_rate:.1f}%")
        print(f"Avg latency: {stats.latency_p50_ms:.0f}ms")
        
        # Alert if issues detected
        if stats.success_rate < 95:
            print("WARNING: High error rate detected!")
        
        if stats.latency_p95_ms > 5000:
            print("WARNING: High latency detected!")
        
        manager.unload_model()
        print()
        
    except Exception as e:
        print(f"Note: {e}\n")


def main():
    """Run worker pool examples."""
    print("\n" + "=" * 60)
    print("LLM Manager - Worker Pools Examples")
    print("=" * 60 + "\n")
    
    example_subprocess_mode()
    example_pool_configuration()
    asyncio.run(example_concurrent_requests())
    example_worker_lifecycle()
    example_error_recovery()
    example_performance_tuning()
    example_monitoring()


if __name__ == "__main__":
    main()
