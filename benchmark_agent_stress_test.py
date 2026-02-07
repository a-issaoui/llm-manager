#!/usr/bin/env python3
"""
Agent Stress Test Benchmark for LLM Manager HTTP Server

This benchmark simulates real-world agent usage patterns:
- All models in models folder
- Context handling with long conversations
- Thinking/reasoning tasks
- Tool calling functionality
- Model swapping like agents do
- Concurrent requests
- Stability and correctness verification

Usage:
    python benchmark_agent_stress_test.py [--duration 300] [--concurrency 5]
"""

import argparse
import asyncio
import json
import logging
import random
import statistics
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import requests

from llm_manager.server import LLMServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8888
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Available models and their best use cases
MODELS_CONFIG = {
    "SmolLM-1.7B-Instruct-Q4_K_M.gguf": {
        "tasks": ["general", "summarization"],
        "max_tokens": 512,
        "temperature": 0.7,
        "n_ctx": 2048,  # Conservative context
    },
    "Qwen2.5-3b-instruct-q4_k_m.gguf": {
        "tasks": ["general", "chat", "instruction"],
        "max_tokens": 1024,
        "temperature": 0.7,
        "n_ctx": 4096,
    },
    "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf": {
        "tasks": ["code", "technical"],
        "max_tokens": 1024,
        "temperature": 0.3,
        "n_ctx": 4096,
    },
    "DeepSeek-R1-Distill-Llama-3B.Q5_K_M.gguf": {
        "tasks": ["reasoning", "math", "logic"],
        "max_tokens": 1024,
        "temperature": 0.5,
        "n_ctx": 4096,
    },
    "Reason-With-Choice-3B.Q4_K_M.gguf": {
        "tasks": ["reasoning", "classification"],
        "max_tokens": 512,
        "temperature": 0.3,
        "n_ctx": 4096,
    },
    "Nanbeige_Nanbeige4-3B-Thinking-2511-Q4_K_M.gguf": {
        "tasks": ["thinking", "analysis"],
        "max_tokens": 1024,
        "temperature": 0.5,
        "n_ctx": 4096,
    },
    # Embedding model is loaded but not used for chat in this benchmark
}

# Test scenarios
REASONING_TASKS = [
    {
        "name": "math_word_problem",
        "prompt": "A train travels 120 km in 2 hours. If it maintains the same speed, how far will it travel in 5 hours? Think step by step.",
        "check_contains": ["300", "km"],
    },
    {
        "name": "logical_deduction",
        "prompt": "If all cats are mammals, and some mammals are pets, can we conclude that some cats are pets? Explain your reasoning.",
        "check_contains": ["yes", "cannot", "conclude"],
    },
    {
        "name": "pattern_recognition",
        "prompt": "What comes next in this sequence: 2, 6, 12, 20, 30, ? Explain the pattern.",
        "check_contains": ["42", "pattern"],
    },]

CODE_TASKS = [
    {
        "name": "python_function",
        "prompt": "Write a Python function that takes a list of integers and returns the sum of all even numbers.",
        "check_contains": ["def", "even", "sum"],
    },
    {
        "name": "code_review",
        "prompt": "Review this code and suggest improvements:\n```python\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n```",
        "check_contains": ["memoization", "optimization", "inefficient"],
    },
]

TOOL_TASKS = [
    {
        "name": "weather_query",
        "prompt": "What's the weather like in Tokyo?",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
    },
    {
        "name": "calculator",
        "prompt": "Calculate 123 * 456 + 789",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Calculate mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ],
    },
]

CONVERSATION_SCENARIOS = [
    {
        "name": "customer_support",
        "turns": [
            {"role": "system", "content": "You are a helpful customer support agent."},
            {"role": "user", "content": "My order hasn't arrived yet."},
            {"role": "assistant", "content": "I'd be happy to help you track your order. Could you provide your order number?"},
            {"role": "user", "content": "It's ORD-12345. I ordered it 2 weeks ago."},
            {"role": "assistant", "content": "Thank you. Let me check the status of order ORD-12345..."},
            {"role": "user", "content": "Can you tell me when it will arrive?"},
        ]
    },
    {
        "name": "coding_assistant",
        "turns": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "How do I read a file in Python?"},
            {"role": "assistant", "content": "You can use the open() function:\n```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\n```"},
            {"role": "user", "content": "What if the file is very large?"},
            {"role": "assistant", "content": "For large files, read line by line:\n```python\nwith open('file.txt', 'r') as f:\n    for line in f:\n        process(line)\n```"},
            {"role": "user", "content": "Can you show me how to handle encoding errors?"},
        ]
    },
]


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    model: str
    success: bool
    latency_ms: float
    tokens_in: int
    tokens_out: int
    error: str | None = None
    response_preview: str = ""


@dataclass
class BenchmarkStats:
    """Aggregated benchmark statistics."""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    latencies: list[float] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    model_usage: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    
    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0
    
    @property
    def p95_latency(self) -> float:
        return statistics.quantiles(self.latencies, n=20)[18] if len(self.latencies) >= 20 else self.avg_latency
    
    @property
    def throughput_tps(self) -> float:
        total_time = sum(self.latencies) / 1000 if self.latencies else 1
        return self.tokens_out / total_time if total_time > 0 else 0


class LLMManagerBenchmark:
    """Benchmark harness for LLM Manager HTTP server."""
    
    def __init__(self, models_dir: Path, duration: int = 300, concurrency: int = 5):
        self.models_dir = models_dir
        self.duration = duration
        self.concurrency = concurrency
        self.server: LLMServer | None = None
        self.server_thread: threading.Thread | None = None
        self.results: list[TestResult] = []
        self.stop_event = threading.Event()
        
    def start_server(self) -> bool:
        """Start the HTTP server."""
        logger.info(f"Starting server on {SERVER_HOST}:{SERVER_PORT}")
        
        self.server = LLMServer(
            models_dir=str(self.models_dir),
            host=SERVER_HOST,
            port=SERVER_PORT,
            log_level="warning",  # Reduce noise during benchmark
        )
        
        self.server_thread = threading.Thread(target=self.server.start, daemon=True)
        self.server_thread.start()
        
        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < 60:
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=2)
                if response.status_code == 200:
                    logger.info("Server is ready!")
                    return True
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)
        
        logger.error("Server failed to start within timeout")
        return False
    
    def stop_server(self):
        """Stop the HTTP server."""
        if self.server:
            logger.info("Stopping server...")
            self.server.stop()
            if self.server_thread:
                self.server_thread.join(timeout=10)
    
    async def wait_for_server(self) -> bool:
        """Wait for server to be ready (async version)."""
        start_time = time.time()
        while time.time() - start_time < 60:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{BASE_URL}/health", timeout=2) as resp:
                        if resp.status == 200:
                            return True
            except Exception:
                await asyncio.sleep(0.5)
        return False
    
    async def make_request(
        self,
        session: aiohttp.ClientSession,
        messages: list[dict],
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        test_name: str = "unknown"
    ) -> TestResult:
        """Make a single chat completion request."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        start_time = time.time()
        try:
            async with session.post(
                f"{BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=120
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return TestResult(
                        test_name=test_name,
                        model=model,
                        success=False,
                        latency_ms=(time.time() - start_time) * 1000,
                        tokens_in=0,
                        tokens_out=0,
                        error=f"HTTP {resp.status}: {error_text[:200]}"
                    )
                
                data = await resp.json()
                latency_ms = (time.time() - start_time) * 1000
                
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                
                usage = data.get("usage", {})
                
                return TestResult(
                    test_name=test_name,
                    model=model,
                    success=True,
                    latency_ms=latency_ms,
                    tokens_in=usage.get("prompt_tokens", 0),
                    tokens_out=usage.get("completion_tokens", 0),
                    response_preview=content[:100] if content else ""
                )
                
        except asyncio.TimeoutError:
            return TestResult(
                test_name=test_name,
                model=model,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_in=0,
                tokens_out=0,
                error="Request timeout"
            )
        except Exception as e:
            return TestResult(
                test_name=test_name,
                model=model,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_in=0,
                tokens_out=0,
                error=str(e)[:200]
            )
    
    async def test_model_with_task(
        self,
        session: aiohttp.ClientSession,
        model: str,
        task: dict,
        task_type: str
    ) -> TestResult:
        """Test a model with a specific task."""
        messages = [{"role": "user", "content": task["prompt"]}]
        
        config = MODELS_CONFIG.get(model, {})
        
        result = await self.make_request(
            session=session,
            messages=messages,
            model=model,
            max_tokens=config.get("max_tokens", 256),
            temperature=config.get("temperature", 0.7),
            tools=task.get("tools"),
            test_name=f"{task_type}/{task['name']}"
        )
        
        # Check if response contains expected content
        if result.success and "check_contains" in task:
            found_any = any(
                check.lower() in result.response_preview.lower()
                for check in task["check_contains"]
            )
            if not found_any:
                result.success = False
                result.error = f"Response missing expected content. Got: {result.response_preview[:50]}..."
        
        return result
    
    async def test_conversation_context(
        self,
        session: aiohttp.ClientSession,
        model: str,
        scenario: dict
    ) -> list[TestResult]:
        """Test conversation with context."""
        results = []
        turns = scenario["turns"]
        
        # Progressive conversation
        for i in range(1, len(turns) + 1):
            messages = turns[:i]
            
            result = await self.make_request(
                session=session,
                messages=messages,
                model=model,
                max_tokens=512,
                temperature=0.7,
                test_name=f"context/{scenario['name']}_turn_{i}"
            )
            results.append(result)
            
            # Don't overwhelm the server
            await asyncio.sleep(0.5)
        
        return results
    
    async def run_single_worker(self, worker_id: int):
        """Run a single worker that continuously makes requests."""
        async with aiohttp.ClientSession() as session:
            # Filter to chat-capable models only
            models = [m for m in MODELS_CONFIG.keys() if "embedding" not in MODELS_CONFIG[m].get("tasks", [])]
            
            while not self.stop_event.is_set():
                results = []
                
                # Pick random model
                model = random.choice(models)
                
                # Random task type based on model capabilities
                config = MODELS_CONFIG[model]
                tasks = config.get("tasks", ["general"])
                
                if "reasoning" in tasks or "math" in tasks or "thinking" in tasks:
                    # Reasoning task
                    task = random.choice(REASONING_TASKS)
                    result = await self.test_model_with_task(session, model, task, "reasoning")
                    results.append(result)
                    
                elif "code" in tasks or "technical" in tasks:
                    # Code task
                    task = random.choice(CODE_TASKS)
                    result = await self.test_model_with_task(session, model, task, "code")
                    results.append(result)
                    
                elif "general" in tasks or "chat" in tasks:
                    # Conversation context test
                    scenario = random.choice(CONVERSATION_SCENARIOS)
                    ctx_results = await self.test_conversation_context(session, model, scenario)
                    results.extend(ctx_results)
                    
                elif "embedding" in tasks:
                    # Skip embedding model for chat tests
                    pass
                
                # Tool calling test (can be done with any capable model)
                if random.random() < 0.3:  # 30% chance
                    tool_task = random.choice(TOOL_TASKS)
                    result = await self.test_model_with_task(
                        session, model, tool_task, "tools"
                    )
                    results.append(result)
                
                # Model swap test - load different model (only one worker does this)
                if worker_id == 0 and random.random() < 0.15:  # 15% chance, only worker 0
                    new_model = random.choice([m for m in models if m in MODELS_CONFIG])
                    try:
                        config = MODELS_CONFIG.get(new_model, {})
                        payload = {
                            "config": {
                                "n_ctx": config.get("n_ctx", 4096),
                                "n_gpu_layers": -1,
                            }
                        }
                        async with session.post(
                            f"{BASE_URL}/v1/models/{new_model}/load",
                            json=payload,
                            timeout=60
                        ) as resp:
                            swap_result = TestResult(
                                test_name="model_swap",
                                model=new_model,
                                success=resp.status == 200,
                                latency_ms=0,
                                tokens_in=0,
                                tokens_out=0,
                                error=None if resp.status == 200 else f"HTTP {resp.status}"
                            )
                            results.append(swap_result)
                            # Extra delay after model swap
                            await asyncio.sleep(2.0)
                    except Exception as e:
                        results.append(TestResult(
                            test_name="model_swap",
                            model=new_model,
                            success=False,
                            latency_ms=0,
                            tokens_in=0,
                            tokens_out=0,
                            error=str(e)[:100]
                        ))
                
                # Store results
                self.results.extend(results)
                
                # Delay between requests to prevent overwhelming and allow model swaps
                await asyncio.sleep(1.0)
    
    async def run_benchmark(self) -> BenchmarkStats:
        """Run the full benchmark."""
        stats = BenchmarkStats()
        
        # Start workers
        logger.info(f"Starting {self.concurrency} concurrent workers for {self.duration}s")
        
        start_time = time.time()
        self.stop_event.clear()
        
        # Run workers
        workers = [
            asyncio.create_task(self.run_single_worker(i))
            for i in range(self.concurrency)
        ]
        
        # Let it run for specified duration
        await asyncio.sleep(self.duration)
        
        # Signal stop
        logger.info("Stopping benchmark...")
        self.stop_event.set()
        
        # Wait for workers to finish
        await asyncio.gather(*workers, return_exceptions=True)
        
        elapsed = time.time() - start_time
        
        # Aggregate results
        stats.total_tests = len(self.results)
        stats.passed = sum(1 for r in self.results if r.success)
        stats.failed = stats.total_tests - stats.passed
        stats.latencies = [r.latency_ms for r in self.results]
        stats.tokens_in = sum(r.tokens_in for r in self.results)
        stats.tokens_out = sum(r.tokens_out for r in self.results)
        stats.errors = [r.error for r in self.results if r.error]
        
        # Model usage
        for r in self.results:
            stats.model_usage[r.model] = stats.model_usage.get(r.model, 0) + 1
        
        logger.info(f"Benchmark completed in {elapsed:.1f}s")
        
        return stats
    
    def print_report(self, stats: BenchmarkStats):
        """Print benchmark report."""
        print("\n" + "="*80)
        print("AGENT STRESS TEST BENCHMARK REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL STATISTICS")
        print(f"  Total Tests:      {stats.total_tests}")
        print(f"  Passed:           {stats.passed} ({stats.passed/stats.total_tests*100:.1f}%)")
        print(f"  Failed:           {stats.failed} ({stats.failed/stats.total_tests*100:.1f}%)")
        
        print(f"\n⏱️  LATENCY")
        if stats.latencies:
            print(f"  Average:          {stats.avg_latency:.1f} ms")
            print(f"  P95:              {stats.p95_latency:.1f} ms")
            print(f"  Min:              {min(stats.latencies):.1f} ms")
            print(f"  Max:              {max(stats.latencies):.1f} ms")
        
        print(f"\n📈 THROUGHPUT")
        print(f"  Total Tokens In:  {stats.tokens_in:,}")
        print(f"  Total Tokens Out: {stats.tokens_out:,}")
        print(f"  Tokens/Second:    {stats.throughput_tps:.1f}")
        
        print(f"\n🤖 MODEL USAGE")
        for model, count in sorted(stats.model_usage.items(), key=lambda x: -x[1]):
            pct = count / stats.total_tests * 100
            print(f"  {model[:40]:<40} {count:>4} ({pct:>5.1f}%)")
        
        if stats.errors:
            print(f"\n❌ ERRORS (showing first 10)")
            error_counts = {}
            for err in stats.errors:
                if err:
                    key = err[:80]
                    error_counts[key] = error_counts.get(key, 0) + 1
            
            for err, count in sorted(error_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  [{count}x] {err}")
        
        # Stability assessment
        print(f"\n🎯 STABILITY ASSESSMENT")
        if stats.failed == 0:
            print("  ✅ EXCELLENT - All tests passed!")
        elif stats.failed / stats.total_tests < 0.05:
            print("  ✅ GOOD - Less than 5% failure rate")
        elif stats.failed / stats.total_tests < 0.10:
            print("  ⚠️  FAIR - 5-10% failure rate")
        else:
            print("  ❌ POOR - More than 10% failure rate")
        
        print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Agent Stress Test Benchmark for LLM Manager HTTP Server"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Benchmark duration in seconds (default: 300)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent workers (default: 5)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Path to models directory (default: ./models)"
    )
    
    args = parser.parse_args()
    
    # Determine models directory
    if args.models_dir:
        models_dir = Path(args.models_dir)
    else:
        models_dir = Path(__file__).parent / "models"
    
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return 1
    
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Found {len(MODELS_CONFIG)} models in config")
    logger.info(f"Benchmark config: duration={args.duration}s, concurrency={args.concurrency}")
    
    # Create and run benchmark
    benchmark = LLMManagerBenchmark(
        models_dir=models_dir,
        duration=args.duration,
        concurrency=args.concurrency
    )
    
    # Start server
    if not benchmark.start_server():
        return 1
    
    try:
        # Run benchmark
        stats = asyncio.run(benchmark.run_benchmark())
        
        # Print report
        benchmark.print_report(stats)
        
        # Return success if less than 10% failures
        return 0 if stats.failed / max(stats.total_tests, 1) < 0.10 else 1
        
    finally:
        benchmark.stop_server()


if __name__ == "__main__":
    exit(main())
