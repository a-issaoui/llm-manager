#!/usr/bin/env python3
"""
Comprehensive Benchmark for LLM Manager HTTP Server

Tests all models with various tasks:
- Reasoning tasks
- Code generation
- Tool calling
- Long context conversations
- Model switching

Usage:
    python benchmark_comprehensive.py [--quick] [--models-dir ./models]
"""

import argparse
import asyncio
import json
import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import requests

from llm_manager.server import LLMServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8889
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


@dataclass
class TestResult:
    test_name: str
    model: str
    success: bool
    latency_ms: float
    tokens_in: int
    tokens_out: int
    error: str | None = None


@dataclass
class BenchmarkStats:
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    latencies: list[float] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    errors: list[str] = field(default_factory=list)
    
    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0
    
    @property
    def success_rate(self) -> float:
        return self.passed / self.total_tests * 100 if self.total_tests > 0 else 0


class ComprehensiveBenchmark:
    """Comprehensive benchmark for LLM Manager."""
    
    def __init__(self, models_dir: Path, quick: bool = False):
        self.models_dir = models_dir
        self.quick = quick
        self.server: LLMServer | None = None
        self.server_thread: threading.Thread | None = None
        self.stats = BenchmarkStats()
        
        # Models to test (skip embedding model for chat)
        self.models = [
            "SmolLM-1.7B-Instruct-Q4_K_M.gguf",
            "Qwen2.5-3b-instruct-q4_k_m.gguf",
            "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf",
            "DeepSeek-R1-Distill-Llama-3B.Q5_K_M.gguf",
        ]
        if not quick:
            self.models.extend([
                "Reason-With-Choice-3B.Q4_K_M.gguf",
                "Nanbeige_Nanbeige4-3B-Thinking-2511-Q4_K_M.gguf",
            ])
    
    def start_server(self) -> bool:
        """Start the HTTP server."""
        logger.info(f"Starting server on {SERVER_HOST}:{SERVER_PORT}")
        
        self.server = LLMServer(
            models_dir=str(self.models_dir),
            host=SERVER_HOST,
            port=SERVER_PORT,
            log_level="warning",
        )
        
        self.server_thread = threading.Thread(target=self.server.start, daemon=True)
        self.server_thread.start()
        
        # Wait for server
        start_time = time.time()
        while time.time() - start_time < 60:
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=2)
                if response.status_code == 200:
                    logger.info("Server is ready!")
                    return True
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)
        
        logger.error("Server failed to start")
        return False
    
    def stop_server(self):
        """Stop the HTTP server."""
        if self.server:
            logger.info("Stopping server...")
            self.server.stop()
            if self.server_thread:
                self.server_thread.join(timeout=10)
    
    async def make_request(
        self,
        session: aiohttp.ClientSession,
        messages: list[dict],
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> dict:
        """Make a chat completion request."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=120
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"HTTP {resp.status}: {error_text[:200]}")
            return await resp.json()
    
    async def test_model_basic(self, session: aiohttp.ClientSession, model: str) -> TestResult:
        """Test basic generation."""
        start = time.time()
        try:
            response = await self.make_request(
                session,
                [{"role": "user", "content": "Say 'Hello' and nothing else."}],
                model,
                max_tokens=32
            )
            latency = (time.time() - start) * 1000
            usage = response.get("usage", {})
            return TestResult(
                test_name="basic_generation",
                model=model,
                success=True,
                latency_ms=latency,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0)
            )
        except Exception as e:
            return TestResult(
                test_name="basic_generation",
                model=model,
                success=False,
                latency_ms=(time.time() - start) * 1000,
                tokens_in=0,
                tokens_out=0,
                error=str(e)[:100]
            )
    
    async def test_reasoning(self, session: aiohttp.ClientSession, model: str) -> TestResult:
        """Test reasoning capability."""
        start = time.time()
        try:
            # Smaller models need more guidance
            is_small_model = "smollm" in model.lower() or "1.7b" in model.lower()
            
            # Use higher max_tokens for thinking models
            max_tokens = 512 if "think" in model.lower() or "nanbeige" in model.lower() else 256
            
            # Adjust prompt for smaller models
            if is_small_model:
                prompt = "What is 15 + 27? Answer with just the number."
                max_tokens = 128
            else:
                prompt = "Calculate 15 + 27 and give me just the final number."
            
            response = await self.make_request(
                session,
                [{"role": "user", "content": prompt}],
                model,
                max_tokens=max_tokens,
                temperature=0.1  # Lower temp for more deterministic answers
            )
            latency = (time.time() - start) * 1000
            usage = response.get("usage", {})
            content = response["choices"][0]["message"].get("content", "")
            
            # Check if answer contains 42 (handle various formats)
            import re
            success = "42" in content
            
            # Also check for the answer in a more flexible way
            if not success:
                # Look for "15 + 27 = 42" or similar patterns
                if re.search(r'15\s*\+\s*27\s*=\s*42', content):
                    success = True
                # Look for standalone 42
                elif re.search(r'\b42\b', content):
                    success = True
            
            return TestResult(
                test_name="reasoning",
                model=model,
                success=success,
                latency_ms=latency,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                error=None if success else f"Answer doesn't contain '42'. Got: {content[:100]}..."
            )
        except Exception as e:
            return TestResult(
                test_name="reasoning",
                model=model,
                success=False,
                latency_ms=(time.time() - start) * 1000,
                tokens_in=0,
                tokens_out=0,
                error=str(e)[:100]
            )
    
    async def test_code(self, session: aiohttp.ClientSession, model: str) -> TestResult:
        """Test code generation."""
        start = time.time()
        try:
            response = await self.make_request(
                session,
                [{"role": "user", "content": "Write a Python function to add two numbers."}],
                model,
                max_tokens=256,
                temperature=0.3
            )
            latency = (time.time() - start) * 1000
            usage = response.get("usage", {})
            content = response["choices"][0]["message"].get("content", "")
            
            # Check if code looks reasonable
            success = "def" in content.lower() and "return" in content.lower()
            
            return TestResult(
                test_name="code_generation",
                model=model,
                success=success,
                latency_ms=latency,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                error=None if success else "Response doesn't look like code"
            )
        except Exception as e:
            return TestResult(
                test_name="code_generation",
                model=model,
                success=False,
                latency_ms=(time.time() - start) * 1000,
                tokens_in=0,
                tokens_out=0,
                error=str(e)[:100]
            )
    
    async def test_conversation_context(self, session: aiohttp.ClientSession, model: str) -> TestResult:
        """Test conversation with context."""
        start = time.time()
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
                {"role": "user", "content": "What is my name?"},
            ]
            
            response = await self.make_request(
                session,
                messages,
                model,
                max_tokens=64
            )
            latency = (time.time() - start) * 1000
            usage = response.get("usage", {})
            content = response["choices"][0]["message"].get("content", "")
            
            # Check if it remembers the name
            success = "alice" in content.lower()
            
            return TestResult(
                test_name="conversation_context",
                model=model,
                success=success,
                latency_ms=latency,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                error=None if success else "Model didn't remember context"
            )
        except Exception as e:
            return TestResult(
                test_name="conversation_context",
                model=model,
                success=False,
                latency_ms=(time.time() - start) * 1000,
                tokens_in=0,
                tokens_out=0,
                error=str(e)[:100]
            )
    
    async def test_tools(self, session: aiohttp.ClientSession, model: str) -> TestResult:
        """Test tool calling."""
        start = time.time()
        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]
            
            response = await self.make_request(
                session,
                [{"role": "user", "content": "What's the weather in Tokyo?"}],
                model,
                max_tokens=256,
                tools=tools
            )
            latency = (time.time() - start) * 1000
            usage = response.get("usage", {})
            
            # Check if model wants to call the tool
            choice = response["choices"][0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])
            
            # Success if either tool call is present or content contains weather info
            success = len(tool_calls) > 0 or "weather" in message.get("content", "").lower()
            
            return TestResult(
                test_name="tool_calling",
                model=model,
                success=success,
                latency_ms=latency,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                error=None if success else "No tool call or weather info"
            )
        except Exception as e:
            return TestResult(
                test_name="tool_calling",
                model=model,
                success=False,
                latency_ms=(time.time() - start) * 1000,
                tokens_in=0,
                tokens_out=0,
                error=str(e)[:100]
            )
    
    async def run_tests_for_model(self, session: aiohttp.ClientSession, model: str) -> list[TestResult]:
        """Run all tests for a single model."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing model: {model}")
        logger.info(f"{'='*60}")
        
        results = []
        
        # The model will be auto-loaded on first request
        logger.info("Running tests (model will auto-load on first request)...")
        
        # Run tests
        tests = [
            ("Basic Generation", self.test_model_basic),
            ("Reasoning", self.test_reasoning),
            ("Code Generation", self.test_code),
            ("Conversation Context", self.test_conversation_context),
        ]
        
        # Only test tool calling on capable models
        if "coder" in model.lower() or "qwen" in model.lower():
            tests.append(("Tool Calling", self.test_tools))
        
        for test_name, test_func in tests:
            logger.info(f"  Running: {test_name}...")
            result = await test_func(session, model)
            results.append(result)
            status = "✅" if result.success else "❌"
            logger.info(f"  {status} {test_name}: {result.latency_ms:.0f}ms")
            await asyncio.sleep(0.5)  # Brief pause between tests
        
        return results
    
    async def run_benchmark(self) -> BenchmarkStats:
        """Run the complete benchmark."""
        self.stats = BenchmarkStats()
        
        async with aiohttp.ClientSession() as session:
            for model in self.models:
                results = await self.run_tests_for_model(session, model)
                self.stats.total_tests += len(results)
                self.stats.passed += sum(1 for r in results if r.success)
                self.stats.failed += sum(1 for r in results if not r.success)
                self.stats.latencies.extend([r.latency_ms for r in results])
                self.stats.tokens_in += sum(r.tokens_in for r in results)
                self.stats.tokens_out += sum(r.tokens_out for r in results)
                self.stats.errors.extend([r.error for r in results if r.error])
                
                # Delay between models
                await asyncio.sleep(2)
        
        return self.stats
    
    def print_report(self):
        """Print benchmark report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL STATISTICS")
        print(f"  Models Tested:    {len(self.models)}")
        print(f"  Total Tests:      {self.stats.total_tests}")
        print(f"  Passed:           {self.stats.passed} ({self.stats.success_rate:.1f}%)")
        print(f"  Failed:           {self.stats.failed}")
        
        print(f"\n⏱️  LATENCY")
        if self.stats.latencies:
            print(f"  Average:          {self.stats.avg_latency:.1f} ms")
            print(f"  Min:              {min(self.stats.latencies):.1f} ms")
            print(f"  Max:              {max(self.stats.latencies):.1f} ms")
        
        print(f"\n📈 THROUGHPUT")
        print(f"  Total Tokens In:  {self.stats.tokens_in:,}")
        print(f"  Total Tokens Out: {self.stats.tokens_out:,}")
        
        if self.stats.errors:
            print(f"\n❌ ERRORS")
            for error in self.stats.errors[:10]:
                if error:
                    print(f"  - {error}")
        
        print(f"\n🎯 STABILITY ASSESSMENT")
        if self.stats.failed == 0:
            print("  ✅ EXCELLENT - All tests passed!")
        elif self.stats.success_rate >= 95:
            print("  ✅ GOOD - High success rate")
        elif self.stats.success_rate >= 80:
            print("  ⚠️  FAIR - Some failures")
        else:
            print("  ❌ POOR - Many failures")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Benchmark for LLM Manager")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer models)")
    parser.add_argument("--models-dir", type=str, default=None, help="Models directory")
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir) if args.models_dir else Path(__file__).parent / "models"
    
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return 1
    
    benchmark = ComprehensiveBenchmark(models_dir, quick=args.quick)
    
    if not benchmark.start_server():
        return 1
    
    try:
        asyncio.run(benchmark.run_benchmark())
        benchmark.print_report()
        return 0 if benchmark.stats.failed == 0 else 1
    finally:
        benchmark.stop_server()


if __name__ == "__main__":
    exit(main())
