#!/usr/bin/env python3
"""
Example Runner - Execute all examples and report results.

This script runs all examples in sequence and reports which ones passed/failed.
"""

import importlib.util
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# List of examples to run with their descriptions
EXAMPLES = [
    ("01_basic_generation.py", "Basic text generation", 120),
    ("02_chat_completion.py", "Chat with conversation history", 120),
    ("03_openai_server.py", "OpenAI-compatible HTTP server", 60),
    ("04_streaming_generation.py", "Streaming text generation", 120),
    ("05_async_generation.py", "Async generation", 120),
    ("06_batch_processing.py", "Batch processing", 180),
    ("07_embeddings.py", "Embeddings generation", 120),
    ("08_model_switching.py", "Model switching", 300),
    ("09_tool_calling.py", "Tool/function calling", 180),
    ("10_subprocess_mode.py", "Subprocess mode", 120),
    ("11_concurrent_http.py", "Concurrent HTTP requests", 60),
    ("12_model_comparison.py", "Model comparison", 300),
]


def run_example(example_file: str, timeout: int) -> tuple[bool, str, float]:
    """Run a single example and return success status."""
    example_path = Path(__file__).parent / example_file
    
    if not example_path.exists():
        return False, f"File not found: {example_path}", 0.0
    
    logger.info(f"Running {example_file}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            return True, "Success", elapsed
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            return False, f"Exit code {result.returncode}: {error_msg}", elapsed
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return False, f"Timeout after {timeout}s", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return False, str(e), elapsed


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("LLM MANAGER EXAMPLES - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"Running {len(EXAMPLES)} examples...\n")
    
    results = []
    total_start = time.time()
    
    for i, (example_file, description, timeout) in enumerate(EXAMPLES, 1):
        print(f"\n[{i}/{len(EXAMPLES)}] {example_file}")
        print(f"      Description: {description}")
        print(f"      Timeout: {timeout}s")
        
        success, message, elapsed = run_example(example_file, timeout)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"      Result: {status} ({elapsed:.1f}s)")
        
        if not success:
            print(f"      Error: {message[:200]}")
        
        results.append({
            "file": example_file,
            "description": description,
            "success": success,
            "message": message,
            "time": elapsed
        })
    
    total_time = time.time() - total_start
    
    # Summary
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total: {len(results)} examples")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.1f}s")
    print("-"*70)
    
    # Detailed results table
    print(f"\n{'Example':<35} {'Status':<10} {'Time(s)':<10}")
    print("-"*70)
    
    for r in results:
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        name = r["file"][:33]
        print(f"{name:<35} {status:<10} {r['time']:<10.1f}")
    
    print("="*70)
    
    # List failed examples
    if failed > 0:
        print("\nFailed Examples:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['file']}: {r['message'][:100]}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
