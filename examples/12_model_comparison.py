#!/usr/bin/env python3
"""
Example 12: Model Comparison

Compares different models on the same prompts to show their strengths.
"""

import logging
import time
from pathlib import Path

from llm_manager import LLMManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def benchmark_model(manager: LLMManager, model_name: str, prompts: list[str]) -> dict:
    """Benchmark a model on multiple prompts."""
    logger.info(f"Loading {model_name}...")
    
    start = time.time()
    success = manager.load_model(model_name, config={
        "n_ctx": 4096,
        "n_gpu_layers": 0,
    })
    load_time = time.time() - start
    
    if not success:
        return {"model": model_name, "error": "Failed to load"}
    
    results = []
    total_tokens = 0
    total_time = 0
    
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        gen_start = time.time()
        response = manager.generate(
            messages=messages,
            max_tokens=128,
            temperature=0.7
        )
        gen_time = time.time() - gen_start
        
        message = response.get("choices", [{}])[0].get("message", {})
        text = message.get("content", "")
        usage = response.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        
        results.append({
            "prompt": prompt[:40] + "...",
            "response": text[:100] + "...",
            "time": gen_time,
            "tokens": tokens
        })
        
        total_tokens += tokens
        total_time += gen_time
    
    return {
        "model": model_name,
        "load_time": load_time,
        "results": results,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "tokens_per_sec": total_tokens / total_time if total_time > 0 else 0
    }


def main():
    """Run model comparison example."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    logger.info("Starting model comparison")
    
    manager = LLMManager(
        models_dir=str(models_dir),
        use_subprocess=False,
        verbose=False
    )
    
    # Test prompts
    prompts = [
        "Explain what is Python programming language:",
        "Write a haiku about technology:",
    ]
    
    # Models to compare
    models = [
        "SmolLM-1.7B-Instruct-Q4_K_M.gguf",
        "Qwen2.5-3b-instruct-q4_k_m.gguf",
    ]
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"Prompts: {len(prompts)}")
    print(f"Models: {len(models)}")
    
    all_results = []
    
    for model_name in models:
        print(f"\n{'─'*70}")
        print(f"Testing: {model_name}")
        print(f"{'─'*70}")
        
        result = benchmark_model(manager, model_name, prompts)
        all_results.append(result)
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
            continue
        
        print(f"Load time: {result['load_time']:.2f}s")
        print(f"Total tokens: {result['total_tokens']}")
        print(f"Total time: {result['total_time']:.2f}s")
        print(f"Speed: {result['tokens_per_sec']:.2f} tokens/sec")
    
    # Summary table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<40} {'Load(s)':<10} {'Tokens':<8} {'Time(s)':<10} {'T/s':<8}")
    print("-"*70)
    
    for r in all_results:
        if "error" not in r:
            name = r['model'][:38]
            print(f"{name:<40} {r['load_time']:<10.2f} {r['total_tokens']:<8} "
                  f"{r['total_time']:<10.2f} {r['tokens_per_sec']:<8.2f}")
    
    print("="*70)
    
    # Cleanup
    manager.unload_model()
    logger.info("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
