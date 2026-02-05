#!/usr/bin/env python3
"""
Example 06: Batch Generation & Variations

Demonstrates:
- Batch generation for multiple prompts
- Generate variations of same prompt
- Parallel execution

No external dependencies required.
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


async def example_batch_generation():
    """Batch generation example (async)."""
    print("=" * 60)
    print("Batch Generation: Multiple Prompts")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    # Use subprocess mode for isolation and limit GPU layers
    manager = LLMManager(models_dir=get_config().models.dir, use_subprocess=True)
    
    try:
        # Load with conservative settings for batch processing
        await manager.load_model_async(model_name, n_gpu_layers=0)
        
        prompts = [
            [{"role": "user", "content": "What is Python?"}],
            [{"role": "user", "content": "What is JavaScript?"}],
            [{"role": "user", "content": "What is Rust?"}],
        ]
        
        print(f"Processing {len(prompts)} prompts...")
        
        results = await manager.generate_batch(prompts, max_tokens=50)
        
        for i, result in enumerate(results):
            if "choices" in result:
                content = result["choices"][0]["message"]["content"][:60]
                print(f"  {i+1}. {content}...")
            else:
                print(f"  {i+1}. Error: {result.get('error', 'Unknown')}")
        
        print("Batch processing complete\n")
        
    except Exception as e:
        print(f"Note: {e}\n")
    finally:
        manager.unload_model()


async def example_async_batch():
    """Async batch generation with semaphore control."""
    print("=" * 60)
    print("Batch Generation: Async with Concurrency Control")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    # Use subprocess mode for better isolation
    manager = LLMManager(models_dir=get_config().models.dir, use_subprocess=True)
    
    try:
        # Load with CPU for stable batch processing
        await manager.load_model_async(model_name, n_gpu_layers=0)
        
        prompts = [
            [{"role": "user", "content": "Explain AI in one sentence"}],
            [{"role": "user", "content": "Explain ML in one sentence"}],
            [{"role": "user", "content": "Explain DL in one sentence"}],
        ]
        
        print(f"Processing {len(prompts)} prompts sequentially...")
        
        # Process sequentially to avoid GPU memory issues
        results = []
        for prompt in prompts:
            result = await manager.generate_async(prompt, max_tokens=50)
            results.append(result)
        
        for i, result in enumerate(results):
            if "choices" in result:
                content = result["choices"][0]["message"]["content"][:60]
                print(f"  {i+1}. {content}...")
            else:
                print(f"  {i+1}. Error")
        
        print("Async batch complete\n")
        
    except Exception as e:
        print(f"Note: {e}\n")
    finally:
        manager.unload_model()


async def example_variations():
    """Generate variations of same prompt."""
    print("=" * 60)
    print("Variations: Multiple Outputs for Same Prompt")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    # Use subprocess mode
    manager = LLMManager(models_dir=get_config().models.dir, use_subprocess=True)
    
    try:
        await manager.load_model_async(model_name, n_gpu_layers=0)
        
        prompt = [{"role": "user", "content": "Give me a creative name for a coffee shop"}]
        
        print("Generating 3 variations...")
        
        # Generate with different seeds
        variations = []
        for seed in [1, 2, 3]:
            result = await manager.generate_async(
                prompt, 
                temperature=0.9, 
                max_tokens=50,
                seed=seed
            )
            variations.append(result)
        
        for i, result in enumerate(variations):
            if "choices" in result:
                content = result["choices"][0]["message"]["content"][:60]
                print(f"  {i+1}. {content}")
            else:
                print(f"  {i+1}. Error")
        
        print()
        
    except Exception as e:
        print(f"Note: {e}\n")
    finally:
        manager.unload_model()


async def example_batch_with_different_params():
    """Batch with different parameters per prompt."""
    print("=" * 60)
    print("Batch: Different Parameters")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    # Use subprocess mode
    manager = LLMManager(models_dir=get_config().models.dir, use_subprocess=True)
    
    try:
        await manager.load_model_async(model_name, n_gpu_layers=0)
        
        # Process different types of requests
        batch_requests = [
            {
                "messages": [{"role": "user", "content": "Say hello"}],
                "temperature": 0.2,
                "max_tokens": 20
            },
            {
                "messages": [{"role": "user", "content": "Greet me"}],
                "temperature": 0.9,
                "max_tokens": 20
            },
            {
                "messages": [{"role": "user", "content": "Hi there"}],
                "temperature": 0.5,
                "max_tokens": 20
            }
        ]
        
        print("Processing batch with different parameters...")
        
        # Process each with its own settings
        results = []
        for req in batch_requests:
            result = await manager.generate_async(**req)
            results.append(result)
        
        for i, result in enumerate(results):
            if "choices" in result:
                content = result["choices"][0]["message"]["content"][:60]
                print(f"  {i+1}. {content}...")
            else:
                print(f"  {i+1}. Error")
        
        print()
        
    except Exception as e:
        print(f"Note: {e}\n")
    finally:
        manager.unload_model()


def example_concurrent_processing():
    """Concurrent processing with pool."""
    print("=" * 60)
    print("Concurrent Processing with Worker Pool")
    print("=" * 60)
    
    example_code = '''
from llm_manager import LLMManager
import asyncio

# Manager with worker pool
manager = LLMManager(
    models_dir=get_config().models.dir,
    use_subprocess=True,  # Enable subprocess workers
    pool_size=4  # 4 concurrent workers
)

async def process_many(items):
    """Process many items concurrently."""
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(4)
    
    async def process_one(item):
        async with semaphore:
            return await manager.generate_async(
                messages=[{"role": "user", "content": item}],
                max_tokens=100
            )
    
    # Process all concurrently
    tasks = [process_one(item) for item in items]
    return await asyncio.gather(*tasks)

# Process 100 items with max 4 concurrent
items = [f"Question {i}" for i in range(100)]
results = asyncio.run(process_many(items))
'''
    print(example_code)


async def example_error_handling():
    """Error handling in batch processing."""
    print("=" * 60)
    print("Batch Error Handling")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    manager = LLMManager(models_dir=get_config().models.dir, use_subprocess=True)
    
    try:
        await manager.load_model_async(model_name, n_gpu_layers=0)
        
        prompts = [
            [{"role": "user", "content": "Valid prompt"}],
            [{"role": "user", "content": "Another valid"}],
            [{"role": "user", "content": "Third valid"}],
        ]
        
        # Process with error handling
        print("Processing with error handling...")
        results = await manager.generate_batch(prompts, max_tokens=30)
        
        # Check results
        for i, result in enumerate(results):
            if "error" in result:
                print(f"Prompt {i} failed: {result['error']}")
            elif "choices" in result:
                content = result["choices"][0]["message"]["content"][:40]
                print(f"Prompt {i} succeeded: {content}...")
            else:
                print(f"Prompt {i}: Unexpected result")
        
    except Exception as e:
        print(f"Note: {e}\n")
    finally:
        manager.unload_model()


async def example_gpu_cpu_offload():
    """Batch processing with GPU/CPU offload (partial GPU layers)."""
    print("=" * 60)
    print("Batch: GPU/CPU Offload (Partial GPU)")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    # Use subprocess mode with partial GPU offloading
    manager = LLMManager(models_dir=get_config().models.dir, use_subprocess=True)
    
    try:
        # Load with partial GPU layers (20 layers on GPU, rest on CPU)
        # This balances speed and memory usage
        print("Loading model with 20 GPU layers (partial offload)...")
        await manager.load_model_async(model_name, n_gpu_layers=20, n_ctx=2048)
        
        prompts = [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "What is 5*5?"}],
            [{"role": "user", "content": "What is 10/2?"}],
        ]
        
        print(f"Processing {len(prompts)} prompts with GPU/CPU hybrid...")
        
        results = await manager.generate_batch(prompts, max_tokens=30)
        
        for i, result in enumerate(results):
            if "choices" in result:
                content = result["choices"][0]["message"]["content"][:60]
                print(f"  {i+1}. {content}...")
            else:
                print(f"  {i+1}. Error")
        
        print("GPU/CPU offload batch complete\n")
        
    except Exception as e:
        print(f"Note: {e}\n")
    finally:
        manager.unload_model()


async def example_gpu_only_small_context():
    """Batch processing with full GPU using a small model (SmolLM-1.7B)."""
    print("=" * 60)
    print("Batch: Full GPU with Small Model (SmolLM-1.7B)")
    print("=" * 60)
    
    # Use the smallest instruction model available
    small_model = "SmolLM-1.7B-Instruct-Q4_K_M.gguf"
    
    # Check if model exists
    models_dir = get_config().models.dir
    model_path = Path(models_dir) / small_model
    
    if not model_path.exists():
        print(f"Small model {small_model} not found, using first available model...")
        small_model = get_first_model()
        if not small_model:
            print("No models found in registry")
            return
    
    print(f"Using small model: {small_model} (~1GB)")
    
    # Use subprocess mode with full GPU
    manager = LLMManager(models_dir=models_dir, use_subprocess=True)
    
    try:
        # Load with all GPU layers - small model should fit easily
        print("Loading model with all GPU layers (n_gpu_layers=-1)...")
        print("Using small context (n_ctx=2048)...")
        await manager.load_model_async(small_model, n_gpu_layers=-1, n_ctx=2048)
        
        prompts = [
            [{"role": "user", "content": "Say 'hello'"}],
            [{"role": "user", "content": "Say 'hi'"}],
            [{"role": "user", "content": "Say 'hey'"}],
        ]
        
        print(f"Processing {len(prompts)} prompts on GPU only...")
        
        # Process with small max_tokens
        results = await manager.generate_batch(prompts, max_tokens=20)
        
        for i, result in enumerate(results):
            if "choices" in result:
                content = result["choices"][0]["message"]["content"][:60]
                print(f"  {i+1}. {content}...")
            else:
                print(f"  {i+1}. Error")
        
        print("Full GPU batch complete\n")
        
    except Exception as e:
        print(f"GPU load failed: {e}\n")
        print("Tip: SmolLM-1.7B is a 1GB model that should fit in most GPUs.")
        print("      If this fails, your GPU may have very limited VRAM.\n")
    finally:
        manager.unload_model()


def example_progress_tracking():
    """Track batch progress."""
    print("=" * 60)
    print("Batch Progress Tracking")
    print("=" * 60)
    
    example_code = '''
import asyncio
from tqdm import tqdm

async def process_with_progress(prompts, manager):
    """Process batch with progress bar."""
    
    results = []
    
    with tqdm(total=len(prompts), desc="Processing") as pbar:
        for i in range(0, len(prompts), 4):  # Batch of 4
            batch = prompts[i:i+4]
            batch_results = await manager.generate_batch(batch)
            results.extend(batch_results)
            pbar.update(len(batch))
    
    return results

# Usage
prompts = [[{"role": "user", "content": f"Q{i}"}] for i in range(100)]
results = asyncio.run(process_with_progress(prompts, manager))
'''
    print(example_code)


async def main_async():
    """Run all batch generation examples."""
    print("\n" + "=" * 60)
    print("LLM Manager - Batch Generation Examples")
    print("=" * 60 + "\n")
    print("Note: Examples demonstrate different GPU/CPU strategies\n")
    
    # CPU-only examples (most stable)
    await example_batch_generation()
    await example_async_batch()
    await example_variations()
    await example_batch_with_different_params()
    await example_error_handling()
    
    # GPU/CPU hybrid examples
    await example_gpu_cpu_offload()
    await example_gpu_only_small_context()
    
    # Documentation examples (code only)
    example_concurrent_processing()
    example_progress_tracking()


def main():
    """Run batch generation examples."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
