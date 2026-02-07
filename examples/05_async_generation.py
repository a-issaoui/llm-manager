#!/usr/bin/env python3
"""
Example 05: Async Generation

Demonstrates asynchronous model loading and text generation.
Uses asyncio for non-blocking operations.
"""

import asyncio
import logging
from pathlib import Path

from llm_manager import LLMManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def generate_with_model(manager: LLMManager, model_name: str, prompt: str, idx: int) -> dict:
    """Generate text with a specific model."""
    logger.info(f"[{idx}] Starting generation with {model_name}")
    
    # Ensure model is loaded
    await manager.get_or_load_async(model_name)
    
    # Create messages from prompt
    messages = [{"role": "user", "content": prompt}]
    
    # Generate
    response = await manager.generate_async(
        messages=messages,
        max_tokens=128,
        temperature=0.7
    )
    
    message = response.get("choices", [{}])[0].get("message", {})
    text = message.get("content", "")
    logger.info(f"[{idx}] Generation complete")
    
    return {"model": model_name, "text": text, "idx": idx}


async def main():
    """Run async generation example."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    logger.info("Initializing LLMManager for async operations")
    
    manager = LLMManager(
        models_dir=str(models_dir),
        use_subprocess=False,
        verbose=True
    )
    
    # Define prompts for different models
    tasks_data = [
        {
            "model": "SmolLM-1.7B-Instruct-Q4_K_M.gguf",
            "prompt": "Write a haiku about nature:"
        },
        {
            "model": "Qwen2.5-3b-instruct-q4_k_m.gguf",
            "prompt": "Explain what machine learning is in one sentence:"
        },
    ]
    
    logger.info(f"Running {len(tasks_data)} async generation tasks")
    
    # Create tasks
    tasks = [
        generate_with_model(manager, data["model"], data["prompt"], idx)
        for idx, data in enumerate(tasks_data)
    ]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Display results
    print("\n" + "="*70)
    print("ASYNC GENERATION RESULTS:")
    print("="*70)
    
    for result in results:
        if isinstance(result, Exception):
            print(f"\nERROR: {result}")
            continue
            
        print(f"\n[{result['idx']}] Model: {result['model']}")
        print(f"Response: {result['text'][:200]}...")
    
    print("="*70)
    
    # Cleanup
    manager.unload_model()
    logger.info("Done!")
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
