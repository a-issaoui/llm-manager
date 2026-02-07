#!/usr/bin/env python3
"""
Example 08: Model Switching

Demonstrates loading different models for different tasks without restarting.
"""

import logging
import time
from pathlib import Path

from llm_manager import LLMManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run model switching example."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    logger.info("Initializing model switching example")
    
    manager = LLMManager(
        models_dir=str(models_dir),
        use_subprocess=False,
        verbose=True
    )
    
    # Define tasks with different models
    tasks = [
        {
            "model": "SmolLM-1.7B-Instruct-Q4_K_M.gguf",
            "name": "Creative Writing",
            "prompt": "Write a creative story opening about a mysterious library:",
            "max_tokens": 150,
            "temperature": 0.8,
        },
        {
            "model": "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf",
            "name": "Code Generation",
            "prompt": "Write a Python function to reverse a linked list:",
            "max_tokens": 200,
            "temperature": 0.2,
        },
        {
            "model": "DeepSeek-R1-Distill-Llama-3B.Q5_K_M.gguf",
            "name": "Reasoning",
            "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets? Think step by step:",
            "max_tokens": 200,
            "temperature": 0.5,
        },
    ]
    
    print("\n" + "="*70)
    print("MODEL SWITCHING DEMONSTRATION")
    print("="*70)
    
    total_start = time.time()
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'─'*70}")
        print(f"Task {i}/{len(tasks)}: {task['name']}")
        print(f"Model: {task['model']}")
        print(f"{'─'*70}")
        
        task_start = time.time()
        
        # Switch to the model for this task
        logger.info(f"Switching to model: {task['model']}")
        success = manager.load_model(task['model'], config={
            "n_ctx": 4096,
            "n_gpu_layers": 0,
        })
        
        if not success:
            logger.error(f"Failed to load model: {task['model']}")
            continue
        
        load_time = time.time() - task_start
        print(f"Model loaded in {load_time:.2f}s")
        
        # Generate
        messages = [{"role": "user", "content": task['prompt']}]
        print(f"\nPrompt: {task['prompt'][:60]}...")
        print("\nResponse:")
        
        gen_start = time.time()
        response = manager.generate(
            messages=messages,
            max_tokens=task['max_tokens'],
            temperature=task['temperature'],
            stop=["\n\n", "User:", "Human:"]
        )
        gen_time = time.time() - gen_start
        
        message = response.get("choices", [{}])[0].get("message", {})
        text = message.get("content", "No response")
        print(text)
        
        usage = response.get("usage", {})
        print(f"\n[Generated in {gen_time:.2f}s, "
              f"tokens: {usage.get('total_tokens', 0)}]")
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"Completed {len(tasks)} tasks with model switching in {total_time:.2f}s")
    print("="*70)
    
    # Cleanup
    manager.unload_model()
    logger.info("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
