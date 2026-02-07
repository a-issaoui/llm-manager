#!/usr/bin/env python3
"""
Example 10: Subprocess Mode

Demonstrates using subprocess isolation for stability and crash protection.
"""

import logging
from pathlib import Path

from llm_manager import LLMManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run subprocess mode example."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    model_name = "SmolLM-1.7B-Instruct-Q4_K_M.gguf"
    
    logger.info("Initializing LLMManager with SUBPROCESS mode")
    logger.info("This provides isolation - model crashes won't affect main process")
    
    # Use subprocess mode for isolation
    manager = LLMManager(
        models_dir=str(models_dir),
        use_subprocess=True,  # Enable subprocess mode
        verbose=True
    )
    
    logger.info(f"Loading model in subprocess: {model_name}")
    
    success = manager.load_model(model_name, config={
        "n_ctx": 4096,
        "n_gpu_layers": 0,
    })
    
    if not success:
        logger.error("Failed to load model!")
        return 1
    
    logger.info("Model loaded in subprocess!")
    
    # Multiple generation requests
    prompts = [
        "List 3 benefits of exercise:",
        "Explain quantum computing in simple terms:",
        "Write a recipe for chocolate chip cookies:",
    ]
    
    print("\n" + "="*70)
    print("SUBPROCESS MODE GENERATIONS")
    print("="*70)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'─'*70}")
        print(f"Generation {i}/{len(prompts)}")
        print(f"{'─'*70}")
        print(f"Prompt: {prompt}")
        print("\nResponse:")
        
        # Generate (communication happens through pipes)
        messages = [{"role": "user", "content": prompt}]
        response = manager.generate(
            messages=messages,
            max_tokens=128,
            temperature=0.7,
            stop=["\n\n", "User:"]
        )
        
        message = response.get("choices", [{}])[0].get("message", {})
        text = message.get("content", "No response")
        print(text[:200] + "..." if len(text) > 200 else text)
        
        usage = response.get("usage", {})
        print(f"\n[Tokens: {usage.get('total_tokens', 0)}]")
    
    print(f"\n{'='*70}")
    print("Subprocess mode provides:")
    print("  • Process isolation (crashes don't affect main process)")
    print("  • Memory cleanup (subprocess memory is freed on exit)")
    print("  • Stability (GIL contention avoided)")
    print("="*70)
    
    # Cleanup
    logger.info("Shutting down subprocess...")
    manager.unload_model()
    logger.info("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
