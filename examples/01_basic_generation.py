#!/usr/bin/env python3
"""
Example 01: Basic Text Generation

Demonstrates the simplest way to load a model and generate text.
Uses the SmolLM-1.7B model for fast inference.
"""

import logging
from pathlib import Path

from llm_manager import LLMManager

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run basic text generation example."""
    # Get project root and models directory
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    # Use the smallest model for fast demonstration
    model_name = "SmolLM-1.7B-Instruct-Q4_K_M.gguf"
    
    logger.info(f"Initializing LLMManager with models_dir: {models_dir}")
    
    # Initialize the manager
    manager = LLMManager(
        models_dir=str(models_dir),
        use_subprocess=False,  # Direct mode for simplicity
        verbose=True
    )
    
    logger.info(f"Loading model: {model_name}")
    
    # Load the model
    success = manager.load_model(model_name, config={
        "n_ctx": 4096,
        "n_gpu_layers": 0,  # CPU-only for compatibility
    })
    
    if not success:
        logger.error("Failed to load model!")
        return 1
    
    logger.info("Model loaded successfully!")
    
    # Generate text
    messages = [{"role": "user", "content": "Write a short poem about artificial intelligence:"}]
    logger.info(f"Generating text for messages: {messages!r}")
    
    response = manager.generate(
        messages=messages,
        max_tokens=256,
        temperature=0.7,
        stop=["\n\n", "User:", "Human:"]
    )
    
    # Display results
    print("\n" + "="*60)
    print("MESSAGES:")
    print("="*60)
    for msg in messages:
        print(f"[{msg['role']}] {msg['content']}")
    print("\n" + "="*60)
    print("RESPONSE:")
    print("="*60)
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    print(message.get("content", "No response"))
    print("="*60)
    
    # Show generation stats
    usage = response.get("usage", {})
    print(f"\nTokens: prompt={usage.get('prompt_tokens', 0)}, "
          f"completion={usage.get('completion_tokens', 0)}, "
          f"total={usage.get('total_tokens', 0)}")
    print(f"Finish reason: {choice.get('finish_reason', 'unknown')}")
    
    # Cleanup
    logger.info("Unloading model...")
    manager.unload_model()
    logger.info("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
