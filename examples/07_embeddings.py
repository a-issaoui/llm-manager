#!/usr/bin/env python3
"""
Example 07: Embedding Model Usage

Demonstrates loading and using an embedding model.
Uses Qwen3-Embedding model for text understanding.
"""

import logging
from pathlib import Path

from llm_manager import LLMManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run embedding model example."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    # Use embedding model
    model_name = "Qwen3-Embedding-0.6B-Q8_0.gguf"
    
    logger.info("Initializing embedding model example")
    
    manager = LLMManager(
        models_dir=str(models_dir),
        use_subprocess=False,
        verbose=True
    )
    
    logger.info(f"Loading embedding model: {model_name}")
    
    success = manager.load_model(model_name, config={
        "n_ctx": 2048,
        "n_gpu_layers": 0,
        "embedding": True,
    })
    
    if not success:
        logger.error("Failed to load model!")
        return 1
    
    logger.info("Model loaded!")
    
    # Test with simple text understanding task
    # Note: Embedding models can still generate text, just optimized for embeddings
    messages = [
        {
            "role": "user",
            "content": "Explain what text embeddings are in one sentence:"
        }
    ]
    
    logger.info("Testing embedding model with generation task...")
    
    response = manager.generate(
        messages=messages,
        max_tokens=64,
        temperature=0.3,
    )
    
    message = response.get("choices", [{}])[0].get("message", {})
    text = message.get("content", "")
    
    print("\n" + "="*70)
    print("EMBEDDING MODEL TEST")
    print("="*70)
    print(f"Query: {messages[0]['content']}")
    print(f"\nResponse: {text}")
    
    usage = response.get("usage", {})
    print(f"\nTokens used: {usage.get('total_tokens', 0)}")
    
    print("="*70)
    print("\nNote: Embedding models are optimized for creating vector")
    print("representations of text for similarity search and clustering.")
    print("="*70)
    
    # Cleanup
    manager.unload_model()
    logger.info("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
