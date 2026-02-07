#!/usr/bin/env python3
"""
Example 04: Streaming Text Generation

Demonstrates streaming generation where tokens are received incrementally.
Uses DeepSeek-R1-Distill model for reasoning tasks.
"""

import logging
import sys
from pathlib import Path

from llm_manager import LLMManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run streaming generation example."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    # Use DeepSeek-R1 which is good for reasoning
    model_name = "DeepSeek-R1-Distill-Llama-3B.Q5_K_M.gguf"
    
    logger.info(f"Initializing LLMManager for streaming")
    
    manager = LLMManager(
        models_dir=str(models_dir),
        use_subprocess=False,
        verbose=False  # Reduce noise for streaming demo
    )
    
    logger.info(f"Loading model: {model_name}")
    
    success = manager.load_model(model_name, config={
        "n_ctx": 4096,
        "n_gpu_layers": 0,
    })
    
    if not success:
        logger.error("Failed to load model!")
        return 1
    
    logger.info("Model loaded! Starting streaming generation...")
    
    # A prompt that benefits from step-by-step reasoning
    messages = [{
        "role": "user",
        "content": """Solve this step by step: If a train travels at 60 km/h and another 
traveling in the opposite direction at 80 km/h, how long until they meet 
if they start 420 km apart?"""
    }]
    
    print("\n" + "="*70)
    print("MESSAGES:")
    print("="*70)
    for msg in messages:
        print(f"[{msg['role']}] {msg['content']}")
    print("\n" + "="*70)
    print("STREAMING RESPONSE:")
    print("="*70)
    
    # Stream the response - generate returns an iterator when stream=True
    full_response = []
    token_count = 0
    
    try:
        stream_iterator = manager.generate(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            stream=True,  # Enable streaming
            stop=["\n\n", "User:"]
        )
        
        for chunk in stream_iterator:
            # Extract text from chunk
            choices = chunk.get("choices", [{}])
            if choices:
                delta = choices[0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    print(text, end="", flush=True)
                    full_response.append(text)
                    token_count += 1
        
        print()  # New line after streaming
        
    except KeyboardInterrupt:
        print("\n[Generation interrupted by user]")
    
    print("="*70)
    print(f"\nStreamed {token_count} tokens")
    
    # Cleanup
    manager.unload_model()
    logger.info("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
