#!/usr/bin/env python3
"""
Example 02: Chat Completion with Conversation History

Demonstrates how to use chat format with system prompts and conversation history.
Uses Qwen2.5-3B-Instruct model which is optimized for chat.
"""

import logging
from pathlib import Path

from llm_manager import LLMManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run chat completion example."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    # Use Qwen2.5 which is good for instruction following
    model_name = "Qwen2.5-3b-instruct-q4_k_m.gguf"
    
    logger.info(f"Initializing LLMManager")
    
    manager = LLMManager(
        models_dir=str(models_dir),
        use_subprocess=False,
        verbose=True
    )
    
    logger.info(f"Loading model: {model_name}")
    
    success = manager.load_model(model_name, config={
        "n_ctx": 8192,
        "n_gpu_layers": 0,
    })
    
    if not success:
        logger.error("Failed to load model!")
        return 1
    
    logger.info("Model loaded successfully!")
    
    # Define a conversation with system prompt
    messages = [
        {
            "role": "system",
            "content": "You are a helpful coding assistant. Provide concise, accurate code examples."
        },
        {
            "role": "user",
            "content": "Write a Python function to calculate Fibonacci numbers."
        },
        {
            "role": "assistant",
            "content": "Here's a Python function to calculate Fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n```"
        },
        {
            "role": "user",
            "content": "Can you optimize it with memoization?"
        }
    ]
    
    logger.info("Generating chat response with conversation history...")
    
    response = manager.generate(
        messages=messages,
        max_tokens=512,
        temperature=0.3,  # Lower temperature for code
        stop=["\n\nUser:", "\n\nHuman:"]
    )
    
    # Display the conversation
    print("\n" + "="*70)
    print("CONVERSATION HISTORY:")
    print("="*70)
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n[{role}]")
        print(content[:200] + "..." if len(content) > 200 else content)
    
    print("\n" + "="*70)
    print("ASSISTANT RESPONSE:")
    print("="*70)
    response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    print(response_text)
    print("="*70)
    
    # Show token usage
    usage = response.get("usage", {})
    print(f"\nToken Usage:")
    print(f"  Prompt tokens: {usage.get('prompt_tokens', 0)}")
    print(f"  Completion tokens: {usage.get('completion_tokens', 0)}")
    print(f"  Total tokens: {usage.get('total_tokens', 0)}")
    
    # Cleanup
    manager.unload_model()
    logger.info("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
