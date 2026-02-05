#!/usr/bin/env python3
"""
Simple basic example of llm_manager usage.

Usage:
    python examples/basic.py
"""

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


def main():
    """Basic usage example."""
    print("=" * 60)
    print("LLM Manager - Basic Example")
    print("=" * 60)
    
    # Get first available model
    model_name = get_first_model()
    if not model_name:
        print("\nNo models found in registry.")
        print(f"Please add models to: {get_config().models.dir}")
        return
    
    print(f"\nUsing model: {model_name}")
    
    # Create manager
    manager = LLMManager(models_dir=get_config().models.dir)
    
    try:
        # Load model with GPU support
        print("Loading model...")
        manager.load_model(model_name, n_gpu_layers=-1)
        print("Model loaded!\n")
        
        # Generate response
        print("Generating response...")
        response = manager.generate(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2 + 2?"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        content = response["choices"][0]["message"]["content"]
        print(f"Response: {content}\n")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have a model in the models directory.")
    finally:
        manager.unload_model()
        print("Model unloaded.")


if __name__ == "__main__":
    main()
