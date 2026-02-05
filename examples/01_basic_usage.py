#!/usr/bin/env python3
"""
Example 01: Basic Usage

Demonstrates:
- Basic configuration
- Model loading
- Simple generation
- Async generation
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_manager import LLMManager, get_config, Config


def get_models_dir():
    """Get models directory from config."""
    return get_config().models.dir


def get_registry_path():
    """Get registry path from config."""
    return get_config().models.get_registry_path()



def get_first_model():
    """Get first available model from registry."""
    import json
    models_file = Path(get_registry_path())
    if models_file.exists():
        with open(models_file) as f:
            models = json.load(f)
        if models:
            return list(models.keys())[0]
    return None


def example_basic_generation():
    """Basic synchronous generation."""
    print("=" * 60)
    print("Example 1: Basic Generation")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        registry_path = get_registry_path()
        print(f"No models found in {registry_path}")
        return
    
    print(f"Using model from registry: {model_name}")
    
    # Initialize manager (subprocess mode is safer)
    manager = LLMManager(models_dir=get_models_dir(), use_subprocess=False)
    
    try:
        # Load a model from registry (use smaller context for demo)
        print("Loading model...")
        manager.load_model(model_name, n_ctx=4096, n_gpu_layers=-1)
        
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
        print(f"Response: {content}")
        
    finally:
        manager.unload_model()
        print("Model unloaded.\n")


async def example_async_generation():
    """Async generation example."""
    print("=" * 60)
    print("Example 2: Async Generation")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        registry_path = get_registry_path()
        print(f"No models found in {registry_path}")
        return
    
    manager = LLMManager(models_dir=get_models_dir(), use_subprocess=False)
    
    try:
        print(f"Loading model (async): {model_name}")
        await manager.load_model_async(model_name, n_ctx=4096, n_gpu_layers=-1)
        
        print("Generating response (async)...")
        response = await manager.generate_async(
            messages=[
                {"role": "user", "content": "Explain quantum computing in one sentence."}
            ],
            max_tokens=100
        )
        
        content = response["choices"][0]["message"]["content"]
        print(f"Response: {content}")
        
    finally:
        manager.unload_model()
        print("Model unloaded.\n")


def example_configuration():
    """Configuration examples."""
    print("=" * 60)
    print("Example 3: Configuration")
    print("=" * 60)
    
    # Get default config
    config = get_config()
    print(f"Default context size: {config.context.default_size}")
    print(f"Default temperature: {config.generation.default_temperature}")
    
    # Create custom config
    custom_config = Config.from_dict({
        "models": {
            "dir": "./custom_models",
            "pool_size": 4
        },
        "generation": {
            "default_temperature": 0.5,
            "default_max_tokens": 512
        }
    })
    
    print(f"Custom models dir: {custom_config.models.dir}")
    print(f"Custom temperature: {custom_config.generation.default_temperature}\n")


async def example_model_switching():
    """Model hot-swapping example."""
    print("=" * 60)
    print("Example 4: Model Hot-Switching")
    print("=" * 60)
    
    import json
    registry_path = get_registry_path()
    if not registry_path.exists():
        print(f"No models.json found at {registry_path}")
        return
    
    with open(registry_path) as f:
        models = json.load(f)
    
    model_list = list(models.keys())
    if len(model_list) < 2:
        print(f"Need 2+ models for switching, found {len(model_list)}")
        return
    
    manager = LLMManager(models_dir=get_models_dir(), use_subprocess=False)
    
    try:
        # Load first model
        print(f"Loading model A: {model_list[0]}")
        await manager.load_model_async(model_list[0], n_ctx=4096, n_gpu_layers=-1)
        
        # Switch to second model (fast hot-swap)
        print(f"Switching to model B: {model_list[1]} (hot-swap)...")
        success = await manager.switch_model_async(model_list[1], n_ctx=4096, n_gpu_layers=-1)
        print(f"Switch successful: {success}")
        
    except Exception as e:
        print(f"Note: {e}")


def main():
    """Run all basic examples."""
    print("\n" + "=" * 60)
    print("LLM Manager - Basic Usage Examples")
    print("=" * 60 + "\n")
    
    # Configuration (no model needed)
    example_configuration()
    
    # These require actual models:
    example_basic_generation()
    asyncio.run(example_async_generation())
    asyncio.run(example_model_switching())


if __name__ == "__main__":
    main()
