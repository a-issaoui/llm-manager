#!/usr/bin/env python3
"""
Example 08: Advanced Configuration

Demonstrates:
- YAML configuration files
- Environment variables
- Multiple configuration profiles
- Runtime configuration changes
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_manager.config import (
    Config, get_config, load_config, reload_config,
    create_default_config, ConfigValidationError
)


def example_yaml_config():
    """Load configuration from YAML."""
    print("=" * 60)
    print("Config: YAML Configuration File")
    print("=" * 60)
    
    # Create sample config file
    config_content = """
models:
  dir: ./my_models
  pool_size: 4

generation:
  default_temperature: 0.5
  default_max_tokens: 1024
  auto_context: true

context:
  default_size: 8192
  auto_resize: true
  safety_margin: 256

gpu:
  type_k: q4_0
  type_v: q4_0
  flash_attention: true

logging:
  level: INFO
  format: "%(asctime)s - %(levelname)s - %(message)s"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    try:
        # Load from file
        config = load_config(config_path)
        
        print(f"Loaded config from: {config_path}")
        print(f"Models dir: {config.models.dir}")
        print(f"Pool size: {config.models.pool_size}")
        print(f"Temperature: {config.generation.default_temperature}")
        print(f"KV cache: {config.gpu.type_k}/{config.gpu.type_v}")
        print()
        
    finally:
        os.unlink(config_path)


def example_environment_variables():
    """Configuration via environment variables."""
    print("=" * 60)
    print("Config: Environment Variables")
    print("=" * 60)
    
    # Show available environment variables
    env_vars = {
        "LLM_MODELS_DIR": "Directory containing models",
        "LLM_DEFAULT_MODEL": "Default model to load",
        "LLM_LOG_LEVEL": "Logging level (DEBUG/INFO/WARNING/ERROR)",
        "LLM_API_KEY": "API key for server authentication",
        "LLM_CONTEXT_SIZE": "Default context window size",
        "LLM_GPU_LAYERS": "Number of GPU layers to use",
        "LLM_BATCH_SIZE": "Processing batch size",
    }
    
    print("Supported environment variables:")
    for var, desc in env_vars.items():
        value = os.getenv(var, "<not set>")
        print(f"  {var}")
        print(f"    Description: {desc}")
        print(f"    Current value: {value}")
    print()


def example_multiple_profiles():
    """Multiple configuration profiles."""
    print("=" * 60)
    print("Config: Multiple Profiles")
    print("=" * 60)
    
    # Development profile
    dev_config = Config.from_dict({
        "models": {"dir": "./dev_models", "pool_size": 1},  # CPU only for dev
        "logging": {"level": "DEBUG"},
        "cache": {"disk_enabled": False}  # Disable cache in dev
    })
    
    # Production profile
    prod_config = Config.from_dict({
        "models": {"dir": "./prod_models", "pool_size": 8},  # All GPU layers
        "logging": {"level": "WARNING"},
        "cache": {"disk_enabled": True, "ttl_seconds": 7200}
    })
    
    print("Development Profile:")
    print(f"  Models dir: {dev_config.models.dir}")
    print(f"  Pool size: {dev_config.models.pool_size}")
    print(f"  Log Level: {dev_config.logging.level}")
    
    print("\nProduction Profile:")
    print(f"  Models dir: {prod_config.models.dir}")
    print(f"  Pool size: {prod_config.models.pool_size}")
    print(f"  Log Level: {prod_config.logging.level}")
    print()


def example_validation():
    """Configuration validation."""
    print("=" * 60)
    print("Config: Validation")
    print("=" * 60)
    
    from llm_manager.config import GenerationConfig, ContextConfig
    
    # Valid configs
    try:
        gen = GenerationConfig(default_temperature=0.5, default_top_p=0.9)
        print(f"✓ Valid generation config: temp={gen.default_temperature}")
    except ConfigValidationError as e:
        print(f"✗ Validation error: {e}")
    
    # Invalid configs
    try:
        GenerationConfig(default_temperature=5.0)  # Too high
    except ConfigValidationError as e:
        print(f"✓ Caught invalid temperature: {e}")
    
    try:
        ContextConfig(min_size=10000, max_size=1000)  # Invalid range
    except ConfigValidationError as e:
        print(f"✓ Caught invalid context range: {e}")
    
    print()


def example_config_merge():
    """Merge configurations."""
    print("=" * 60)
    print("Config: Merging")
    print("=" * 60)
    
    # Base config
    base = Config.from_dict({
        "models": {"pool_size": 2},
        "generation": {"default_temperature": 0.7}
    })
    
    # Override
    override = {
        "models": {"pool_size": 8},  # Change this
        # generation keeps base value
    }
    
    # Manual merge
    merged = Config.from_dict(base.to_dict())
    merged.models.pool_size = override["models"]["pool_size"]
    
    print("Base config:")
    print(f"  Pool size: {base.models.pool_size}")
    print(f"  Temperature: {base.generation.default_temperature}")
    
    print("\nMerged config:")
    print(f"  Pool size: {merged.models.pool_size} (overridden)")
    print(f"  Temperature: {merged.generation.default_temperature} (inherited)")
    print()


def example_serialization():
    """Config serialization."""
    print("=" * 60)
    print("Config: Serialization")
    print("=" * 60)
    
    config = Config.from_dict({
        "models": {"pool_size": 4},
        "gpu": {"type_k": "q4_0", "type_v": "q4_0"}
    })
    
    # To dict
    data = config.to_dict()
    print(f"Serialized keys: {list(data.keys())}")
    
    # To JSON
    import json
    json_str = json.dumps(data, indent=2)
    print(f"JSON size: {len(json_str)} bytes")
    
    # Round-trip
    config2 = Config.from_dict(json.loads(json_str))
    print(f"Round-trip successful: {config == config2}")
    print()


def example_create_default():
    """Create default configuration file."""
    print("=" * 60)
    print("Config: Create Default File")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "llm_manager.yaml")
        
        create_default_config(config_path)
        
        print(f"Created: {config_path}")
        print(f"File size: {os.path.getsize(config_path)} bytes")
        
        # Show first few lines
        with open(config_path) as f:
            lines = f.readlines()[:10]
            print("First 10 lines:")
            for line in lines:
                print(f"  {line.rstrip()}")
    
    print()


def main():
    """Run all config examples."""
    print("\n" + "=" * 60)
    print("LLM Manager - Advanced Configuration Examples")
    print("=" * 60 + "\n")
    
    example_yaml_config()
    example_environment_variables()
    example_multiple_profiles()
    example_validation()
    example_config_merge()
    example_serialization()
    example_create_default()


if __name__ == "__main__":
    main()
