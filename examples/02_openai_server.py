#!/usr/bin/env python3
"""
Example 02: OpenAI-Compatible Server

Demonstrates:
- Starting the OpenAI-compatible server
- Using OpenAI SDK client
- Streaming responses
- Chat completions
- Legacy completions
- Model listing

Prerequisites:
    pip install openai

Usage:
    # Terminal 1: Start server
    llm-manager --port 8000
    
    # Terminal 2: Run example
    python examples/02_openai_server.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_manager import get_config


def get_first_model():
    """Get first available model from registry."""
    models_file = get_config().models.get_registry_path()
    if models_file.exists():
        with open(models_file) as f:
            models = json.load(f)
        if models:
            return list(models.keys())[0]
    return None


def example_basic_chat_completion():
    """Basic chat completion with OpenAI SDK."""
    print("=" * 60)
    print("OpenAI Server: Basic Chat Completion")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    try:
        from openai import OpenAI
        
        # Connect to local server
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="not-needed"  # Local server doesn't require auth
        )
        
        # Simple chat completion
        print("Sending request...")
        response = client.chat.completions.create(
            model=model_name,  # Model name from your registry
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning?"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens}\n")
        
    except Exception as e:
        print(f"Note: {e}")
        print("Make sure server is running: llm-manager\n")


def example_streaming_chat():
    """Streaming chat completion."""
    print("=" * 60)
    print("OpenAI Server: Streaming Chat")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="not-needed"
        )
        
        print("Streaming response:")
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Count from 1 to 5"}
            ],
            stream=True
        )
        
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
        
        print("\n")
        
    except Exception as e:
        print(f"Note: {e}\n")


def example_list_models():
    """List available models."""
    print("=" * 60)
    print("OpenAI Server: List Models")
    print("=" * 60)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="not-needed"
        )
        
        print("Available models:")
        models = client.models.list()
        for model in models.data:
            print(f"  - {model.id}")
        print()
        
    except Exception as e:
        print(f"Note: {e}\n")


def example_legacy_completion():
    """Legacy text completion endpoint."""
    print("=" * 60)
    print("OpenAI Server: Legacy Completion")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="not-needed"
        )
        
        response = client.completions.create(
            model=model_name,
            prompt="Once upon a time",
            max_tokens=50
        )
        
        print(f"Completion: {response.choices[0].text}\n")
        
    except Exception as e:
        print(f"Note: {e}\n")


def example_with_curl():
    """Example curl commands."""
    print("=" * 60)
    print("OpenAI Server: curl Examples")
    print("=" * 60)
    
    model_name = get_first_model() or "your-model"
    
    examples = f"""
# Chat completion
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{model_name}",
    "messages": [{{"role": "user", "content": "Hello!"}}],
    "temperature": 0.7
  }}'

# List models
curl http://localhost:8000/v1/models

# Health check
curl http://localhost:8000/health

# Server info
curl http://localhost:8000/info
"""
    print(examples)


def example_with_auth():
    """Example with API key authentication."""
    print("=" * 60)
    print("OpenAI Server: With Authentication")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    try:
        from openai import OpenAI
        
        # If server started with --api-key
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="your-secret-key"  # Must match server's --api-key
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        print(f"Response: {response.choices[0].message.content}\n")
        
    except Exception as e:
        print(f"Note: {e}")
        print("Server must be started with: llm-manager server --api-key your-secret-key\n")


def main():
    """Run OpenAI server examples."""
    print("\n" + "=" * 60)
    print("LLM Manager - OpenAI-Compatible Server Examples")
    print("=" * 60)
    print("\nPrerequisites:")
    print("  1. Start server: llm-manager")
    print("  2. Install: pip install openai\n")
    
    # Run examples (will fail gracefully if server not running)
    example_list_models()
    example_basic_chat_completion()
    example_streaming_chat()
    example_legacy_completion()
    example_with_curl()
    example_with_auth()


if __name__ == "__main__":
    main()
