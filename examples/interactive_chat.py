#!/usr/bin/env python3
"""
Interactive Chat Demo

A simple interactive chat that uses the OpenAI-compatible server.

Usage:
    1. Start server: llm-manager --model qwen2.5-7b
    2. Run: python examples/interactive_chat.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_first_model():
    """Get first model from registry."""
    from llm_manager import get_config
    import json
    models_file = get_config().models.get_registry_path()
    if models_file.exists():
        with open(models_file) as f:
            models = json.load(f)
        if models:
            return list(models.keys())[0]
    return None


def main():
    print("\n" + "="*60)
    print("  LLM Manager - Interactive Chat")
    print("="*60)
    
    # Get model from registry
    from llm_manager import get_config
    config = get_config()
    model_name = get_first_model()
    if not model_name:
        registry_path = config.models.get_registry_path()
        print(f"\nNo models found in {registry_path}")
        print("Please download a model first.")
        return
    
    print(f"\nUsing model: {model_name}")
    print("\nMake sure server is running:")
    print(f"  llm-manager --model {model_name}")
    print("\nPress Ctrl+C to exit\n")
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="not-needed"
        )
        
        # Get available models from server
        try:
            models = client.models.list()
            model_id = models.data[0].id if models.data else model_name
            print(f"Server model: {model_id}\n")
        except:
            model_id = model_name
            print(f"Using model: {model_id}\n")
        
        # Chat history
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        
        while True:
            # Get user input
            try:
                user_input = input("You: ")
            except EOFError:
                break
            
            if not user_input.strip():
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            
            # Add to history
            messages.append({"role": "user", "content": user_input})
            
            # Get response with streaming
            print("Assistant: ", end="", flush=True)
            
            try:
                stream = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    stream=True,
                    max_tokens=500
                )
                
                response_text = ""
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        print(content, end="", flush=True)
                        response_text += content
                
                print()  # Newline
                
                # Add to history
                messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                print(f"\n[Error: {e}]")
                print("Make sure server is running!")
    
    except ImportError:
        print("\nError: openai package not installed")
        print("Install: pip install openai")
        return
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    
    print("\n" + "="*60)
    print("Chat session ended.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
