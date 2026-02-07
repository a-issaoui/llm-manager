#!/usr/bin/env python3
"""
Example 09: Tool/Function Calling

Demonstrates using models with tool calling capabilities for agent-like behavior.
Uses Qwen2.5 which supports function calling.
"""

import json
import logging
from pathlib import Path

from llm_manager import LLMManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Define available tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to calculate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


def get_weather(location: str, unit: str = "celsius") -> str:
    """Mock weather function."""
    return f"The weather in {location} is sunny, 22°{unit[0].upper()}"


def calculate(expression: str) -> str:
    """Mock calculator function."""
    try:
        # Safe evaluation for simple math
        allowed = {"__builtins__": {}}
        result = eval(expression, allowed, {"abs": abs, "max": max, "min": min, "round": round})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def main():
    """Run tool calling example."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    model_name = "Qwen2.5-3b-instruct-q4_k_m.gguf"
    
    logger.info("Initializing tool calling example")
    
    manager = LLMManager(
        models_dir=str(models_dir),
        use_subprocess=False,
        verbose=True
    )
    
    logger.info(f"Loading model: {model_name}")
    
    success = manager.load_model(model_name, config={
        "n_ctx": 4096,
        "n_gpu_layers": 0,
    })
    
    if not success:
        logger.error("Failed to load model!")
        return 1
    
    # Test queries that should trigger tool use
    test_queries = [
        "What's the weather like in Paris?",
        "Calculate 15 * 24 + 7",
    ]
    
    print("\n" + "="*70)
    print("TOOL CALLING DEMONSTRATION")
    print("="*70)
    
    for query in test_queries:
        print(f"\n{'─'*70}")
        print(f"User: {query}")
        print(f"{'─'*70}")
        
        # Create messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to tools. "
                           "Use the available functions when appropriate."
            },
            {"role": "user", "content": query}
        ]
        
        # Generate with tools
        response = manager.generate(
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=256,
            temperature=0.3
        )
        
        # Check if model wants to call a tool
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])
        
        if tool_calls:
            print("Assistant wants to call tools:")
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = function.get("name")
                arguments = json.loads(function.get("arguments", "{}"))
                
                print(f"  → {name}({json.dumps(arguments)})")
                
                # Execute the tool
                if name == "get_weather":
                    result = get_weather(**arguments)
                elif name == "calculate":
                    result = calculate(**arguments)
                else:
                    result = f"Unknown tool: {name}"
                
                print(f"  ← Result: {result}")
                
                # Add tool response to messages
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": result
                })
            
            # Get final response
            final_response = manager.generate(
                messages=messages,
                max_tokens=256,
                temperature=0.7
            )
            
            final_text = final_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"\nFinal Answer: {final_text}")
        else:
            # Direct response
            text = message.get("content", "")
            print(f"Assistant: {text}")
    
    print(f"\n{'='*70}")
    
    # Cleanup
    manager.unload_model()
    logger.info("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
