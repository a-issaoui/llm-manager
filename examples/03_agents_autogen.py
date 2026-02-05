#!/usr/bin/env python3
"""
Example 03: AutoGen Agent Integration

Demonstrates:
- Using llm_manager server with Microsoft AutoGen
- Multi-agent conversations
- Agent tools and functions

Prerequisites:
    pip install pyautogen openai

Usage:
    # Start server
    llm-manager --port 8000
    
    # Run example
    python examples/03_agents_autogen.py
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


def example_basic_autogen():
    """Basic AutoGen conversation."""
    print("=" * 60)
    print("AutoGen: Basic Multi-Agent Conversation")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    try:
        from autogen import AssistantAgent, UserProxyAgent
        
        # Configure LLM to use local server
        llm_config = {
            "config_list": [{
                "model": model_name,
                "base_url": "http://localhost:8000/v1",
                "api_key": "not-needed",
            }],
            "temperature": 0.7,
        }
        
        # Create agents
        assistant = AssistantAgent(
            name="assistant",
            llm_config=llm_config,
            system_message="You are a helpful coding assistant."
        )
        
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )
        
        print("Starting conversation...")
        user_proxy.initiate_chat(
            assistant,
            message="Write a Python function to calculate fibonacci numbers."
        )
        
    except ImportError:
        print("Install AutoGen: pip install pyautogen")
    except Exception as e:
        print(f"Note: {e}")
        print("Make sure server is running\n")


def example_group_chat():
    """Group chat with multiple agents."""
    print("=" * 60)
    print("AutoGen: Group Chat")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    example_code = f'''
from autogen import AssistantAgent, GroupChat, GroupChatManager

llm_config = {{
    "config_list": [{{
        "model": "{model_name}",
        "base_url": "http://localhost:8000/v1",
        "api_key": "not-needed",
    }}]
}}

# Create multiple agents
coder = AssistantAgent(
    name="coder",
    llm_config=llm_config,
    system_message="You are a Python expert."
)

reviewer = AssistantAgent(
    name="reviewer", 
    llm_config=llm_config,
    system_message="You review code for best practices."
)

tester = AssistantAgent(
    name="tester",
    llm_config=llm_config,
    system_message="You write unit tests."
)

# Group chat
groupchat = GroupChat(
    agents=[coder, reviewer, tester],
    messages=[],
    max_round=6
)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Start discussion
manager.initiate_chat(
    manager,
    message="Create a calculator class with add, subtract, multiply, divide methods."
)
'''
    print(example_code)


def example_function_calling():
    """AutoGen with function calling."""
    print("=" * 60)
    print("AutoGen: Function Calling")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    example_code = f'''
from autogen import AssistantAgent, UserProxyAgent

# Define functions for the agent
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {{location}}: Sunny, 22°C"

def calculate(expression: str) -> str:
    """Calculate mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {{result}}"
    except:
        return "Invalid expression"

llm_config = {{
    "config_list": [{{
        "model": "{model_name}",
        "base_url": "http://localhost:8000/v1",
        "api_key": "not-needed",
    }}]
}}

# Agent with function calling
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="Use functions when needed."
)

user = UserProxyAgent(
    name="user",
    function_map={{
        "get_weather": get_weather,
        "calculate": calculate
    }}
)

user.initiate_chat(
    assistant,
    message="What's the weather in London and what's 25 * 4?"
)
'''
    print(example_code)


def example_code_executor():
    """AutoGen with code execution."""
    print("=" * 60)
    print("AutoGen: Code Execution")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    example_code = f'''
from autogen import AssistantAgent, UserProxyAgent

llm_config = {{
    "config_list": [{{
        "model": "{model_name}",
        "base_url": "http://localhost:8000/v1",
        "api_key": "not-needed",
    }}]
}}

# Assistant that writes code
assistant = AssistantAgent(
    name="coder",
    llm_config=llm_config,
    system_message="Write Python code to solve problems."
)

# User proxy that executes code
user_proxy = UserProxyAgent(
    name="executor",
    human_input_mode="NEVER",
    code_execution_config={{
        "work_dir": "coding",
        "use_docker": False,  # Set to True if available
    }}
)

# Solve a problem with code
user_proxy.initiate_chat(
    assistant,
    message="""
    Plot a sine wave using matplotlib and save it to 'sine.png'.
    Install matplotlib if needed.
    """
)
'''
    print(example_code)


def main():
    """Run AutoGen examples."""
    print("\n" + "=" * 60)
    print("LLM Manager - AutoGen Agent Examples")
    print("=" * 60)
    print("\nPrerequisites:")
    print("  pip install pyautogen openai")
    print("  llm-manager --port 8000\n")
    
    example_basic_autogen()
    example_group_chat()
    example_function_calling()
    example_code_executor()


if __name__ == "__main__":
    main()
