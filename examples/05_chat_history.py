#!/usr/bin/env python3
"""
Example 05: Chat History Management

Demonstrates:
- ChatHistory class for agent conversations
- Automatic truncation strategies
- Rollback and branching
- Export/import

No external dependencies required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_manager.history import ChatHistory, HistoryConfig


def example_basic_history():
    """Basic chat history usage."""
    print("=" * 60)
    print("Chat History: Basic Usage")
    print("=" * 60)
    
    # Create history with default config
    history = ChatHistory()
    
    # Add messages
    history.add_system("You are a helpful coding assistant.")
    history.add_user("How do I create a list in Python?")
    history.add_assistant("You can create a list like this: my_list = [1, 2, 3]")
    history.add_user("How do I append to it?")
    
    # Get messages for API
    messages = history.get_messages()
    
    print(f"History has {len(history)} messages:")
    for msg in messages:
        print(f"  [{msg['role']}] {msg['content'][:50]}...")
    
    print(f"Token count: {history.get_token_count()}\n")


def example_truncation_middle():
    """Middle truncation strategy."""
    print("=" * 60)
    print("Chat History: Middle Truncation")
    print("=" * 60)
    
    # Config with middle truncation
    config = HistoryConfig(
        max_tokens=500,  # Low limit to trigger truncation
        truncation_strategy="middle",
        keep_first_n=2,
        keep_last_n=2
    )
    
    history = ChatHistory(config)
    history.add_system("System prompt")
    
    # Add many messages
    for i in range(20):
        history.add_user(f"Question {i}: How does Python work?")
        history.add_assistant(f"Answer {i}: Python is a programming language.")
    
    print(f"Added 40+ messages, history has {len(history)} messages")
    print(f"Token count: {history.get_token_count()} / {config.max_tokens}")
    
    messages = history.get_messages()
    print(f"Kept {len(messages)} messages after truncation")
    print(f"First message: {messages[0]['content'][:30]}...")
    print(f"Last message: {messages[-1]['content'][:30]}...\n")


def example_truncation_oldest():
    """Oldest-first truncation strategy."""
    print("=" * 60)
    print("Chat History: Oldest Truncation")
    print("=" * 60)
    
    config = HistoryConfig(
        max_tokens=500,
        truncation_strategy="oldest",
        keep_last_n=3
    )
    
    history = ChatHistory(config)
    history.add_system("Important system prompt - must be kept")
    
    for i in range(20):
        history.add_user(f"Old message {i}")
    
    print(f"History has {len(history)} messages")
    messages = history.get_messages()
    
    # System message should be preserved
    if messages[0]["role"] == "system":
        print("✓ System message preserved")
    
    print(f"Recent messages kept: {len([m for m in messages if m['role'] == 'user'])}\n")


def example_rollback():
    """Rollback functionality."""
    print("=" * 60)
    print("Chat History: Rollback")
    print("=" * 60)
    
    history = ChatHistory()
    history.add_system("System")
    history.add_user("Question 1")
    history.add_assistant("Answer 1")
    history.add_user("Question 2")
    history.add_assistant("Answer 2")
    history.add_user("Question 3")
    
    print(f"Before rollback: {len(history)} messages")
    
    # Rollback last 2 messages (user question + assistant answer)
    history.rollback(2)
    
    print(f"After rollback: {len(history)} messages")
    
    messages = history.get_messages()
    print(f"Last message: [{messages[-1]['role']}] {messages[-1]['content']}\n")


def example_export_import():
    """Export and import history."""
    print("=" * 60)
    print("Chat History: Export/Import")
    print("=" * 60)
    
    # Create and populate history
    history1 = ChatHistory()
    history1.add_system("System prompt")
    history1.add_user("Hello")
    history1.add_assistant("Hi there!")
    
    # Export
    data = history1.export()
    print(f"Exported {len(data['messages'])} messages")
    print(f"Export format: {list(data.keys())}")
    
    # Import into new history
    history2 = ChatHistory()
    history2.import_(data)
    
    print(f"Imported {len(history2)} messages")
    print(f"Token count: {history2.get_token_count()}\n")


def example_metadata():
    """History metadata."""
    print("=" * 60)
    print("Chat History: Metadata")
    print("=" * 60)
    
    history = ChatHistory()
    history.add_system("System")
    
    for i in range(5):
        history.add_user(f"Question {i}")
        history.add_assistant(f"Answer {i}")
    
    metadata = history.get_metadata()
    
    print("Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print()


def example_branching_conversation():
    """Branching conversation example."""
    print("=" * 60)
    print("Chat History: Branching Conversation")
    print("=" * 60)
    
    # Main conversation
    main_history = ChatHistory()
    main_history.add_system("You are a helpful assistant.")
    main_history.add_user("Tell me about Python")
    main_history.add_assistant("Python is a versatile programming language...")
    main_history.add_user("What about data science?")
    
    # Create branch at this point
    branch_point = main_history.export()
    
    # Continue main conversation
    main_history.add_assistant("Python is great for data science with pandas...")
    main_history.add_user("Tell me about machine learning")
    
    # Create branch for web development topic
    web_branch = ChatHistory()
    web_branch.import_(branch_point)
    web_branch.add_assistant("Python has Django and Flask for web development...")
    web_branch.add_user("Which is better?")
    
    print(f"Main conversation: {len(main_history)} messages")
    print(f"Web dev branch: {len(web_branch)} messages")
    
    print("\nMain conversation topics:")
    for msg in main_history.get_messages():
        if msg['role'] == 'user':
            print(f"  - {msg['content']}")
    
    print("\nWeb dev branch topics:")
    for msg in web_branch.get_messages():
        if msg['role'] == 'user':
            print(f"  - {msg['content']}")
    print()


def main():
    """Run chat history examples."""
    print("\n" + "=" * 60)
    print("LLM Manager - Chat History Examples")
    print("=" * 60 + "\n")
    
    example_basic_history()
    example_truncation_middle()
    example_truncation_oldest()
    example_rollback()
    example_export_import()
    example_metadata()
    example_branching_conversation()


if __name__ == "__main__":
    main()
