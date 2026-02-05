#!/usr/bin/env python3
"""
Example 10: Complete Agent Workflow

A comprehensive example combining all features:
- Configuration management
- Model loading with metrics
- Chat history management
- Batch processing
- OpenAI-compatible API
- Agent integration

This is a reference implementation showing best practices.
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_manager import (
    LLMManager, Config, get_config, get_global_metrics,
    ChatHistory, HistoryConfig,
)


def get_first_model():
    """Get first available model from registry."""
    models_file = get_config().models.get_registry_path()
    if models_file.exists():
        with open(models_file) as f:
            models = json.load(f)
        if models:
            return list(models.keys())[0]
    return None


class AgentSystem:
    """Complete agent system using llm_manager."""
    
    def __init__(self, models_dir: str = None):
        from llm_manager import get_config
        if models_dir is None:
            models_dir = get_config().models.dir
        """Initialize the agent system."""
        self.models_dir = models_dir
        self.manager: LLMManager = None
        self.conversations: Dict[str, ChatHistory] = {}
        self.metrics = get_global_metrics()
        
    async def initialize(self, model_name: str):
        """Initialize with a model."""
        print(f"Initializing agent system with {model_name}...")
        
        # Create manager with metrics tracking
        self.manager = LLMManager(
            models_dir=self.models_dir,
            use_subprocess=True,  # Safer
            pool_size=4  # Concurrent workers
        )
        
        # Load model
        await self.manager.load_model_async(model_name, n_gpu_layers=-1)
        print(f"✓ Model loaded: {model_name}")
        
    def create_conversation(self, conv_id: str, system_prompt: str = None) -> ChatHistory:
        """Create a new conversation."""
        config = HistoryConfig(
            max_tokens=4000,
            truncation_strategy="middle",
            keep_system=True
        )
        
        history = ChatHistory(config)
        
        if system_prompt:
            history.add_system(system_prompt)
        
        self.conversations[conv_id] = history
        return history
    
    async def chat(self, conv_id: str, user_message: str) -> str:
        """Send a message and get response."""
        # Get or create conversation
        if conv_id not in self.conversations:
            self.create_conversation(conv_id, "You are a helpful assistant.")
        
        history = self.conversations[conv_id]
        
        # Add user message
        history.add_user(user_message)
        
        # Generate with metrics tracking
        start_time = asyncio.get_event_loop().time()
        
        with self.metrics.record_callback(
            model_name=self.manager.model_path.stem if self.manager.model_path else "unknown",
            output_tokens=0  # Will be updated
        ) as cb:
            
            response = await self.manager.generate_async(
                messages=history.get_messages(),
                max_tokens=500,
                temperature=0.7
            )
            
            content = response["choices"][0]["message"]["content"]
            
            # Update callback with actual output
            cb.set_output_info(
                output_tokens=len(content) // 4,
                success=True
            )
        
        # Add assistant response to history
        history.add_assistant(content)
        
        return content
    
    async def chat_batch(self, conv_id: str, messages: List[str]) -> List[str]:
        """Process multiple messages in batch."""
        if conv_id not in self.conversations:
            self.create_conversation(conv_id)
        
        history = self.conversations[conv_id]
        
        # Prepare prompts
        prompts = []
        for msg in messages:
            temp_history = ChatHistory()
            temp_history._messages = history._messages.copy()
            temp_history.add_user(msg)
            prompts.append(temp_history.get_messages())
        
        # Generate batch
        results = await self.manager.generate_batch(prompts)
        
        # Extract responses
        responses = []
        for result in results:
            if "choices" in result:
                content = result["choices"][0]["message"]["content"]
                responses.append(content)
            else:
                responses.append("Error generating response")
        
        return responses
    
    def get_conversation_summary(self, conv_id: str) -> Dict:
        """Get summary of a conversation."""
        if conv_id not in self.conversations:
            return {"error": "Conversation not found"}
        
        history = self.conversations[conv_id]
        metadata = history.get_metadata()
        
        return {
            "message_count": len(history),
            "token_count": history.get_token_count(),
            "truncations": metadata["truncations"]
        }
    
    def export_conversation(self, conv_id: str) -> Dict:
        """Export conversation data."""
        if conv_id not in self.conversations:
            return None
        
        return self.conversations[conv_id].export()
    
    def get_system_stats(self) -> Dict:
        """Get system statistics."""
        stats = self.metrics.get_stats()
        
        return {
            "total_requests": stats.total_requests,
            "total_tokens": stats.total_tokens,
            "tokens_per_second": stats.tokens_per_second,
            "success_rate": stats.success_rate,
            "avg_latency_ms": stats.latency_p50_ms,
            "conversations": len(self.conversations)
        }
    
    async def shutdown(self):
        """Shutdown the system."""
        if self.manager:
            self.manager.unload_model()
            print("✓ Model unloaded")


async def demo_basic_chat():
    """Demo: Basic chat."""
    print("\n" + "=" * 60)
    print("Demo: Basic Chat")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    agent = AgentSystem(models_dir=get_config().models.dir)
    
    try:
        await agent.initialize(model_name)
        
        # Single conversation
        response = await agent.chat(
            conv_id="user_123",
            user_message="What is Python?"
        )
        print(f"User: What is Python?")
        print(f"Agent: {response[:100]}...")
        
        # Continue conversation
        response = await agent.chat(
            conv_id="user_123",
            user_message="What can I build with it?"
        )
        print(f"User: What can I build with it?")
        print(f"Agent: {response[:100]}...")
        
    except Exception as e:
        print(f"Note: {e} (expected if model not available)")
    finally:
        await agent.shutdown()


async def demo_multiple_conversations():
    """Demo: Multiple concurrent conversations."""
    print("\n" + "=" * 60)
    print("Demo: Multiple Conversations")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    agent = AgentSystem(models_dir=get_config().models.dir)
    
    try:
        await agent.initialize(model_name)
        
        # Create multiple conversations
        agent.create_conversation(
            "coder",
            "You are an expert Python programmer."
        )
        agent.create_conversation(
            "writer",
            "You are a creative writing assistant."
        )
        
        print("Created 2 conversations with different personas")
        
        # Chat in both
        code_response = await agent.chat("coder", "How do I read a file?")
        write_response = await agent.chat("writer", "Write a haiku about coding")
        
        print(f"Coder response: {code_response[:80]}...")
        print(f"Writer response: {write_response[:80]}...")
        
        # Get summaries
        print("\nConversation summaries:")
        for conv_id in ["coder", "writer"]:
            summary = agent.get_conversation_summary(conv_id)
            print(f"  {conv_id}: {summary}")
        
    except Exception as e:
        print(f"Note: {e}")
    finally:
        await agent.shutdown()


async def demo_batch_processing():
    """Demo: Batch processing."""
    print("\n" + "=" * 60)
    print("Demo: Batch Processing")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    agent = AgentSystem(models_dir=get_config().models.dir)
    
    try:
        await agent.initialize(model_name)
        
        questions = [
            "What is AI?",
            "What is ML?",
            "What is DL?",
            "What is NN?"
        ]
        
        print(f"Processing {len(questions)} questions...")
        
        responses = await agent.chat_batch("batch_demo", questions)
        
        for q, r in zip(questions, responses):
            print(f"Q: {q}")
            print(f"A: {r[:60]}...")
            print()
        
    except Exception as e:
        print(f"Note: {e}")
    finally:
        await agent.shutdown()


def demo_configuration():
    """Demo: Configuration."""
    print("\n" + "=" * 60)
    print("Demo: Configuration")
    print("=" * 60)
    
    # Custom configuration
    config = Config.from_dict({
        "models": {
            "dir": "./my_models",
            "pool_size": 4
        },
        "generation": {
            "default_temperature": 0.5,
            "default_max_tokens": 1000
        },
        "context": {
            "auto_resize": True,
            "safety_margin": 256
        }
    })
    
    print("Custom configuration:")
    print(f"  Models dir: {config.models.dir}")
    print(f"  Pool size: {config.models.pool_size}")
    print(f"  Temperature: {config.generation.default_temperature}")
    print(f"  Auto resize: {config.context.auto_resize}")
    print()


def demo_metrics_dashboard():
    """Demo: Metrics dashboard."""
    print("\n" + "=" * 60)
    print("Demo: Metrics Dashboard")
    print("=" * 60)
    
    from llm_manager.metrics import MetricsCollector
    
    metrics = MetricsCollector()
    
    # Simulate some requests
    import random
    for i in range(20):
        model_choice = get_first_model() or "qwen2.5-7b"
        metrics.record_request(
            model_name=model_choice,
            input_tokens=random.randint(50, 200),
            output_tokens=random.randint(50, 300),
            duration_ms=random.randint(50, 500),
            success=random.random() > 0.1  # 90% success
        )
    
    stats = metrics.get_stats()
    
    print("System Metrics:")
    print(f"  Total Requests: {stats.total_requests}")
    print(f"  Success Rate: {stats.success_rate:.1f}%")
    print(f"  Tokens/sec: {stats.tokens_per_second:.2f}")
    print(f"  Latency P50: {stats.latency_p50_ms:.0f}ms")
    print(f"  Latency P95: {stats.latency_p95_ms:.0f}ms")
    print()
    
    # Model breakdown
    breakdown = metrics.get_model_breakdown()
    print("Requests by Model:")
    for model, count in breakdown.items():
        print(f"  {model}: {count}")
    print()


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("LLM Manager - Complete Agent Workflow")
    print("=" * 60)
    print("\nThis example shows a complete agent system combining all features.\n")
    
    # Config and metrics demos (no model needed)
    demo_configuration()
    demo_metrics_dashboard()
    
    # Model-dependent demos (run with actual models)
    await demo_basic_chat()
    await demo_multiple_conversations()
    await demo_batch_processing()
    
    print("\nAll demos completed!")


if __name__ == "__main__":
    asyncio.run(main())
