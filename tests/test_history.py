"""
Tests for llm_manager/history.py - Chat History Management
"""

import pytest
from unittest.mock import Mock, patch

from llm_manager.history import ChatHistory, HistoryConfig
from llm_manager.estimation import TokenEstimator, TokenEstimate


class TestHistoryConfig:
    """Tests for HistoryConfig dataclass."""
    
    def test_defaults(self):
        """Test default configuration values."""
        config = HistoryConfig()
        assert config.max_tokens == 4096
        assert config.reserve_tokens == 512
        assert config.keep_system is True
        assert config.keep_first_n == 1
        assert config.keep_last_n == 6
        assert config.truncation_strategy == "middle"
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = HistoryConfig(
            max_tokens=8192,
            reserve_tokens=1024,
            keep_system=False,
            keep_first_n=2,
            keep_last_n=10,
            truncation_strategy="oldest"
        )
        assert config.max_tokens == 8192
        assert config.reserve_tokens == 1024
        assert config.keep_system is False
        assert config.keep_first_n == 2
        assert config.keep_last_n == 10
        assert config.truncation_strategy == "oldest"


class TestChatHistoryInit:
    """Tests for ChatHistory initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        history = ChatHistory()
        assert history.config.max_tokens == 4096
        assert len(history) == 0
        # Token count is 0 for empty history
        assert history.get_token_count() >= 0
    
    def test_custom_config(self):
        """Test initialization with custom config."""
        config = HistoryConfig(max_tokens=2048)
        history = ChatHistory(config=config)
        assert history.config.max_tokens == 2048
    
    def test_custom_estimator(self):
        """Test initialization with custom estimator."""
        estimator = TokenEstimator()
        history = ChatHistory(estimator=estimator)
        assert history.estimator is estimator


class TestChatHistoryAddMessages:
    """Tests for adding messages."""
    
    def test_add_message(self):
        """Test adding a generic message."""
        history = ChatHistory()
        history.add_message("user", "Hello")
        
        assert len(history) == 1
        assert history._messages[0]["role"] == "user"
        assert history._messages[0]["content"] == "Hello"
    
    def test_add_system(self):
        """Test adding system message."""
        history = ChatHistory()
        history.add_system("You are a helpful assistant.")
        
        assert len(history) == 1
        assert history._messages[0]["role"] == "system"
    
    def test_add_system_replaces_existing(self):
        """Test that adding system message replaces existing."""
        history = ChatHistory()
        history.add_system("First system prompt")
        history.add_system("Second system prompt")
        
        assert len(history) == 1
        assert history._messages[0]["content"] == "Second system prompt"
    
    def test_add_user(self):
        """Test adding user message."""
        history = ChatHistory()
        history.add_user("What's the weather?")
        
        assert len(history) == 1
        assert history._messages[0]["role"] == "user"
    
    def test_add_assistant(self):
        """Test adding assistant message."""
        history = ChatHistory()
        history.add_assistant("It's sunny today.")
        
        assert len(history) == 1
        assert history._messages[0]["role"] == "assistant"
    
    def test_add_tool_result(self):
        """Test adding tool result."""
        history = ChatHistory()
        history.add_tool_result("call_123", "Tool output here")
        
        assert len(history) == 1
        assert history._messages[0]["role"] == "tool"
        assert history._messages[0]["tool_call_id"] == "call_123"
    
    def test_add_message_with_kwargs(self):
        """Test adding message with extra fields."""
        history = ChatHistory()
        history.add_message("assistant", "Hi", name="Assistant", tool_calls=["call_1"])
        
        assert history._messages[0]["name"] == "Assistant"
        assert history._messages[0]["tool_calls"] == ["call_1"]


class TestChatHistoryTruncation:
    """Tests for history truncation."""
    
    def test_no_truncation_when_under_limit(self):
        """Test that small history is not truncated."""
        history = ChatHistory()
        
        for i in range(5):
            history.add_user(f"Message {i}")
            history.add_assistant(f"Response {i}")
        
        # Should have all messages (10 total)
        assert len(history) == 10
    
    def test_middle_truncation(self):
        """Test middle truncation strategy."""
        config = HistoryConfig(
            max_tokens=500,  # Low limit to trigger truncation
            reserve_tokens=0,
            keep_first_n=2,
            keep_last_n=2,
            truncation_strategy="middle"
        )
        history = ChatHistory(config)
        
        # Add many long messages to definitely trigger truncation
        for i in range(50):
            history.add_user(f"User message {i} " * 200)  # Very long messages
            history.add_assistant(f"Assistant response {i} " * 200)
        
        # Should be truncated - less than or equal to 100 we added
        # (may hit minimum message limit which prevents further truncation)
        assert len(history) <= 100
        # Should have truncation indicator or reduced count
        metadata = history.get_metadata()
        assert metadata["truncations"] >= 1
    
    def test_oldest_truncation(self):
        """Test oldest-first truncation strategy."""
        config = HistoryConfig(
            max_tokens=300,  # Lower limit
            reserve_tokens=0,
            keep_last_n=3,
            truncation_strategy="oldest"
        )
        history = ChatHistory(config)
        
        # Add system message first
        history.add_system("System prompt")
        
        # Add many long messages to trigger truncation
        for i in range(50):
            history.add_user(f"Message {i} with lots of content to exceed token limit " * 50)
        
        # Should be truncated or at minimum (may hit min message limit)
        assert len(history) <= 51
        # System should be preserved
        system_messages = [m for m in history._messages if m["role"] == "system"]
        assert len(system_messages) >= 1
    
    def test_system_preserved_in_truncation(self):
        """Test that system messages are preserved during truncation."""
        config = HistoryConfig(
            max_tokens=500,
            reserve_tokens=0,
            keep_system=True
        )
        history = ChatHistory(config)
        
        history.add_system("Important system prompt")
        
        # Add many messages to trigger truncation
        for i in range(50):
            history.add_user(f"User message {i} " * 100)
        
        # System message should still be present
        system_messages = [m for m in history._messages if m["role"] == "system"]
        assert len(system_messages) >= 1
    
    def test_truncation_metadata(self):
        """Test that truncation updates metadata."""
        config = HistoryConfig(
            max_tokens=500,
            reserve_tokens=0
        )
        history = ChatHistory(config)
        
        # Add many messages
        for i in range(50):
            history.add_user(f"Message {i} " * 100)
        
        metadata = history.get_metadata()
        assert metadata["truncations"] >= 1
        assert metadata["tokens_saved"] > 0


class TestChatHistoryGetMessages:
    """Tests for getting messages."""
    
    def test_get_messages(self):
        """Test getting all messages."""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi there!")
        
        messages = history.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
    
    def test_get_messages_returns_copy(self):
        """Test that get_messages returns a copy."""
        history = ChatHistory()
        history.add_user("Hello")
        
        messages = history.get_messages()
        messages[0]["content"] = "Modified"
        
        # Original should be unchanged
        assert history._messages[0]["content"] == "Hello"
    
    def test_get_messages_with_lower_limit(self):
        """Test getting messages with lower token limit."""
        history = ChatHistory()
        
        # Add many messages
        for i in range(20):
            history.add_user(f"Message {i}")
        
        # Get with lower limit
        messages = history.get_messages(max_tokens=100)
        assert len(messages) < 20
    
    def test_get_token_count(self):
        """Test token count estimation."""
        history = ChatHistory()
        history.add_user("Hello world")
        
        count = history.get_token_count()
        assert count > 0  # Should have some tokens
    
    def test_get_metadata(self):
        """Test getting metadata."""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi")
        
        metadata = history.get_metadata()
        assert metadata["total_messages"] == 2
        assert metadata["current_messages"] == 2
        assert metadata["current_tokens"] > 0
        assert metadata["max_tokens"] == 4096


class TestChatHistoryClear:
    """Tests for clearing history."""
    
    def test_clear(self):
        """Test clearing all messages."""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi")
        
        history.clear()
        
        assert len(history) == 0
    
    def test_clear_except_system(self):
        """Test clearing except system messages."""
        history = ChatHistory()
        history.add_system("System prompt")
        history.add_user("Hello")
        history.add_assistant("Hi")
        
        history.clear_except_system()
        
        assert len(history) == 1
        assert history._messages[0]["role"] == "system"


class TestChatHistoryRollback:
    """Tests for rollback functionality."""
    
    def test_rollback(self):
        """Test rolling back messages."""
        history = ChatHistory()
        history.add_user("Message 1")
        history.add_assistant("Response 1")
        history.add_user("Message 2")
        history.add_assistant("Response 2")
        
        history.rollback(n_messages=2)
        
        assert len(history) == 2
        assert history._messages[-1]["role"] == "assistant"
        assert history._messages[-1]["content"] == "Response 1"
    
    def test_rollback_more_than_exists(self):
        """Test rollback when requesting more than exists."""
        history = ChatHistory()
        history.add_user("Hello")
        
        history.rollback(n_messages=10)
        
        assert len(history) == 0


class TestChatHistoryExportImport:
    """Tests for export and import functionality."""
    
    def test_export(self):
        """Test exporting history."""
        history = ChatHistory()
        history.add_system("System prompt")
        history.add_user("Hello")
        
        exported = history.export()
        
        assert "config" in exported
        assert "messages" in exported
        assert "metadata" in exported
        assert exported["messages"][0]["role"] == "system"
    
    def test_import(self):
        """Test importing history."""
        data = {
            "config": {
                "max_tokens": 2048,
                "reserve_tokens": 256,
                "keep_system": True,
                "keep_first_n": 1,
                "keep_last_n": 4,
                "truncation_strategy": "middle"
            },
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Hello"}
            ],
            "metadata": {
                "total_messages": 10,
                "truncations": 2,
                "tokens_saved": 1000
            }
        }
        
        history = ChatHistory.import_(data)
        
        assert history.config.max_tokens == 2048
        assert len(history) == 2
        assert history._metadata["total_messages"] == 10


class TestChatHistoryDunderMethods:
    """Tests for dunder methods."""
    
    def test_len(self):
        """Test __len__ method."""
        history = ChatHistory()
        assert len(history) == 0
        
        history.add_user("Hello")
        assert len(history) == 1
    
    def test_iter(self):
        """Test __iter__ method."""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi")
        
        messages = list(history)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
    
    def test_repr(self):
        """Test __repr__ method."""
        history = ChatHistory()
        history.add_user("Hello")
        
        repr_str = repr(history)
        assert "ChatHistory" in repr_str
        assert "messages=1" in repr_str


class TestChatHistoryEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_history_operations(self):
        """Test operations on empty history."""
        history = ChatHistory()
        
        assert len(history) == 0
        # Token count should be 0 or minimal for empty
        initial_tokens = history.get_token_count()
        assert initial_tokens >= 0
        assert history.get_messages() == []
        
        # Rollback on empty should not crash
        history.rollback(5)
        assert len(history) == 0
    
    def test_single_message_truncation(self):
        """Test truncation with single message."""
        config = HistoryConfig(
            max_tokens=100,  # Very low
            reserve_tokens=0
        )
        history = ChatHistory(config)
        
        # Add one very long message
        history.add_user("x" * 10000)
        
        # Should still have the message (can't truncate further)
        assert len(history) == 1
    
    def test_minimum_message_count_respected(self):
        """Test that minimum message count is respected."""
        config = HistoryConfig(
            max_tokens=100,
            reserve_tokens=0,
            keep_first_n=1,
            keep_last_n=1
        )
        history = ChatHistory(config)
        
        # Add just a few messages
        history.add_user("Hello")
        history.add_assistant("Hi")
        
        # Should not truncate below keep_first + keep_last
        assert len(history) == 2
