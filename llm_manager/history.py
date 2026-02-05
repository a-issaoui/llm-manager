"""
Chat history management for agent workflows.

Provides automatic context window management with smart truncation,
preserving system prompts and recent context while removing older messages.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from copy import deepcopy

from .estimation import TokenEstimator
from .context import ContextManager

logger = logging.getLogger(__name__)


@dataclass
class HistoryConfig:
    """Configuration for chat history management."""
    max_tokens: int = 4096
    reserve_tokens: int = 512  # Tokens to reserve for response
    keep_system: bool = True  # Always keep system messages
    keep_first_n: int = 1  # Keep first N messages (often contains instructions)
    keep_last_n: int = 6  # Keep last N messages (recent context)
    truncation_strategy: str = "middle"  # "middle", "oldest", "summary"


class ChatHistory:
    """
    Manages chat history with automatic context window management.
    
    Perfect for long-running agents that need to maintain conversation
    state without exceeding context limits.
    
    Example:
        >>> history = ChatHistory(max_tokens=4096)
        >>> history.add_system("You are a coding assistant.")
        >>> history.add_user("Write a function to sort a list")
        >>> history.add_assistant("Here's a Python function...")
        >>> 
        >>> # Auto-truncates when adding would exceed limit
        >>> for i in range(100):
        ...     history.add_user(f"Task {i}")
        ...     response = manager.generate(history.get_messages())
        ...     history.add_assistant(response)
    """
    
    def __init__(
        self,
        config: Optional[HistoryConfig] = None,
        estimator: Optional[TokenEstimator] = None
    ):
        """
        Initialize chat history manager.
        
        Args:
            config: History configuration
            estimator: Token estimator for size calculations
        """
        self.config = config or HistoryConfig()
        self.estimator = estimator or TokenEstimator()
        self._messages: List[Dict[str, str]] = []
        self._metadata: Dict[str, Any] = {
            "total_messages": 0,
            "truncations": 0,
            "tokens_saved": 0
        }
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """
        Add a message to history with automatic truncation.
        
        Args:
            role: Message role (system, user, assistant, tool)
            content: Message content
            **kwargs: Additional message fields (name, tool_calls, etc.)
        """
        message = {"role": role, "content": content, **kwargs}
        self._messages.append(message)
        self._metadata["total_messages"] += 1
        
        # Check if we need to truncate
        self._maybe_truncate()
    
    def add_system(self, content: str) -> None:
        """Add a system message."""
        # Remove existing system messages if configured
        if self.config.keep_system:
            self._messages = [m for m in self._messages if m["role"] != "system"]
        self.add_message("system", content)
    
    def add_user(self, content: str, **kwargs) -> None:
        """Add a user message."""
        self.add_message("user", content, **kwargs)
    
    def add_assistant(self, content: str, **kwargs) -> None:
        """Add an assistant message."""
        self.add_message("assistant", content, **kwargs)
    
    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Add a tool/function result message."""
        self.add_message("tool", content, tool_call_id=tool_call_id)
    
    def _maybe_truncate(self) -> None:
        """Truncate history if it exceeds token limit."""
        estimate = self.estimator.estimate_heuristic(self._messages)
        available_tokens = self.config.max_tokens - self.config.reserve_tokens
        
        if estimate.total_tokens <= available_tokens:
            return
        
        # Need to truncate
        logger.debug(f"History exceeds limit ({estimate.total_tokens} > {available_tokens}), truncating")
        self._truncate(available_tokens)
    
    def _truncate(self, max_tokens: int) -> None:
        """
        Truncate history to fit within token limit.
        
        Uses configured strategy to determine which messages to remove.
        """
        if len(self._messages) <= self.config.keep_first_n + self.config.keep_last_n:
            logger.warning("Cannot truncate further, minimum message count reached")
            return
        
        strategy = self.config.truncation_strategy
        
        if strategy == "middle":
            self._truncate_middle(max_tokens)
        elif strategy == "oldest":
            self._truncate_oldest(max_tokens)
        elif strategy == "summary":
            self._truncate_with_summary(max_tokens)
        else:
            self._truncate_middle(max_tokens)
        
        self._metadata["truncations"] += 1
    
    def _truncate_middle(self, max_tokens: int) -> None:
        """
        Remove messages from the middle, keeping first N and last N.
        This preserves instructions and recent context.
        """
        keep_first = self.config.keep_first_n
        keep_last = self.config.keep_last_n
        
        # Always keep system messages
        system_messages = [m for m in self._messages if m["role"] == "system"]
        non_system = [m for m in self._messages if m["role"] != "system"]
        
        if len(non_system) <= keep_first + keep_last:
            return
        
        # Keep first N and last N non-system messages
        kept = non_system[:keep_first] + non_system[-keep_last:]
        removed = non_system[keep_first:-keep_last]
        
        # Calculate tokens saved
        removed_estimate = self.estimator.estimate_heuristic(removed)
        self._metadata["tokens_saved"] += removed_estimate.total_tokens
        
        self._messages = system_messages + kept
        
        # Add truncation indicator
        self._messages.insert(
            keep_first + len(system_messages),
            {"role": "system", "content": f"... {len(removed)} earlier messages truncated ..."}
        )
        
        logger.info(f"Truncated {len(removed)} messages from middle of history")
    
    def _truncate_oldest(self, max_tokens: int) -> None:
        """Remove oldest messages first."""
        while len(self._messages) > self.config.keep_last_n:
            removed = self._messages.pop(0)
            if removed["role"] == "system" and self.config.keep_system:
                # Put system message back at start
                self._messages.insert(0, removed)
                break
            
            removed_estimate = self.estimator.estimate_heuristic([removed])
            self._metadata["tokens_saved"] += removed_estimate.total_tokens
            
            # Check if we're under limit
            current = self.estimator.estimate_heuristic(self._messages)
            if current.total_tokens <= max_tokens:
                break
    
    def _truncate_with_summary(self, max_tokens: int) -> None:
        """
        Replace truncated messages with a summary.
        Note: This is a placeholder - actual summarization would require
        calling the LLM itself.
        """
        # For now, just use middle truncation
        self._truncate_middle(max_tokens)
    
    def get_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get current messages, optionally with a different token limit.
        
        Args:
            max_tokens: Optional override for max tokens
            
        Returns:
            List of messages
        """
        if max_tokens is None or max_tokens >= self.config.max_tokens:
            return deepcopy(self._messages)
        
        # Create temporary history with different limit
        temp_config = HistoryConfig(
            max_tokens=max_tokens,
            reserve_tokens=self.config.reserve_tokens,
            keep_system=self.config.keep_system,
            keep_first_n=self.config.keep_first_n,
            keep_last_n=self.config.keep_last_n
        )
        temp_history = ChatHistory(config=temp_config, estimator=self.estimator)
        temp_history._messages = deepcopy(self._messages)
        temp_history._maybe_truncate()
        
        return temp_history._messages
    
    def get_token_count(self) -> int:
        """Get current token estimate."""
        estimate = self.estimator.estimate_heuristic(self._messages)
        return estimate.total_tokens
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get history metadata."""
        return {
            **self._metadata,
            "current_messages": len(self._messages),
            "current_tokens": self.get_token_count(),
            "max_tokens": self.config.max_tokens
        }
    
    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        logger.debug("History cleared")
    
    def clear_except_system(self) -> None:
        """Clear all messages except system messages."""
        self._messages = [m for m in self._messages if m["role"] == "system"]
        logger.debug("History cleared except system messages")
    
    def rollback(self, n_messages: int = 1) -> None:
        """
        Remove last N messages (for retry scenarios).
        
        Args:
            n_messages: Number of messages to remove
        """
        for _ in range(min(n_messages, len(self._messages))):
            self._messages.pop()
        logger.debug(f"Rolled back {n_messages} messages")
    
    def export(self) -> Dict[str, Any]:
        """
        Export history to dictionary for persistence.
        
        Returns:
            Dictionary with messages and metadata
        """
        return {
            "config": {
                "max_tokens": self.config.max_tokens,
                "reserve_tokens": self.config.reserve_tokens,
                "keep_system": self.config.keep_system,
                "keep_first_n": self.config.keep_first_n,
                "keep_last_n": self.config.keep_last_n,
                "truncation_strategy": self.config.truncation_strategy
            },
            "messages": self._messages,
            "metadata": self._metadata
        }
    
    @classmethod
    def import_(cls, data: Dict[str, Any], estimator: Optional[TokenEstimator] = None) -> "ChatHistory":
        """
        Import history from dictionary.
        
        Args:
            data: Dictionary from export()
            estimator: Optional token estimator
            
        Returns:
            Restored ChatHistory instance
        """
        config = HistoryConfig(**data["config"])
        history = cls(config=config, estimator=estimator)
        history._messages = deepcopy(data["messages"])
        history._metadata = deepcopy(data["metadata"])
        return history
    
    def __len__(self) -> int:
        """Return number of messages."""
        return len(self._messages)
    
    def __iter__(self):
        """Iterate over messages."""
        return iter(self._messages)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ChatHistory(messages={len(self._messages)}, "
            f"tokens={self.get_token_count()}/{self.config.max_tokens})"
        )
