"""
Tests for llm_manager/estimation.py - Token estimation.
"""

import pytest
from unittest.mock import Mock, patch

from llm_manager.estimation import (
    TokenEstimator,
    TokenEstimate,
    ContentType,
    ConversationType,
    detect_conversation_type,
)
from llm_manager.exceptions import ValidationError


class TestTokenEstimate:
    """Test TokenEstimate dataclass."""

    def test_estimate_creation(self):
        """Test creating TokenEstimate."""
        estimate = TokenEstimate(
            total_tokens=100,
            content_tokens=50,
            template_tokens=30,
            special_tokens=20,
            content_type=ContentType.TEXT,
            is_accurate=False
        )

        assert estimate.total_tokens == 100
        assert estimate.is_accurate is False

    def test_estimate_repr(self):
        """Test string representation."""
        estimate = TokenEstimate(
            total_tokens=100,
            content_tokens=50,
            template_tokens=30,
            special_tokens=20,
            content_type=ContentType.TEXT,
            is_accurate=True
        )

        repr_str = repr(estimate)
        assert "total=100" in repr_str
        assert "accurate" in repr_str


class TestTokenEstimator:
    """Test TokenEstimator functionality."""

    def test_heuristic_simple_text(self):
        """Test heuristic estimation for text."""
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "Hello world"}]

        estimate = estimator.estimate_heuristic(messages)

        assert estimate.total_tokens > 0
        assert estimate.content_tokens > 0
        assert estimate.is_accurate is False
        assert estimate.content_type == ContentType.TEXT

    def test_heuristic_code(self):
        """Test heuristic estimation for code."""
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "def hello(): pass"}]

        estimate = estimator.estimate_heuristic(messages)

        assert estimate.content_type == ContentType.CODE
        assert estimate.content_tokens >= 10

    def test_heuristic_cjk(self):
        """Test heuristic estimation for CJK."""
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "你好世界"}]

        estimate = estimator.estimate_heuristic(messages)

        assert estimate.content_type == ContentType.CJK

    def test_heuristic_base64_content(self):
        """Test estimation with base64 content."""
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "data:image/png;base64," + "A" * 1000}]

        estimate = estimator.estimate_heuristic(messages)
        assert estimate.content_type == ContentType.IMAGE

    def test_heuristic_document_content(self):
        """Test estimation with document content."""
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "data:application/pdf;base64," + "A" * 1000}]

        estimate = estimator.estimate_heuristic(messages)
        assert estimate.content_type == ContentType.DOCUMENT

    def test_heuristic_multimodal_list(self):
        """Test estimation with multimodal list content."""
        estimator = TokenEstimator()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image", "image_url": "..."}
            ]
        }]

        estimate = estimator.estimate_heuristic(messages)
        assert estimate.content_type == ContentType.IMAGE

    def test_heuristic_long_messages(self):
        """Test estimation with many messages (cache key sampling)."""
        estimator = TokenEstimator()
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(15)
        ]

        estimate = estimator.estimate_heuristic(messages)
        assert estimate.total_tokens > 0

    def test_caching(self):
        """Test cache hit/miss tracking."""
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "Hello world"}]

        estimate1 = estimator.estimate_heuristic(messages)
        stats1 = estimator.get_stats()

        estimate2 = estimator.estimate_heuristic(messages)
        stats2 = estimator.get_stats()

        assert estimate1.total_tokens == estimate2.total_tokens
        assert stats2["hits"] > stats1["hits"]

    def test_clear_cache(self):
        """Test clearing cache."""
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "Hello"}]

        estimator.estimate_heuristic(messages)
        estimator.clear_cache()

        stats = estimator.get_stats()
        estimator.estimate_heuristic(messages)
        stats_after = estimator.get_stats()

        assert stats_after["misses"] > stats["misses"]

    def test_estimate_accurate_success(self):
        """Test accurate estimation with valid tokenizer."""
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "Hello world"}]

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        estimate = estimator.estimate_accurate(messages, mock_tokenizer)

        assert estimate.is_accurate is True
        assert estimate.total_tokens == 5

    def test_estimate_accurate_no_encode_method(self):
        """Test error when tokenizer has no encode method."""
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "Hello"}]

        class FakeTokenizer:
            pass

        with pytest.raises(ValidationError) as exc_info:
            estimator.estimate_accurate(messages, FakeTokenizer())
        assert "encode" in str(exc_info.value).lower()

    def test_estimate_accurate_fallback(self):
        """Test fallback to heuristic on tokenizer error."""
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "Hello world"}]

        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Tokenize error")

        estimate = estimator.estimate_accurate(messages, mock_tokenizer)

        assert estimate.is_accurate is False

    def test_estimate_accurate_with_template(self):
        """Test accurate estimation with template."""
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "Hello"}]

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(10))

        estimate = estimator.estimate_accurate(messages, mock_tokenizer, template="{role}: {content}")

        assert estimate.is_accurate is True

    def test_estimate_accurate_cache_hit(self):
        """Test accurate estimation cache hit."""
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "Hello"}]

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        estimator.estimate_accurate(messages, mock_tokenizer)
        estimator.estimate_accurate(messages, mock_tokenizer)

        assert mock_tokenizer.encode.call_count == 1

    def test_detect_content_type_image(self):
        """Test image detection."""
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "data:image/png;base64," + "A" * 1000}]

        content_type = estimator._detect_content_type(messages)
        assert content_type == ContentType.IMAGE

    def test_detect_content_type_code_majority(self):
        """Test code detection with majority."""
        estimator = TokenEstimator()
        messages = [
            {"role": "user", "content": "def f1(): pass"},
            {"role": "user", "content": "def f2(): pass"},
            {"role": "user", "content": "def f3(): pass"},
            {"role": "user", "content": "hello"},
        ]

        content_type = estimator._detect_content_type(messages)
        assert content_type == ContentType.CODE

    def test_detect_content_type_cjk_majority(self):
        """Test CJK detection with majority."""
        estimator = TokenEstimator()
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "user", "content": "世界"},
            {"role": "user", "content": "hello"},
        ]

        content_type = estimator._detect_content_type(messages)
        assert content_type == ContentType.CJK

    def test_get_stats(self):
        """Test statistics collection."""
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "Hello"}]

        estimator.estimate_heuristic(messages)
        stats = estimator.get_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "heuristic_calls" in stats


class TestDetectConversationType:
    """Test conversation type detection."""

    def test_detect_chat(self):
        """Test chat detection."""
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        conv_type = detect_conversation_type(messages)
        assert conv_type == ConversationType.CHAT

    def test_detect_coding(self):
        """Test coding detection."""
        messages = [
            {"role": "user", "content": "def hello(): pass"},
            {"role": "user", "content": "class MyClass: pass"},
        ]
        conv_type = detect_conversation_type(messages)
        assert conv_type == ConversationType.CODING

    def test_detect_reasoning(self):
        """Test reasoning detection."""
        messages = [
            {"role": "user", "content": "Let's think step by step about this problem"}
        ]
        conv_type = detect_conversation_type(messages)
        assert conv_type == ConversationType.REASONING

    def test_detect_reasoning_various_keywords(self):
        """Test reasoning detection with various keywords."""
        keywords = [
            "think about this",
            "analyze the problem",
            "reason through this",
            "proof by contradiction",
            "demonstrate that",
            "explain why",
            "derive the formula",
            "chain of thought",
        ]

        for keyword in keywords:
            messages = [{"role": "user", "content": keyword}]
            conv_type = detect_conversation_type(messages)
            assert conv_type == ConversationType.REASONING, f"Failed for: {keyword}"

    def test_detect_multimodal(self):
        """Test multimodal detection."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": "..."}
                ]
            }
        ]
        conv_type = detect_conversation_type(messages)
        assert conv_type == ConversationType.MULTIMODAL

    def test_detect_multimodal_base64(self):
        """Test multimodal detection via base64."""
        messages = [
            {"role": "user", "content": "data:image/png;base64," + "A" * 10000}
        ]
        conv_type = detect_conversation_type(messages)
        assert conv_type == ConversationType.MULTIMODAL

    def test_detect_tool_tool_role(self):
        """Test tool detection via tool role."""
        messages = [
            {"role": "user", "content": "Call a function"},
            {"role": "tool", "content": "{\"result\": 42}"}
        ]
        conv_type = detect_conversation_type(messages)
        assert conv_type == ConversationType.TOOL

    def test_detect_tool_function_role(self):
        """Test tool detection via function role."""
        messages = [
            {"role": "user", "content": "Call a function"},
            {"role": "function", "content": "result"}
        ]
        conv_type = detect_conversation_type(messages)
        assert conv_type == ConversationType.TOOL

    def test_detect_tool_tool_calls(self):
        """Test tool detection via tool_calls."""
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}
        ]
        conv_type = detect_conversation_type(messages)
        assert conv_type == ConversationType.TOOL

    def test_detect_tool_function_call(self):
        """Test tool detection via function_call."""
        messages = [
            {"role": "assistant", "content": "", "function_call": {"name": "test"}}
        ]
        conv_type = detect_conversation_type(messages)
        assert conv_type == ConversationType.TOOL

    def test_detect_coding_not_majority(self):
        """Test chat when coding is not majority."""
        messages = [
            {"role": "user", "content": "def hello(): pass"},
            {"role": "user", "content": "What is the weather?"},
            {"role": "user", "content": "Tell me a joke"},
        ]
        conv_type = detect_conversation_type(messages)
        assert conv_type == ConversationType.CHAT

    def test_detect_empty_messages(self):
        """Test with empty messages."""
        conv_type = detect_conversation_type([])
        assert conv_type == ConversationType.CHAT


class TestTokenEstimatorFeatures:
    """Tests for specific estimator features."""

    def test_disk_cache_integration(self, tmp_path):
        """Cover disk cache integration."""
        db_path = tmp_path / "cache.db"

        # Test initialization with disk cache
        with patch('llm_manager.cache.DiskCache') as MockDiskCache:
            mock_db = Mock()
            MockDiskCache.return_value = mock_db

            estimator = TokenEstimator(disk_cache_path=str(db_path))
            messages = [{"role": "user", "content": "hello"}]

            # Setup mock behavior
            mock_db.get.return_value = None # Miss on disk

            # First call: miss memory, miss disk, compute, save to both
            est1 = estimator.estimate_heuristic(messages)
            assert estimator._stats["misses"] == 1
            mock_db.set.assert_called()

            # Second call: hit memory
            est2 = estimator.estimate_heuristic(messages)
            assert estimator._stats["hits"] == 1

            # Clear memory cache to force disk check
            estimator._cache.clear()

            # Third call: miss memory, hit disk
            # Mock disk cache to return a valid record
            mock_db.get.return_value = {
                "total_tokens": est1.total_tokens,
                "content_tokens": est1.content_tokens,
                "template_tokens": est1.template_tokens,
                "special_tokens": est1.special_tokens,
                "content_type": est1.content_type.value,
                "is_accurate": est1.is_accurate,
            }

            est3 = estimator.estimate_heuristic(messages)
            assert estimator._stats["disk_hits"] == 1
            assert est3.total_tokens == est1.total_tokens

    def test_estimate_text_tokens_helpers(self):
        """Cover _estimate_text_tokens helpers."""
        estimator = TokenEstimator()

        # Code path - force is_code=True for text
        with patch('llm_manager.estimation.is_code', return_value=True):
            code_text = "def foo(): pass"
            tokens_code = estimator._estimate_text_tokens(code_text)
            assert tokens_code > 0

        # CJK path - force is_cjk=True
        with patch('llm_manager.estimation.is_code', return_value=False):
            with patch('llm_manager.estimation.is_cjk', return_value=True):
                cjk_text = "こんにちは"
                tokens_cjk = estimator._estimate_text_tokens(cjk_text)
                assert tokens_cjk > 0
