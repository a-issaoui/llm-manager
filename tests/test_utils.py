"""
Tests for llm_manager/utils.py - Utility functions.
"""

import logging
from unittest.mock import patch

import pytest

from llm_manager.exceptions import ValidationError
from llm_manager.utils import (
    LRUCache,
    Timer,
    check_disk_space,
    compute_file_hash,
    is_base64_content,
    is_cjk,
    is_code,
    validate_max_tokens,
    validate_messages,
    validate_model_path,
    validate_temperature,
)


class TestContentDetection:
    """Test content type detection functions."""

    def test_is_code_python(self):
        assert is_code("def hello(): pass") is True
        assert is_code("class MyClass: pass") is True
        assert is_code("import numpy as np") is True

    def test_is_code_javascript(self):
        assert is_code("const x = 5;") is True
        assert is_code("let y = () => {}") is True
        assert is_code("function test() {}") is True

    def test_is_code_negative(self):
        assert is_code("Hello world") is False
        assert is_code("This is plain text") is False
        assert is_code("") is False
        assert is_code(None) is False

    def test_is_code_braces(self):
        assert is_code("function test() { return 1; }") is True

    def test_is_code_semicolon(self):
        assert is_code("x = 1; y = 2;") is True

    def test_is_code_operators(self):
        assert is_code("x += 1") is True
        assert is_code("y -= 1") is True
        assert is_code("z *= 2") is True

    def test_is_cjk_chinese(self):
        assert is_cjk("你好世界") is True

    def test_is_cjk_japanese(self):
        assert is_cjk("こんにちは") is True

    def test_is_cjk_korean(self):
        assert is_cjk("안녕하세요") is True

    def test_is_cjk_negative(self):
        assert is_cjk("Hello world") is False
        assert is_cjk("") is False
        assert is_cjk(None) is False

    def test_is_cjk_mixed(self):
        assert is_cjk("Hello 你好") is True

    def test_is_cjk_after_100_chars(self):
        """Test CJK after 100 char limit (should not detect)."""
        text = "a" * 100 + "你好"
        assert is_cjk(text) is False

    def test_is_base64_data_url(self):
        assert is_base64_content("data:image/png;base64,iVBOR...") is True

    def test_is_base64_application_data(self):
        assert is_base64_content("data:application/pdf;base64,JVBERi0...") is True

    def test_is_base64_video_data(self):
        assert is_base64_content("data:video/mp4;base64,AAAA...") is True

    def test_is_base64_long_string(self):
        long_str = "A" * 6000
        assert is_base64_content(long_str) is True

    def test_is_base64_negative(self):
        assert is_base64_content("Regular text") is False
        assert is_base64_content("") is False
        assert is_base64_content(None) is False

    def test_is_base64_with_spaces(self):
        text = "A" * 6000
        text_with_spaces = " ".join([text[i : i + 10] for i in range(0, len(text), 10)])
        assert is_base64_content(text_with_spaces) is False

    def test_is_base64_with_indicator(self):
        text = "base64," + "A" * 1001
        assert is_base64_content(text) is True


class TestValidation:
    """Test input validation functions."""

    def test_validate_temperature_valid(self):
        assert validate_temperature(0.7) == 0.7
        assert validate_temperature(0.0) == 0.0
        assert validate_temperature(2.0) == 2.0
        assert validate_temperature(1) == 1.0

    def test_validate_temperature_invalid(self):
        with pytest.raises(ValidationError):
            validate_temperature(-0.1)
        with pytest.raises(ValidationError):
            validate_temperature(2.1)
        with pytest.raises(ValidationError):
            validate_temperature("not a number")

    def test_validate_max_tokens_valid(self):
        assert validate_max_tokens(100) == 100
        assert validate_max_tokens(1) == 1

    def test_validate_max_tokens_invalid(self):
        with pytest.raises(ValidationError):
            validate_max_tokens(0)
        with pytest.raises(ValidationError):
            validate_max_tokens(-1)
        with pytest.raises(ValidationError):
            validate_max_tokens(1.5)

    def test_validate_max_tokens_large_warning(self, caplog):
        """Test warning for large max_tokens."""
        with caplog.at_level(logging.WARNING):
            result = validate_max_tokens(200000)
        assert result == 200000
        assert "large max_tokens" in caplog.text.lower()

    def test_validate_messages_valid(self):
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        result = validate_messages(messages)
        assert result == messages

    def test_validate_messages_empty(self):
        with pytest.raises(ValidationError):
            validate_messages([])

    def test_validate_messages_not_list(self):
        with pytest.raises(ValidationError):
            validate_messages("not a list")

    def test_validate_messages_missing_role(self):
        with pytest.raises(ValidationError):
            validate_messages([{"content": "Hello"}])

    def test_validate_messages_missing_content(self):
        with pytest.raises(ValidationError):
            validate_messages([{"role": "user"}])

    def test_validate_messages_invalid_role(self):
        with pytest.raises(ValidationError):
            validate_messages([{"role": "invalid", "content": "Hello"}])

    def test_validate_messages_tool_role(self):
        """Test tool role is valid."""
        messages = [{"role": "tool", "content": "Result"}]
        result = validate_messages(messages)
        assert result == messages

    def test_validate_messages_not_dict(self):
        """Test error when message is not a dict."""
        with pytest.raises(ValidationError) as exc_info:
            validate_messages(["not a dict"])
        assert "must be dict" in str(exc_info.value).lower()


class TestModelPathValidation:
    """Test validate_model_path."""

    def test_validate_model_path_not_found(self, tmp_path):
        with pytest.raises(ValidationError) as exc_info:
            validate_model_path(tmp_path / "nonexistent.gguf")
        assert "not found" in str(exc_info.value).lower()

    def test_validate_model_path_not_file(self, tmp_path):
        with pytest.raises(ValidationError) as exc_info:
            validate_model_path(tmp_path)
        assert "not a file" in str(exc_info.value).lower()

    def test_validate_model_path_invalid_magic(self, tmp_path):
        invalid_file = tmp_path / "invalid.gguf"
        invalid_file.write_bytes(b"NOTGGUF")

        with pytest.raises(ValidationError) as exc_info:
            validate_model_path(invalid_file)
        assert "not a valid gguf" in str(exc_info.value).lower()

    def test_validate_model_path_valid(self, tmp_path):
        valid_file = tmp_path / "model.gguf"
        valid_file.write_bytes(b"GGUF" + b"\x00" * 100)

        result = validate_model_path(valid_file)
        assert result == valid_file.resolve()

    def test_validate_model_path_read_error(self, tmp_path):
        test_file = tmp_path / "model.gguf"
        test_file.write_bytes(b"GGUF")

        with patch("builtins.open", side_effect=OSError("Denied")):
            with pytest.raises(ValidationError) as exc_info:
                validate_model_path(test_file)
            assert "cannot read" in str(exc_info.value).lower()

    def test_validate_model_path_string_input(self, tmp_path):
        valid_file = tmp_path / "model.gguf"
        valid_file.write_bytes(b"GGUF" + b"\x00" * 100)

        result = validate_model_path(str(valid_file))
        assert result == valid_file.resolve()


class TestDiskSpace:
    """Test check_disk_space."""

    def test_check_disk_space_success(self, tmp_path):
        has_space, available = check_disk_space(str(tmp_path), required_mb=1)
        assert has_space is True
        assert available > 0

    def test_check_disk_space_nonexistent_path(self, tmp_path):
        nonexistent = tmp_path / "nonexistent" / "path"
        has_space, available = check_disk_space(str(nonexistent), required_mb=1)
        assert isinstance(has_space, bool)

    def test_check_disk_space_error(self):
        has_space, available = check_disk_space("/nonexistent/path/that/fails")
        assert has_space is True
        assert available == -1


class TestFileHash:
    """Test compute_file_hash."""

    def test_compute_file_hash_md5(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash_result = compute_file_hash(test_file, "md5")
        assert len(hash_result) == 32

    def test_compute_file_hash_sha256(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash_result = compute_file_hash(test_file, "sha256")
        assert len(hash_result) == 64

    def test_compute_file_hash_large_file(self, tmp_path):
        test_file = tmp_path / "large.bin"
        test_file.write_bytes(b"x" * (100 * 1024))

        hash_result = compute_file_hash(test_file, "md5")
        assert len(hash_result) == 32


class TestLRUCache:
    """Test LRU cache implementation."""

    def test_basic_operations(self):
        cache = LRUCache(maxsize=2)
        cache["a"] = 1
        cache["b"] = 2

        assert cache["a"] == 1
        assert cache["b"] == 2
        assert len(cache) == 2

    def test_eviction(self):
        cache = LRUCache(maxsize=2)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3

        assert "a" not in cache
        assert "b" in cache
        assert "c" in cache

    def test_lru_order(self):
        cache = LRUCache(maxsize=2)
        cache["a"] = 1
        cache["b"] = 2

        _ = cache["a"]

        cache["c"] = 3

        assert "a" in cache
        assert "b" not in cache
        assert "c" in cache

    def test_update_existing(self):
        cache = LRUCache(maxsize=2)
        cache["a"] = 1
        cache["a"] = 2

        assert cache["a"] == 2
        assert len(cache) == 1

    def test_contains(self):
        cache = LRUCache(maxsize=2)
        cache["a"] = 1
        assert "a" in cache
        assert "b" not in cache


class TestTimer:
    """Test Timer utility."""

    def test_timer_measures_time(self):
        import time

        with Timer("test") as timer:
            time.sleep(0.01)

        assert timer.elapsed_ms is not None
        assert timer.elapsed_ms >= 10.0

    def test_timer_logs_debug(self, caplog):
        import logging

        logger = logging.getLogger("llm_manager.utils")
        logger.setLevel(logging.DEBUG)

        with Timer("test_operation"):
            pass

        assert any("test_operation" in record.message for record in caplog.records)

    def test_timer_no_start(self):
        timer = Timer("test")
        timer.__exit__(None, None, None)
        assert timer.elapsed_ms is None
