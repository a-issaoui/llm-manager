"""Tests for tool_parser module."""

import json

from llm_manager.tool_parser import (
    _normalize_tool_call,
    extract_tool_names,
    has_tool_calls,
    parse_tool_calls,
)


class TestParseToolCalls:
    """Tests for parse_tool_calls function."""

    def test_single_tool_call(self):
        """Test parsing a single tool call."""
        content = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        )

        cleaned, calls = parse_tool_calls(content)

        assert cleaned == ""
        assert len(calls) == 1
        assert calls[0]["type"] == "function"
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
        assert "id" in calls[0]  # Generated ID

    def test_multiple_tool_calls(self):
        """Test parsing multiple tool calls."""
        content = """<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>
<tool_call>
{"name": "get_time", "arguments": {"timezone": "UTC"}}
</tool_call>"""

        cleaned, calls = parse_tool_calls(content)

        assert cleaned == ""
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "get_weather"
        assert calls[1]["function"]["name"] == "get_time"

    def test_tool_call_with_text(self):
        """Test parsing tool call mixed with text."""
        content = 'I will check the weather for you.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'

        cleaned, calls = parse_tool_calls(content)

        assert "I will check the weather" in cleaned
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"

    def test_no_tool_calls(self):
        """Test content with no tool calls."""
        content = "This is just regular text."

        cleaned, calls = parse_tool_calls(content)

        assert cleaned == content
        assert calls == []

    def test_invalid_json_in_tool_call(self):
        """Test handling invalid JSON in tool call."""
        content = "<tool_call>\n{invalid json}\n</tool_call>"

        cleaned, calls = parse_tool_calls(content)

        assert calls == []  # Invalid JSON is skipped

    def test_parameters_alias(self):
        """Test that 'parameters' is accepted as alias for 'arguments'."""
        content = '<tool_call>\n{"name": "test_fn", "parameters": {"x": 1}}\n</tool_call>'

        cleaned, calls = parse_tool_calls(content)

        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}

    def test_nested_function_format(self):
        """Test nested function format."""
        content = (
            '<tool_call>\n{"function": {"name": "test_fn", "arguments": {"x": 1}}}\n</tool_call>'
        )

        cleaned, calls = parse_tool_calls(content)

        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "test_fn"

    def test_empty_arguments(self):
        """Test tool call with no arguments."""
        content = '<tool_call>\n{"name": "get_time"}\n</tool_call>'

        cleaned, calls = parse_tool_calls(content)

        assert len(calls) == 1
        assert calls[0]["function"]["arguments"] == "{}"


class TestHasToolCalls:
    """Tests for has_tool_calls function."""

    def test_has_tool_calls_true(self):
        """Test detection when tool calls present."""
        content = '<tool_call>{"name": "test"}</tool_call>'
        assert has_tool_calls(content) is True

    def test_has_tool_calls_false(self):
        """Test detection when no tool calls."""
        content = "Just regular text"
        assert has_tool_calls(content) is False

    def test_has_tool_calls_empty(self):
        """Test empty content."""
        assert has_tool_calls("") is False


class TestExtractToolNames:
    """Tests for extract_tool_names function."""

    def test_extract_single_name(self):
        """Test extracting single tool name."""
        content = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
        names = extract_tool_names(content)
        assert names == ["get_weather"]

    def test_extract_multiple_names(self):
        """Test extracting multiple tool names."""
        content = """
<tool_call>{"name": "fn1", "arguments": {}}</tool_call>
<tool_call>{"name": "fn2", "arguments": {}}</tool_call>
"""
        names = extract_tool_names(content)
        assert names == ["fn1", "fn2"]


class TestNormalizeToolCall:
    """Tests for _normalize_tool_call function."""

    def test_missing_name(self):
        """Test handling missing name field."""
        result = _normalize_tool_call({"arguments": {}}, 0)
        assert result is None

    def test_string_arguments(self):
        """Test handling string arguments."""
        result = _normalize_tool_call({"name": "test", "arguments": '{"x": 1}'}, 0)
        assert result is not None
        assert json.loads(result["function"]["arguments"]) == {"x": 1}

    def test_unique_ids(self):
        """Test that each call gets unique ID."""
        call1 = _normalize_tool_call({"name": "fn1", "arguments": {}}, 0)
        call2 = _normalize_tool_call({"name": "fn2", "arguments": {}}, 1)

        assert call1["id"] != call2["id"]
