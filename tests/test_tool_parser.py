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

        assert "I will check the weather for you" in cleaned
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"

    def test_no_tool_calls(self):
        """Test parsing content with no tool calls."""
        content = "This is just regular text with no tool calls."

        cleaned, calls = parse_tool_calls(content)

        assert cleaned == content
        assert calls == []

    def test_invalid_json_in_tool_call(self):
        """Test handling invalid JSON in tool call."""
        content = '<tool_call>{"name": "broken", "arguments": </tool_call>'

        cleaned, calls = parse_tool_calls(content)

        # Should return original content on parse failure
        assert calls == []

    def test_parameters_alias(self):
        """Test handling 'parameters' instead of 'arguments'."""
        content = '<tool_call>{"name": "test", "parameters": {"x": 1}}</tool_call>'

        cleaned, calls = parse_tool_calls(content)

        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}

    def test_nested_function_format(self):
        """Test handling nested function format."""
        content = '<tool_call>{"function": {"name": "test", "arguments": {"x": 1}}}</tool_call>'

        cleaned, calls = parse_tool_calls(content)

        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "test"

    def test_empty_arguments(self):
        """Test handling empty arguments."""
        content = '<tool_call>{"name": "test"}</tool_call>'

        cleaned, calls = parse_tool_calls(content)

        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {}

    def test_raw_json_tool_call(self):
        """Test parsing raw JSON tool call without XML tags."""
        content = '{"name": "get_weather", "arguments": {"location": "Tokyo"}}'

        cleaned, calls = parse_tool_calls(content)

        assert cleaned == ""
        assert len(calls) == 1
        assert calls[0]["type"] == "function"
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {"location": "Tokyo"}

    def test_raw_json_with_parameters(self):
        """Test parsing raw JSON with 'parameters' instead of 'arguments'."""
        content = '{"name": "calculate", "parameters": {"expression": "1+1"}}'

        cleaned, calls = parse_tool_calls(content)

        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "calculate"
        assert json.loads(calls[0]["function"]["arguments"]) == {"expression": "1+1"}


class TestHasToolCalls:
    """Tests for has_tool_calls function."""

    def test_has_tool_calls_true(self):
        """Test detecting tool calls."""
        content = '<tool_call>{"name": "test"}</tool_call>'
        assert has_tool_calls(content) is True

    def test_has_tool_calls_raw_json(self):
        """Test detecting raw JSON tool calls."""
        content = '{"name": "get_weather", "arguments": {"location": "Tokyo"}}'
        assert has_tool_calls(content) is True

    def test_has_tool_calls_false(self):
        """Test detecting no tool calls."""
        content = "Just regular text with {braces} but no tool call"
        assert has_tool_calls(content) is False

    def test_has_tool_calls_empty(self):
        """Test with empty string."""
        assert has_tool_calls("") is False


class TestExtractToolNames:
    """Tests for extract_tool_names function."""

    def test_extract_single_name(self):
        """Test extracting single tool name."""
        content = '<tool_call>{"name": "get_weather"}</tool_call>'
        names = extract_tool_names(content)
        assert names == ["get_weather"]

    def test_extract_multiple_names(self):
        """Test extracting multiple tool names."""
        content = """<tool_call>{"name": "func1"}</tool_call>
        <tool_call>{"name": "func2"}</tool_call>"""
        names = extract_tool_names(content)
        assert "func1" in names
        assert "func2" in names


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
