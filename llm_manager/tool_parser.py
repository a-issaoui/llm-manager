"""
Tool call parsing utilities.

Parses <tool_call> XML format from model output into OpenAI-compatible format.
"""

import json
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Pattern to match <tool_call>...</tool_call> blocks
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?})\s*</tool_call>", re.DOTALL)

# Pattern for raw JSON tool calls (models that output JSON directly)
# Matches: {"name": "func", "arguments": {...}} or {"name": "func", "parameters": {...}}
TOOL_CALL_JSON_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"(?:arguments|parameters)"\s*:\s*(\{[^}]*\})\s*\}',
    re.DOTALL
)

# Alternative pattern for models that use different format
TOOL_CALL_ALT_PATTERN = re.compile(
    r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{.*?})}', re.DOTALL
)


def parse_tool_calls(content: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse tool calls from model output.

    Supports multiple formats:
    - <tool_call>{"name": "fn", "arguments": {...}}</tool_call>
    - Raw JSON: {"name": "fn", "arguments": {...}}

    Args:
        content: Raw model output string

    Returns:
        Tuple of (cleaned_content, tool_calls_list)
        - cleaned_content: Content with tool_call tags removed
        - tool_calls_list: List of OpenAI-compatible tool_call dicts

    Example:
        >>> content = 'Hello <tool_call>{"name": "get_weather"}</tool_call>'
        >>> cleaned, calls = parse_tool_calls(content)
        >>> cleaned
        'Hello'
        >>> calls[0]['function']['name']
        'get_weather'
    """
    tool_calls: list[dict[str, Any]] = []
    cleaned_content = content

    # Try XML format first (<tool_call>...</tool_call>)
    matches = list(TOOL_CALL_PATTERN.finditer(content))

    if matches:
        for i, match in enumerate(matches):
            try:
                call_json = json.loads(match.group(1))
                tool_call = _normalize_tool_call(call_json, i)
                if tool_call:
                    tool_calls.append(tool_call)
                # Remove the tool_call tag from content
                cleaned_content = cleaned_content.replace(match.group(0), "")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {e}")
                continue
    else:
        # Try raw JSON format (models that output JSON directly without XML tags)
        # Look for JSON object with "name" and "arguments" or "parameters"
        json_match = TOOL_CALL_JSON_PATTERN.search(content)
        if json_match:
            try:
                name = json_match.group(1)
                args_str = json_match.group(2)
                args = json.loads(args_str)
                
                call_data = {"name": name, "arguments": args}
                tool_call = _normalize_tool_call(call_data, 0)
                if tool_call:
                    tool_calls.append(tool_call)
                    # Remove the JSON from content
                    cleaned_content = cleaned_content.replace(json_match.group(0), "")
            except (json.JSONDecodeError, IndexError) as e:
                logger.debug(f"Failed to parse raw JSON tool call: {e}")

    # Clean up whitespace
    cleaned_content = cleaned_content.strip()

    # Remove trailing whitespace and newlines from between tool calls
    cleaned_content = re.sub(r"\s*$", "", cleaned_content)

    return cleaned_content, tool_calls


def _normalize_tool_call(call_data: dict[str, Any], index: int) -> dict[str, Any] | None:
    """Normalize tool call data to OpenAI format.

    Handles various input formats:
    - {"name": "fn", "arguments": {...}}
    - {"name": "fn", "parameters": {...}}
    - {"function": {"name": "fn", "arguments": {...}}}

    Args:
        call_data: Parsed JSON from tool call
        index: Index for generating call ID

    Returns:
        Normalized tool call dict or None if invalid
    """
    # Handle nested function format
    if "function" in call_data and isinstance(call_data["function"], dict):
        call_data = call_data["function"]

    # Get function name
    name = call_data.get("name")
    if not name:
        logger.warning("Tool call missing 'name' field")
        return None

    # Get arguments (support both 'arguments' and 'parameters')
    arguments = call_data.get("arguments") or call_data.get("parameters") or {}

    # Ensure arguments is a JSON string (OpenAI format)
    if isinstance(arguments, dict):
        arguments_str = json.dumps(arguments)
    elif isinstance(arguments, str):
        # Validate it's valid JSON
        try:
            json.loads(arguments)
            arguments_str = arguments
        except json.JSONDecodeError:
            arguments_str = json.dumps({"raw": arguments})
    else:
        arguments_str = json.dumps({})

    # Generate unique call ID using index for determinism
    call_id = f"call_{index}_{uuid.uuid4().hex[:6]}"

    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments_str},
    }


def has_tool_calls(content: str) -> bool:
    """Check if content contains tool calls.

    Args:
        content: Model output string

    Returns:
        True if content contains tool_call patterns
    """
    # Check for XML format
    if TOOL_CALL_PATTERN.search(content):
        return True
    # Check for raw JSON format
    if TOOL_CALL_JSON_PATTERN.search(content):
        return True
    return False


def extract_tool_names(content: str) -> list[str]:
    """Extract tool names from content.

    Args:
        content: Model output string

    Returns:
        List of tool function names
    """
    _, tool_calls = parse_tool_calls(content)
    return [tc["function"]["name"] for tc in tool_calls]
