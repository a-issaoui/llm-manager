"""
Capability detection logic.
"""

from typing import Any

from .types import ModelCapabilities


class CapabilityDetector:
    """Detects model capabilities from metadata and filename conventions."""

    @staticmethod
    def detect(filename: str, arch: str, metadata: dict[str, Any]) -> ModelCapabilities:
        caps = ModelCapabilities()
        fname_lower = filename.lower()
        arch_lower = arch.lower()
        template = str(metadata.get("tokenizer.chat_template", "")).lower()

        # Embedding detection
        embed_keywords = ["embedding", "embed", "bert", "nomic"]
        if (
            any(x in fname_lower or x in arch_lower for x in embed_keywords)
            or arch_lower in ["bert", "nomic-bert", "clip"]
            or "pooling_type" in metadata
        ):
            caps.embed = True

        # Vision detection
        vision_keywords = ["vision", "clip", "llava", "multimodal", "vl"]
        if (
            any(x in fname_lower or x in arch_lower for x in vision_keywords)
            or "clip.vision.image_size" in metadata
            or "clip.patch_size" in metadata
        ):
            caps.vision = True
            # Vision models often have embedding capabilities for the image part
            if "clip" in arch_lower:
                caps.embed = True

        # Chat detection
        # Logic: Has chat template OR known chat model keywords
        chat_keywords = ["chat", "instruct", "dialog", "hermes", "dolphin", "nous"]
        has_template = len(template) > 50 or "user" in template or "system" in template

        if (
            any(x in fname_lower for x in chat_keywords)
            or has_template
            or caps.vision  # Most vision models are chat-tuned
        ):
            caps.chat = True

        # Pure base models
        if "base" in fname_lower and "chat" not in fname_lower:
            caps.chat = False

        # Tools/Function calling
        tool_keywords = ["tool", "function", "fc", "agent"]
        tool_tokens = ["<tool_call>", "<function", "<|tool_call|>", "tools"]
        if (
            any(x in fname_lower for x in tool_keywords)
            or "tool_use" in template
            or "function" in template
            or any(t in template for t in tool_tokens)
        ):
            caps.tools = True

        # Reasoning
        reasoning_keywords = ["reasoning", "cot", "chain-of-thought", "deepseek-r1"]
        if any(x in fname_lower for x in reasoning_keywords):
            caps.reasoning = True
            caps.chat = True  # Reasoning implies chat

        return caps

    @staticmethod
    def is_custom_template(template: str) -> bool:
        if not template or len(template) < 50:
            return False
        known = ["<|start_header_id|>", "<|im_start|>", "[INST]", "<start_of_turn>", "{%"]
        return not any(p in template.lower() for p in known)
