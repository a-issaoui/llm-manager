"""
Metadata extraction logic.
"""

import logging
import re
from pathlib import Path
from typing import Any

from .constants import (
    _PARAM_PATTERNS,
    _QUANT_PATTERNS,
    MAX_REASONABLE_CONTEXT,
)
from .detector import CapabilityDetector
from .reader import get_file_hash
from .types import ArchitectureDefaults, ModelSpecs, QuantizationType

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extracts detailed model specifications from GGUF metadata."""

    @staticmethod
    def parse_parameters(
        size_label: str, filename: str, metadata: dict[str, Any]
    ) -> float | None:
        """Extract parameter count from various sources"""
        param_count = metadata.get("general.parameter_count")
        if param_count and isinstance(param_count, (int, float)):
            return float(param_count) / 1e9

        fname_clean = filename.replace(".gguf", "").lower()
        if "bge-m3" in fname_clean:
            return 0.568

        for pattern in _PARAM_PATTERNS:
            match = pattern.search(fname_clean)
            if match and match.group(1):
                return float(match.group(1))

        label_map = {
            "smollm-1.7b": 1.7,
            "smollm-360m": 0.36,
            "phi-3.5-mini": 3.8,
            "phi-3.5-small": 7.0,
            "phi-2": 2.7,
            "mistral-7b": 7.0,
            "mixtral-8x7b": 46.7,
            "mixtral-8x22b": 141.0,
            "deepseek-r1-distill-qwen-1.5b": 1.5,
            "deepseek-r1-distill-llama-3b": 3.0,
            "qwen2.5-0.5b": 0.5,
            "qwen2.5-1.5b": 1.5,
            "qwen2.5-3b": 3.0,
            "qwen2.5-7b": 7.0,
            "qwen2.5-14b": 14.0,
            "qwen2.5-32b": 32.0,
            "qwen2.5-72b": 72.0,
            "qwen3-0.6b": 0.6,
            "qwen3-1.7b": 1.7,
            "gemma-2-2b": 2.0,
            "gemma-2-9b": 9.0,
            "gemma-3-4b": 4.0,
            "llama-3.2-1b": 1.0,
            "llama-3.2-3b": 3.0,
            "nomic-embed": 0.137,
        }
        for label_pattern, params in label_map.items():
            if label_pattern in fname_clean:
                return params

        if size_label and size_label != "Unknown":
            match = re.search(r"(\d+(?:\.\d+)?)\s*B", size_label, re.IGNORECASE)
            if match:
                return float(match.group(1))

        return None

    @staticmethod
    def detect_quantization(metadata: dict[str, Any], filename: str) -> str:
        """Detect quantization from metadata or filename"""
        file_type = metadata.get("general.file_type")
        if file_type is not None:
            try:
                return QuantizationType(file_type).name
            except ValueError:
                pass

        fname_lower = filename.lower()
        for quant, pattern in _QUANT_PATTERNS.items():
            if pattern.search(fname_lower):
                return quant
        return "Unknown"

    @classmethod
    def extract_specs(
        cls, filepath: str, metadata: dict[str, Any], arch: str
    ) -> ModelSpecs:
        """Extract comprehensive model specifications"""

        def get_val(keys: list[str], cast_type: type | None = None) -> Any | None:
            for key in keys:
                if key in metadata:
                    val = metadata[key]
                    if cast_type and val is not None:
                        try:
                            return cast_type(val)
                        except (TypeError, ValueError):
                            continue
                    return val
            return None

        arch_lower = arch.lower()

        layer_count = get_val([f"{arch}.block_count", "block_count", "n_layer"], int)
        hidden_size = get_val(
            [f"{arch}.embedding_length", "hidden_size", "n_embd"], int
        )
        ffn_size = get_val([f"{arch}.feed_forward_length", "intermediate_size"], int)

        vocab_size = get_val([f"{arch}.vocab_size", "vocab_size", "n_vocab"], int)
        if not vocab_size:
            tokens = metadata.get("tokenizer.ggml.tokens")
            if isinstance(tokens, list):
                vocab_size = len(tokens)
            elif isinstance(tokens, str) and tokens.startswith("<array:"):
                match = re.search(r"<array:(\d+)>", tokens)
                if match:
                    vocab_size = int(match.group(1))

        head_count = get_val([f"{arch}.head_count", "n_head"], int)
        head_count_kv = get_val([f"{arch}.head_count_kv", "num_key_value_heads"], int)

        if not head_count:
            head_count = ArchitectureDefaults.HEAD_COUNT.get(arch_lower, 32)
        if not head_count_kv:
            head_count_kv = ArchitectureDefaults.HEAD_COUNT_KV.get(
                arch_lower, head_count
            )

        ctx_keys = [
            f"{arch}.context_length",
            f"{arch}.max_position_embeddings",
            "context_length",
            "n_ctx",
        ]

        ctx_window = get_val(ctx_keys, int)
        if not ctx_window:
            ctx_window = ArchitectureDefaults.CONTEXT_WINDOW.get(arch_lower, 32768)

        if ctx_window > MAX_REASONABLE_CONTEXT:
            logger.warning(
                f"Capping context_window: {ctx_window} -> {MAX_REASONABLE_CONTEXT}"
            )
            ctx_window = MAX_REASONABLE_CONTEXT

        tokenizer_raw = {
            "bos_token_id": get_val(["tokenizer.ggml.bos_token_id"], int),
            "eos_token_id": get_val(["tokenizer.ggml.eos_token_id"], int),
            "padding_token_id": get_val(["tokenizer.ggml.padding_token_id"], int),
            "model": get_val(["tokenizer.ggml.model"]),
            "pre": get_val(["tokenizer.ggml.pre"]),
        }
        tokenizer: dict[str, Any] | None = (
            {k: v for k, v in tokenizer_raw.items() if v is not None} or None
        )

        expert_count = get_val([f"{arch}.expert_count", "num_experts"], int)
        active_experts = get_val([f"{arch}.expert_used_count"], int)

        file_size = Path(filepath).stat().st_size

        return ModelSpecs(
            architecture=arch,
            quantization=cls.detect_quantization(metadata, Path(filepath).name),
            size_label=metadata.get("general.size_label", "Unknown"),
            parameters_b=cls.parse_parameters(
                metadata.get("general.size_label", "Unknown"),
                Path(filepath).name,
                metadata,
            ),
            layer_count=layer_count,
            context_window=ctx_window,
            file_size_mb=file_size // (1024 * 1024),
            hidden_size=hidden_size,
            head_count=head_count,
            head_count_kv=head_count_kv,
            feed_forward_size=ffn_size,
            vocab_size=vocab_size,
            expert_count=expert_count,
            active_expert_count=active_experts,
            rope_freq_base=get_val([f"{arch}.rope.freq_base"], float),
            rope_freq_scale=get_val([f"{arch}.rope.scale"], float),
            rope_scaling_type=get_val([f"{arch}.rope.scaling.type"]),
            rope_scaling_factor=get_val([f"{arch}.rope.scaling.factor"], float),
            attention_layer_norm_rms_epsilon=get_val(
                [f"{arch}.attention.layer_norm_rms_epsilon"], float
            ),
            sliding_window=get_val([f"{arch}.sliding_window"], int),
            tokenizer=tokenizer,
            custom_chat_template=CapabilityDetector.is_custom_template(
                metadata.get("tokenizer.chat_template", "")
            ),
            file_hash=get_file_hash(filepath),
        )
