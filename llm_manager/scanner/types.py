"""
Data types and constants for the scanner module.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar


class QuantizationType(Enum):
    """GGUF quantization types"""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K_S = 11
    Q3_K_M = 12
    Q3_K_L = 13
    Q4_K_S = 14
    Q4_K_M = 15
    Q5_K_S = 16
    Q5_K_M = 17
    Q6_K = 18
    Q8_K = 19
    IQ2_XXS = 20
    IQ2_XS = 21
    IQ3_XXS = 22
    IQ1_S = 23
    IQ4_NL = 24
    IQ3_S = 25
    IQ2_S = 26
    IQ4_XS = 27
    I8 = 28
    I16 = 29
    I32 = 30
    BF16 = 31


class GGUFConstants:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

    TYPE_SIZES: ClassVar[dict[int, int]] = {
        UINT8: 1,
        INT8: 1,
        UINT16: 2,
        INT16: 2,
        UINT32: 4,
        INT32: 4,
        FLOAT32: 4,
        BOOL: 1,
        UINT64: 8,
        INT64: 8,
        FLOAT64: 8,
    }


class ArchitectureDefaults:
    HEAD_COUNT: ClassVar[dict[str, int]] = {
        "llama": 32,
        "llama4": 32,
        "qwen2": 32,
        "qwen3": 32,
        "qwen2vl": 32,
        "qwen2audio": 32,
        "phi3": 32,
        "phi4": 32,
        "gemma": 16,
        "gemma2": 16,
        "gemma3": 16,
        "mistral": 32,
        "mixtral": 32,
        "deepseek": 32,
        "deepseek2": 32,
        "command-r": 32,
        "cohere": 32,
        "yi": 32,
        "internlm": 32,
        "internlm2": 32,
        "baichuan": 32,
        "orion": 32,
        "smollm": 32,
        "olmo": 32,
        "arctic": 32,
        "jamba": 32,
        "dbrx": 32,
        "minicpm": 32,
        "minicpm-v": 32,
        "rwkv": 32,
        "bert": 12,
        "nomic-bert": 12,
        "clip": 12,
        "whisper": 12,
        "mamba": 32,
        "falcon": 71,
        "gpt2": 12,
        "gptj": 16,
        "gptneox": 32,
        "bloom": 32,
        "stablelm": 32,
        "mpt": 32,
        "persimmon": 32,
        "refact": 32,
        "starcoder": 32,
        "starcoder2": 32,
        "codellama": 32,
        "granite": 32,
        "exaone": 32,
    }

    HEAD_COUNT_KV: ClassVar[dict[str, int]] = {
        "llama": 32,
        "llama4": 8,
        "qwen2": 32,
        "qwen3": 32,
        "qwen2vl": 32,
        "qwen2audio": 32,
        "phi3": 32,
        "phi4": 32,
        "gemma": 16,
        "gemma2": 16,
        "gemma3": 16,
        "mistral": 8,
        "mixtral": 8,
        "deepseek": 32,
        "deepseek2": 32,
        "command-r": 8,
        "cohere": 8,
        "yi": 4,
        "internlm": 8,
        "internlm2": 8,
        "baichuan": 32,
        "orion": 32,
        "smollm": 32,
        "olmo": 32,
        "arctic": 8,
        "jamba": 8,
        "dbrx": 8,
        "minicpm": 32,
        "minicpm-v": 32,
        "rwkv": 32,
        "bert": 12,
        "nomic-bert": 12,
        "clip": 12,
        "whisper": 12,
        "mamba": 32,
        "falcon": 71,
        "gpt2": 12,
        "gptj": 16,
        "gptneox": 32,
        "bloom": 32,
        "stablelm": 32,
        "mpt": 32,
        "persimmon": 32,
        "refact": 32,
        "starcoder": 32,
        "starcoder2": 32,
        "codellama": 32,
        "granite": 8,
        "exaone": 8,
    }

    CONTEXT_WINDOW: ClassVar[dict[str, int]] = {
        "llama": 8192,
        "llama4": 131072,
        "qwen2": 32768,
        "qwen3": 32768,
        "qwen2vl": 32768,
        "qwen2audio": 32768,
        "phi3": 131072,
        "phi4": 16384,
        "gemma": 8192,
        "gemma2": 8192,
        "gemma3": 131072,
        "mistral": 32768,
        "mixtral": 32768,
        "deepseek": 16384,
        "deepseek2": 128000,
        "command-r": 128000,
        "cohere": 128000,
        "yi": 200000,
        "internlm": 200000,
        "internlm2": 200000,
        "baichuan": 4096,
        "orion": 4096,
        "smollm": 8192,
        "olmo": 4096,
        "arctic": 4096,
        "jamba": 256000,
        "dbrx": 32768,
        "minicpm": 4096,
        "minicpm-v": 4096,
        "rwkv": 8192,
        "bert": 512,
        "nomic-bert": 8192,
        "clip": 77,
        "whisper": 1500,
        "mamba": 2048,
        "falcon": 2048,
        "gpt2": 1024,
        "gptj": 2048,
        "gptneox": 2048,
        "bloom": 2048,
        "stablelm": 4096,
        "mpt": 2048,
        "persimmon": 16384,
        "refact": 16384,
        "starcoder": 8192,
        "starcoder2": 16384,
        "codellama": 16384,
        "granite": 8192,
        "exaone": 32768,
    }


@dataclass
class ContextTestResult:
    max_context: int
    recommended_context: int
    buffer_tokens: int
    buffer_percent: int
    tested: bool
    verified_stable: bool
    error: str | None = None
    test_config: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    confidence: float = 1.0


@dataclass
class ModelSpecs:
    architecture: str
    quantization: str
    size_label: str
    parameters_b: float | None
    layer_count: int | None
    context_window: int
    file_size_mb: int
    hidden_size: int | None = None
    head_count: int | None = None
    head_count_kv: int | None = None
    feed_forward_size: int | None = None
    vocab_size: int | None = None
    expert_count: int | None = None
    active_expert_count: int | None = None
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None
    rope_scaling_type: str | None = None
    rope_scaling_factor: float | None = None
    attention_layer_norm_rms_epsilon: float | None = None
    attention_type: str | None = None
    gqa_ratio: int = 1
    moe_shared_expert_count: int | None = None
    moe_router_type: str | None = None
    moe_shared_expert_intermediate_size: int | None = None
    sliding_window: int | None = None
    temporal_patch_size: int | None = None
    spatial_patch_size: int | None = None
    tokenizer: dict[str, Any] | None = None
    audio: dict[str, Any] | None = None
    custom_chat_template: bool = False
    context_test: ContextTestResult = field(
        default_factory=lambda: ContextTestResult(
            max_context=8192,
            recommended_context=4096,
            buffer_tokens=4096,
            buffer_percent=50,
            tested=False,
            verified_stable=False,
            error=None,
            test_config={},
            timestamp="",
            confidence=1.0,
        )
    )
    optimized_kv_quant: bool = False
    file_hash: str = ""


@dataclass
class ModelCapabilities:
    chat: bool = False
    embed: bool = False
    vision: bool = False
    audio_in: bool = False
    reasoning: bool = False
    tools: bool = False


@dataclass
class ModelEntry:
    specs: ModelSpecs
    capabilities: ModelCapabilities
    prompt: dict[str, str]
    mmproj: dict[str, Any] | None = None
    path: str = ""
