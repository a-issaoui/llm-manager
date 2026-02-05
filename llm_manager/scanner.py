#!/usr/bin/env python3
"""
Perfect GGUF Scanner v8.0.0 - Unified Scanner for llm_manager

Combines metadata extraction, GPU context testing, and llm_manager integration.

Features:
- Complete GGUF metadata extraction (architecture, quantization, MoE, vision, audio)
- Automated mmproj/vision adapter linking
- GPU context limit testing with binary search + stability verification
- Progress logging during testing phases
- Incremental saving (resume interrupted scans)
- Subprocess isolation for perfect VRAM cleanup
- Atomic file writes (corruption-proof)
- Simple API for llm_manager integration

Usage:
    # CLI usage:
    python -m llm_manager.scanner ./models -o models.json --test-context
    python -m llm_manager.scanner ./models --resume

    # API usage:
    from llm_manager import ModelScanner, scan_models
    
    scanner = ModelScanner("./models")
    results = scanner.scan_and_save(test_context=True)
    
    # Or the convenience function:
    results = scan_models("./models", test_context=True)
"""

import argparse
import atexit
import hashlib
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import IO, Any, ClassVar, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

__version__ = "8.0.0"

# Optional imports
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Import llm_manager components
try:
    from .models import ModelRegistry
    from .exceptions import ValidationError
    _HAS_LLM_MANAGER = True
except ImportError:
    _HAS_LLM_MANAGER = False
    ValidationError = Exception  # type: ignore
    ModelRegistry = None  # type: ignore

# =============================================================================
# Constants & Configuration
# =============================================================================

CONTEXT_SAFETY_MARGIN = 0.80
MIN_BINARY_SEARCH_GAP = 2048
TEST_TIMEOUT = 300
STABILITY_RETRIES = 2
CLEANUP_DELAY_SUCCESS = 0.2
CLEANUP_DELAY_FAILURE = 1.0
MAX_REASONABLE_CONTEXT = 1_000_000
DEFAULT_UNTESTED_CONTEXT = 32768
STABILITY_REDUCTION_FACTOR = 0.1
LARGE_ARRAY_THRESHOLD = 1000

_temp_files: List[str] = []


def cleanup() -> None:
    """Clean up temporary files on exit"""
    for f in _temp_files:
        try:
            if os.path.exists(f):
                os.unlink(f)
        except OSError:
            pass


atexit.register(cleanup)

# Pre-compiled regex patterns
_PARAM_PATTERNS = [
    re.compile(r"(?:embedding[_-])?(\d+(?:\.\d+)?)[_-]?b[_-]?(?:embedding)?"),
    re.compile(r"(\d+(?:\.\d+)?)[_-]?b(?:[_-]|$)"),
    re.compile(r"(\d+\.?\d*)[Bb](?![a-zA-Z])"),
]

_QUANT_PATTERNS = {
    "Q2_K": re.compile(r"q2[_-]?k"),
    "Q3_K_S": re.compile(r"q3[_-]?k[_-]?s"),
    "Q3_K_M": re.compile(r"q3[_-]?k[_-]?m"),
    "Q3_K_L": re.compile(r"q3[_-]?k[_-]?l"),
    "Q4_0": re.compile(r"q4[_-]?0"),
    "Q4_1": re.compile(r"q4[_-]?1"),
    "Q4_K_S": re.compile(r"q4[_-]?k[_-]?s"),
    "Q4_K_M": re.compile(r"q4[_-]?k[_-]?m"),
    "Q5_0": re.compile(r"q5[_-]?0"),
    "Q5_1": re.compile(r"q5[_-]?1"),
    "Q5_K_S": re.compile(r"q5[_-]?k[_-]?s"),
    "Q5_K_M": re.compile(r"q5[_-]?k[_-]?m"),
    "Q6_K": re.compile(r"q6[_-]?k"),
    "Q8_0": re.compile(r"q8[_-]?0"),
    "F16": re.compile(r"f16"),
    "F32": re.compile(r"f32"),
}

_REASONING_PATTERNS = [
    re.compile(r"deepseek-r1"),
    re.compile(r"deepseek-r1-distill"),
    re.compile(r"qwq"),
    re.compile(r"thinking"),
    re.compile(r"deepthink"),
    re.compile(r"[-_]r1[-_]"),
    re.compile(r"chain[_-]of[_-]thought"),
    re.compile(r"\bcot\b"),
]


# =============================================================================
# Data Classes
# =============================================================================

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

    TYPE_SIZES: ClassVar[Dict[int, int]] = {
        UINT8: 1, INT8: 1, UINT16: 2, INT16: 2, UINT32: 4,
        INT32: 4, FLOAT32: 4, BOOL: 1, UINT64: 8,
    }


class ArchitectureDefaults:
    HEAD_COUNT: ClassVar[Dict[str, int]] = {
        "llama": 32, "llama4": 32, "qwen2": 32, "qwen3": 32, "qwen2vl": 32,
        "qwen2audio": 32, "phi3": 32, "phi4": 32, "gemma": 16, "gemma2": 16,
        "gemma3": 16, "mistral": 32, "mixtral": 32, "deepseek": 32, "deepseek2": 32,
        "command-r": 32, "cohere": 32, "yi": 32, "internlm": 32, "internlm2": 32,
        "baichuan": 32, "orion": 32, "smollm": 32, "olmo": 32, "arctic": 32,
        "jamba": 32, "dbrx": 32, "minicpm": 32, "minicpm-v": 32, "rwkv": 32,
        "bert": 12, "nomic-bert": 12, "clip": 12, "whisper": 12, "mamba": 32,
        "falcon": 71, "gpt2": 12, "gptj": 16, "gptneox": 32, "bloom": 32,
        "stablelm": 32, "mpt": 32, "persimmon": 32, "refact": 32, "starcoder": 32,
        "starcoder2": 32, "codellama": 32, "granite": 32, "exaone": 32,
    }

    HEAD_COUNT_KV: ClassVar[Dict[str, int]] = {
        "llama": 32, "llama4": 8, "qwen2": 32, "qwen3": 32, "qwen2vl": 32,
        "qwen2audio": 32, "phi3": 32, "phi4": 32, "gemma": 16, "gemma2": 16,
        "gemma3": 16, "mistral": 8, "mixtral": 8, "deepseek": 32, "deepseek2": 32,
        "command-r": 8, "cohere": 8, "yi": 4, "internlm": 8, "internlm2": 8,
        "baichuan": 32, "orion": 32, "smollm": 32, "olmo": 32, "arctic": 8,
        "jamba": 8, "dbrx": 8, "minicpm": 32, "minicpm-v": 32, "rwkv": 32,
        "bert": 12, "nomic-bert": 12, "clip": 12, "whisper": 12, "mamba": 32,
        "falcon": 71, "gpt2": 12, "gptj": 16, "gptneox": 32, "bloom": 32,
        "stablelm": 32, "mpt": 32, "persimmon": 32, "refact": 32, "starcoder": 32,
        "starcoder2": 32, "codellama": 32, "granite": 8, "exaone": 8,
    }

    CONTEXT_WINDOW: ClassVar[Dict[str, int]] = {
        "llama": 8192, "llama4": 131072, "qwen2": 32768, "qwen3": 32768,
        "qwen2vl": 32768, "qwen2audio": 32768, "phi3": 131072, "phi4": 16384,
        "gemma": 8192, "gemma2": 8192, "gemma3": 131072, "mistral": 32768,
        "mixtral": 32768, "deepseek": 16384, "deepseek2": 128000,
        "command-r": 128000, "cohere": 128000, "yi": 200000, "internlm": 200000,
        "internlm2": 200000, "baichuan": 4096, "orion": 4096, "smollm": 8192,
        "olmo": 4096, "arctic": 4096, "jamba": 256000, "dbrx": 32768,
        "minicpm": 4096, "minicpm-v": 4096, "rwkv": 8192, "bert": 512,
        "nomic-bert": 8192, "clip": 77, "whisper": 1500, "mamba": 2048,
        "falcon": 2048, "gpt2": 1024, "gptj": 2048, "gptneox": 2048,
        "bloom": 2048, "stablelm": 4096, "mpt": 2048, "persimmon": 16384,
        "refact": 16384, "starcoder": 8192, "starcoder2": 16384,
        "codellama": 16384, "granite": 8192, "exaone": 32768,
    }


@dataclass
class ContextTestResult:
    max_context: int
    recommended_context: int
    buffer_tokens: int
    buffer_percent: int
    tested: bool
    verified_stable: bool
    error: Optional[str] = None
    test_config: Dict = field(default_factory=dict)
    timestamp: str = ""
    confidence: float = 1.0


@dataclass
class ModelSpecs:
    architecture: str
    quantization: str
    size_label: str
    parameters_b: Optional[float]
    layer_count: Optional[int]
    context_window: int
    file_size_mb: int
    hidden_size: Optional[int] = None
    head_count: Optional[int] = None
    head_count_kv: Optional[int] = None
    feed_forward_size: Optional[int] = None
    vocab_size: Optional[int] = None
    expert_count: Optional[int] = None
    active_expert_count: Optional[int] = None
    rope_freq_base: Optional[float] = None
    rope_freq_scale: Optional[float] = None
    rope_scaling_type: Optional[str] = None
    rope_scaling_factor: Optional[float] = None
    attention_layer_norm_rms_epsilon: Optional[float] = None
    attention_type: Optional[str] = None
    gqa_ratio: int = 1
    moe_shared_expert_count: Optional[int] = None
    moe_router_type: Optional[str] = None
    moe_shared_expert_intermediate_size: Optional[int] = None
    sliding_window: Optional[int] = None
    temporal_patch_size: Optional[int] = None
    spatial_patch_size: Optional[int] = None
    tokenizer: Optional[Dict[str, Any]] = None
    audio: Optional[Dict[str, Any]] = None
    custom_chat_template: bool = False
    context_test: ContextTestResult = field(default_factory=lambda: ContextTestResult(
        max_context=8192, recommended_context=4096, buffer_tokens=4096, buffer_percent=50,
        tested=False, verified_stable=False, error=None, test_config={}, timestamp="",
        confidence=1.0
    ))
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
    prompt: Dict[str, str]
    mmproj: Optional[Dict[str, Any]] = None
    path: str = ""


# =============================================================================
# Context Testing Worker
# =============================================================================

CONTEXT_TEST_WORKER = '''
import sys
import json
import gc
import time
import os

def test_context(model_path, ctx_size, gpu_layers, flash_attn, kv_quant, gpu_device):
    if gpu_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    try:
        from llama_cpp import Llama
    except ImportError as e:
        return {"success": False, "error": str(e), "error_type": "ImportError"}

    try:
        import llama_cpp
        version = getattr(llama_cpp, '__version__', 'unknown')
    except:
        version = 'unknown'

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except:
        pass

    gc.collect()
    time.sleep(0.3)

    cache_type = "q4_0" if kv_quant else "f16"

    try:
        try:
            llm = Llama(
                model_path=str(model_path),
                n_ctx=ctx_size,
                n_gpu_layers=gpu_layers,
                flash_attn=flash_attn,
                cache_type_k=cache_type,
                cache_type_v=cache_type,
                verbose=False,
                n_batch=512,
                n_ubatch=512,
                use_mlock=False,
                use_mmap=True,
                logits_all=False,
            )
        except TypeError:
            llm = Llama(
                model_path=str(model_path),
                n_ctx=ctx_size,
                n_gpu_layers=gpu_layers,
                verbose=False,
                n_batch=512,
                n_ubatch=512,
                use_mlock=False,
                use_mmap=True,
                logits_all=False,
            )

        actual_ctx = llm.n_ctx()
        if actual_ctx < ctx_size:
            return {
                "success": False,
                "error": f"Context reduced: requested {ctx_size}, got {actual_ctx}",
                "error_type": "ContextReduced"
            }

        prompt_tokens = min(ctx_size - 100, 8000)
        words = max(10, prompt_tokens // 2)
        test_prompt = "Test " * words
        max_gen = min(50, ctx_size - prompt_tokens)

        output = llm(
            test_prompt, 
            max_tokens=max_gen, 
            echo=False, 
            temperature=0.01,
            top_p=0.95
        )

        if not output or 'choices' not in output:
            raise RuntimeError("No output generated")

        result = {
            "success": True,
            "actual_ctx": actual_ctx,
            "llama_cpp_version": version
        }

        try:
            import torch
            if torch.cuda.is_available():
                result["gpu_allocated_gb"] = torch.cuda.memory_allocated(0) / 1e9
        except:
            pass

        del llm
        gc.collect()
        return result

    except Exception as e:
        error_msg = str(e).lower()
        if any(kw in error_msg for kw in [
            "out of memory", "cuda error", "failed to allocate", 
            "cudamalloc", "insufficient memory", "failed to create llama_context"
        ]):
            error_type = "OOM"
        elif "n_ctx_per_seq" in error_msg and "n_ctx_train" in error_msg:
            error_type = "ContextTooLow"
        else:
            error_type = "Error"

        return {
            "success": False,
            "error": str(e)[:400],
            "error_type": error_type
        }

if __name__ == "__main__":
    args = json.loads(sys.argv[1])
    result = test_context(**args)
    print(json.dumps(result))
    sys.exit(0 if result["success"] else 1)
'''


# =============================================================================
# GGUF Parser
# =============================================================================

class GGUFReader:
    @staticmethod
    def read_value(f: IO, value_type: int) -> Any:
        if value_type == GGUFConstants.UINT8:
            return struct.unpack("<B", f.read(1))[0]
        elif value_type == GGUFConstants.INT8:
            return struct.unpack("<b", f.read(1))[0]
        elif value_type == GGUFConstants.UINT16:
            return struct.unpack("<H", f.read(2))[0]
        elif value_type == GGUFConstants.INT16:
            return struct.unpack("<h", f.read(2))[0]
        elif value_type == GGUFConstants.UINT32:
            return struct.unpack("<I", f.read(4))[0]
        elif value_type == GGUFConstants.INT32:
            return struct.unpack("<i", f.read(4))[0]
        elif value_type == GGUFConstants.FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif value_type == GGUFConstants.BOOL:
            return struct.unpack("<?", f.read(1))[0]
        elif value_type == GGUFConstants.STRING:
            str_len = struct.unpack("<Q", f.read(8))[0]
            return f.read(str_len).decode("utf-8", errors="replace")
        elif value_type == GGUFConstants.ARRAY:
            array_type = struct.unpack("<I", f.read(4))[0]
            array_len = struct.unpack("<Q", f.read(8))[0]
            if array_len > LARGE_ARRAY_THRESHOLD:
                for _ in range(array_len):
                    GGUFReader.read_value(f, array_type)
                return f"<array:{array_len}>"
            return [GGUFReader.read_value(f, array_type) for _ in range(array_len)]
        elif value_type == GGUFConstants.UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        return None

    @staticmethod
    def extract_metadata(filepath: str) -> Optional[Dict[str, Any]]:
        try:
            with open(filepath, "rb") as f:
                magic = f.read(4)
                if magic != b"GGUF":
                    return None

                version = struct.unpack("<I", f.read(4))[0]
                if version not in (2, 3):
                    logger.warning(f"Unsupported GGUF version: {version}")
                    return None

                tensor_count = struct.unpack("<Q", f.read(8))[0]
                metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

                metadata = {}
                for _ in range(metadata_kv_count):
                    key_len = struct.unpack("<Q", f.read(8))[0]
                    key = f.read(key_len).decode("utf-8", errors="replace")
                    value_type = struct.unpack("<I", f.read(4))[0]
                    metadata[key] = GGUFReader.read_value(f, value_type)

                return metadata
        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            return None


def get_file_hash(filepath: str) -> str:
    """Quick hash of first/last 1MB for change detection"""
    try:
        with open(filepath, "rb") as f:
            start = f.read(1024 * 1024)
            f.seek(-1024 * 1024, 2)
            end = f.read(1024 * 1024)
        return hashlib.sha256(start + end).hexdigest()[:16]
    except Exception:
        return ""


# =============================================================================
# Perfect Scanner
# =============================================================================

class PerfectScanner:
    """Low-level GGUF scanner with metadata extraction and context testing."""

    def __init__(self):
        self.results: Dict[str, ModelEntry] = {}
        self.mmproj_files: Dict[str, Dict[str, Any]] = {}
        self.stats = {
            "total": 0, "parsed": 0, "failed": 0,
            "context_tested": 0, "context_skipped": 0, "context_failed": 0
        }
        self._save_counter = 0

    def parse_parameters(self, size_label: str, filename: str, metadata: Dict) -> Optional[float]:
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
            "smollm-1.7b": 1.7, "smollm-360m": 0.36, "phi-3.5-mini": 3.8,
            "phi-3.5-small": 7.0, "phi-2": 2.7, "mistral-7b": 7.0,
            "mixtral-8x7b": 46.7, "mixtral-8x22b": 141.0,
            "deepseek-r1-distill-qwen-1.5b": 1.5, "deepseek-r1-distill-llama-3b": 3.0,
            "qwen2.5-0.5b": 0.5, "qwen2.5-1.5b": 1.5, "qwen2.5-3b": 3.0,
            "qwen2.5-7b": 7.0, "qwen2.5-14b": 14.0, "qwen2.5-32b": 32.0,
            "qwen2.5-72b": 72.0, "qwen3-0.6b": 0.6, "qwen3-1.7b": 1.7,
            "gemma-2-2b": 2.0, "gemma-2-9b": 9.0, "gemma-3-4b": 4.0,
            "llama-3.2-1b": 1.0, "llama-3.2-3b": 3.0, "nomic-embed": 0.137,
        }
        for pattern, params in label_map.items():
            if pattern in fname_clean:
                return params

        if size_label and size_label != "Unknown":
            match = re.search(r"(\d+(?:\.\d+)?)\s*B", size_label, re.IGNORECASE)
            if match:
                return float(match.group(1))

        return None

    def detect_quantization(self, metadata: Dict, filename: str) -> str:
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

    def extract_specs(self, filepath: str, metadata: Dict, arch: str) -> ModelSpecs:
        """Extract comprehensive model specifications"""

        def get_val(keys: List[str], cast_type=None):
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
        hidden_size = get_val([f"{arch}.embedding_length", "hidden_size", "n_embd"], int)
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
            head_count_kv = ArchitectureDefaults.HEAD_COUNT_KV.get(arch_lower, head_count)

        gqa_ratio = 1
        if head_count and head_count_kv and head_count_kv > 0:
            gqa_ratio = head_count // head_count_kv

        ctx_keys = [f"{arch}.context_length", f"{arch}.max_position_embeddings",
                    "context_length", "n_ctx"]
        ctx_window = get_val(ctx_keys, int)
        if not ctx_window:
            ctx_window = ArchitectureDefaults.CONTEXT_WINDOW.get(arch_lower, 32768)

        if ctx_window > MAX_REASONABLE_CONTEXT:
            logger.warning(f"Capping suspicious context_window: {ctx_window} -> {MAX_REASONABLE_CONTEXT}")
            ctx_window = MAX_REASONABLE_CONTEXT

        tokenizer = {
            "bos_token_id": get_val(["tokenizer.ggml.bos_token_id"], int),
            "eos_token_id": get_val(["tokenizer.ggml.eos_token_id"], int),
            "padding_token_id": get_val(["tokenizer.ggml.padding_token_id"], int),
            "model": get_val(["tokenizer.ggml.model"]),
            "pre": get_val(["tokenizer.ggml.pre"]),
        }
        tokenizer = {k: v for k, v in tokenizer.items() if v is not None} or None

        expert_count = get_val([f"{arch}.expert_count", "num_experts"], int)
        active_experts = get_val([f"{arch}.expert_used_count"], int)

        file_size = Path(filepath).stat().st_size

        return ModelSpecs(
            architecture=arch,
            quantization=self.detect_quantization(metadata, Path(filepath).name),
            size_label=metadata.get("general.size_label", "Unknown"),
            parameters_b=self.parse_parameters(
                metadata.get("general.size_label", "Unknown"),
                Path(filepath).name,
                metadata
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
            custom_chat_template=self.is_custom_template(
                metadata.get("tokenizer.chat_template", "")
            ),
            file_hash=get_file_hash(filepath),
        )

    def detect_capabilities(self, filename: str, arch: str, metadata: Dict) -> ModelCapabilities:
        caps = ModelCapabilities()
        fname_lower = filename.lower()
        arch_lower = arch.lower()
        template = str(metadata.get("tokenizer.chat_template", "")).lower()

        if (any(x in fname_lower or x in arch_lower for x in ["embedding", "embed", "bert", "nomic"])
                and "chat" not in fname_lower):
            caps.embed = True
            return caps

        vision_archs = ["qwen2vl", "llava", "minicpm-v", "gemma3", "internvl", "cogvlm"]
        if any(x in arch_lower for x in vision_archs):
            caps.vision = True
            caps.chat = True

        audio_archs = ["qwen2audio", "whisper"]
        if any(x in arch_lower for x in audio_archs) or "omni" in fname_lower:
            caps.audio_in = True
            caps.chat = True

        for pattern in _REASONING_PATTERNS:
            if pattern.search(fname_lower):
                caps.reasoning = True
                caps.chat = True
                break

        tool_tokens = ["<tool_call>", "<function", "<|tool_call|>", "tools"]
        if any(t in template for t in tool_tokens) or "instruct" in fname_lower:
            caps.tools = True
            caps.chat = True

        if not any([caps.chat, caps.embed, caps.vision, caps.audio_in]):
            chat_indicators = ["instruct", "chat", "hermes", "vicuna", "dolphin"]
            if any(x in fname_lower for x in chat_indicators):
                caps.chat = True

        return caps

    def is_custom_template(self, template: str) -> bool:
        if not template or len(template) < 50:
            return False
        known = ["<|start_header_id|>", "<|im_start|>", "[INST]", "<start_of_turn>", "{%"]
        return not any(p in template.lower() for p in known)

    def scan_mmproj(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Extract vision adapter metadata"""
        metadata = GGUFReader.extract_metadata(filepath)
        if not metadata:
            return None

        arch = metadata.get("general.architecture", "clip")
        if arch != "clip" and "clip" not in arch:
            return None

        mmproj = {
            "architecture": arch,
            "quantization": self.detect_quantization(metadata, Path(filepath).name),
            "file_size_mb": Path(filepath).stat().st_size // (1024 * 1024),
        }

        vision_params = {
            "vision_embedding_length": "clip.vision_embedding_length",
            "projection_dim": "clip.projection_dim",
            "patch_size": "clip.patch_size",
            "image_size": "clip.image_size",
        }
        for param, key in vision_params.items():
            if key in metadata:
                val = metadata[key]
                if not (isinstance(val, str) and val.startswith("<array:")):
                    mmproj[param] = val

        return mmproj

    def find_parent_model(self, mmproj_name: str) -> Optional[str]:
        """Fuzzy match mmproj to base model"""
        base = mmproj_name.replace("mmproj-", "").replace("-mmproj", "").replace(".gguf", "")
        base = re.sub(r"-(f16|f32|q\d+[_k]*)", "", base, flags=re.IGNORECASE).lower()
        base_tokens = [t for t in re.split(r"[-_.]", base) if t]

        best_match = None
        best_score = 0.0

        for filename, entry in self.results.items():
            if "mmproj" in filename or entry.capabilities.embed:
                continue

            model_lower = filename.lower()
            model_tokens = [t for t in re.split(r"[-_.]", model_lower) if t]

            score = sum(1.5 if len(t) > 3 else 1.0
                        for t in base_tokens if t in model_tokens)

            if entry.capabilities.vision:
                score += 2.0

            if score > best_score and score >= 3.0:
                best_score = score
                best_match = filename

        return best_match

    def save_atomic(self, filepath: str, data: Dict):
        """Atomic write with backup"""
        if os.path.exists(filepath):
            try:
                shutil.copy2(filepath, f"{filepath}.backup")
            except (OSError, shutil.Error):
                pass

        dir_name = os.path.dirname(filepath) or '.'
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, filepath)
        except Exception as e:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e

    def run_context_test(
            self,
            filepath: str,
            min_ctx: int,
            max_ctx: int,
            gpu_layers: int,
            gpu_device: Optional[int],
            kv_quant: bool,
            flash_attn: bool
    ) -> ContextTestResult:
        """Binary search for maximum stable context with progress logging"""

        def test_single(ctx: int, ctx_label: str = "") -> Tuple[bool, Optional[str]]:
            """Run isolated test at specific context"""
            logger.info(f"    Testing {ctx:,} tokens... {ctx_label}")

            worker_file = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(CONTEXT_TEST_WORKER)
                    worker_file = f.name
                    _temp_files.append(worker_file)

                args = {
                    "model_path": str(filepath),
                    "ctx_size": ctx,
                    "gpu_layers": gpu_layers,
                    "flash_attn": flash_attn,
                    "kv_quant": kv_quant,
                    "gpu_device": gpu_device,
                }

                result = subprocess.run(
                    [sys.executable, worker_file, json.dumps(args)],
                    capture_output=True,
                    text=True,
                    timeout=TEST_TIMEOUT,
                )

                time.sleep(CLEANUP_DELAY_SUCCESS if result.returncode == 0 else CLEANUP_DELAY_FAILURE)

                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    return data.get("success", False), data.get("error")
                return False, result.stderr[:200]

            except subprocess.TimeoutExpired:
                return False, "Timeout"
            except Exception as e:
                return False, str(e)[:200]
            finally:
                if worker_file and os.path.exists(worker_file):
                    try:
                        os.unlink(worker_file)
                        _temp_files.remove(worker_file)
                    except (OSError, ValueError):
                        pass

        # Test minimum first
        logger.info(f"  [Phase 1/3] Testing minimum context {min_ctx:,}...")
        success, err = test_single(min_ctx, "(minimum)")
        if not success:
            error_msg = err or "Unknown error"
            if "n_ctx_per_seq" in error_msg and "n_ctx_train" in error_msg:
                match = re.search(r'n_ctx_train \((\d+)\)', error_msg)
                required_ctx = int(match.group(1)) if match else "unknown"
                error_msg = f"Model requires min context {required_ctx} (hardware may be insufficient)"

            return ContextTestResult(
                max_context=0,
                recommended_context=0,
                buffer_tokens=0,
                buffer_percent=20,
                tested=True,
                verified_stable=False,
                error=error_msg,
                timestamp=datetime.now().isoformat(),
                confidence=0.0
            )

        logger.info(f"    ✓ Minimum {min_ctx:,} works")

        # Binary search with progress logging
        logger.info(f"  [Phase 2/3] Binary searching optimal context...")
        low, high = min_ctx, max_ctx
        best_working = min_ctx
        iteration = 0

        while high - low > MIN_BINARY_SEARCH_GAP:
            iteration += 1
            gap = high - low

            if gap > 32768:
                step = 4096
            elif gap > 16384:
                step = 2048
            else:
                step = 1024

            mid = (low + high) // 2
            mid = (mid // step) * step

            if mid <= low:
                mid = low + step
            if mid >= high:
                mid = high - step

            if mid <= low or mid >= high or mid == best_working:
                logger.debug(f"    Binary search converged at gap={gap:,}")
                break

            success, err = test_single(mid, f"(iteration {iteration}, gap {gap:,})")

            if success:
                best_working = mid
                low = mid
                logger.info(f"      ✓ {mid:,} works (new best)")
            else:
                high = mid
                logger.info(f"      ✗ {mid:,} failed")

        logger.info(f"  Best found: {best_working:,} tokens")

        # Stability verification with progress
        logger.info(f"  [Phase 3/3] Verifying stability at {best_working:,}...")
        verified = True
        attempts = 0
        max_attempts = STABILITY_RETRIES + 2

        while attempts < max_attempts and best_working > min_ctx:
            attempts += 1
            success, err = test_single(best_working, f"(stability check {attempts}/{STABILITY_RETRIES})")

            if success:
                success2, _ = test_single(best_working, f"(confirmation)")
                if success2:
                    verified = True
                    break

            verified = False
            reduction = int(best_working * STABILITY_REDUCTION_FACTOR)
            new_working = max(min_ctx, best_working - reduction)

            if new_working >= best_working:
                break

            best_working = new_working
            logger.warning(f"    Unstable, reduced to {best_working:,}")

        recommended = int(best_working * CONTEXT_SAFETY_MARGIN)

        confidence = 1.0
        if not verified:
            confidence *= 0.7
        if best_working < max_ctx * 0.2:
            confidence *= 0.8

        logger.info(f"  Result: {best_working:,} max → {recommended:,} recommended "
                    f"({best_working - recommended:,} buffer) "
                    f"[{'Stable' if verified else 'Unstable'}, {confidence * 100:.0f}% confidence]")

        return ContextTestResult(
            max_context=best_working,
            recommended_context=recommended,
            buffer_tokens=best_working - recommended,
            buffer_percent=int((1 - CONTEXT_SAFETY_MARGIN) * 100),
            tested=True,
            verified_stable=verified,
            test_config={
                "kv_quant": "q4_0" if kv_quant else "f16",
                "flash_attn": flash_attn,
                "gpu_layers": gpu_layers,
            },
            timestamp=datetime.now().isoformat(),
            confidence=confidence
        )

    def scan_and_test(
            self,
            folder: str,
            output: str,
            test_context: bool = False,
            resume: bool = False,
            min_ctx: int = 8192,
            max_ctx: int = 131072,
            gpu_layers: int = -1,
            gpu_device: Optional[int] = None,
            kv_quant: bool = True,
            flash_attn: bool = True,
            model_filter: Optional[str] = None,
    ):
        """Main scanning and testing workflow"""

        path = Path(folder)
        if not path.exists():
            logger.error(f"Directory not found: {folder}")
            return

        gguf_files = list(path.rglob("*.gguf"))
        if not gguf_files:
            logger.error("No GGUF files found")
            return

        if model_filter:
            gguf_files = [f for f in gguf_files if model_filter.lower() in f.name.lower()]

        self.stats["total"] = len(gguf_files)

        if resume and os.path.exists(output):
            try:
                with open(output) as f:
                    existing = json.load(f)

                for fname, data in existing.items():
                    ctx_data = data.get("specs", {}).get("context_test", {})
                    if isinstance(ctx_data, dict):
                        context_test = ContextTestResult(**ctx_data)
                    else:
                        context_test = ContextTestResult(
                            max_context=0, recommended_context=0,
                            buffer_tokens=0, buffer_percent=20,
                            tested=False, verified_stable=False
                        )

                    specs_data = {k: v for k, v in data.get("specs", {}).items()
                                  if k not in ("context_test", "optimized_kv_quant")}

                    specs = ModelSpecs(**specs_data)
                    specs.context_test = context_test
                    specs.optimized_kv_quant = data.get("specs", {}).get("optimized_kv_quant", False)

                    self.results[fname] = ModelEntry(
                        specs=specs,
                        capabilities=ModelCapabilities(**data.get("capabilities", {})),
                        prompt=data.get("prompt", {}),
                        mmproj=data.get("mmproj"),
                        path=data.get("path", "")
                    )

                logger.info(f"Loaded {len(self.results)} existing results")
            except Exception as e:
                logger.warning(f"Could not resume: {e}")

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Perfect GGUF Scanner v{__version__}")
        logger.info(f"{'=' * 70}")
        logger.info(f"Found {len(gguf_files)} GGUF files")
        logger.info(f"Context testing: {'Enabled' if test_context else 'Disabled'}")
        if test_context:
            logger.info(f"KV Quantization: {'Q4_0' if kv_quant else 'FP16'}")
        logger.info(f"{'=' * 70}\n")

        regular_files = []
        mmproj_files = []

        for f in sorted(gguf_files):
            (mmproj_files if "mmproj" in f.name.lower() else regular_files).append(f)

        for idx, filepath in enumerate(regular_files, 1):
            fname = filepath.name

            if fname in self.results:
                logger.info(f"[{idx}/{len(regular_files)}] {fname} - Already scanned")
            else:
                logger.info(f"[{idx}/{len(regular_files)}] Scanning {fname}...")

                metadata = GGUFReader.extract_metadata(str(filepath))
                if not metadata:
                    self.stats["failed"] += 1
                    continue

                arch = metadata.get("general.architecture", "unknown")
                specs = self.extract_specs(str(filepath), metadata, arch)
                capabilities = self.detect_capabilities(fname, arch, metadata)
                template = metadata.get("tokenizer.chat_template", "")
                if isinstance(template, (list, dict)):
                    template = str(template[0] if isinstance(template, list) else template.get("default", ""))

                entry = ModelEntry(
                    specs=specs,
                    capabilities=capabilities,
                    prompt={"template": template},
                    path=str(filepath)
                )

                self.results[fname] = entry
                self.stats["parsed"] += 1

                caps = [k for k, v in vars(capabilities).items() if v]
                logger.info(
                    f"  ├─ {specs.architecture} | {specs.parameters_b}B | {specs.quantization} | {specs.context_window // 1024}k")
                logger.info(f"  └─ Capabilities: {', '.join(caps) if caps else 'none'}")

            if test_context and not self.results[fname].capabilities.embed:
                existing_test = self.results[fname].specs.context_test

                if existing_test.tested and resume:
                    self.stats["context_skipped"] += 1
                    logger.info(f"  └─ Context: Already tested ({existing_test.recommended_context})")
                    continue

                current_hash = get_file_hash(str(filepath))
                if existing_test.tested and self.results[fname].specs.file_hash == current_hash:
                    self.stats["context_skipped"] += 1
                    logger.info(f"  └─ Context: Skipped (hash match)")
                    continue

                try:
                    result = self.run_context_test(
                        str(filepath),
                        min_ctx=min_ctx,
                        max_ctx=min(max_ctx, self.results[fname].specs.context_window),
                        gpu_layers=gpu_layers,
                        gpu_device=gpu_device,
                        kv_quant=kv_quant,
                        flash_attn=flash_attn,
                    )

                    self.results[fname].specs.context_test = result
                    self.results[fname].specs.optimized_kv_quant = kv_quant
                    self.results[fname].specs.file_hash = current_hash

                    if result.error:
                        self.stats["context_failed"] += 1
                        logger.warning(f"  └─ Context test failed: {result.error[:80]}")
                    else:
                        self.stats["context_tested"] += 1
                        confidence_str = f" [{result.confidence * 100:.0f}%]" if result.confidence < 1.0 else ""
                        logger.info(
                            f"  └─ Context: {result.max_context} max → {result.recommended_context} stable{confidence_str}")

                    self._save_counter += 1
                    if self._save_counter % 5 == 0:
                        self.save_results(output)

                except KeyboardInterrupt:
                    logger.info("\nInterrupted - saving progress...")
                    self.save_results(output)
                    raise
                except Exception as e:
                    self.stats["context_failed"] += 1
                    logger.error(f"  └─ Context test error: {e}")

        if mmproj_files:
            logger.info(f"\nLinking {len(mmproj_files)} vision adapters...")
            for mmproj_path in mmproj_files:
                mmproj_data = self.scan_mmproj(str(mmproj_path))
                if not mmproj_data:
                    continue

                parent = self.find_parent_model(mmproj_path.name)
                if parent and parent in self.results:
                    self.results[parent].mmproj = mmproj_data

                    parent_lower = parent.lower()
                    if "whisper" in parent_lower:
                        self.results[parent].capabilities.audio_in = True
                    else:
                        self.results[parent].capabilities.vision = True

                    logger.info(f"  {mmproj_path.name} → {parent}")

        self.save_results(output)
        self.print_summary()

    def save_results(self, output: str):
        """Serialize results to JSON with proper defaults for untested models"""
        data = {}
        for fname, entry in self.results.items():
            specs_dict = asdict(entry.specs)

            if not specs_dict["context_test"]["tested"]:
                native = specs_dict["context_window"]
                default_max = min(DEFAULT_UNTESTED_CONTEXT, native // 2)
                specs_dict["context_test"]["max_context"] = default_max
                specs_dict["context_test"]["recommended_context"] = int(default_max * CONTEXT_SAFETY_MARGIN)
                specs_dict["context_test"]["buffer_tokens"] = default_max - specs_dict["context_test"][
                    "recommended_context"]

            data[fname] = {
                "specs": specs_dict,
                "capabilities": vars(entry.capabilities),
                "prompt": entry.prompt,
                "mmproj": entry.mmproj,
                "path": entry.path,
            }

        self.save_atomic(output, data)

    def print_summary(self):
        """Print final statistics"""
        logger.info(f"\n{'=' * 70}")
        logger.info("SCAN COMPLETE")
        logger.info(f"{'=' * 70}")
        logger.info(f"Total files:         {self.stats['total']}")
        logger.info(f"Parsed successfully: {self.stats['parsed']}")
        logger.info(f"Failed to parse:     {self.stats['failed']}")
        if self.stats['context_tested'] or self.stats['context_failed']:
            logger.info(f"Context tested:      {self.stats['context_tested']}")
            logger.info(f"Context skipped:     {self.stats['context_skipped']}")
            logger.info(f"Context failed:      {self.stats['context_failed']}")
        logger.info(f"{'=' * 70}")


# =============================================================================
# ModelScanner - High-level API for llm_manager
# =============================================================================

class ModelScanner:
    """
    High-level scanner API for llm_manager integration.
    
    Wraps PerfectScanner with simplified interface:
    - Generates models.json compatible with ModelRegistry
    - Creates llm_manager.yaml with optimized settings
    - Provides simple API for scanning and configuration
    """

    def __init__(
        self,
        models_dir: Union[str, Path],
        registry_file: str = "models.json",
        config_file: str = "llm_manager.yaml"
    ):
        """
        Initialize scanner.

        Args:
            models_dir: Directory containing GGUF files
            registry_file: Name of registry JSON file (default: models.json)
            config_file: Name of config YAML file (default: llm_manager.yaml)
        """
        self.models_dir = Path(models_dir)
        self.registry_file = self.models_dir / registry_file
        self.config_file = self.models_dir / config_file
        self._scanner = PerfectScanner()

    def scan_and_save(
        self,
        test_context: bool = False,
        resume: bool = False,
        min_context: int = 8192,
        max_context: int = 131072,
        gpu_layers: int = -1,
        gpu_device: Optional[int] = None,
        kv_quant: bool = True,
        flash_attn: bool = True,
        model_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scan models directory and save registry.

        Args:
            test_context: Enable GPU context testing (slow but accurate)
            resume: Resume from previous scan
            min_context: Minimum context to test
            max_context: Maximum context to test
            gpu_layers: GPU layers (-1 = all)
            gpu_device: GPU device index
            kv_quant: Use quantized KV cache
            flash_attn: Use flash attention
            model_filter: Filter models by name substring

        Returns:
            Scan results dictionary

        Raises:
            ValidationError: If models directory not found
        """
        if not self.models_dir.exists():
            if _HAS_LLM_MANAGER:
                raise ValidationError(f"Models directory not found: {self.models_dir}")
            else:
                raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        logger.info(f"Scanning {self.models_dir}...")

        self._scanner.scan_and_test(
            folder=str(self.models_dir),
            output=str(self.registry_file),
            test_context=test_context,
            resume=resume,
            min_ctx=min_context,
            max_ctx=max_context,
            gpu_layers=gpu_layers,
            gpu_device=gpu_device,
            kv_quant=kv_quant,
            flash_attn=flash_attn,
            model_filter=model_filter,
        )

        # Generate llm_manager.yaml if yaml is available
        if YAML_AVAILABLE:
            self._generate_config()

        return {
            "registry_file": str(self.registry_file),
            "config_file": str(self.config_file) if YAML_AVAILABLE else None,
            "stats": self._scanner.stats,
            "models_found": len(self._scanner.results),
        }

    def _generate_config(self) -> None:
        """Generate llm_manager.yaml with optimized settings from scan."""
        if not self._scanner.results:
            logger.warning("No models scanned, skipping config generation")
            return

        config = {
            "llm_manager": {
                "version": "5.0.0",
                "models_dir": str(self.models_dir),
                "registry_file": self.registry_file.name,
            },
            "scan_results": {
                "total_models": len(self._scanner.results),
                "context_tested": self._scanner.stats.get("context_tested", 0),
            },
            "models": {},
            "recommended_defaults": {},
        }

        # Analyze models and extract recommendations
        total_params = []
        max_contexts = []

        for fname, entry in self._scanner.results.items():
            specs = entry.specs

            model_config = {
                "filename": fname,
                "architecture": specs.architecture,
                "parameters_b": specs.parameters_b,
                "quantization": specs.quantization,
                "context_window": specs.context_window,
                "capabilities": {
                    "vision": entry.capabilities.vision,
                    "embedding": entry.capabilities.embed,
                    "reasoning": entry.capabilities.reasoning,
                    "multilingual": entry.capabilities.reasoning,
                    "tools": entry.capabilities.tools,
                },
            }

            if specs.context_test and specs.context_test.tested:
                model_config["recommended_context"] = specs.context_test.recommended_context
                model_config["max_tested_context"] = specs.context_test.max_context
                max_contexts.append(specs.context_test.recommended_context)

            if specs.parameters_b:
                total_params.append(specs.parameters_b)

            config["models"][fname] = model_config

        # Generate recommended defaults based on scanned models
        if max_contexts:
            config["recommended_defaults"]["typical_context"] = int(sum(max_contexts) / len(max_contexts))
            config["recommended_defaults"]["max_safe_context"] = min(max_contexts)

        if total_params:
            avg_params = sum(total_params) / len(total_params)
            config["recommended_defaults"]["avg_parameters_b"] = round(avg_params, 2)

        # Write config
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Generated config: {self.config_file}")
        except Exception as e:
            logger.warning(f"Failed to write llm_manager.yaml: {e}")

    def get_registry(self):
        """
        Get ModelRegistry from scanned results.

        Returns:
            ModelRegistry instance loaded from scan results

        Raises:
            ValidationError: If registry not found or llm_manager not available
        """
        if not _HAS_LLM_MANAGER:
            raise RuntimeError("llm_manager not available, cannot create ModelRegistry")

        if not self.registry_file.exists():
            raise ValidationError(f"Registry not found: {self.registry_file}")

        return ModelRegistry(str(self.models_dir))

    def quick_scan(self) -> Dict[str, Any]:
        """
        Quick scan without context testing.

        Returns:
            Scan results dictionary
        """
        return self.scan_and_save(test_context=False)


# =============================================================================
# Convenience Functions
# =============================================================================

def scan_models(
    models_dir: Union[str, Path],
    test_context: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to scan models directory.

    Args:
        models_dir: Directory containing GGUF files
        test_context: Enable GPU context testing
        **kwargs: Additional arguments passed to ModelScanner.scan_and_save

    Returns:
        Scan results dictionary

    Example:
        >>> from llm_manager.scanner import scan_models
        >>> results = scan_models("./models", test_context=True)
        >>> print(f"Found {results['models_found']} models")
    """
    scanner = ModelScanner(models_dir)
    return scanner.scan_and_save(test_context=test_context, **kwargs)


async def scan_models_async(
    models_dir: Union[str, Path],
    test_context: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Async wrapper for scan_models.

    Args:
        models_dir: Directory containing GGUF files
        test_context: Enable GPU context testing
        **kwargs: Additional arguments passed to ModelScanner.scan_and_save

    Returns:
        Scan results dictionary
    """
    import asyncio
    return await asyncio.to_thread(scan_models, models_dir, test_context, **kwargs)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"Perfect GGUF Scanner v{__version__} - Metadata & Context Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Metadata scan only
  python -m llm_manager.scanner ./models

  # Full scan with context testing (Q4_0 KV cache)
  python -m llm_manager.scanner ./models --test-context

  # Resume interrupted scan
  python -m llm_manager.scanner ./models --test-context --resume

  # Test with FP16 KV (slower, more VRAM)
  python -m llm_manager.scanner ./models --test-context --no-kv-quant

  # Specific GPU device
  python -m llm_manager.scanner ./models --test-context --gpu-device 1
        """
    )

    parser.add_argument("folder", help="Directory containing GGUF files")
    parser.add_argument("-o", "--output", default="models.json", help="Output JSON file")
    parser.add_argument("--test-context", action="store_true", help="Enable GPU context testing")
    parser.add_argument("--resume", action="store_true", help="Resume from previous results")
    parser.add_argument("--min-context", type=int, default=8192, help="Minimum context to test")
    parser.add_argument("--max-context", type=int, default=131072, help="Maximum context to test")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers (-1 = all)")
    parser.add_argument("--gpu-device", type=int, help="GPU device index")
    parser.add_argument("--no-kv-quant", action="store_true", help="Use FP16 for KV cache (more VRAM)")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention")
    parser.add_argument("--model-filter", help="Filter models by name substring")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.test_context:
        try:
            import llama_cpp
            logger.info(f"Using llama-cpp-python {getattr(llama_cpp, '__version__', 'unknown')}")
        except ImportError:
            logger.error("llama-cpp-python required for context testing")
            logger.error("Install: pip install llama-cpp-python")
            return 1

        try:
            import torch
            if torch.cuda.is_available():
                dev = args.gpu_device or 0
                logger.info(f"GPU {dev}: {torch.cuda.get_device_name(dev)}")
            else:
                logger.warning("CUDA not available - testing may fail")
        except ImportError:
            pass

    scanner = PerfectScanner()
    scanner.scan_and_test(
        folder=args.folder,
        output=args.output,
        test_context=args.test_context,
        resume=args.resume,
        min_ctx=args.min_context,
        max_ctx=args.max_context,
        gpu_layers=args.gpu_layers,
        gpu_device=args.gpu_device,
        kv_quant=not args.no_kv_quant,
        flash_attn=not args.no_flash_attn,
        model_filter=args.model_filter,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
