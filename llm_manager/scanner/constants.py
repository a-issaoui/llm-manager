"""
Constants for the scanner module.
"""

import re

__version__ = "8.0.0"

STABILITY_RETRIES = 2
CLEANUP_DELAY_SUCCESS = 0.2
CLEANUP_DELAY_FAILURE = 1.0
MAX_REASONABLE_CONTEXT = 1_000_000
DEFAULT_UNTESTED_CONTEXT = 32768
STABILITY_REDUCTION_FACTOR = 0.1
LARGE_ARRAY_THRESHOLD = 1000
CONTEXT_SAFETY_MARGIN = 0.95  # Default margin for recommended context
MIN_BINARY_SEARCH_GAP = 256
TEST_TIMEOUT = 120

# Pre-compiled regex patterns
_PARAM_PATTERNS = [
    re.compile(r"(?:embedding[_-])?(\d+(?:\.\d+)?)[_-]?b[_-]?(?:embedding)?"),
    re.compile(r"(\d+(?:\.\d+)?)[_-]?b(?:[_-]|$)"),
    re.compile(r"(\d+\.?\d*)[Bb](?![a-zA-Z])"),
    re.compile(r"(\d+)m\b"),
]

_QUANT_PATTERNS = {
   "Q4_0": re.compile(r"q4_0", re.IGNORECASE),
   "Q4_K_M": re.compile(r"q4_k_m", re.IGNORECASE),
   "Q4_K_S": re.compile(r"q4_k_s", re.IGNORECASE),
   "Q5_0": re.compile(r"q5_0", re.IGNORECASE),
   "Q5_K_M": re.compile(r"q5_k_m", re.IGNORECASE),
   "Q5_K_S": re.compile(r"q5_k_s", re.IGNORECASE),
   "Q8_0": re.compile(r"q8_0", re.IGNORECASE),
   "F16": re.compile(r"f16", re.IGNORECASE),
   "Q6_K": re.compile(r"q6_k", re.IGNORECASE),
   "Q3_K_M": re.compile(r"q3_k_m", re.IGNORECASE),
   "Q3_K_L": re.compile(r"q3_k_l", re.IGNORECASE),
   "Q3_K_S": re.compile(r"q3_k_s", re.IGNORECASE),
   "Q2_K": re.compile(r"q2_k", re.IGNORECASE),
   "IQ4_XS": re.compile(r"iq4_xs", re.IGNORECASE),
}

CONTEXT_TEST_WORKER = """
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
    except Exception:
        version = 'unknown'

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
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
        except Exception:
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
"""
