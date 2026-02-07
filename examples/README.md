# LLM Manager Examples

This directory contains comprehensive examples demonstrating all features of the LLM Manager library.

## Available Models

The examples use real models from the `../models/` directory:

| Model | Size | Best For |
|-------|------|----------|
| `SmolLM-1.7B-Instruct-Q4_K_M.gguf` | ~1GB | Fast inference, basic tasks |
| `Qwen2.5-3b-instruct-q4_k_m.gguf` | ~2GB | Instruction following, chat |
| `Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf` | ~2GB | Code generation |
| `DeepSeek-R1-Distill-Llama-3B.Q5_K_M.gguf` | ~2GB | Reasoning tasks |
| `Reason-With-Choice-3B.Q4_K_M.gguf` | ~2GB | Multiple choice reasoning |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | ~600MB | Embeddings, semantic search |

## Examples

### 01_basic_generation.py
Basic text generation with a simple prompt.
```bash
python examples/01_basic_generation.py
```

### 02_chat_completion.py
Chat completion with system prompts and conversation history.
```bash
python examples/02_chat_completion.py
```

### 03_openai_server.py
OpenAI-compatible HTTP server with API endpoints.
```bash
python examples/03_openai_server.py
```

### 04_streaming_generation.py
Real-time streaming of generated tokens.
```bash
python examples/04_streaming_generation.py
```

### 05_async_generation.py
Async/await pattern for non-blocking operations.
```bash
python examples/05_async_generation.py
```

### 06_batch_processing.py
Efficient batch processing of multiple prompts.
```bash
python examples/06_batch_processing.py
```

### 07_embeddings.py
Text embeddings for similarity and semantic search.
```bash
python examples/07_embeddings.py
```

### 08_model_switching.py
Dynamically switching between models for different tasks.
```bash
python examples/08_model_switching.py
```

### 09_tool_calling.py
Function calling for agent-like behavior.
```bash
python examples/09_tool_calling.py
```

### 10_subprocess_mode.py
Process isolation for stability and crash protection.
```bash
python examples/10_subprocess_mode.py
```

### 11_concurrent_http.py
Multiple concurrent requests to the HTTP server.
```bash
python examples/11_concurrent_http.py
```

### 12_model_comparison.py
Compare different models on the same prompts.
```bash
python examples/12_model_comparison.py
```

## Running All Examples

To run all examples in sequence:

```bash
python examples/run_all_examples.py
```

This will execute each example and report pass/fail status.

## Requirements

All examples require:
- Python 3.10+
- llm_manager installed (`pip install -e .`)
- Models downloaded in `../models/` directory

For HTTP examples:
```bash
pip install requests aiohttp
```

## Expected Runtime

| Example | Approximate Time |
|---------|-----------------|
| 01_basic_generation.py | 30-60s |
| 02_chat_completion.py | 30-60s |
| 03_openai_server.py | 60-120s |
| 04_streaming_generation.py | 30-60s |
| 05_async_generation.py | 60-120s |
| 06_batch_processing.py | 60-120s |
| 07_embeddings.py | 30-60s |
| 08_model_switching.py | 120-240s |
| 09_tool_calling.py | 60-120s |
| 10_subprocess_mode.py | 30-60s |
| 11_concurrent_http.py | 60-120s |
| 12_model_comparison.py | 120-240s |

**Total: ~15-30 minutes for all examples**
