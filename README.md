# LLM Manager v5.0 🚀

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-747%20passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-89%25-green.svg)]()
[![Code Quality](https://img.shields.io/badge/code%20quality-9.42%2F10-brightgreen.svg)]()

**Production-grade LLM management system with OpenAI-compatible API, intelligent context sizing, process isolation, and multi-agent support.**

## 🌟 Features

### Core Capabilities
- ✅ **OpenAI-Compatible REST API** - Drop-in replacement for OpenAI API
- ✅ **Tool Calling** - XML-based function calling with `<tool_call>` syntax
- ✅ **Intelligent Context Management** - Automatic context sizing based on conversation
- ✅ **Process Isolation** - Safe subprocess execution prevents crashes
- ✅ **Model Registry** - Centralized metadata with GPU-optimized configs
- ✅ **Rate Limiting** - Token bucket rate limiter per client
- ✅ **Async Support** - First-class async/await for high concurrency
- ✅ **Model Hot-Switching** - Switch models without restarting
- ✅ **Metrics & Monitoring** - Prometheus-compatible metrics and telemetry
- ✅ **Worker Pools** - Managed worker processes for parallel generation
- ✅ **Token Estimation** - Fast heuristic and accurate estimation
- ✅ **Comprehensive Testing** - 747 tests, 89% coverage

### Security Features
- 🔒 **Path Traversal Protection** - Blocks `../` and absolute path attacks
- 🔒 **Request Queue** - Backpressure with semaphore-based concurrency limiting
- 🔒 **Input Validation** - Message limits, injection pattern detection
- 🔒 **File Type Validation** - Only `.gguf` files allowed

### Agent Features
- 🤖 **Multi-Agent Support** - AutoGen and LangChain integration
- 📊 **Metrics Collection** - Request tracking and performance stats
- 🔧 **Tool Calling** - Function calling via XML parsing

### Safety & Robustness
- 🛡️ **Timeout Protection** - Hard limits on all operations
- 🛡️ **Resource Cleanup** - Automatic cleanup on exit
- 🛡️ **Error Recovery** - Graceful failure handling
- 🛡️ **Input Validation** - Type checking and bounds validation
- 🛡️ **Rate Limiting** - Per-client request throttling
- 🛡️ **CORS Support** - Cross-origin request handling

## 📦 Installation

```bash
# Install llama-cpp-python with GPU support (recommended)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python

# Install from source
git clone https://github.com/a-issaoui/llm-manager
cd llm-manager
pip install -e .

# For agent features
pip install pyautogen langchain langchain-openai

# For HTTP examples
pip install requests aiohttp

# Run tests
pytest tests/ -v
```

## 🚀 Quick Start

### 1. Start the OpenAI-Compatible Server

```bash
# Start server with default settings
llm-manager

# Or with custom settings
llm-manager --port 8000 --host 0.0.0.0 --models-dir ./models
```

### 2. Use with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="your-model.gguf",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7
)

print(response.choices[0].message.content)
```

### 3. Tool Calling

Models that support tool calling can output XML that gets parsed automatically:

```xml
<tool_call>
  {"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call>
```

The API will return this as an OpenAI-compatible tool_calls response.

### 4. Direct Python API

```python
from llm_manager import LLMManager

# Initialize with GPU support
manager = LLMManager(models_dir="./models")

# Load model
manager.load_model(
    "model.gguf",
    config={
        "n_ctx": 4096,
        "n_gpu_layers": -1  # All layers on GPU
    }
)

# Generate
response = manager.generate(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
    temperature=0.7
)

print(response["choices"][0]["message"]["content"])

# Cleanup
manager.unload_model()
```

### 5. Async Usage

```python
import asyncio
from llm_manager import LLMManager

async def main():
    manager = LLMManager(models_dir="./models")
    await manager.load_model_async("model.gguf")
    
    response = await manager.generate_async(
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=256
    )
    print(response["choices"][0]["message"]["content"])
    
    manager.unload_model()

asyncio.run(main())
```

## 📚 Documentation

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (OpenAI-compatible) |
| `POST /v1/completions` | Text completions |
| `GET /v1/models` | List available models |
| `GET /v1/models/{id}` | Get model details |
| `POST /v1/models/{id}/load` | Load a model |
| `POST /v1/models/{id}/unload` | Unload a model |
| `GET /health` | Health check |
| `GET /health/detailed` | Detailed health status |
| `GET /metrics` | Prometheus metrics |
| `GET /docs` | API documentation (Swagger UI) |

### OpenAI API Compatibility

| Parameter | Support | Notes |
|-----------|---------|-------|
| `model` | ✅ Full | Model ID or filename |
| `messages` | ✅ Full | Chat history |
| `temperature` | ✅ Full | 0.0 - 2.0 |
| `top_p` | ✅ Full | 0.0 - 1.0 |
| `max_tokens` | ✅ Full | Maximum generation length |
| `stop` | ✅ Full | Stop sequences |
| `seed` | ✅ Full | Reproducible sampling |
| `presence_penalty` | ✅ Full | -2.0 - 2.0 |
| `frequency_penalty` | ✅ Full | -2.0 - 2.0 |
| `logit_bias` | ✅ Full | Token bias (-100 to 100) |
| `stream` | ✅ Full | Server-sent events |
| `stream_options` | ✅ Full | `include_usage` support |
| `tools` | ✅ Full | Function calling supported |
| `tool_choice` | ✅ Full | Auto/none/function selection |
| `n` | ⚠️ Limited | Only `n=1` supported (will error if > 1) |
| `response_format` | ⚠️ Parsed only | Accepted in API but not used in generation |

### CLI Commands

```bash
# Start server
llm-manager server

# Scan models
llm-manager scan ./models --test-context

# Load/unload models via API
curl http://localhost:8000/v1/models/my-model/load -X POST
curl http://localhost:8000/v1/models/my-model/unload -X POST
```

### Configuration

Create `llm_manager.yaml`:

```yaml
models:
  dir: "./models"
  default_ctx: 8192
  use_subprocess: true

generation:
  max_tokens: 4096
  temperature: 0.7
  top_p: 0.9

gpu:
  n_gpu_layers: -1  # All layers on GPU

cache:
  enabled: true
  dir: "./cache"
  max_size_mb: 5120

server:
  port: 8000
  host: "0.0.0.0"
  workers: 1
  cors_origins: ["*"]
  rate_limit: 60  # Requests per minute per client
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenAI-Compatible API                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ /v1/chat     │  │ /v1/completions│  │ /v1/models        │  │
│  │ completions  │  │              │  │ /health            │  │
│  │              │  │              │  │ /metrics           │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Server Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐    │
│  │ Rate Limiter│  │ RequestQueue│  │ Tool Parser          │    │
│  │             │  │ (Backpressure)│  │ Metrics Collector  │    │
│  └─────────────┘  └─────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLMManager (Stateless)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐    │
│  │ Model       │  │ Generation  │  │ Async Worker Pool    │    │
│  │ Loading     │  │ Engine      │  │ Process Isolation    │    │
│  │ Registry    │  │ Context Mgr │  │                      │    │
│  └─────────────┘  └─────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    llama-cpp-python                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  GGUF Model Loading | GPU Acceleration | Tokenization   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Stateless Design

LLM Manager follows a **stateless architecture** like OpenAI's API:
- **No session management** - Agents own their own memory/state
- **No chat history** - Clients manage conversation history
- **Request-scoped** - Each request is independent
- **Scalable** - Easy to deploy multiple instances behind a load balancer

## 🧪 Examples

See `examples/` directory for comprehensive examples using real models:

| Example | Description |
|---------|-------------|
| `01_basic_generation.py` | Basic text generation |
| `02_chat_completion.py` | Chat with conversation history |
| `03_openai_server.py` | OpenAI-compatible HTTP server |
| `04_streaming_generation.py` | Streaming text generation |
| `05_async_generation.py` | Async generation |
| `06_batch_processing.py` | Batch processing |
| `07_embeddings.py` | Embedding model usage |
| `08_model_switching.py` | Dynamic model switching |
| `09_tool_calling.py` | Tool/function calling |
| `10_subprocess_mode.py` | Process isolation mode |
| `11_concurrent_http.py` | Concurrent HTTP requests |
| `12_model_comparison.py` | Model comparison |

### Running Examples

```bash
# Run individual example
python examples/01_basic_generation.py

# Run all examples with reporting
python examples/run_all_examples.py
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=llm_manager --cov-report=html

# Run specific test file
pytest tests/test_core.py -v

# Run server tests
pytest tests/test_server*.py -v

# Run security tests
pytest tests/test_security_features.py -v

# Run with linting
ruff check llm_manager/
pylint llm_manager/ --disable=all --enable=E,W
```

## 🔧 Model Scanner

The included model scanner tests and records optimal GPU context sizes:

```bash
# Basic scan
llm-manager scan ./models

# With context testing (requires GPU)
llm-manager scan ./models --test-context

# With KV quantization (less VRAM)
llm-manager scan ./models --test-context --kv-quant

# Resume interrupted scan
llm-manager scan ./models --test-context --resume
```

## 📊 Metrics

Prometheus-compatible metrics available at `/metrics`:

- `llm_requests_total` - Total requests (labeled by model, status)
- `llm_request_duration_seconds` - Request latency histogram
- `llm_tokens_input_total` - Input tokens processed
- `llm_tokens_output_total` - Output tokens generated
- `llm_model_load_duration_seconds` - Model load time
- `llm_active_models` - Currently loaded models
- `llm_context_size_bytes` - Context size per model

## 🔒 Security Features

### Path Traversal Protection
- Blocks `../` and absolute path attacks
- Only `.gguf` files allowed
- All paths resolved within models directory

### Request Queue with Backpressure
- Semaphore-based concurrency limiting (default: 10)
- Returns HTTP 503 when overloaded
- Timeout handling for queued requests

### Input Validation
- Max 100 messages per request
- Max 100KB per message
- Max 1MB total content
- Blocks `<script>` and `javascript:` patterns

## 🔒 Rate Limiting

Built-in token bucket rate limiter (default: 60 requests/minute per client):

```python
# Configure via config file
server:
  rate_limit: 60  # requests per minute
  rate_limit_window: 60  # window in seconds

# Or programmatically
from llm_manager.server.rate_limiter import get_rate_limiter

limiter = get_rate_limiter(requests_per_minute=100)
```

## 🛠️ Code Quality

- **Linting**: Ruff + Pylint (9.42/10)
- **Type Checking**: MyPy with strict mode
- **Test Coverage**: 89% (747 tests)
- **CI/CD**: Automated testing on Python 3.10-3.12

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Run linting: `ruff check llm_manager/`
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Python bindings for llama.cpp
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [pydantic](https://docs.pydantic.dev/) - Data validation
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - High-performance LLM inference

---

**Made with ❤️ for the AI community**
