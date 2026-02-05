# LLM Manager v5.0 🚀

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-696%20passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-92%25-green.svg)]()

**Production-grade LLM management system with OpenAI-compatible API, intelligent context sizing, process isolation, and multi-agent support.**

## 🌟 Features

### Core Capabilities
- ✅ **OpenAI-Compatible REST API** - Drop-in replacement for OpenAI API
- ✅ **Intelligent Context Management** - Automatic context sizing based on conversation
- ✅ **Process Isolation** - Safe subprocess execution prevents crashes
- ✅ **Model Registry** - Centralized metadata with GPU-optimized configs
- ✅ **Async Support** - First-class async/await for high concurrency
- ✅ **Model Hot-Switching** - Switch models without reloading
- ✅ **Chat History** - Automatic truncation and conversation branching
- ✅ **Metrics & Monitoring** - Performance tracking and telemetry
- ✅ **Worker Pools** - Managed worker processes for parallel generation
- ✅ **Token Estimation** - Fast heuristic and accurate estimation
- ✅ **Comprehensive Testing** - 696 tests, 92% coverage

### Agent Features
- 🤖 **Multi-Agent Support** - AutoGen and LangChain integration
- 💬 **Conversation Memory** - Smart context window management
- 📊 **Metrics Collection** - Request tracking and performance stats
- 🔧 **Tool Calling** - Function calling capabilities

### Safety & Robustness
- 🛡️ **Timeout Protection** - Hard limits on all operations
- 🛡️ **Resource Cleanup** - Automatic cleanup on exit
- 🛡️ **Error Recovery** - Graceful failure handling
- 🛡️ **Input Validation** - Type checking and bounds validation
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

### 3. Direct Python API

```python
from llm_manager import LLMManager

# Initialize with GPU support
manager = LLMManager(models_dir="./models")

# Load model
manager.load_model(
    "model.gguf",
    n_ctx=4096,
    n_gpu_layers=-1  # All layers on GPU
)

# Generate
response = manager.generate(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
    temperature=0.7
)

print(response['choices'][0]['message']['content'])
manager.unload_model()
```

### 4. Async Usage

```python
import asyncio
from llm_manager import LLMManager

async def main():
    async with LLMManager() as manager:
        await manager.load_model_async("model.gguf")
        
        # Concurrent generation
        tasks = [
            manager.generate_async(
                messages=[{"role": "user", "content": f"Question {i}"}]
            )
            for i in range(5)
        ]
        
        responses = await asyncio.gather(*tasks)
        for r in responses:
            print(r['choices'][0]['message']['content'])

asyncio.run(main())
```

## 📚 API Reference

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming supported) |
| `/v1/completions` | POST | Legacy completion |
| `/v1/models` | GET | List available models |
| `/v1/models/{id}` | GET | Get model info |
| `/health` | GET | Health check |
| `/info` | GET | Server info |

### Configuration (llm_manager.yaml)

```yaml
version: "5.0.0"
models:
  dir: ./models
  registry_file: models.json
  use_subprocess: true
  pool_size: 4

server:
  host: 127.0.0.1
  port: 8000
  cors_origins:
    - http://localhost:3000
  request_timeout: 120.0

generation:
  default_temperature: 0.7
  default_max_tokens: 2048

context:
  default_size: 4096
  tiers: [2048, 4096, 8192, 16384, 32768, 65536, 131072]
```

## 🏗️ Architecture

```
llm_manager/
├── core.py              # Main LLMManager class
├── models.py            # Model registry & metadata
├── context.py           # Context management
├── estimation.py        # Token estimation
├── workers.py           # Subprocess workers
├── pool.py              # Worker pools
├── history.py           # Chat history
├── metrics.py           # Metrics collection
├── config.py            # Configuration system
├── server/              # REST API server
│   ├── app.py          # FastAPI application
│   ├── routes/         # API endpoints
│   └── dependencies.py # FastAPI dependencies
└── exceptions.py        # Custom exceptions
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=llm_manager --cov-report=html

# Run specific module
pytest tests/test_core.py -v

# Run with timeout
pytest tests/ --timeout=60
```

**Test Coverage: 696 tests, 92% coverage**

## 📊 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Model Load (3B Q4) | ~2s | SSD, GPU |
| Context Calculation | ~1ms | Auto-sizing |
| Token Estimation | ~0.5ms | Heuristic |
| Generation (100 tokens) | ~3s | GPU, 4096 ctx |
| Model Hot-Switch | ~1s | Same size models |

## 🛠️ Examples

All examples use real LLMs from the models directory:

```bash
# Basic usage
python examples/01_basic_usage.py

# OpenAI server
python examples/02_openai_server.py

# AutoGen agents
python examples/03_agents_autogen.py

# LangChain integration
python examples/04_agents_langchain.py

# Chat history
python examples/05_chat_history.py

# Batch generation
python examples/06_batch_generation.py

# Metrics monitoring
python examples/07_metrics_monitoring.py

# Advanced config
python examples/08_advanced_config.py

# Worker pools
python examples/09_worker_pools.py

# Complete workflow
python examples/10_complete_agent_workflow.py
```

## ⚙️ Environment Variables

```bash
# Models
LLM_MODELS_DIR=./models
LLM_DEFAULT_MODEL=model.gguf

# Server
LLM_HOST=127.0.0.1
LLM_PORT=8000
LLM_API_KEY=secret

# GPU
LLM_GPU_LAYERS=-1
LLM_FLASH_ATTN=true

# Timeouts
LLM_REQUEST_TIMEOUT=120.0
LLM_WORKER_TIMEOUT=30.0
```

## 🐛 Troubleshooting

### Model Won't Load
```python
# Check registry
from llm_manager import get_config
print(get_config().models.get_registry_path())

# Use conservative settings
manager.load_model("model.gguf", n_ctx=2048, n_gpu_layers=0)
```

### Out of Memory
```python
# Reduce context
manager.load_model("model.gguf", n_ctx=2048)

# Use CPU only
manager.load_model("model.gguf", n_gpu_layers=0)
```

### Server Not Responding
```bash
# Check if port is free
lsof -i :8000

# Use different port
llm-manager --port 8080
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all 696 tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built on [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- OpenAI-compatible API design
- Production-ready patterns

---

**Made with ❤️ for the LLM community**
