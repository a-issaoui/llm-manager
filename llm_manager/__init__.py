"""
Production-Grade LLM Manager v5.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A robust, production-ready LLM management system.

Basic usage (sync):

    >>> from llm_manager import LLMManager
    >>>
    >>> with LLMManager(models_dir="./models") as manager:
    ...     manager.load_model("my-model.gguf")
    ...     response = manager.generate(
    ...         messages=[{"role": "user", "content": "Hello!"}]
    ...     )
    ...     print(response['choices'][0]['message']['content'])

Basic usage (async):

    >>> import asyncio
    >>> from llm_manager import LLMManager
    >>>
    >>> async def main():
    ...     async with LLMManager(models_dir="./models") as manager:
    ...         await manager.load_model_async("my-model.gguf")
    ...         response = await manager.generate_async(
    ...             messages=[{"role": "user", "content": "Hello!"}]
    ...         )
    ...         print(response['choices'][0]['message']['content'])
    >>> asyncio.run(main())

:copyright: (c) 2026 by Production-Grade LLM Systems
:license: MIT
"""

__version__ = "5.0.0"
__author__ = "Production-Grade LLM Systems"
__license__ = "MIT"

# Core
# Caching
from .cache import DiskCache

# Configuration System
from .config import (
    CacheConfig,
    Config,
    ConfigValidationError,
    ContextConfig,
    EstimationConfig,
    GenerationConfig,
    GPUConfig,
    LoggingConfig,
    ModelConfig,
    ResourceConfig,
    ScannerConfig,
    SecurityConfig,
    ServerConfig,
    WorkerConfig,
    clear_config_cache,
    create_default_config,
    get_config,
    load_config,
    reload_config,
)

# Context Management
from .context import ContextManager, ContextStats
from .core import DEFAULT_CONTEXT_CONFIG, LLMManager

# Token Estimation
from .estimation import ContentType, ConversationType, TokenEstimate, TokenEstimator

# Exceptions
from .exceptions import (
    ContextError,
    GenerationError,
    LLMManagerError,
    ModelLoadError,
    ModelNotFoundError,
    ValidationError,
    WorkerError,
)

# Metrics & Telemetry
from .metrics import MetricsCollector, PerformanceStats, RequestMetrics, get_global_metrics

# Models & Registry
from .models import ModelMetadata, ModelRegistry

# Worker Pools
from .pool import AsyncWorkerPool, WorkerPool

# Model Scanner
from .scanner import (
    ContextTestResult,
    ModelCapabilities,
    ModelEntry,
    ModelScanner,
    ModelSpecs,
    PerfectScanner,
    scan_models,
)

# Tool Calling (Function Calling)
from .tool_parser import extract_tool_names, has_tool_calls, parse_tool_calls

# Worker Processes
from .workers import AsyncWorkerProcess, WorkerProcess

# Server (optional, import may fail if fastapi not installed)
try:
    from .server import LLMServer, create_app

    _server_available = True
except ImportError:
    _server_available = False
    LLMServer = None  # type: ignore
    create_app = None  # type: ignore

__all__ = [
    "AsyncWorkerPool",
    "AsyncWorkerProcess",
    "CacheConfig",
    "clear_config_cache",
    "Config",
    "ConfigValidationError",
    "ContentType",
    "ContextConfig",
    "ContextError",
    "ContextManager",
    "ContextStats",
    "ContextTestResult",
    "ConversationType",
    "create_app",
    "create_default_config",
    "DEFAULT_CONTEXT_CONFIG",
    "DiskCache",
    "EstimationConfig",
    "extract_tool_names",
    "GenerationConfig",
    "GenerationError",
    "get_config",
    "get_global_metrics",
    "GPUConfig",
    "has_tool_calls",
    "LLMManager",
    "LLMManagerError",
    "LLMServer",
    "load_config",
    "LoggingConfig",
    "MetricsCollector",
    "ModelCapabilities",
    "ModelConfig",
    "ModelEntry",
    "ModelLoadError",
    "ModelMetadata",
    "ModelNotFoundError",
    "ModelRegistry",
    "ModelScanner",
    "ModelSpecs",
    "parse_tool_calls",
    "PerfectScanner",
    "PerformanceStats",
    "reload_config",
    "RequestMetrics",
    "ResourceConfig",
    "scan_models",
    "ScannerConfig",
    "SecurityConfig",
    "ServerConfig",
    "TokenEstimate",
    "TokenEstimator",
    "ValidationError",
    "WorkerConfig",
    "WorkerError",
    "WorkerPool",
    "WorkerProcess",
]
