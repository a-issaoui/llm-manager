"""
Production-Grade LLM Manager v5.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A robust, production-ready LLM management system.

Basic usage:

    >>> from llm_manager import LLMManager, get_config
    >>> 
    >>> config = get_config()
    >>> async with LLMManager(config=config) as manager:
    ...     await manager.load_model_async("my-model.gguf")
    ...     response = await manager.generate_async(
    ...         messages=[{"role": "user", "content": "Hello!"}]
    ...     )
    ...     print(response['choices'][0]['message']['content'])

:copyright: (c) 2026 by Production-Grade LLM Systems
:license: MIT
"""

__version__ = "5.0.0"
__author__ = "Production-Grade LLM Systems"
__license__ = "MIT"

# Core
from .core import LLMManager, DEFAULT_CONTEXT_CONFIG

# Models & Registry
from .models import ModelRegistry, ModelMetadata

# Context Management
from .context import ContextManager, ContextStats

# Token Estimation
from .estimation import (
    TokenEstimator, 
    TokenEstimate, 
    ContentType, 
    ConversationType
)

# Worker Processes
from .workers import WorkerProcess, AsyncWorkerProcess

# Worker Pools
from .pool import WorkerPool, AsyncWorkerPool

# Caching
from .cache import DiskCache

# Chat History (Agent Feature)
from .history import ChatHistory, HistoryConfig

# Metrics & Telemetry (Agent Feature)
from .metrics import MetricsCollector, PerformanceStats, RequestMetrics, get_global_metrics

# Model Scanner
from .scanner import (
    ModelScanner,
    scan_models,
    PerfectScanner,
    ModelEntry,
    ModelSpecs,
    ModelCapabilities,
    ContextTestResult,
)

# Configuration System
from .config import (
    Config,
    ModelConfig,
    GenerationConfig,
    ContextConfig,
    WorkerConfig,
    GPUConfig,
    CacheConfig,
    LoggingConfig,
    ScannerConfig,
    SecurityConfig,
    EstimationConfig,
    ResourceConfig,
    ServerConfig,
    get_config,
    load_config,
    reload_config,
    clear_config_cache,
    create_default_config,
    ConfigValidationError,
)

# Exceptions
from .exceptions import (
    LLMManagerError,
    ModelLoadError,
    ModelNotFoundError,
    GenerationError,
    WorkerError,
    ContextError,
    ValidationError,
)

# Server (optional, import may fail if fastapi not installed)
try:
    from .server import LLMServer, create_app
    _server_available = True
except ImportError:
    _server_available = False
    LLMServer = None
    create_app = None

__all__ = [
    # Core
    "LLMManager",
    # Models & Registry
    "ModelRegistry",
    "ModelMetadata",
    # Context
    "ContextManager",
    "ContextStats",
    # Estimation
    "TokenEstimator",
    "TokenEstimate",
    "ContentType",
    "ConversationType",
    # Workers
    "WorkerProcess",
    "AsyncWorkerProcess",
    # Pools
    "WorkerPool",
    "AsyncWorkerPool",
    # Cache
    "DiskCache",
    # Chat History (Agent Feature)
    "ChatHistory",
    "HistoryConfig",
    # Metrics (Agent Feature)
    "MetricsCollector",
    "PerformanceStats",
    "RequestMetrics",
    "get_global_metrics",
    # Scanner
    "ModelScanner",
    "scan_models",
    "PerfectScanner",
    "ModelEntry",
    "ModelSpecs",
    "ModelCapabilities",
    "ContextTestResult",
    # Configuration
    "Config",
    "ModelConfig",
    "GenerationConfig",
    "ContextConfig",
    "WorkerConfig",
    "GPUConfig",
    "CacheConfig",
    "LoggingConfig",
    "ScannerConfig",
    "SecurityConfig",
    "EstimationConfig",
    "ResourceConfig",
    "ServerConfig",
    "DEFAULT_CONTEXT_CONFIG",
    "get_config",
    "load_config",
    "reload_config",
    "clear_config_cache",
    "create_default_config",
    "ConfigValidationError",
    # Exceptions
    "LLMManagerError",
    "ModelLoadError",
    "ModelNotFoundError",
    "GenerationError",
    "WorkerError",
    "ContextError",
    "ValidationError",
    # Server (optional)
    "LLMServer",
    "create_app",
]
