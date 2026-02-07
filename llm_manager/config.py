"""
Unified Configuration System for llm_manager.

Supports YAML config files with environment variable overrides.

Usage:
    >>> from llm_manager.config import get_config
    >>> config = get_config()
    >>> config.models.dir  # doctest: +SKIP
    './models'
    >>> config.worker.critical_timeout  # doctest: +SKIP
    120
"""

import json
import logging
import os
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


# =============================================================================
# Configuration Sections
# =============================================================================


@dataclass
class ModelConfig:
    """Model loading configuration."""

    dir: str = "./models"
    registry_file: str = "models.json"  # Always relative to models dir
    use_subprocess: bool = True
    pool_size: int = 0

    def __post_init__(self) -> None:
        if self.pool_size < 0:
            raise ConfigValidationError("models.pool_size must be >= 0")

    def get_registry_path(self) -> Path:
        """Get full path to registry file (always inside models dir)."""
        return Path(self.dir) / self.registry_file


@dataclass
class GenerationConfig:
    """Text generation defaults."""

    default_max_tokens: int = 2048
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 40
    default_repeat_penalty: float = 1.1
    auto_context: bool = True
    stream_chunk_size: int = 16
    response_buffer: int = 2048
    reasoning_buffer: int = 4096
    tool_buffer: int = 1024

    def __post_init__(self) -> None:
        if not 0 <= self.default_temperature <= 2:
            raise ConfigValidationError("generation.default_temperature must be in [0, 2]")
        if not 0 <= self.default_top_p <= 1:
            raise ConfigValidationError("generation.default_top_p must be in [0, 1]")
        if self.default_max_tokens < 1:
            raise ConfigValidationError("generation.default_max_tokens must be >= 1")


@dataclass
class ContextConfig:
    """Context window configuration."""

    default_size: int = 4096
    min_size: int = 2048
    max_size: int = 131072
    upsize_threshold: float = 0.9
    downsize_threshold: float = 0.5
    safety_margin: int = 512
    auto_resize: bool = True
    min_downsize: int = 4096
    tiers: list[int] = field(
        default_factory=lambda: [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    )

    def __post_init__(self) -> None:
        if self.min_size >= self.max_size:
            raise ConfigValidationError("context.min_size must be < context.max_size")
        if not 0 < self.upsize_threshold <= 1:
            raise ConfigValidationError("context.upsize_threshold must be in (0, 1]")


@dataclass
class WorkerConfig:
    """Worker process configuration."""

    critical_timeout: float = 30.0
    start_timeout: float = 10.0
    idle_timeout: int = 3600
    reuse_limit: int = 100
    max_pool_size: int = 8

    def __post_init__(self) -> None:
        if self.critical_timeout < 1:
            raise ConfigValidationError("worker.critical_timeout must be >= 1")


@dataclass
class GPUConfig:
    """GPU acceleration configuration."""

    default_gpu_layers: int = -1
    flash_attention: bool = True
    kv_cache_quantization: bool = True
    type_k: str | None = "q4_0"  # Default: 4-bit quantized KV cache (K)
    type_v: str | None = "q4_0"  # Default: 4-bit quantized KV cache (V)

    VALID_TYPES: ClassVar[set[str | None]] = {None, "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"}

    def __post_init__(self) -> None:
        if self.type_k not in self.VALID_TYPES:
            raise ConfigValidationError(f"gpu.type_k must be one of {self.VALID_TYPES}")
        if self.type_v not in self.VALID_TYPES:
            raise ConfigValidationError(f"gpu.type_v must be one of {self.VALID_TYPES}")


@dataclass
class CacheConfig:
    """Caching configuration."""

    token_max_size: int = 1000
    template_max_size: int = 100
    sequence_max_size: int = 500
    disk_enabled: bool = True
    disk_dir: str = "~/.cache/llm_manager"
    ttl_seconds: int = 3600
    cleanup_delay_success: float = 0.3
    cleanup_delay_failure: float = 1.0

    def __post_init__(self) -> None:
        if self.token_max_size < 1:
            raise ConfigValidationError("cache.token_max_size must be >= 1")


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str | None = None

    def __post_init__(self) -> None:
        self.level = self.level.upper()


@dataclass
class ScannerConfig:
    """Model scanner configuration."""

    test_context: bool = False
    min_context: int = 8192
    max_context: int = 131072
    kv_quant: bool = True
    flash_attn: bool = True

    def __post_init__(self) -> None:
        if self.min_context >= self.max_context:
            raise ConfigValidationError("scanner.min_context must be < scanner.max_context")


@dataclass
class SecurityConfig:
    """Security settings."""

    allow_external_paths: bool = False
    max_model_size_mb: int = 100_000
    sandbox_workers: bool = True


@dataclass
class EstimationConfig:
    """Token estimation heuristics."""

    tokens_per_word_text: float = 1.3
    tokens_per_word_code: float = 3.5
    tokens_per_char_cjk: float = 1.5
    template_overhead_per_message: int = 30
    special_tokens_base: int = 50
    image_tokens: int = 1000


@dataclass
class ResourceConfig:
    """Resource management settings."""

    min_disk_space_mb: int = 100
    batch_size_small: int = 1024
    batch_size_medium: int = 512
    batch_size_large: int = 256


@dataclass
class ServerConfig:
    """OpenAI-compatible server configuration."""

    enabled: bool = True  # Enable REST API server
    host: str = "127.0.0.1"  # Bind address (use 0.0.0.0 for remote)
    port: int = 8000  # HTTP port
    api_key: str | None = None  # Optional API key for auth
    cors_origins: list[str] = field(
        default_factory=lambda: [  # Allowed CORS origins
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
        ]
    )
    request_timeout: float = 120.0  # Request timeout in seconds
    max_request_size_mb: int = 100  # Max request body size
    enable_docs: bool = True  # Enable /docs endpoint
    enable_metrics_endpoint: bool = True  # Enable /metrics endpoint
    log_requests: bool = True  # Log all requests

    def __post_init__(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ConfigValidationError("server.port must be between 1 and 65535")
        if self.request_timeout < 1:
            raise ConfigValidationError("server.request_timeout must be >= 1")
        if self.max_request_size_mb < 1:
            raise ConfigValidationError("server.max_request_size_mb must be >= 1")


# =============================================================================
# Main Configuration
# =============================================================================


@dataclass
class Config:
    """Complete llm_manager configuration."""

    version: str = "5.0.0"
    profile: str = "default"

    models: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for model_name, overrides in self.model_overrides.items():
            if not isinstance(overrides, dict):
                raise ConfigValidationError(f"model_overrides['{model_name}'] must be a dict")

    def _merge_configs(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two configurations."""
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            elif isinstance(value, list) and isinstance(result.get(key), list):
                result[key] = result[key] + value
            else:
                result[key] = deepcopy(value)
        return result

    def get_model_config(self, model_name: str) -> dict[str, Any]:
        """Get effective configuration for a specific model."""
        base = {
            "gpu_layers": self.gpu.default_gpu_layers,
            "flash_attention": self.gpu.flash_attention,
            "context_size": self.context.default_size,
            "temperature": self.generation.default_temperature,
            "max_tokens": self.generation.default_max_tokens,
            "top_p": self.generation.default_top_p,
            "top_k": self.generation.default_top_k,
        }
        if model_name in self.model_overrides:
            base.update(self.model_overrides[model_name])
        return base

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "version": self.version,
            "profile": self.profile,
            "models": asdict(self.models),
            "generation": asdict(self.generation),
            "context": asdict(self.context),
            "worker": asdict(self.worker),
            "gpu": asdict(self.gpu),
            "cache": asdict(self.cache),
            "logging": asdict(self.logging),
            "scanner": asdict(self.scanner),
            "security": asdict(self.security),
            "estimation": asdict(self.estimation),
            "resource": asdict(self.resource),
            "server": asdict(self.server),
            "model_overrides": self.model_overrides,
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in (".yaml", ".yml"):
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML required for YAML export")
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        elif path.suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        try:
            return cls(
                version=data.get("version", "5.0.0"),
                profile=data.get("profile", "default"),
                models=ModelConfig(**data.get("models", {})),
                generation=GenerationConfig(**data.get("generation", {})),
                context=ContextConfig(**data.get("context", {})),
                worker=WorkerConfig(**data.get("worker", {})),
                gpu=GPUConfig(**data.get("gpu", {})),
                cache=CacheConfig(**data.get("cache", {})),
                logging=LoggingConfig(**data.get("logging", {})),
                scanner=ScannerConfig(**data.get("scanner", {})),
                security=SecurityConfig(**data.get("security", {})),
                estimation=EstimationConfig(**data.get("estimation", {})),
                resource=ResourceConfig(**data.get("resource", {})),
                server=ServerConfig(**data.get("server", {})),
                model_overrides=data.get("model_overrides", {}),
            )
        except TypeError as e:
            raise ConfigValidationError(f"Invalid configuration: {e}") from e

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load Config from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML required")
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls.from_dict(data)


# =============================================================================
# Configuration Loader
# =============================================================================

_config_cache: Config | None = None
_config_lock = threading.Lock()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif isinstance(value, list) and isinstance(result.get(key), list):
            result[key] = result[key] + value
        else:
            result[key] = deepcopy(value)
    return result


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides."""
    mappings = {
        "LLM_MODELS_DIR": ("models", "dir"),
        "LLM_REGISTRY_FILE": ("models", "registry_file"),
        "LLM_USE_SUBPROCESS": ("models", "use_subprocess"),
        "LLM_POOL_SIZE": ("models", "pool_size"),
        "LLM_DEFAULT_TEMPERATURE": ("generation", "default_temperature"),
        "LLM_DEFAULT_MAX_TOKENS": ("generation", "default_max_tokens"),
        "LLM_DEFAULT_CONTEXT": ("context", "default_size"),
        "LLM_CRITICAL_TIMEOUT": ("worker", "critical_timeout"),
        "LLM_GPU_LAYERS": ("gpu", "default_gpu_layers"),
        "LLM_FLASH_ATTENTION": ("gpu", "flash_attention"),
        "LLM_LOG_LEVEL": ("logging", "level"),
    }

    for env_name, (section, key) in mappings.items():
        env_value = os.getenv(env_name)
        if env_value is None:
            continue

        if section not in data:
            data[section] = {}

        # Type coercion
        coerced_value: str | bool | int | float
        if key in ["use_subprocess", "flash_attention"]:
            coerced_value = env_value.lower() in ("true", "1", "yes", "on")
        elif key in ["pool_size", "default_gpu_layers", "reuse_limit"]:
            try:
                coerced_value = int(env_value)
            except ValueError:
                continue
        elif key in ["default_temperature", "critical_timeout"]:
            try:
                coerced_value = float(env_value)
            except ValueError:
                continue
        else:
            coerced_value = env_value

        data[section][key] = coerced_value
        logger.debug(f"Applied env override: {env_name}")

    return data


def _get_config_paths(profile: str | None = None) -> list[Path]:
    """Get config file search paths."""
    paths = []

    if profile and profile != "default":
        paths.extend(
            [
                Path(f"llm_manager.{profile}.yaml"),
                Path(f"config.{profile}.yaml"),
                Path.home() / ".config" / "llm_manager" / f"config.{profile}.yaml",
            ]
        )

    paths.extend(
        [
            Path("llm_manager.yaml"),  # Project root
            Path.home() / ".config" / "llm_manager" / "llm_manager.yaml",  # User config
            Path("/etc/llm_manager/llm_manager.yaml"),  # System-wide
        ]
    )

    return [p for p in paths if p.exists()]


def load_config(
    path: str | Path | None = None, profile: str | None = None, use_env: bool = True
) -> Config:
    """Load configuration from file(s)."""
    data: dict[str, Any] = {}

    if path:
        config_path = Path(path)
        if config_path.exists():
            config = Config.from_file(config_path)
            data = config.to_dict()
    else:
        for config_path in _get_config_paths(profile):
            try:
                config = Config.from_file(config_path)
                data = _deep_merge(data, config.to_dict())
            except Exception as e:
                logger.warning(f"Failed to load {config_path}: {e}")

    if use_env:
        data = _apply_env_overrides(data)

    if profile:
        data["profile"] = profile

    return Config.from_dict(data)


def get_config(
    path: str | Path | None = None,
    profile: str | None = None,
    use_env: bool = True,
    reload: bool = False,
) -> Config:
    """Get or load configuration with caching."""
    global _config_cache

    with _config_lock:
        if not reload and _config_cache is not None:
            return _config_cache

        _config_cache = load_config(path, profile, use_env)
        return _config_cache


def reload_config(**kwargs: Any) -> Config:
    """Reload configuration from disk."""
    global _config_cache
    _config_cache = None
    return get_config(**kwargs)


def clear_config_cache() -> None:
    """Clear configuration cache."""
    global _config_cache
    _config_cache = None


def create_default_config(path: str | Path) -> Path:
    """Create a default configuration file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = Config()
    config.save(path)

    logger.info(f"Created configuration: {path}")
    return path


__all__ = [
    "CacheConfig",
    "Config",
    "ConfigValidationError",
    "ContextConfig",
    "EstimationConfig",
    "GPUConfig",
    "GenerationConfig",
    "LoggingConfig",
    "ModelConfig",
    "ResourceConfig",
    "ScannerConfig",
    "SecurityConfig",
    "ServerConfig",
    "WorkerConfig",
    "clear_config_cache",
    "create_default_config",
    "get_config",
    "load_config",
    "reload_config",
]
