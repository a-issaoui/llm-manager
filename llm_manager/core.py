"""
Core LLM Manager implementation.

Main class that provides high-level interface for model management
and text generation.
"""

__all__ = ["DEFAULT_CONTEXT_CONFIG", "LLMManager"]

import asyncio
import logging
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import Any, cast

from .config import Config, get_config
from .context import ContextManager, ContextStats
from .estimation import ConversationType, TokenEstimate, TokenEstimator, detect_conversation_type
from .exceptions import (
    GenerationError,
    ModelLoadError,
    ModelNotFoundError,
)
from .models import ModelRegistry
from .pool import AsyncWorkerPool, WorkerPool
from .utils import (
    Timer,
    validate_max_tokens,
    validate_messages,
    validate_model_path,
    validate_temperature,
)
from .workers import AsyncWorkerProcess, WorkerProcess

logger = logging.getLogger(__name__)

# Default registry filename
_DEFAULT_REGISTRY_FILE = "models.json"

# Default context parameters for GPU-stable operation
# These are conservative defaults that work on most GPUs
DEFAULT_CONTEXT_CONFIG: dict[str, Any] = {
    "max_context": 8192,
    "recommended_context": 4096,
    "buffer_tokens": 4096,
    "buffer_percent": 50,
    "tested": False,
    "verified_stable": False,
    "error": None,
    "test_config": {},
    "timestamp": "",
    "confidence": 1.0,
}

# Optional imports
try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None  # type: ignore
    logger.warning("llama-cpp-python not available, only subprocess mode supported")

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


class LLMManager:
    """
    Production-grade LLM Manager.

    Manages LLM model loading, generation, and context optimization
    with support for both direct and subprocess modes.

    Features:
    - Automatic context sizing based on conversation
    - Dynamic context resizing
    - Process isolation for safety
    - Token estimation and caching
    - Model registry integration
    - Async support for high concurrency

    Examples:
        Synchronous usage:

        >>> with LLMManager(models_dir="./models") as manager:
        ...     manager.load_model("model.gguf")
        ...     response = manager.generate(
        ...         messages=[{"role": "user", "content": "Hello!"}]
        ...     )

        Async usage:

        >>> import asyncio
        >>> async def main():
        ...     async with LLMManager(models_dir="./models") as manager:
        ...         await manager.load_model_async("model.gguf")
        ...         response = await manager.generate_async(
        ...             messages=[{"role": "user", "content": "Hello!"}]
        ...         )
        >>> asyncio.run(main())  # doctest: +SKIP
    """

    def __init__(
        self,
        models_dir: str | None = None,
        use_subprocess: bool | None = None,
        enable_registry: bool = True,
        pool_size: int | None = None,
        config: Config | None = None,
        **kwargs: Any,  # Absorb extra kwargs from server/CLI
    ):
        """
        Initialize LLM Manager.

        Args:
            models_dir: Directory containing models and metadata
            use_subprocess: Use subprocess isolation (recommended)
            enable_registry: Load model registry if available
            pool_size: Worker pool size (None = from config)
            config: Configuration object (loads default if not provided)
        """
        # Load configuration
        self.config = config or get_config()

        # Core attributes (initialize early for safe cleanup)
        self.model = None
        self.model_path = None
        self.model_name = None
        self.model_config = {}
        self.pool = None
        self.worker = None
        self.async_pool = None
        self.async_worker = None
        self._conversation_type = None

        # Apply config values with parameter overrides
        self.models_dir = Path(models_dir or self.config.models.dir)
        self.use_subprocess = (
            use_subprocess if use_subprocess is not None else self.config.models.use_subprocess
        )
        self._default_load_kwargs = kwargs
        _reg_file = self.config.models.registry_file or _DEFAULT_REGISTRY_FILE
        self.registry_file = self.models_dir / _reg_file

        # Core components
        self.estimator = TokenEstimator()
        self.context_manager = ContextManager(estimator=self.estimator)

        # Registry (optional)
        self.registry: ModelRegistry | None = None
        if enable_registry:
            if self.registry_file.exists():
                try:
                    self.registry = ModelRegistry(str(self.models_dir))
                    logger.info(f"Loaded registry with {len(self.registry)} models")
                except Exception as e:
                    logger.warning(f"Failed to load registry: {e}")

        # Model state
        self.model: Any | None = None
        self.model_path: Path | None = None
        self.model_name: str | None = None
        self.model_config: dict[str, Any] = {}
        self._conversation_type: ConversationType | None = None
        self._last_used_tokens: int = 0  # Track last generation's token usage

        # Worker process or pool (if using subprocess mode)
        self.worker: WorkerProcess | None = None
        self.async_worker: AsyncWorkerProcess | None = None
        self.pool: WorkerPool | None = None
        self.async_pool: AsyncWorkerPool | None = None

        # Determine pool size from config if not specified
        actual_pool_size = pool_size if pool_size is not None else self.config.models.pool_size

        if self.use_subprocess:
            if actual_pool_size > 1:
                self.pool = WorkerPool(size=actual_pool_size)
                self.async_pool = AsyncWorkerPool(size=actual_pool_size)
            else:
                self.worker = WorkerProcess()
                self.async_worker = AsyncWorkerProcess()

        logger.info(
            f"LLMManager initialized (subprocess={self.use_subprocess}, "
            f"registry={'enabled' if self.registry else 'disabled'})"
        )

    # ==========================================================================
    # Model Loading
    # ==========================================================================

    def _prepare_load_config(
        self,
        model_path: str,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        auto_context: bool = True,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[Path, dict[str, Any]]:
        """Prepare model path and configuration for loading."""
        # Use defaults from __init__ if not specified
        if n_ctx is None:
            n_ctx = self._default_load_kwargs.get("n_ctx")
        if n_gpu_layers is None:
            n_gpu_layers = self._default_load_kwargs.get("n_gpu_layers", -1)

        # Merge other kwargs with defaults
        current_kwargs = {**self._default_load_kwargs, **kwargs}
        # Filter out core params we already handled
        for key in ["n_ctx", "n_gpu_layers", "auto_context"]:
            current_kwargs.pop(key, None)
        kwargs = current_kwargs

        # Resolve model path
        resolved_path = self._resolve_model_path(model_path)

        # Validate file
        validate_model_path(resolved_path)

        # Get metadata from registry
        metadata = None
        if self.registry:
            try:
                metadata = self.registry.get(resolved_path.name)
            except Exception as e:
                logger.warning(f"Failed to get metadata: {e}")

        # Determine context size
        if n_ctx is None and auto_context:
            if messages:
                max_context = (
                    metadata.specs.context_window
                    if metadata
                    else DEFAULT_CONTEXT_CONFIG["max_context"]
                )
                n_ctx = self.context_manager.calculate_context_size(messages, max_context)
            # Use registry recommended if tested, otherwise use default
            elif metadata and metadata.specs.context_test and metadata.specs.context_test.tested:
                n_ctx = metadata.specs.context_test.recommended_context
            else:
                # Use GPU-stable default context
                n_ctx = DEFAULT_CONTEXT_CONFIG["recommended_context"]

        elif n_ctx is None:
            # Use GPU-stable default context
            n_ctx = DEFAULT_CONTEXT_CONFIG["recommended_context"]

        # Get optimal config from metadata or use defaults
        if metadata:
            optimal = metadata.get_optimal_config()
            # Apply if not explicitly overridden
            if "flash_attn" not in kwargs:
                kwargs["flash_attn"] = optimal.get("flash_attn", False)
            # Note: type_k/type_v are config/gpu settings only
            # They are not passed to Llama() constructor

        # Note: type_k/type_v are stored in config.gpu, not passed to Llama constructor
        # They can be accessed via manager.config.gpu if needed

        # Calculate batch sizes
        vram_gb = self._get_vram_gb()
        n_batch, n_ubatch = self.context_manager.calculate_batch_size(
            n_ctx if n_ctx is not None else DEFAULT_CONTEXT_CONFIG["recommended_context"], vram_gb
        )

        return resolved_path, {
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch,
            "n_ubatch": n_ubatch,
            "use_mmap": kwargs.get("use_mmap", True),
            "verbose": kwargs.get("verbose", False),
            **kwargs,
        }

    def load_model(
        self,
        model_path: str,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        auto_context: bool = True,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> bool:
        """
        Load a model for generation.

        Args:
            model_path: Path to model file
            n_ctx: Context size
            n_gpu_layers: GPU layers
            auto_context: Automatically determine optimal context
            messages: Optional messages for context calculation
            **kwargs: Additional parameters

        Returns:
            True if loaded successfully
        """
        resolved_path, config = self._prepare_load_config(
            model_path, n_ctx, n_gpu_layers, auto_context, messages, **kwargs
        )

        # Check if same model already loaded with same config
        if (
            self.is_loaded()
            and self.model_path == resolved_path
            and self.model_config.get("n_ctx") == config["n_ctx"]
            and self.model_config.get("n_gpu_layers") == config["n_gpu_layers"]
        ):
            logger.info(f"Model {resolved_path.name} already loaded with same config")
            return True

        # Unload existing model if different model or config
        if self.is_loaded():
            logger.info(f"Unloading current model before loading {resolved_path.name}")
            self.unload_model()

        logger.info(
            f"Loading {resolved_path.name} with context={config['n_ctx']}, "
            f"gpu_layers={config['n_gpu_layers']}, batch={config['n_batch']}"
        )

        # Load based on mode
        if self.use_subprocess:
            return self._load_subprocess(resolved_path, config)
        else:
            return self._load_direct(resolved_path, config)

    async def load_model_async(
        self,
        model_path: str,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        auto_context: bool = True,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> bool:
        """
        Async version of load_model.
        """
        resolved_path, config = self._prepare_load_config(
            model_path, n_ctx, n_gpu_layers, auto_context, messages, **kwargs
        )

        # Check if same model already loaded with same config
        if (
            self.is_loaded()
            and self.model_path == resolved_path
            and self.model_config.get("n_ctx") == config["n_ctx"]
            and self.model_config.get("n_gpu_layers") == config["n_gpu_layers"]
        ):
            logger.info(f"Model {resolved_path.name} already loaded with same config")
            return True

        # Unload existing model if different model or config
        if self.is_loaded():
            logger.info(f"Unloading current model before loading {resolved_path.name}")
            await asyncio.to_thread(self.unload_model)

        if self.use_subprocess:
            return await self._load_subprocess_async(resolved_path, config)
        else:
            return await asyncio.to_thread(self._load_direct, resolved_path, config)

    async def _load_subprocess_async(self, path: Path, config: dict[str, Any]) -> bool:
        """Load model in subprocess asynchronously."""
        try:
            # Ensure worker/pool is started
            if self.async_pool:
                await self.async_pool.start()
            elif self.async_worker:
                await self.async_worker.start()

            # Send load command to all workers in pool, or single worker
            if self.async_pool:
                # Load model in all workers in parallel
                workers = self.async_pool._workers
                load_commands = [
                    worker.send_command(
                        {"operation": "load", "model_path": str(path), "config": config}
                    )
                    for worker in workers
                ]
                responses = await asyncio.gather(*load_commands, return_exceptions=True)

                for i, response in enumerate(responses):
                    if isinstance(response, BaseException):  # Changed from Exception
                        raise ModelLoadError(f"Worker {i} load failed: {response}")
                    # Ensure response is a dict before accessing its keys
                    if not isinstance(response, dict) or not response.get("success"):
                        error_msg = (
                            response.get("error") if isinstance(response, dict)
                            else "Unknown error"
                        )
                        raise ModelLoadError(f"Worker {i} load failed: {error_msg}")
            elif self.async_worker:
                response = await self.async_worker.send_command(
                    {"operation": "load", "model_path": str(path), "config": config}
                )
                if not response.get("success"):
                    raise ModelLoadError(f"Subprocess load failing: {response.get('error')}")

            self.model_path = path
            self.model_name = path.stem
            self.model_config = config

            logger.info(f"✓ Model ready (async subprocess mode): {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Async subprocess load error: {e}")
            raise ModelLoadError(f"Async subprocess load failed: {e}") from e

    def _load_direct(self, path: Path, config: dict[str, Any]) -> bool:
        """Load model directly (in-process)."""
        if not LLAMA_CPP_AVAILABLE:
            raise ModelLoadError(
                "llama-cpp-python not available, cannot load directly. Use use_subprocess=True"
            )

        try:
            with Timer(f"load_model_{path.name}"):
                self.model = Llama(model_path=str(path), **config)

            self.model_path = path
            self.model_name = path.stem
            self.model_config = config

            logger.info(f"✓ Model loaded: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Model loading failed: {e}", {"path": str(path)}) from e

    def _load_subprocess(self, path: Path, config: dict[str, Any]) -> bool:
        """Load model in subprocess."""
        try:
            # Ensure worker/pool is started
            if self.pool:
                self.pool.start()
            elif not self.worker:
                self.worker = WorkerProcess()
                self.worker.start()
            elif not self.worker.is_alive():
                self.worker.start()

            # Send load command to all workers in pool, or single worker
            if self.pool:
                # Load model in all workers in the pool
                workers = self.pool._workers
                for i, worker in enumerate(workers):
                    response = worker.send_command(
                        {"operation": "load", "model_path": str(path), "config": config}
                    )
                    if not response.get("success"):
                        raise ModelLoadError(
                            f"Subprocess load failing for worker {i}: {response.get('error')}"
                        )
            elif self.worker:
                response = self.worker.send_command(
                    {"operation": "load", "model_path": str(path), "config": config}
                )
                if not response.get("success"):
                    raise ModelLoadError(f"Subprocess load failing: {response.get('error')}")

            self.model_path = path
            self.model_name = path.stem
            self.model_config = config

            logger.info(f"✓ Model ready (subprocess mode): {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Subprocess load error: {e}")
            raise ModelLoadError(f"Failed to load in subprocess: {e}") from e

    def _resolve_model_path(self, model_path: str) -> Path:
        """Resolve model path (handle relative paths).

        Security: Strict validation to prevent path traversal attacks.
        Only files within models_dir with .gguf extension are allowed.
        """
        # Normalize the models directory
        models_dir = self.models_dir.resolve()

        # Handle absolute paths - reject if outside models_dir
        path = Path(model_path)
        if path.is_absolute():
            resolved_path = path.resolve()
            try:
                resolved_path.relative_to(models_dir)
            except ValueError:
                raise ModelNotFoundError(
                    f"Access denied: Model path outside models directory: {model_path}",
                    {"path": str(path), "models_dir": str(models_dir)}
                ) from None
        else:
            # Relative path - resolve within models_dir
            resolved_path = (models_dir / path).resolve()

        # Security: Ensure resolved path is still within models_dir
        # (protects against traversal like ../../../etc/passwd)
        try:
            resolved_path.relative_to(models_dir)
        except ValueError:
            raise ModelNotFoundError(
                f"Invalid model path (traversal detected): {model_path}",
                {"attempted_path": str(resolved_path), "models_dir": str(models_dir)},
            ) from None

        # Security: Only allow .gguf files
        if resolved_path.suffix.lower() != ".gguf":
            raise ModelNotFoundError(
                f"Invalid model file type: {model_path}. Only .gguf files are allowed.",
                {"path": str(resolved_path)}
            )

        # Check if file exists
        if resolved_path.exists():
            if not resolved_path.is_file():
                raise ModelNotFoundError(
                    f"Model path is not a file: {model_path}",
                    {"path": str(resolved_path)}
                )
            return resolved_path

        # Try recursively in models_dir (safe - rglob stays within)
        for candidate in self.models_dir.rglob(path.name):
            if candidate.is_file() and candidate.suffix.lower() == ".gguf":
                # Validate found candidate is within models_dir
                try:
                    candidate.resolve().relative_to(models_dir)
                    return candidate
                except ValueError:
                    continue

        # Not found
        raise ModelNotFoundError(
            f"Model not found: {model_path}",
            {"searched": [str(resolved_path)], "models_dir": str(self.models_dir)},
        )

    def unload_model(self) -> None:
        """Unload current model and free resources."""
        if getattr(self, "use_subprocess", False):
            if self.pool or self.async_pool:
                # No explicit unload for pools (workers stay idle)
                pass
            elif self.worker and self.worker.is_alive():
                try:
                    self.worker.send_command({"operation": "unload"})
                except Exception as e:
                    logger.warning(f"Error unloading in subprocess: {e}")
            # Note: async_worker doesnt support sync unload easily, handled in async_cleanup

        elif self.model:
            del self.model
            import gc

            gc.collect()

            if TORCH_AVAILABLE and torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.model = None
        self.model_path = None
        self.model_name = None
        self.model_config = {}
        self._conversation_type = None

        logger.info("Model unloaded")

    def switch_model(
        self,
        model_path: str,
        n_ctx: int | None = None,
        n_gpu_layers: int = -1,
        auto_context: bool = True,
        **kwargs: Any,
    ) -> bool:
        """
        Hot-swap to a different model without full cleanup.

        Much faster than unload + load for switching between models
        for different tasks (coding → reasoning → fast inference).

        Args:
            model_path: Path to new model
            n_ctx: Context size (auto if None)
            n_gpu_layers: GPU layers (-1 = all)
            auto_context: Auto-determine optimal context
            **kwargs: Additional load parameters

        Returns:
            True if switch successful

        Example:
            >>> # Fast model for simple tasks
            >>> manager.switch_model("qwen2.5-7b.gguf")
            >>>
            >>> # Large model for complex reasoning
            >>> manager.switch_model("deepseek-r1-32b.gguf")
        """
        logger.info(f"Hot-swapping model: {self.model_name or 'none'} -> {model_path}")

        # Unload current model
        self.unload_model()

        # Clear CUDA cache for clean state
        if TORCH_AVAILABLE and torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load new model
        return self.load_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            auto_context=auto_context,
            **kwargs,
        )

    async def switch_model_async(
        self,
        model_path: str,
        n_ctx: int | None = None,
        n_gpu_layers: int = -1,
        auto_context: bool = True,
        **kwargs: Any,
    ) -> bool:
        """Async version of switch_model."""
        logger.info(f"Hot-swapping model (async): {self.model_name or 'none'} -> {model_path}")

        # Unload current model
        self.unload_model()

        # Clear CUDA cache
        if TORCH_AVAILABLE and torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load new model
        return await self.load_model_async(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            auto_context=auto_context,
            **kwargs,
        )

    async def get_or_load_async(self, model_name: str) -> bool:
        """
        Ensures a specific model is loaded, switching if necessary.

        Args:
            model_name: Name of the model to load (filename, registry key, or partial match)

        Returns:
            True if the model is successfully loaded (or was already loaded)

        Raises:
            ModelNotFoundError: If model cannot be found
            ModelLoadError: If model fails to load
            ValueError: If model name is ambiguous
        """
        # 1. Check if currently loaded model matches
        if self.is_loaded():
            current_path = self.model_path
            if current_path:
                if model_name in (current_path.stem, current_path.name):
                    return True

        # 2. Resolve the target model path
        target_path: str | None = None

        # Try registry first
        if self.registry:
            metadata = self.registry.get(model_name)
            if metadata and metadata.path:
                target_path = str(metadata.path)
            else:
                # Handle partial matching / ambiguity
                all_models = self.registry.list_models()
                matching = [m for m in all_models if m.startswith(model_name)]

                if len(matching) > 1:
                    # check for exact match in the list
                    if model_name in matching:
                         metadata = self.registry.get(model_name)
                         if metadata and metadata.path:
                             target_path = str(metadata.path)
                    else:
                        raise ValueError(
                            f"Ambiguous model name '{model_name}' matches: {', '.join(matching)}"
                        )
                elif len(matching) == 1:
                    metadata = self.registry.get(matching[0])
                    if metadata and metadata.path:
                        target_path = str(metadata.path)

        # Fallback to direct file path construction
        if not target_path:
            # Construct path assuming it's in models_dir
            potential_path = self.models_dir / (
                model_name if model_name.endswith(".gguf") else f"{model_name}.gguf"
            )
            if potential_path.exists():
                target_path = str(potential_path)

        if not target_path:
             raise ModelNotFoundError(f"Model not found: {model_name}")

        # 3. Switch to the new model
        return await self.switch_model_async(target_path)

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        if self.use_subprocess:
            if self.pool or self.async_pool:
                return self.model_path is not None

            sync_alive = bool(self.worker and self.worker.is_alive())
            async_alive = bool(self.async_worker and self.async_worker.is_alive())
            return self.model_path is not None and (sync_alive or async_alive)
        return self.model is not None

    # ==========================================================================
    # Generation
    # ==========================================================================

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | Iterator[Any]:
        """
        Generate completion for messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            stream: Whether to stream response
            **kwargs: Additional generation parameters

        Returns:
            Response dict or iterator if streaming

        Raises:
            GenerationError: If generation fails
            ValidationError: If parameters are invalid

        Examples:
            >>> response = manager.generate(
            ...     messages=[{"role": "user", "content": "Hello!"}],
            ...     max_tokens=100,
            ...     temperature=0.7
            ... )
            >>> print(response['choices'][0]['message']['content'])
            Hello! How can I help you today?
        """
        if not self.is_loaded():
            raise GenerationError("No model loaded")

        # Validate inputs
        messages = validate_messages(messages)
        max_tokens = validate_max_tokens(max_tokens)
        temperature = validate_temperature(temperature)

        # Detect conversation type
        self._conversation_type = detect_conversation_type(messages)

        logger.debug(
            f"Generating: {len(messages)} messages, "
            f"max_tokens={max_tokens}, temp={temperature}, "
            f"type={self._conversation_type.value}"
        )

        # Estimate input tokens for tracking
        input_estimate = self.estimator.estimate_heuristic(messages)
        self._last_used_tokens = input_estimate.total_tokens

        # Generate based on mode
        if self.use_subprocess:
            if stream:
                # Streaming in subprocess mode
                return self._generate_subprocess_streaming(
                    messages, max_tokens, temperature, **kwargs
                )
            return self._generate_subprocess(messages, max_tokens, temperature, **kwargs)
        else:
            return self._generate_direct(messages, max_tokens, temperature, stream, **kwargs)

    async def generate_async(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | Any:
        """
        Async version of generate.
        """
        # Validate inputs
        validate_messages(messages)
        validate_temperature(temperature)
        validate_max_tokens(max_tokens)

        # Detect conversation type
        self._conversation_type = detect_conversation_type(messages)

        # Generate based on mode
        if self.use_subprocess:
            if stream:
                return self._generate_subprocess_streaming_async(
                    messages, max_tokens, temperature, **kwargs
                )
            return await self._generate_subprocess_async(
                messages, max_tokens, temperature, **kwargs
            )
        else:
            return await asyncio.to_thread(
                self.generate, messages, max_tokens, temperature, stream, **kwargs
            )

    def _build_command(
        self,
        operation: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build command dict for worker (DRY helper)."""
        return {
            "operation": operation,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "model_path": str(self.model_path) if self.model_path else None,
            "config": self.model_config,
            **kwargs,
        }

    async def _generate_subprocess_async(
        self, messages: list[dict[str, Any]], max_tokens: int, temperature: float, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate using subprocess asynchronously."""
        try:
            command = self._build_command(
                "generate", messages, max_tokens, temperature, stream=False, **kwargs
            )

            if self.async_pool:
                worker: AsyncWorkerProcess
                async with self.async_pool.acquire() as worker:
                    response = await worker.send_command(command)
            elif self.async_worker:
                response = await self.async_worker.send_command(command)
            else:
                raise GenerationError("No async worker available")

            if response.get("success"):
                return cast(dict[str, Any], response["response"])
            else:
                error = response.get("error", "Unknown error")
                raise GenerationError(f"Async subprocess generation failed: {error}")

        except Exception as e:
            logger.error(f"Async subprocess generation error: {e}")
            raise GenerationError(f"Async generation error: {e}") from e

    async def _generate_subprocess_streaming_async(
        self, messages: list[dict[str, Any]], max_tokens: int, temperature: float, **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        """Generate using subprocess with streaming asynchronously."""
        try:
            command = self._build_command(
                "generate", messages, max_tokens, temperature, stream=True, **kwargs
            )

            if self.async_pool:
                worker: AsyncWorkerProcess
                async with self.async_pool.acquire() as worker:
                    async for chunk in worker.send_streaming_command_gen(command):
                        yield chunk
            elif self.async_worker:
                async for chunk in self.async_worker.send_streaming_command_gen(command):
                    yield chunk
            else:
                raise GenerationError("No async worker available")

        except Exception as e:
            logger.error(f"Async subprocess streaming error: {e}")
            raise GenerationError(f"Async streaming error: {e}") from e

    def _generate_direct(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        stream: bool,
        **kwargs: Any,
    ) -> dict[str, Any] | Iterator[Any]:
        """Generate using direct model."""
        if self.model is None:
            raise GenerationError("Model not loaded")

        try:
            with Timer("generation"):
                response = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                    **kwargs,
                )

            return cast(dict[str, Any] | Iterator[Any], response)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise GenerationError(f"Generation error: {e}") from e

    def _generate_subprocess(
        self, messages: list[dict[str, Any]], max_tokens: int, temperature: float, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate using subprocess."""
        try:
            command = self._build_command(
                "generate", messages, max_tokens, temperature, stream=False, **kwargs
            )

            if self.pool:
                worker: WorkerProcess
                with self.pool.acquire() as worker:
                    response = worker.send_command(command)
            elif self.worker:
                response = self.worker.send_command(command)
            else:
                raise GenerationError("No worker available")

            if response.get("success"):
                return cast(dict[str, Any], response["response"])
            else:
                error = response.get("error", "Unknown error")
                raise GenerationError(f"Subprocess generation failed: {error}")

        except Exception as e:
            logger.error(f"Subprocess generation error: {e}")
            raise GenerationError(f"Generation error: {e}") from e

    def _generate_subprocess_streaming(
        self, messages: list[dict[str, Any]], max_tokens: int, temperature: float, **kwargs: Any
    ) -> Iterator[Any]:
        """Generate using subprocess with streaming."""
        try:
            command = self._build_command(
                "generate", messages, max_tokens, temperature, stream=True, **kwargs
            )

            if self.pool:
                worker: WorkerProcess
                with self.pool.acquire() as worker:
                    yield from worker.send_streaming_command(command)
            elif self.worker:
                yield from self.worker.send_streaming_command(command)
            else:
                raise GenerationError("No worker available")

        except Exception as e:
            logger.error(f"Subprocess streaming error: {e}")
            raise GenerationError(f"Streaming error: {e}") from e

    # ==========================================================================
    # Batch Generation
    # ==========================================================================

    async def generate_batch(
        self,
        prompts: list[list[dict[str, Any]]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate responses for multiple prompts in parallel.

        Perfect for agents that need to evaluate multiple hypotheses,
        generate variations, or run parallel tool calls.

        Args:
            prompts: List of message lists, one per generation
            max_tokens: Max tokens per generation
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            List of response dicts in same order as prompts

        Example:
            >>> import asyncio
            >>> async def batch_example(manager):
            ...     prompts = [
            ...         [{"role": "user", "content": "Approach A: ..."}],
            ...         [{"role": "user", "content": "Approach B: ..."}],
            ...     ]
            ...     responses = await manager.generate_batch(prompts)
            ...     return responses
            >>> # asyncio.run(batch_example(manager))  # doctest: +SKIP
        """
        if not self.is_loaded():
            raise GenerationError("No model loaded")

        if not prompts:
            return []

        logger.info(f"Batch generation: {len(prompts)} prompts")

        # Create tasks for parallel execution
        tasks = [
            self.generate_async(
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                **kwargs,
            )
            for prompt in prompts
        ]

        # Execute all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error responses
        processed_results: list[dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(f"Batch item {i} failed: {result}")
                processed_results.append(
                    {
                        "error": str(result),
                        "error_type": type(result).__name__,
                        "choices": [{"message": {"content": ""}}],
                    }
                )
            # Result is Dict[str, Any] or Any, safe to append
            elif isinstance(result, dict):
                processed_results.append(result)
            else:
                logger.warning(f"Unexpected result type in batch: {type(result)}")
                processed_results.append({"error": "Unexpected return type"})

        return processed_results

    async def generate_variations(
        self,
        prompt: list[dict[str, Any]],
        n_variations: int = 3,
        temperature: float = 0.8,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> list[str]:
        """
        Generate N variations of a response to the same prompt.

        Useful for tree-of-thought, diverse sampling, or selecting
        the best response from multiple attempts.

        Args:
            prompt: Base prompt messages
            n_variations: Number of variations to generate
            temperature: Sampling temperature (higher = more diverse)
            max_tokens: Max tokens per generation
            **kwargs: Additional parameters

        Returns:
            List of response contents
        """
        prompts = [prompt for _ in range(n_variations)]

        responses = await self.generate_batch(
            prompts=prompts, max_tokens=max_tokens, temperature=temperature, **kwargs
        )

        # Extract content from each response
        contents = []
        for resp in responses:
            if "error" not in resp:
                try:
                    content = resp["choices"][0]["message"]["content"]
                    contents.append(content)
                except (KeyError, IndexError):
                    contents.append("")
            else:
                contents.append("")

        return contents

    # ==========================================================================
    # Context Management
    # ==========================================================================

    def get_context_stats(self) -> ContextStats:
        """
        Get current context statistics.

        Returns:
            ContextStats object
        """
        if not self.is_loaded():
            return ContextStats(
                loaded=False,
                model_name=None,
                allocated_context=0,
                used_tokens=0,
                max_context=0,
                utilization_percent=0.0,
                allocated_percent=0.0,
                can_grow_to=0,
                conversation_type=None,
                n_batch=None,
                n_ubatch=None,
                flash_attn=None,
            )

        # Get max context
        max_ctx = 32768
        if self.registry and self.model_name:
            max_ctx = self.registry.get_max_context(self.model_name)

        # Get current config
        current_ctx = self.model_config.get("n_ctx", 0)

        # Get tracked token usage
        used_tokens = self._last_used_tokens

        return ContextStats(
            loaded=True,
            model_name=self.model_name,
            allocated_context=current_ctx,
            used_tokens=used_tokens,
            max_context=max_ctx,
            utilization_percent=(used_tokens / current_ctx * 100) if current_ctx else 0.0,
            allocated_percent=(current_ctx / max_ctx * 100) if max_ctx else 0.0,
            can_grow_to=max_ctx - current_ctx,
            conversation_type=self._conversation_type,
            n_batch=self.model_config.get("n_batch"),
            n_ubatch=self.model_config.get("n_ubatch"),
            flash_attn=self.model_config.get("flash_attn"),
        )

    def print_context_stats(self) -> None:
        """Print context statistics to console."""
        stats = self.get_context_stats()
        print(stats)

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def estimate_tokens(
        self, messages: list[dict[str, Any]], use_heuristic: bool = True
    ) -> TokenEstimate:
        """
        Estimate token count for messages.

        Args:
            messages: List of message dicts
            use_heuristic: Use fast heuristic (True) or accurate (False)

        Returns:
            TokenEstimate
        """
        if use_heuristic:
            return self.estimator.estimate_heuristic(messages)
        else:
            # For accurate counting, would need model's tokenizer
            # Fall back to heuristic with warning
            logger.debug("Accurate token estimation requires tokenizer, using heuristic")
            return self.estimator.estimate_heuristic(messages)

    def _get_vram_gb(self) -> float:
        """Get available VRAM in GB."""
        if not TORCH_AVAILABLE or not torch or not torch.cuda.is_available():
            return 0.0

        try:
            free_bytes, _ = torch.cuda.mem_get_info(0)
            return free_bytes / 1e9
        except Exception:
            return 0.0

    # ==========================================================================
    # Context Manager Protocol
    # ==========================================================================

    def __enter__(self) -> "LLMManager":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit."""
        self.cleanup()

    async def __aenter__(self) -> "LLMManager":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.async_cleanup()

    def cleanup(self) -> None:
        """Cleanup resources (synchronous)."""
        try:
            self.unload_model()

            if self.pool:
                self.pool.shutdown()
                self.pool = None

            if self.worker:
                self.worker.stop()
                self.worker = None

            logger.info("LLMManager cleanup complete")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def scan_models(self, test_context: bool = False, **kwargs: Any) -> dict[str, Any]:
        """
        Scan models directory and generate/update registry.

        Args:
            test_context: Enable GPU context testing (slow but accurate)
            **kwargs: Additional scanner options

        Returns:
            Scan results dictionary

        Example:
            >>> manager = LLMManager(models_dir="./models")
            >>> results = manager.scan_models(test_context=True)
            >>> print(f"Found {results['models_found']} models")
        """
        from .scanner import ModelScanner

        scanner = ModelScanner(
            models_dir=self.models_dir,
            registry_file=self.registry_file.name if self.registry_file else "models.json",
        )

        results = scanner.scan_and_save(test_context=test_context, **kwargs)

        # Reload registry after scan
        if self.registry_file and self.registry_file.exists() and self.registry is not None:
            try:
                self.registry.load()
                logger.info(f"Reloaded registry with {len(self.registry)} models")
            except Exception as e:
                logger.warning(f"Failed to reload registry: {e}")

        return results

    async def scan_models_async(self, test_context: bool = False, **kwargs: Any) -> dict[str, Any]:
        """
        Async version of scan_models.

        Args:
            test_context: Enable GPU context testing
            **kwargs: Additional scanner options

        Returns:
            Scan results dictionary
        """
        return await asyncio.to_thread(self.scan_models, test_context, **kwargs)

    async def async_cleanup(self) -> None:
        """Cleanup resources (asynchronous)."""
        try:
            # Sync unload for state clearing
            self.unload_model()

            if self.async_pool:
                await self.async_pool.shutdown()
                self.async_pool = None

            if self.async_worker:
                await self.async_worker.stop()
                self.async_worker = None

            # Call sync cleanup for sync workers/pools
            self.cleanup()

            logger.info("LLMManager async cleanup complete")

        except Exception as e:
            logger.error(f"Async cleanup error: {e}")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LLMManager(model={self.model_name or 'none'}, "
            f"subprocess={self.use_subprocess}, "
            f"loaded={self.is_loaded()})"
        )
