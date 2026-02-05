"""
Model registry for managing model metadata and test configurations.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

from .exceptions import ModelNotFoundError, ValidationError

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MODELS_DIR = "./models"
REGISTRY_FILE_NAME = "models.json"


@dataclass(slots=True)
class MetadataTestConfig:
    """Test configuration from context testing."""
    kv_quant: str
    flash_attn: bool
    gpu_layers: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetadataTestConfig":
        """Create from dictionary."""
        return cls(
            kv_quant=data.get("kv_quant", "q4_0"),
            flash_attn=data.get("flash_attn", False),
            gpu_layers=data.get("gpu_layers", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "kv_quant": self.kv_quant,
            "flash_attn": self.flash_attn,
            "gpu_layers": self.gpu_layers,
        }


@dataclass(slots=True)
class ContextTest:
    """Context test results from model metadata."""
    max_context: int
    recommended_context: int
    buffer_tokens: int
    buffer_percent: int
    tested: bool
    verified_stable: bool
    error: Optional[str]
    test_config: MetadataTestConfig
    timestamp: str
    confidence: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextTest":
        """Create from dictionary."""
        return cls(
            max_context=data.get("max_context", 2048),
            recommended_context=data.get("recommended_context", 2048),
            buffer_tokens=data.get("buffer_tokens", 512),
            buffer_percent=data.get("buffer_percent", 20),
            tested=data.get("tested", False),
            verified_stable=data.get("verified_stable", False),
            error=data.get("error"),
            test_config=MetadataTestConfig.from_dict(data.get("test_config", {})),
            timestamp=data.get("timestamp", ""),
            confidence=data.get("confidence", 0.5),
        )


@dataclass(slots=True)
class ModelCapabilities:
    """Model capabilities."""
    chat: bool = True
    embed: bool = False
    vision: bool = False
    audio_in: bool = False
    reasoning: bool = False
    tools: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCapabilities":
        """Create from dictionary."""
        return cls(
            chat=data.get("chat", True),
            embed=data.get("embed", False),
            vision=data.get("vision", False),
            audio_in=data.get("audio_in", False),
            reasoning=data.get("reasoning", False),
            tools=data.get("tools", False),
        )


@dataclass(slots=True)
class ModelSpecs:
    """Model technical specifications."""
    architecture: str
    quantization: str
    size_label: str
    parameters_b: float
    layer_count: int
    context_window: int
    file_size_mb: int
    hidden_size: int
    head_count: int
    head_count_kv: int
    context_test: Optional[ContextTest] = None
    vocab_size: Optional[int] = None
    rope_freq_base: Optional[float] = None
    file_hash: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSpecs":
        """Create from dictionary."""
        context_test = None
        if "context_test" in data and data["context_test"]:
            context_test = ContextTest.from_dict(data["context_test"])

        return cls(
            architecture=data.get("architecture", "llama"),
            quantization=data.get("quantization", "Q4_K_M"),
            size_label=data.get("size_label", "unknown"),
            parameters_b=data.get("parameters_b", 0.0),
            layer_count=data.get("layer_count", 0),
            context_window=data.get("context_window", 2048),
            file_size_mb=data.get("file_size_mb", 0),
            hidden_size=data.get("hidden_size", 0),
            head_count=data.get("head_count", 0),
            head_count_kv=data.get("head_count_kv", 0),
            context_test=context_test,
            vocab_size=data.get("vocab_size"),
            rope_freq_base=data.get("rope_freq_base"),
            file_hash=data.get("file_hash"),
        )


@dataclass(slots=True)
class ModelMetadata:
    """
    Complete model metadata.

    Attributes:
        filename: Model filename
        specs: Technical specifications
        capabilities: Model capabilities
        prompt_template: Chat template for formatting
        path: Full path to model file
    """
    filename: str
    specs: ModelSpecs
    capabilities: ModelCapabilities
    prompt_template: str
    path: str

    @classmethod
    def from_dict(cls, filename: str, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from registry dictionary entry."""
        return cls(
            filename=filename,
            specs=ModelSpecs.from_dict(data.get("specs", {})),
            capabilities=ModelCapabilities.from_dict(data.get("capabilities", {})),
            prompt_template=data.get("prompt", {}).get("template", ""),
            path=data.get("path", ""),
        )

    def get_optimal_config(self) -> Dict[str, Any]:
        """
        Get optimal model configuration from test results.

        Returns:
            Dict with recommended n_ctx, flash_attn, gpu_layers, etc
        """
        config = {}

        if self.specs.context_test and self.specs.context_test.verified_stable:
            # Use tested configuration
            config["n_ctx"] = self.specs.context_test.recommended_context
            config["flash_attn"] = self.specs.context_test.test_config.flash_attn
            config["n_gpu_layers"] = self.specs.context_test.test_config.gpu_layers
            config["type_k"] = self.specs.context_test.test_config.kv_quant
            config["type_v"] = self.specs.context_test.test_config.kv_quant
        else:
            # Use conservative defaults
            config["n_ctx"] = min(self.specs.context_window, 4096)
            config["flash_attn"] = False
            config["n_gpu_layers"] = 0

        return config


class ModelRegistry:
    """
    Registry of available models with metadata.

    Loads model information from models.json and provides lookup
    and configuration management.
    """

    def __init__(self, models_dir: str = DEFAULT_MODELS_DIR):
        """
        Initialize model registry.

        Args:
            models_dir: Directory containing models and models.json
        """
        self.models_dir = Path(models_dir)
        self.registry_file = self.models_dir / REGISTRY_FILE_NAME
        self._models: Dict[str, ModelMetadata] = {}

        if self.registry_file.exists():
            self.load()
        else:
            logger.warning(f"Registry file not found: {self.registry_file}")

    def load(self) -> None:
        """
        Load model registry from file.

        Raises:
            ValidationError: If registry file is invalid
        """
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._models.clear()

            for filename, metadata in data.items():
                try:
                    model = ModelMetadata.from_dict(filename, metadata)
                    self._models[filename] = model
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {filename}: {e}")

            logger.info(f"Loaded {len(self._models)} models from registry")

        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Invalid JSON in registry file: {e}",
                {"file": str(self.registry_file)}
            )
        except OSError as e:
            raise ValidationError(
                f"Failed to read registry file: {e}",
                {"file": str(self.registry_file)}
            )

    def save(self) -> None:
        """
        Save registry to file.

        Raises:
            OSError: If file cannot be written
        """
        # Convert models back to dict format with full serialization
        data = {}
        for filename, model in self._models.items():
            # Serialize specs
            specs_dict = {
                "architecture": model.specs.architecture,
                "quantization": model.specs.quantization,
                "size_label": model.specs.size_label,
                "parameters_b": model.specs.parameters_b,
                "layer_count": model.specs.layer_count,
                "context_window": model.specs.context_window,
                "file_size_mb": model.specs.file_size_mb,
                "hidden_size": model.specs.hidden_size,
                "head_count": model.specs.head_count,
                "head_count_kv": model.specs.head_count_kv,
            }

            # Add optional fields
            if model.specs.vocab_size:
                specs_dict["vocab_size"] = model.specs.vocab_size
            if model.specs.rope_freq_base:
                specs_dict["rope_freq_base"] = model.specs.rope_freq_base
            if model.specs.file_hash:
                specs_dict["file_hash"] = model.specs.file_hash

            # Serialize context_test if exists
            if model.specs.context_test:
                ct = model.specs.context_test
                specs_dict["context_test"] = {
                    "max_context": ct.max_context,
                    "recommended_context": ct.recommended_context,
                    "buffer_tokens": ct.buffer_tokens,
                    "buffer_percent": ct.buffer_percent,
                    "tested": ct.tested,
                    "verified_stable": ct.verified_stable,
                    "error": ct.error,
                    "test_config": ct.test_config.to_dict(),
                    "timestamp": ct.timestamp,
                    "confidence": ct.confidence,
                }

            # Serialize capabilities
            caps = model.capabilities
            caps_dict = {
                "chat": caps.chat,
                "embed": caps.embed,
                "vision": caps.vision,
                "audio_in": caps.audio_in,
                "reasoning": caps.reasoning,
                "tools": caps.tools,
            }

            data[filename] = {
                "specs": specs_dict,
                "capabilities": caps_dict,
                "prompt": {"template": model.prompt_template},
                "path": model.path,
            }

        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self._models)} models to registry")

    def get(self, filename: str) -> Optional[ModelMetadata]:
        """
        Get model metadata by filename.

        Args:
            filename: Model filename (with or without .gguf)

        Returns:
            ModelMetadata or None if not found
        """
        # Try exact match
        if filename in self._models:
            return self._models[filename]

        # Try with .gguf extension
        if not filename.endswith(".gguf"):
            filename_with_ext = f"{filename}.gguf"
            if filename_with_ext in self._models:
                return self._models[filename_with_ext]

        # Try without extension
        if filename.endswith(".gguf"):
            filename_without_ext = filename[:-5]
            if filename_without_ext in self._models:
                return self._models[filename_without_ext]

        return None

    def get_or_raise(self, filename: str) -> ModelMetadata:
        """
        Get model metadata or raise exception.

        Args:
            filename: Model filename

        Returns:
            ModelMetadata

        Raises:
            ModelNotFoundError: If model not in registry
        """
        metadata = self.get(filename)
        if metadata is None:
            raise ModelNotFoundError(
                f"Model not found in registry: {filename}",
                {"filename": filename, "available": list(self._models.keys())}
            )
        return metadata

    def list_models(self) -> List[str]:
        """
        Get list of all model filenames.

        Returns:
            List of model filenames
        """
        return list(self._models.keys())

    def search(self, **criteria) -> List[ModelMetadata]:
        """
        Search for models matching criteria.

        Args:
            **criteria: Attributes to match (e.g., reasoning=True)

        Returns:
            List of matching ModelMetadata

        Examples:
            >>> registry.search(reasoning=True, architecture="llama")
            [ModelMetadata(...), ...]
        """
        results = []

        for model in self._models.values():
            match = True

            for key, value in criteria.items():
                # Check capabilities
                if hasattr(model.capabilities, key):
                    if getattr(model.capabilities, key) != value:
                        match = False
                        break

                # Check specs
                elif hasattr(model.specs, key):
                    if getattr(model.specs, key) != value:
                        match = False
                        break

            if match:
                results.append(model)

        return results

    def get_max_context(self, filename: str) -> int:
        """
        Get maximum context for a model.

        Args:
            filename: Model filename

        Returns:
            Maximum context size
        """
        metadata = self.get(filename)
        if metadata is None:
            return 32768  # Default

        if metadata.specs.context_test and metadata.specs.context_test.verified_stable:
            return metadata.specs.context_test.max_context

        return metadata.specs.context_window

    def get_recommended_context(self, filename: str) -> int:
        """
        Get recommended context for a model.

        Args:
            filename: Model filename

        Returns:
            Recommended context size
        """
        metadata = self.get(filename)
        if metadata is None:
            return 2048  # Conservative default

        if metadata.specs.context_test and metadata.specs.context_test.verified_stable:
            return metadata.specs.context_test.recommended_context

        # Use 50% of max as conservative default
        return min(metadata.specs.context_window // 2, 4096)

    def __len__(self) -> int:
        """Get number of models in registry."""
        return len(self._models)

    def __contains__(self, filename: str) -> bool:
        """Check if model is in registry."""
        return self.get(filename) is not None

    def __iter__(self):
        """Iterate over model metadata."""
        return iter(self._models.values())
