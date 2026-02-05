"""
Tests for llm_manager/models.py - Model registry.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llm_manager.models import (
    ModelRegistry,
    ModelMetadata,
    ModelSpecs,
    ModelCapabilities,
    MetadataTestConfig,
    ContextTest,
)
from llm_manager.exceptions import ModelNotFoundError, ValidationError


class TestMetadataTestConfig:
    """Tests for MetadataTestConfig."""

    def test_from_dict_defaults(self):
        """Test from_dict with defaults."""
        config = MetadataTestConfig.from_dict({})
        assert config.kv_quant == "q4_0"
        assert config.flash_attn is False
        assert config.gpu_layers == 0

    def test_from_dict_full(self):
        """Test from_dict with all fields."""
        data = {"kv_quant": "q8_0", "flash_attn": True, "gpu_layers": -1}
        config = MetadataTestConfig.from_dict(data)
        assert config.kv_quant == "q8_0"
        assert config.flash_attn is True
        assert config.gpu_layers == -1

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = MetadataTestConfig(kv_quant="q8_0", flash_attn=True, gpu_layers=-1)
        data = config.to_dict()
        assert data["kv_quant"] == "q8_0"
        assert data["flash_attn"] is True
        assert data["gpu_layers"] == -1


class TestContextTest:
    """Tests for ContextTest."""

    def test_from_dict_defaults(self):
        """Test from_dict with defaults."""
        ct = ContextTest.from_dict({})
        assert ct.max_context == 2048
        assert ct.recommended_context == 2048
        assert ct.tested is False
        assert ct.verified_stable is False

    def test_from_dict_full(self):
        """Test from_dict with all fields."""
        data = {
            "max_context": 8192,
            "recommended_context": 6553,
            "buffer_tokens": 1639,
            "buffer_percent": 20,
            "tested": True,
            "verified_stable": True,
            "error": None,
            "test_config": {"kv_quant": "q4_0", "flash_attn": True, "gpu_layers": -1},
            "timestamp": "2026-01-01",
            "confidence": 0.95
        }
        ct = ContextTest.from_dict(data)
        assert ct.max_context == 8192
        assert ct.confidence == 0.95


class TestModelCapabilities:
    """Tests for ModelCapabilities."""

    def test_from_dict_defaults(self):
        """Test from_dict with defaults."""
        caps = ModelCapabilities.from_dict({})
        assert caps.chat is True
        assert caps.embed is False
        assert caps.vision is False

    def test_from_dict_full(self):
        """Test from_dict with all fields."""
        data = {
            "chat": False,
            "embed": True,
            "vision": True,
            "audio_in": True,
            "reasoning": True,
            "tools": True
        }
        caps = ModelCapabilities.from_dict(data)
        assert caps.chat is False
        assert caps.embed is True
        assert caps.vision is True


class TestModelSpecs:
    """Tests for ModelSpecs."""

    def test_from_dict_defaults(self):
        """Test from_dict with defaults."""
        specs = ModelSpecs.from_dict({})
        assert specs.architecture == "llama"
        assert specs.context_window == 2048
        assert specs.parameters_b == 0.0

    def test_from_dict_no_context_test(self):
        """Test from_dict without context_test."""
        data = {"architecture": "mistral", "context_window": 32768}
        specs = ModelSpecs.from_dict(data)
        assert specs.context_test is None

    def test_from_dict_with_context_test(self):
        """Test from_dict with context_test."""
        data = {
            "architecture": "llama",
            "context_window": 32768,
            "context_test": {
                "max_context": 8192,
                "recommended_context": 6553,
                "buffer_tokens": 1639,
                "buffer_percent": 20,
                "tested": True,
                "verified_stable": True,
                "error": None,
                "test_config": {"kv_quant": "q4_0", "flash_attn": True, "gpu_layers": -1},
                "timestamp": "2026-01-01",
                "confidence": 1.0
            }
        }
        specs = ModelSpecs.from_dict(data)
        assert specs.context_test.max_context == 8192


class TestModelMetadata:
    """Tests for ModelMetadata."""

    def test_optimal_config_not_verified(self):
        """Test optimal config when not verified."""
        specs = ModelSpecs(
            architecture="llama",
            quantization="Q4_K_M",
            size_label="3B",
            parameters_b=3.0,
            layer_count=28,
            context_window=32768,
            file_size_mb=2000,
            hidden_size=3072,
            head_count=32,
            head_count_kv=32,
            context_test=ContextTest(
                max_context=8192,
                recommended_context=6553,
                buffer_tokens=1639,
                buffer_percent=20,
                tested=True,
                verified_stable=False,
                error=None,
                test_config=MetadataTestConfig(kv_quant="q4_0", flash_attn=True, gpu_layers=-1),
                timestamp="2026-01-01",
                confidence=1.0
            )
        )
        metadata = ModelMetadata(
            filename="test.gguf",
            specs=specs,
            capabilities=ModelCapabilities(),
            prompt_template="",
            path=""
        )

        config = metadata.get_optimal_config()
        assert config["n_ctx"] == 4096
        assert config["flash_attn"] is False

    def test_optimal_config_no_context_test(self):
        """Test optimal config when no context test."""
        specs = ModelSpecs(
            architecture="llama",
            quantization="Q4_K_M",
            size_label="3B",
            parameters_b=3.0,
            layer_count=28,
            context_window=32768,
            file_size_mb=2000,
            hidden_size=3072,
            head_count=32,
            head_count_kv=32,
            context_test=None
        )
        metadata = ModelMetadata(
            filename="test.gguf",
            specs=specs,
            capabilities=ModelCapabilities(),
            prompt_template="",
            path=""
        )

        config = metadata.get_optimal_config()
        assert config["n_ctx"] == 4096

    def test_from_dict_full(self):
        """Test from_dict with all fields."""
        data = {
            "specs": {
                "architecture": "llama",
                "quantization": "Q4_K_M",
                "size_label": "3B",
                "parameters_b": 3.0,
                "layer_count": 28,
                "context_window": 32768,
                "file_size_mb": 2000,
                "hidden_size": 3072,
                "head_count": 32,
                "head_count_kv": 32,
                "vocab_size": 32000,
                "rope_freq_base": 10000.0,
                "file_hash": "abc123"
            },
            "capabilities": {
                "chat": True,
                "embed": False,
                "vision": True,
                "audio_in": False,
                "reasoning": False,
                "tools": True
            },
            "prompt": {
                "template": "{% for message in messages %}{{ message.content }}{% endfor %}"
            },
            "path": "models/test.gguf"
        }

        metadata = ModelMetadata.from_dict("test.gguf", data)

        assert metadata.filename == "test.gguf"
        assert metadata.specs.vocab_size == 32000
        assert metadata.specs.file_hash == "abc123"
        assert metadata.capabilities.vision is True

    def test_model_metadata_defaults(self):
        """Cover models.py default values."""
        specs = ModelSpecs(
            architecture="llama",
            quantization="q4_k_m",
            size_label="7B",
            parameters_b=7.0,
            layer_count=32,
            context_window=32768,
            file_size_mb=4000.0,
            hidden_size=4096,
            head_count=32,
            head_count_kv=8
        )
        assert specs.context_window == 32768
        assert specs.context_test is None

        metadata = ModelMetadata(
            filename="test.gguf",
            specs=specs,
            capabilities=ModelCapabilities(),
            prompt_template="",
            path="/tmp/test.gguf"
        )
        assert metadata.capabilities.chat is True

class TestModelRegistry:
    """Tests for ModelRegistry."""

    @pytest.fixture
    def sample_registry_data(self):
        """Sample registry data."""
        return {
            "test-model.gguf": {
                "specs": {
                    "architecture": "llama",
                    "quantization": "Q4_K_M",
                    "size_label": "3B",
                    "parameters_b": 3.0,
                    "layer_count": 28,
                    "context_window": 32768,
                    "file_size_mb": 2000,
                    "hidden_size": 3072,
                    "head_count": 32,
                    "head_count_kv": 32,
                    "context_test": {
                        "max_context": 8192,
                        "recommended_context": 6553,
                        "buffer_tokens": 1639,
                        "buffer_percent": 20,
                        "tested": True,
                        "verified_stable": True,
                        "error": None,
                        "test_config": {
                            "kv_quant": "q4_0",
                            "flash_attn": True,
                            "gpu_layers": -1
                        },
                        "timestamp": "2026-01-01T00:00:00",
                        "confidence": 1.0
                    }
                },
                "capabilities": {
                    "chat": True,
                    "reasoning": False,
                    "tools": True
                },
                "prompt": {
                    "template": "{% for message in messages %}..."
                },
                "path": "models/test-model.gguf"
            }
        }

    @pytest.fixture
    def temp_registry(self, sample_registry_data, tmp_path):
        """Create temporary registry file."""
        registry_file = tmp_path / "models.json"
        with open(registry_file, "w") as f:
            json.dump(sample_registry_data, f)

        return ModelRegistry(str(tmp_path))

    def test_load_registry(self, temp_registry):
        """Test loading registry from file."""
        assert len(temp_registry) == 1
        assert "test-model.gguf" in temp_registry

    def test_get_model(self, temp_registry):
        """Test retrieving model metadata."""
        metadata = temp_registry.get("test-model.gguf")

        assert metadata is not None
        assert metadata.filename == "test-model.gguf"
        assert metadata.specs.architecture == "llama"
        assert metadata.capabilities.chat is True

    def test_get_model_without_extension(self, temp_registry):
        """Test getting model without .gguf extension."""
        metadata = temp_registry.get("test-model")
        assert metadata is not None

    def test_get_nonexistent_model(self, temp_registry):
        """Test getting nonexistent model."""
        metadata = temp_registry.get("nonexistent.gguf")
        assert metadata is None

    def test_get_or_raise(self, temp_registry):
        """Test get_or_raise method."""
        metadata = temp_registry.get_or_raise("test-model.gguf")
        assert metadata is not None

        with pytest.raises(ModelNotFoundError):
            temp_registry.get_or_raise("nonexistent.gguf")

    def test_list_models(self, temp_registry):
        """Test listing all models."""
        models = temp_registry.list_models()
        assert len(models) == 1
        assert "test-model.gguf" in models

    def test_search_by_capability(self, temp_registry):
        """Test searching by capability."""
        results = temp_registry.search(chat=True)
        assert len(results) == 1

        results = temp_registry.search(reasoning=True)
        assert len(results) == 0

    def test_search_by_specs(self, temp_registry):
        """Test searching by specs attribute."""
        results = temp_registry.search(architecture="llama")
        assert len(results) == 1

        results = temp_registry.search(architecture="mistral")
        assert len(results) == 0

    def test_get_max_context(self, temp_registry):
        """Test getting max context."""
        max_ctx = temp_registry.get_max_context("test-model.gguf")
        assert max_ctx == 8192

    def test_get_recommended_context(self, temp_registry):
        """Test getting recommended context."""
        rec_ctx = temp_registry.get_recommended_context("test-model.gguf")
        assert rec_ctx == 6553

    def test_optimal_config(self, temp_registry):
        """Test getting optimal configuration."""
        metadata = temp_registry.get("test-model.gguf")
        config = metadata.get_optimal_config()

        assert config["n_ctx"] == 6553
        assert config["flash_attn"] is True

    def test_registry_file_not_found(self, tmp_path, caplog):
        """Test initialization when registry file doesn't exist."""
        with caplog.at_level("WARNING"):
            registry = ModelRegistry(str(tmp_path))
        assert len(registry) == 0

    def test_registry_invalid_json(self, tmp_path):
        """Test error on invalid JSON."""
        registry_file = tmp_path / "models.json"
        registry_file.write_text("not valid json")

        with pytest.raises(ValidationError) as exc_info:
            ModelRegistry(str(tmp_path))
        assert "invalid json" in str(exc_info.value).lower()

    def test_get_max_context_no_registry(self, tmp_path):
        """Test get_max_context when model not in registry."""
        registry_file = tmp_path / "models.json"
        registry_file.write_text('{}')
        registry = ModelRegistry(str(tmp_path))

        max_ctx = registry.get_max_context("unknown.gguf")
        assert max_ctx == 32768

    def test_get_recommended_context_no_registry(self, tmp_path):
        """Test get_recommended_context when model not in registry."""
        registry_file = tmp_path / "models.json"
        registry_file.write_text('{}')
        registry = ModelRegistry(str(tmp_path))

        rec_ctx = registry.get_recommended_context("unknown.gguf")
        assert rec_ctx == 2048

    def test_iter_models(self, tmp_path):
        """Test iterating over models."""
        registry_file = tmp_path / "models.json"
        data = {
            "model1.gguf": {"specs": {}, "capabilities": {}, "prompt": {}, "path": ""},
            "model2.gguf": {"specs": {}, "capabilities": {}, "prompt": {}, "path": ""}
        }
        registry_file.write_text(json.dumps(data))
        registry = ModelRegistry(str(tmp_path))

        models = list(registry)
        assert len(models) == 2

    def test_save_registry(self, tmp_path):
        """Test saving registry to file."""
        registry_file = tmp_path / "models.json"
        registry_file.write_text('{}')
        registry = ModelRegistry(str(tmp_path))

        metadata = ModelMetadata(
            filename="test.gguf",
            specs=ModelSpecs(
                architecture="llama",
                quantization="Q4_K_M",
                size_label="3B",
                parameters_b=3.0,
                layer_count=28,
                context_window=32768,
                file_size_mb=2000,
                hidden_size=3072,
                head_count=32,
                head_count_kv=32
            ),
            capabilities=ModelCapabilities(chat=True),
            prompt_template="test",
            path="models/test.gguf"
        )
        registry._models["test.gguf"] = metadata

        registry.save()

        saved_data = json.loads(registry_file.read_text())
        assert "test.gguf" in saved_data


    def test_models_registry_load_exception_handling(self, tmp_path):
        """Cover exception handling inside registry loading loop."""
        registry_dir = tmp_path / "registry_load_test"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.json"

        data = {
            "valid.gguf": {
                "specs": {
                    "architecture": "llama", "quantization": "q4", "size_label": "7B",
                    "parameters_b": 7, "layer_count": 1, "context_window": 1024,
                    "file_size_mb": 100, "hidden_size": 1, "head_count": 1, "head_count_kv": 1
                }
            },
            "broken.gguf": None
        }
        registry_file.write_text(json.dumps(data))

        registry = ModelRegistry(str(registry_dir))
        assert len(registry._models) == 1
        assert "valid.gguf" in registry._models

    def test_models_optional_specs_and_save(self, tmp_path):
        """Cover optional specs fields and saving."""
        registry_dir = tmp_path / "registry_save_test"
        registry_dir.mkdir()

        specs = ModelSpecs(
            architecture="llama", quantization="q4", size_label="7B",
            parameters_b=7, layer_count=1, context_window=1024,
            file_size_mb=100, hidden_size=1, head_count=1, head_count_kv=1
        )
        specs.vocab_size = 32000
        specs.rope_freq_base = 10000.0
        specs.file_hash = "abc123hash"

        metadata = ModelMetadata("test.gguf", specs, ModelCapabilities(), "", "")

        registry = ModelRegistry(str(registry_dir))
        registry._models["test.gguf"] = metadata
        registry.save()

        content = json.loads((registry_dir / "models.json").read_text())
        saved_specs = content["test.gguf"]["specs"]
        assert saved_specs["vocab_size"] == 32000
        assert saved_specs["rope_freq_base"] == 10000.0
        assert saved_specs["file_hash"] == "abc123hash"


class TestModelRegistryContextTest:
    """Tests for context_test serialization (lines 287-288)."""
    
    def test_save_with_context_test(self, tmp_path):
        """Cover context_test serialization with test data."""
        registry_dir = tmp_path / "models"
        registry_dir.mkdir()
        
        specs = ModelSpecs(
            architecture="llama",
            quantization="Q4_K_M",
            size_label="7B",
            parameters_b=7.0,
            layer_count=32,
            context_window=32768,
            file_size_mb=4000,
            hidden_size=4096,
            head_count=32,
            head_count_kv=32
        )
        
        # Add context test data using models.ContextTest
        specs.context_test = ContextTest(
            max_context=32768,
            recommended_context=26214,
            buffer_tokens=6554,
            buffer_percent=20,
            tested=True,
            verified_stable=True,
            error=None,
            test_config=MetadataTestConfig(
                kv_quant="q4_0",
                flash_attn=True,
                gpu_layers=-1
            ),
            timestamp="2024-01-01T00:00:00",
            confidence=0.95
        )
        
        metadata = ModelMetadata(
            filename="test.gguf",
            specs=specs,
            capabilities=ModelCapabilities(chat=True),
            prompt_template="",
            path=str(registry_dir / "test.gguf")
        )
        
        registry = ModelRegistry(str(registry_dir))
        registry._models["test.gguf"] = metadata
        registry.save()
        
        # Verify context_test was serialized correctly
        content = json.loads((registry_dir / "models.json").read_text())
        saved_context_test = content["test.gguf"]["specs"]["context_test"]
        
        assert saved_context_test["max_context"] == 32768
        assert saved_context_test["recommended_context"] == 26214
        assert saved_context_test["tested"] is True
        assert saved_context_test["verified_stable"] is True
        assert saved_context_test["confidence"] == 0.95


class TestModelRegistryErrors:
    """Tests for error handling in ModelRegistry."""

    def test_load_os_error(self, tmp_path):
        """Cover OSError handling in registry load (lines 244-245)."""
        from unittest.mock import patch
        from llm_manager.exceptions import ValidationError
        
        registry_file = tmp_path / "models.json"
        registry_file.write_text('{}')
        
        registry = ModelRegistry(str(tmp_path))
        
        with patch('builtins.open', side_effect=OSError("Permission denied")):
            with pytest.raises(ValidationError) as exc_info:
                registry.load()
            
            assert "Failed to read registry file" in str(exc_info.value)
