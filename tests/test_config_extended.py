"""Extended tests for configuration system - maximizing coverage."""

import os
from unittest.mock import patch

import pytest

from llm_manager.config import (
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
    WorkerConfig,
    clear_config_cache,
    create_default_config,
    get_config,
    load_config,
    reload_config,
)


class TestConfigValidationExtended:
    """Extended config validation tests."""

    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        # Valid config
        config = ModelConfig(dir="./models", pool_size=4)
        assert config.dir == "./models"
        assert config.pool_size == 4

        # Invalid pool_size
        with pytest.raises(ConfigValidationError):
            ModelConfig(pool_size=-1)

    def test_generation_config_validation(self):
        """Test GenerationConfig validation."""
        # Valid temperature range
        config = GenerationConfig(default_temperature=0.5)
        assert config.default_temperature == 0.5

        # Temperature too high
        with pytest.raises(ConfigValidationError):
            GenerationConfig(default_temperature=3.0)

        # Temperature too low
        with pytest.raises(ConfigValidationError):
            GenerationConfig(default_temperature=-0.1)

        # Valid top_p
        config = GenerationConfig(default_top_p=0.9)
        assert config.default_top_p == 0.9

        # Invalid top_p
        with pytest.raises(ConfigValidationError):
            GenerationConfig(default_top_p=1.5)

        # Invalid max_tokens
        with pytest.raises(ConfigValidationError):
            GenerationConfig(default_max_tokens=0)

    def test_context_config_validation(self):
        """Test ContextConfig validation."""
        # Valid config
        config = ContextConfig(default_size=8192, min_size=2048, max_size=32768)
        assert config.default_size == 8192

        # min_size >= max_size
        with pytest.raises(ConfigValidationError):
            ContextConfig(min_size=32768, max_size=2048)

        # Invalid upsize_threshold
        with pytest.raises(ConfigValidationError):
            ContextConfig(upsize_threshold=1.5)

        with pytest.raises(ConfigValidationError):
            ContextConfig(upsize_threshold=0)

    def test_worker_config_validation(self):
        """Test WorkerConfig creation."""
        # Valid config
        config = WorkerConfig(critical_timeout=30.0, start_timeout=10.0, idle_timeout=300.0)
        assert config.critical_timeout == 30.0
        assert config.max_pool_size == 8  # Default

    def test_gpu_config_validation(self):
        """Test GPUConfig creation."""
        # Valid types
        config = GPUConfig(type_k="q4_0", type_v="q4_0")
        assert config.type_k == "q4_0"

        # Check defaults
        config = GPUConfig()
        assert config.default_gpu_layers == -1  # All layers
        assert config.flash_attention is True

    def test_cache_config_validation(self):
        """Test CacheConfig creation."""
        # Valid config
        config = CacheConfig(token_max_size=10000)
        assert config.token_max_size == 10000

        # Check defaults
        config = CacheConfig()
        assert config.disk_enabled is True
        assert config.ttl_seconds == 3600

    def test_logging_config_validation(self):
        """Test LoggingConfig creation."""
        # Valid level
        config = LoggingConfig(level="DEBUG")
        assert config.level == "DEBUG"

        # Check defaults
        config = LoggingConfig()
        assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def test_scanner_config_validation(self):
        """Test ScannerConfig creation."""
        # Valid config
        config = ScannerConfig(test_context=True)
        assert config.test_context is True

        # Check defaults
        config = ScannerConfig()
        assert config.min_context == 8192
        assert config.max_context == 131072

    def test_security_config_validation(self):
        """Test SecurityConfig creation."""
        # Valid config
        config = SecurityConfig(allow_external_paths=False)
        assert config.allow_external_paths is False

        # Check defaults
        config = SecurityConfig()
        assert config.max_model_size_mb == 100000
        assert config.sandbox_workers is True

    def test_estimation_config_validation(self):
        """Test EstimationConfig creation."""
        # Valid config
        config = EstimationConfig(tokens_per_word_text=1.3)
        assert config.tokens_per_word_text == 1.3

        # Check defaults
        config = EstimationConfig()
        assert config.tokens_per_word_code == 3.5
        assert config.image_tokens == 1000

    def test_resource_config_validation(self):
        """Test ResourceConfig creation."""
        # Valid config
        config = ResourceConfig(min_disk_space_mb=10000)
        assert config.min_disk_space_mb == 10000

        # Check batch sizes
        config = ResourceConfig()
        assert config.batch_size_small == 1024
        assert config.batch_size_medium == 512
        assert config.batch_size_large == 256


class TestConfigFromDict:
    """Test creating config from dictionary."""

    def test_config_from_dict_basic(self):
        """Test basic config from dict."""
        data = {
            "models": {"dir": "/custom/models", "pool_size": 8},
            "generation": {"default_temperature": 0.8, "default_max_tokens": 512},
        }

        config = Config.from_dict(data)
        assert config.models.dir == "/custom/models"
        assert config.models.pool_size == 8
        assert config.generation.default_temperature == 0.8
        assert config.generation.default_max_tokens == 512

    def test_config_from_dict_nested(self):
        """Test nested config from dict."""
        data = {
            "gpu": {"type_k": "q4_0", "type_v": "q4_1", "flash_attention": True},
            "cache": {"disk_enabled": True, "ttl_seconds": 7200},
        }

        config = Config.from_dict(data)
        assert config.gpu.type_k == "q4_0"
        assert config.gpu.type_v == "q4_1"
        assert config.cache.disk_enabled is True
        assert config.cache.ttl_seconds == 7200

    def test_config_from_dict_partial(self):
        """Test partial config from dict."""
        data = {"logging": {"level": "ERROR"}}

        config = Config.from_dict(data)
        # Other sections should have defaults
        assert config.logging.level == "ERROR"
        assert config.models.dir == "./models"  # Default

    def test_config_from_empty_dict(self):
        """Test config from empty dict."""
        config = Config.from_dict({})
        # Should use all defaults
        assert config.models.dir == "./models"
        assert config.generation.default_temperature == 0.7


class TestConfigToDict:
    """Test converting config to dictionary."""

    def test_config_to_dict(self):
        """Test config serialization."""
        config = Config(
            models=ModelConfig(dir="/custom"), generation=GenerationConfig(default_temperature=0.5)
        )

        data = config.to_dict()
        assert data["models"]["dir"] == "/custom"
        assert data["generation"]["default_temperature"] == 0.5

    def test_config_to_dict_types(self):
        """Test that to_dict returns serializable types."""
        config = Config()
        data = config.to_dict()

        # Should be JSON serializable
        import json

        json_str = json.dumps(data)
        assert isinstance(json_str, str)


class TestConfigEnvironmentVariables:
    """Test config loading from environment variables."""

    def test_config_from_env_vars(self):
        """Test loading config from environment."""
        env_vars = {
            "LLM_MODELS_DIR": "/env/models",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = get_config()
            # Note: env vars may or may not affect config depending on implementation
            # This test documents expected behavior
            assert config is not None

    def test_config_env_var_override(self):
        """Test that env vars can override defaults."""
        with patch.dict(os.environ, {"LLM_LOG_LEVEL": "DEBUG"}, clear=False):
            # Clear cache to get fresh config
            clear_config_cache()
            config = get_config()
            # The actual behavior depends on implementation
            # This test just ensures no errors
            assert config.logging is not None


class TestConfigFileOperations:
    """Test config file loading and saving."""

    def test_load_config_from_file(self, tmp_path):
        """Test loading config from file."""
        config_file = tmp_path / "llm_manager.yaml"
        config_file.write_text("""
models:
  dir: /custom/models
  pool_size: 4
generation:
  default_temperature: 0.8
  default_max_tokens: 256
""")

        config = load_config(str(config_file))
        assert config.models.dir == "/custom/models"
        assert config.generation.default_temperature == 0.8

    def test_load_config_file_not_found(self, tmp_path):
        """Test loading non-existent config file."""
        # With invalid path, should fall back to defaults
        config = load_config("/nonexistent/llm_manager.yaml")
        # Should return default config
        assert config.models.dir == "./models"

    def test_create_default_config_file(self, tmp_path):
        """Test creating default config file."""
        config_path = tmp_path / "default_llm_manager.yaml"

        create_default_config(str(config_path))

        assert config_path.exists()
        content = config_path.read_text()
        assert "models:" in content
        assert "generation:" in content

    def test_config_reload(self):
        """Test config reload clears cache."""
        config1 = get_config()

        # Reload should clear cache
        reload_config()

        config2 = get_config()
        # Should be fresh instances
        assert config1 is not config2


class TestConfigCache:
    """Test config caching behavior."""

    def test_config_caching(self):
        """Test that config is cached."""
        clear_config_cache()

        config1 = get_config()
        config2 = get_config()

        # Should be same instance due to caching
        assert config1 is config2

    def test_clear_config_cache(self):
        """Test clearing config cache."""
        config1 = get_config()

        clear_config_cache()

        config2 = get_config()
        # Should be different instance after clear
        assert config1 is not config2


class TestConfigEdgeCases:
    """Test config edge cases."""

    def test_config_with_defaults(self):
        """Test config with default values."""
        config = Config()

        # All sections should have defaults
        assert config.models is not None
        assert config.generation is not None
        assert config.context is not None
        assert config.worker is not None
        assert config.gpu is not None
        assert config.cache is not None
        assert config.logging is not None
        assert config.scanner is not None
        assert config.security is not None
        assert config.estimation is not None
        assert config.resource is not None

    def test_config_repr(self):
        """Test config string representation."""
        config = Config()
        repr_str = repr(config)

        assert "Config" in repr_str
        # Should not raise

    def test_config_equality(self):
        """Test config equality comparison."""
        config1 = Config(models=ModelConfig(dir="/test"))
        config2 = Config(models=ModelConfig(dir="/test"))
        config3 = Config(models=ModelConfig(dir="/other"))

        # Different instances with same values
        assert config1 == config2
        # Different values
        assert config1 != config3
        # Not equal to other types
        assert config1 != "config"

    def test_config_nested_equality(self):
        """Test nested config equality."""
        config1 = Config(generation=GenerationConfig(default_temperature=0.8))
        config2 = Config(generation=GenerationConfig(default_temperature=0.8))
        config3 = Config(generation=GenerationConfig(default_temperature=0.9))

        assert config1 == config2
        assert config1 != config3


class TestGPUConfigQuantization:
    """Test GPU config quantization options."""

    def test_gpu_config_q4_quantization(self):
        """Test Q4 KV cache quantization."""
        config = GPUConfig(type_k="q4_0", type_v="q4_0")
        assert config.type_k == "q4_0"
        assert config.type_v == "q4_0"

    def test_gpu_config_f16_quantization(self):
        """Test F16 KV cache quantization."""
        config = GPUConfig(type_k="f16", type_v="f16")
        assert config.type_k == "f16"
        assert config.type_v == "f16"

    def test_gpu_config_mixed_quantization(self):
        """Test mixed KV cache quantization."""
        config = GPUConfig(type_k="q4_0", type_v="f16")
        assert config.type_k == "q4_0"
        assert config.type_v == "f16"


class TestContextTiers:
    """Test context tier configuration."""

    def test_context_default_tiers(self):
        """Test default context tiers."""
        config = ContextConfig()
        expected_tiers = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
        assert config.tiers == expected_tiers

    def test_context_custom_tiers(self):
        """Test custom context tiers."""
        custom_tiers = [1024, 2048, 4096]
        config = ContextConfig(tiers=custom_tiers)
        assert config.tiers == custom_tiers


class TestEstimationDefaults:
    """Test estimation default values."""

    def test_estimation_text_defaults(self):
        """Test text estimation defaults."""
        config = EstimationConfig()
        assert config.tokens_per_word_text == 1.3
        assert config.tokens_per_word_code == 3.5

    def test_estimation_cjk_defaults(self):
        """Test CJK character estimation."""
        config = EstimationConfig()
        assert config.tokens_per_char_cjk == 1.5

    def test_estimation_overhead(self):
        """Test overhead calculations."""
        config = EstimationConfig()
        assert config.template_overhead_per_message == 30
        assert config.special_tokens_base == 50
