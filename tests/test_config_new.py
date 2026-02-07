"""
Tests for llm_manager/config.py - Unified Configuration System
"""

import json
import os
from unittest.mock import patch

import pytest

from llm_manager.config import (
    Config,
    # Exceptions
    ConfigValidationError,
    ContextConfig,
    GenerationConfig,
    GPUConfig,
    LoggingConfig,
    ModelConfig,
    clear_config_cache,
    create_default_config,
    get_config,
    # Functions
    load_config,
    reload_config,
)


class TestModelConfig:
    """Tests for ModelConfig validation."""

    def test_valid_defaults(self):
        config = ModelConfig()
        assert config.dir == "./models"
        assert config.pool_size == 0

    def test_invalid_pool_size(self):
        with pytest.raises(ConfigValidationError):
            ModelConfig(pool_size=-1)

    def test_custom_values(self):
        config = ModelConfig(dir="/custom", pool_size=4)
        assert config.dir == "/custom"
        assert config.pool_size == 4


class TestGenerationConfig:
    """Tests for GenerationConfig validation."""

    def test_temperature_out_of_range(self):
        with pytest.raises(ConfigValidationError):
            GenerationConfig(default_temperature=5.0)
        with pytest.raises(ConfigValidationError):
            GenerationConfig(default_temperature=-0.1)

    def test_top_p_out_of_range(self):
        with pytest.raises(ConfigValidationError):
            GenerationConfig(default_top_p=1.5)


class TestContextConfig:
    """Tests for ContextConfig validation."""

    def test_min_greater_than_max(self):
        with pytest.raises(ConfigValidationError):
            ContextConfig(min_size=10000, max_size=1000)


class TestGPUConfig:
    """Tests for GPUConfig validation."""

    def test_valid_types(self):
        config = GPUConfig(type_k="q4_0", type_v="q8_0")
        assert config.type_k == "q4_0"

    def test_invalid_type_k(self):
        with pytest.raises(ConfigValidationError):
            GPUConfig(type_k="invalid")


class TestLoggingConfig:
    """Tests for LoggingConfig validation."""

    def test_valid_levels(self):
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LoggingConfig(level=level)
            assert config.level == level

    def test_case_insensitive(self):
        config = LoggingConfig(level="info")
        assert config.level == "INFO"


class TestConfig:
    """Tests for main Config class."""

    def test_default_creation(self):
        config = Config()
        assert config.version == "5.0.0"
        assert config.models.dir == "./models"

    def test_to_dict(self):
        config = Config()
        data = config.to_dict()
        assert "models" in data
        assert "generation" in data

    def test_get_model_config_defaults(self):
        config = Config()
        model_cfg = config.get_model_config("unknown.gguf")
        assert model_cfg["temperature"] == 0.7

    def test_get_model_config_with_override(self):
        config = Config(model_overrides={"special.gguf": {"temperature": 0.9}})
        model_cfg = config.get_model_config("special.gguf")
        assert model_cfg["temperature"] == 0.9


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_defaults(self):
        config = load_config(use_env=False)
        assert isinstance(config, Config)

    def test_load_from_file(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
models:
  dir: "/custom/models"
  pool_size: 4
""")
        config = load_config(config_file, use_env=False)
        assert config.models.dir == "/custom/models"
        assert config.models.pool_size == 4


class TestConfigCache:
    """Tests for config caching."""

    def setup_method(self):
        clear_config_cache()

    def test_get_config_caches(self):
        config1 = get_config(use_env=False)
        config2 = get_config(use_env=False)
        assert config1 is config2

    def test_reload_config(self):
        config1 = get_config(use_env=False)
        config2 = reload_config(use_env=False)
        assert config1 is not config2


class TestSaveConfig:
    """Tests for config.save method."""

    def test_save_yaml(self, tmp_path):
        config = Config(models=ModelConfig(dir="/test"))
        config_file = tmp_path / "llm_manager.yaml"
        config.save(config_file)

        assert config_file.exists()
        loaded = load_config(config_file, use_env=False)
        assert loaded.models.dir == "/test"

    def test_save_json(self, tmp_path):
        config = Config(models=ModelConfig(dir="/test"))
        config_file = tmp_path / "config.json"
        config.save(config_file)

        assert config_file.exists()
        with open(config_file) as f:
            data = json.load(f)
        assert data["models"]["dir"] == "/test"


class TestCreateDefaultConfig:
    """Tests for create_default_config function."""

    def test_creates_yaml(self, tmp_path):
        config_file = tmp_path / "llm_manager.yaml"
        result = create_default_config(config_file)

        assert result == config_file
        assert config_file.exists()


class TestEnvironmentVariables:
    """Tests for environment variable overrides."""

    def test_env_override_models_dir(self):
        with patch.dict(os.environ, {"LLM_MODELS_DIR": "/custom/models"}):
            config = load_config(use_env=True)
        assert config.models.dir == "/custom/models"

    def test_env_override_temperature(self):
        with patch.dict(os.environ, {"LLM_DEFAULT_TEMPERATURE": "0.9"}):
            config = load_config(use_env=True)
        assert config.generation.default_temperature == 0.9
