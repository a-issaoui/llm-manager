"""
Tests for llm_manager/core.py - Main LLMManager class.
"""

import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from llm_manager.core import LLMManager
from llm_manager.estimation import ConversationType
from llm_manager.exceptions import (
    GenerationError,
    ModelLoadError,
    ModelNotFoundError,
)
from llm_manager.models import (
    ContextTest,
    MetadataTestConfig,
    ModelCapabilities,
    ModelMetadata,
    ModelSpecs,
)

# ============================================
# Initialization Tests
# ============================================


class TestLLMManagerInit:
    """Tests for LLMManager initialization."""

    def test_default_init(self, tmp_path):
        """Test default initialization."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

        assert manager.use_subprocess is True
        assert manager.models_dir == Path(tmp_path)
        assert manager.model is None

    def test_init_creates_components(self, tmp_path):
        """Test initialization creates core components."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

        assert manager.estimator is not None
        assert manager.context_manager is not None

    def test_init_with_pool(self, tmp_path):
        """Test initialization with worker pool."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            with patch("llm_manager.core.WorkerPool") as mock_pool:
                with patch("llm_manager.core.AsyncWorkerPool") as mock_async_pool:
                    manager = LLMManager(models_dir=str(tmp_path), pool_size=4)

        assert manager.pool is not None
        assert manager.async_pool is not None

    def test_init_without_subprocess(self, tmp_path):
        """Test initialization without subprocess."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)

        assert manager.worker is None
        assert manager.async_worker is None

    def test_init_loads_registry(self, tmp_path):
        """Test initialization loads registry if exists."""
        registry_file = tmp_path / "models.json"
        registry_file.write_text("{}")

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = Mock()
            mock_registry.return_value.__len__ = Mock(return_value=5)
            manager = LLMManager(models_dir=str(tmp_path))

        mock_registry.assert_called_once()

    def test_init_registry_load_failure(self, tmp_path, caplog):
        """Test initialization handles registry load failure."""
        registry_file = tmp_path / "models.json"
        registry_file.write_text("{}")

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.side_effect = Exception("Load error")
            manager = LLMManager(models_dir=str(tmp_path))

        assert manager.registry is None


# ============================================
# Path Resolution Tests
# ============================================


class TestResolveModelPath:
    """Tests for _resolve_model_path."""

    def test_absolute_path_in_models_dir(self, tmp_path):
        """Test absolute path within models_dir."""
        model_file = tmp_path / "model.gguf"
        model_file.write_text("test")

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))
            resolved = manager._resolve_model_path(str(model_file))

        assert resolved == model_file

    def test_relative_path(self, tmp_path):
        """Test relative path resolution."""
        model_file = tmp_path / "model.gguf"
        model_file.write_text("test")

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))
            resolved = manager._resolve_model_path("model.gguf")

        assert resolved == model_file

    def test_path_traversal_blocked(self, tmp_path):
        """Test path traversal attack is blocked."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            with pytest.raises(ModelNotFoundError) as exc_info:
                manager._resolve_model_path("../outside.gguf")

            assert "traversal" in str(exc_info.value).lower()

    def test_external_path_blocked(self, tmp_path, caplog):
        """Test external path is blocked for security (P0 fix)."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        external_dir = tmp_path / "external"
        external_dir.mkdir()
        external_file = external_dir / "external.gguf"
        external_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(models_dir))

            with pytest.raises(ModelNotFoundError) as exc_info:
                manager._resolve_model_path(str(external_file))

        assert "access denied" in str(exc_info.value).lower()
        assert "outside models directory" in str(exc_info.value).lower()

    def test_recursive_search(self, tmp_path):
        """Test recursive search in models_dir."""
        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()
        model_file = nested_dir / "model.gguf"
        model_file.write_text("test")

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))
            resolved = manager._resolve_model_path("model.gguf")

        assert resolved == model_file

    def test_not_found_error(self, tmp_path):
        """Test error when model not found."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            with pytest.raises(ModelNotFoundError):
                manager._resolve_model_path("nonexistent.gguf")

    def test_resolve_model_path_not_found_fallback(self, tmp_path):
        """Cover _resolve_model_path fallback logic."""
        manager = LLMManager(models_dir=str(tmp_path))
        with pytest.raises(Exception):
            manager._resolve_model_path("nonexistent")


# ============================================
# Load Config Tests
# ============================================


class TestPrepareLoadConfig:
    """Tests for _prepare_load_config."""

    def test_auto_context_with_messages(self, tmp_path):
        """Test auto context calculation with messages."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            path, config = manager._prepare_load_config(
                "model.gguf",
                n_ctx=None,
                auto_context=True,
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert config["n_ctx"] >= 2048

    def test_explicit_n_ctx(self, tmp_path):
        """Test explicit n_ctx is respected."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            path, config = manager._prepare_load_config("model.gguf", n_ctx=8192)

        assert config["n_ctx"] == 8192

    def test_metadata_load_error(self, tmp_path, caplog):
        """Test handling of metadata load error."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        mock_registry = Mock()
        mock_registry.get.side_effect = Exception("Registry error")

        with patch("llm_manager.core.ModelRegistry") as mock_registry_class:
            mock_registry_class.return_value = mock_registry
            manager = LLMManager(models_dir=str(tmp_path))

            with caplog.at_level(logging.WARNING):
                path, config = manager._prepare_load_config("model.gguf")

    def test_prepare_config_with_context_test(self, tmp_path):
        """Test prepare config using context_test."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        mock_metadata = Mock()
        mock_metadata.specs.context_window = 32768
        mock_metadata.specs.context_test = Mock()
        mock_metadata.specs.context_test.recommended_context = 8192

        mock_registry = Mock()
        mock_registry.get.return_value = mock_metadata

        with patch("llm_manager.core.ModelRegistry") as mock_registry_class:
            mock_registry_class.return_value = mock_registry
            manager = LLMManager(models_dir=str(tmp_path))

            path, config = manager._prepare_load_config("model.gguf", auto_context=True)

            assert config["n_ctx"] >= 4096

    def test_prepare_config_default_n_ctx(self, tmp_path):
        """Test default n_ctx when no auto_context."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry_class:
            mock_registry_class.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            path, config = manager._prepare_load_config(
                "model.gguf", auto_context=False, n_ctx=None
            )

            assert config["n_ctx"] == 4096  # DEFAULT_CONTEXT_CONFIG["recommended_context"]

    def test_prepare_config_optimal_config(self, tmp_path):
        """Test applying optimal config from metadata."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        mock_metadata = Mock()
        mock_metadata.specs.context_window = 32768
        mock_metadata.specs.context_test = None
        mock_metadata.get_optimal_config.return_value = {
            "flash_attn": True,
            "type_k": "q4_0",
            "type_v": "q4_1",
        }

        # Create mock registry with __len__ for _count_models
        mock_registry = Mock()
        mock_registry.__len__ = Mock(return_value=1)
        mock_registry.get.return_value = mock_metadata

        # Ensure models.json exists to avoid file not found error
        registry_file = tmp_path / "models.json"
        registry_file.write_text('{"models": {}}')

        with patch("llm_manager.core.ModelRegistry", return_value=mock_registry):
            manager = LLMManager(models_dir=str(tmp_path))

            # Ensure registry was set
            assert manager.registry is mock_registry

            path, config = manager._prepare_load_config("model.gguf")

            # flash_attn is applied to kwargs and then merged into config
            assert config.get("flash_attn") is True
            # type_k/type_v are removed from config (not supported by llama_cpp Llama constructor)
            assert "type_k" not in config

    def test_core_registry_recommendation(self, tmp_path):
        """Cover registry context recommendation (core.py line 233)."""
        # Setup registry
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()

        # Create a model entry with context_test
        specs = ModelSpecs(
            architecture="llama",
            quantization="q4",
            size_label="7B",
            parameters_b=7,
            layer_count=1,
            context_window=1024,
            file_size_mb=100,
            hidden_size=1,
            head_count=1,
            head_count_kv=1,
        )
        context_test = ContextTest(
            max_context=8192,
            recommended_context=6000,
            buffer_tokens=0,
            buffer_percent=0,
            tested=True,
            verified_stable=True,
            error=None,
            test_config=MetadataTestConfig("q4_0", False, 0),
            timestamp="2023-01-01",
            confidence=1.0,
        )
        specs.context_test = context_test
        metadata = ModelMetadata("test_rec.gguf", specs, ModelCapabilities(), "", "")

        registry_file = registry_dir / "models.json"

        # Manually create the JSON structure that matches what ModelRegistry loads
        registry_content = {"test_rec.gguf": asdict(metadata)}
        registry_file.write_text(json.dumps(registry_content))

        # Initialize manager with this registry
        manager = LLMManager(models_dir=str(registry_dir))

        # Verify metadata loaded correctly
        loaded_meta = manager.registry.get("test_rec.gguf")
        assert loaded_meta is not None, "Metadata not loaded"
        assert loaded_meta.specs.context_test is not None, "ContextTest not loaded"
        assert loaded_meta.specs.context_test.recommended_context == 6000

        # Mock _resolve_model_path/validate/etc to pass
        manager._resolve_model_path = Mock(return_value=Path("test_rec.gguf"))
        with patch("llm_manager.core.validate_model_path"):
            # Mock config.py or assume defaults
            # The key is _prepare_load_config uses registry
            path, config = manager._prepare_load_config("test_rec.gguf", auto_context=True)

            # n_ctx should be 6000 (recommended)
            assert config["n_ctx"] == 6000


# ============================================
# Direct Load Tests
# ============================================


class TestLoadModelDirect:
    """Tests for _load_direct."""

    @patch("llm_manager.core.LLAMA_CPP_AVAILABLE", True)
    @patch("llm_manager.core.Llama")
    def test_load_direct_success(self, mock_llama_class, tmp_path):
        """Test successful direct model load."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)

            result = manager._load_direct(model_file, {"n_ctx": 2048, "n_gpu_layers": 0})

        assert result is True
        assert manager.model is mock_llama_class.return_value

    @patch("llm_manager.core.LLAMA_CPP_AVAILABLE", False)
    def test_load_direct_not_available(self, tmp_path):
        """Test error when llama-cpp not available."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)

            with pytest.raises(ModelLoadError) as exc_info:
                manager._load_direct(Path("/tmp/model.gguf"), {})

            assert "not available" in str(exc_info.value).lower()

    @patch("llm_manager.core.LLAMA_CPP_AVAILABLE", True)
    @patch("llm_manager.core.Llama")
    def test_load_direct_failure(self, mock_llama_class, tmp_path):
        """Test error handling on load failure."""
        mock_llama_class.side_effect = Exception("Load failed")

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)

            with pytest.raises(ModelLoadError) as exc_info:
                manager._load_direct(Path("/tmp/model.gguf"), {})

            assert "failed" in str(exc_info.value).lower()


# ============================================
# Subprocess Load Tests
# ============================================


class TestLoadModelSubprocess:
    """Tests for _load_subprocess."""

    def test_load_subprocess_success(self, tmp_path):
        """Test successful subprocess load."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), use_subprocess=True)

            mock_worker = Mock()
            mock_worker.is_alive.return_value = True
            mock_worker.send_command.return_value = {"success": True}
            manager.worker = mock_worker

            result = manager._load_subprocess(model_file, {"n_ctx": 2048})

        assert result is True

    def test_load_subprocess_pool_start(self, tmp_path):
        """Test pool start in sync load."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), pool_size=2)

            # Create mock workers for the pool
            mock_worker1 = Mock()
            mock_worker1.send_command.return_value = {"success": True}
            mock_worker2 = Mock()
            mock_worker2.send_command.return_value = {"success": True}

            mock_pool = Mock()
            mock_pool._workers = [mock_worker1, mock_worker2]
            manager.pool = mock_pool

            result = manager._load_subprocess(model_file, {"n_ctx": 2048})

            mock_pool.start.assert_called_once()
            # Verify load command was sent to all workers
            mock_worker1.send_command.assert_called_once()
            mock_worker2.send_command.assert_called_once()

    def test_load_subprocess_worker_creation(self, tmp_path):
        """Test worker creation in sync load."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))
            # Ensure no pool is used
            manager.pool = None
            # Ensure no existing worker
            manager.worker = None

            mock_worker = Mock()
            mock_worker.is_alive.return_value = True
            mock_worker.send_command.return_value = {"success": True}

            with patch("llm_manager.core.WorkerProcess", return_value=mock_worker):
                result = manager._load_subprocess(model_file, {"n_ctx": 2048})

                assert manager.worker is mock_worker

    def test_load_subprocess_worker_restart(self, tmp_path):
        """Test worker restart when dead."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            mock_worker = Mock()
            mock_worker.is_alive.return_value = False
            mock_worker.send_command.return_value = {"success": True}
            manager.worker = mock_worker

            result = manager._load_subprocess(model_file, {"n_ctx": 2048})

            mock_worker.start.assert_called_once()

    def test_load_subprocess_worker_command(self, tmp_path):
        """Test worker explicit load command."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            mock_worker = Mock()
            mock_worker.send_command.return_value = {"success": False, "error": "Load failed"}
            manager.worker = mock_worker

            with pytest.raises(ModelLoadError) as exc_info:
                manager._load_subprocess(model_file, {"n_ctx": 2048})

            assert "load failing" in str(exc_info.value).lower()

    def test_load_subprocess_success_logging(self, tmp_path, caplog):
        """Test subprocess load success logging."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            mock_worker = Mock()
            mock_worker.send_command.return_value = {"success": True}
            manager.worker = mock_worker

            with caplog.at_level(logging.INFO):
                manager._load_subprocess(model_file, {"n_ctx": 2048})

            assert "subprocess mode" in caplog.text.lower()

    def test_load_subprocess_error(self, tmp_path, caplog):
        """Test subprocess load error."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            mock_worker = Mock()
            mock_worker.send_command.side_effect = Exception("Worker error")
            manager.worker = mock_worker

            with pytest.raises(ModelLoadError) as exc_info:
                with caplog.at_level(logging.ERROR):
                    manager._load_subprocess(model_file, {"n_ctx": 2048})

            assert "subprocess load error" in caplog.text.lower()

    def test_resolve_model_path_external_blocked(self, tmp_path):
        """Test external path is blocked for security (P0 fix)."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            external_path = "/nonexistent/external.gguf"

            with pytest.raises(ModelNotFoundError) as exc_info:
                manager._resolve_model_path(external_path)

            # Security: External paths are blocked before checking existence
            assert "access denied" in str(exc_info.value).lower()


# ============================================
# Async Load Tests
# ============================================


class TestAsyncLoad:
    """Tests for async load methods."""

    @pytest.mark.asyncio
    async def test_load_model_async_direct_mode(self, tmp_path):
        """Test async load model in direct mode."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)

            with patch.object(manager, "_load_direct", return_value=True):
                result = await manager.load_model_async("model.gguf")
                assert result is True

    @pytest.mark.asyncio
    async def test_load_subprocess_async_pool_start(self, tmp_path):
        """Test async pool start."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), pool_size=2)

            # Create mock workers for the pool
            mock_worker1 = Mock()
            mock_worker1.send_command = AsyncMock(return_value={"success": True})
            mock_worker2 = Mock()
            mock_worker2.send_command = AsyncMock(return_value={"success": True})

            mock_pool = Mock()
            mock_pool.start = AsyncMock()
            mock_pool._workers = [mock_worker1, mock_worker2]
            manager.async_pool = mock_pool

            result = await manager._load_subprocess_async(model_file, {"n_ctx": 2048})

            mock_pool.start.assert_called_once()
            # Verify load command was sent to all workers
            mock_worker1.send_command.assert_called_once()
            mock_worker2.send_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_subprocess_async_worker_command(self, tmp_path):
        """Test async worker explicit load command."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            mock_worker = Mock()
            mock_worker.send_command = AsyncMock(
                return_value={"success": False, "error": "Load failed"}
            )
            mock_worker.start = AsyncMock()
            manager.async_worker = mock_worker

            with pytest.raises(ModelLoadError) as exc_info:
                await manager._load_subprocess_async(model_file, {"n_ctx": 2048})

            assert "load" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_load_subprocess_async_success_logging(self, tmp_path, caplog):
        """Test async load success logging."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            mock_worker = Mock()
            mock_worker.send_command = AsyncMock(return_value={"success": True})
            mock_worker.start = AsyncMock()
            manager.async_worker = mock_worker

            with caplog.at_level(logging.INFO):
                await manager._load_subprocess_async(model_file, {"n_ctx": 2048})

            assert "async subprocess mode" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_load_subprocess_async_error(self, tmp_path, caplog):
        """Test async load error handling."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            mock_worker = Mock()
            mock_worker.send_command = AsyncMock(side_effect=Exception("Connection failed"))
            mock_worker.start = AsyncMock()
            manager.async_worker = mock_worker

            with pytest.raises(ModelLoadError) as exc_info:
                with caplog.at_level(logging.ERROR):
                    await manager._load_subprocess_async(model_file, {"n_ctx": 2048})

            assert "async subprocess load error" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_core_load_model_async_not_found(self, tmp_path):
        """Cover ModelNotFoundError in load_model_async."""
        # Ensure registry is initialized (mocked)
        with patch("llm_manager.core.ModelRegistry") as mock_registry_cls:
            mock_registry = Mock()
            mock_registry_cls.return_value = mock_registry
            manager = LLMManager(models_dir=str(tmp_path))

            # Now manager.registry is our mock_registry
            mock_registry.get.return_value = None

            with pytest.raises(ModelNotFoundError):
                await manager.load_model_async("nonexistent")


# ============================================
# Generation Tests
# ============================================


class TestGenerate:
    """Tests for generate()."""

    def test_generate_not_loaded(self, tmp_path):
        """Test error when no model loaded."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            with pytest.raises(GenerationError) as exc_info:
                manager.generate([{"role": "user", "content": "Hello"}])

            assert "no model loaded" in str(exc_info.value).lower()

    def test_generate_subprocess(self, tmp_path):
        """Test generate using subprocess."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), use_subprocess=True)
            manager.model_path = model_file
            manager.model_name = "test"
            manager.model_config = {"n_ctx": 2048}

            mock_worker = Mock()
            mock_worker.is_alive.return_value = True
            mock_worker.send_command.return_value = {
                "success": True,
                "response": {"choices": [{"message": {"content": "Hi!"}}]},
            }
            manager.worker = mock_worker

            result = manager.generate([{"role": "user", "content": "Hello"}])

        assert result["choices"][0]["message"]["content"] == "Hi!"

    def test_core_generate_failure(self, tmp_path):
        """Cover generate failure paths in core.py."""
        manager = LLMManager(models_dir=str(tmp_path))
        with patch.object(manager, "_generate_subprocess", side_effect=Exception("Gen error")):
            with pytest.raises(GenerationError):
                manager.generate([{"role": "user", "content": "hi"}])

    def test_core_generate_direct_success_and_fail(self, tmp_path):
        """Cover direct generation."""
        manager = LLMManager(models_dir=str(tmp_path))
        manager.model = Mock()

        # Failure
        manager.model.create_chat_completion.side_effect = Exception("Direct Boom")
        msgs = [{"role": "user", "content": "hi"}]
        with pytest.raises(GenerationError):
            manager._generate_direct(msgs, 100, 0.7, False)

        # Success
        manager.model.create_chat_completion.side_effect = None
        manager.model.create_chat_completion.return_value = {"choices": []}
        res = manager._generate_direct(msgs, 100, 0.7, False)
        assert res == {"choices": []}

    def test_core_generate_subprocess_streaming_sync(self, tmp_path):
        """Cover sync subprocess streaming."""
        manager = LLMManager(models_dir=str(tmp_path))
        manager.worker = Mock()
        manager.pool = None

        def stream_gen(*args, **kwargs):
            yield "chunk1"

        manager.worker.send_streaming_command = stream_gen
        msgs = [{"role": "user", "content": "hi"}]

        gen = manager._generate_subprocess_streaming(msgs, 100, 0.7)
        chunks = list(gen)
        assert chunks == ["chunk1"]

        # With Pool - FIX: Use MagicMock for context manager support
        manager.pool = MagicMock()
        manager.pool.acquire.return_value.__enter__.return_value = manager.worker
        manager.worker.send_streaming_command = stream_gen

        gen = manager._generate_subprocess_streaming(msgs, 100, 0.7)
        chunks = list(gen)
        assert chunks == ["chunk1"]

    def test_core_generate_subprocess_error_handling(self):
        """Cover _generate_subprocess error paths (core.py lines 738-743)."""
        manager = LLMManager(use_subprocess=True)
        manager.worker = MagicMock()
        manager.pool = None  # Disable pool for direct worker access
        manager.is_loaded = Mock(return_value=True)  # Bypass is_loaded check which checks worker

        # 1. Verification of success=False
        manager.worker.send_command.return_value = {"success": False, "error": "Planned fail"}
        with pytest.raises(GenerationError, match="Subprocess generation failed: Planned fail"):
            manager._generate_subprocess([], 10, 0.7)

        # 2. Verification of Exception
        manager.worker.send_command.side_effect = Exception("Crash")
        with pytest.raises(GenerationError, match="Generation error: Crash"):
            manager._generate_subprocess([], 10, 0.7)

    def test_core_generate_stream_call(self):
        """Cover generate() calling _generate_subprocess_streaming (core.py line 570)."""
        manager = LLMManager(use_subprocess=True)
        manager.is_loaded = Mock(return_value=True)
        manager.worker = MagicMock()  # Ensure is_loaded check passes if it checks worker
        # Mock internal methods to avoid real execution
        manager._generate_subprocess_streaming = Mock(return_value=iter(["chunk"]))

        # This call should route to _generate_subprocess_streaming
        result = manager.generate([{"role": "user", "content": "hi"}], stream=True)
        assert list(result) == ["chunk"]
        manager._generate_subprocess_streaming.assert_called_once()

    def test_core_generate_subprocess_streaming_exception(self):
        """Cover exception handling in _generate_subprocess_streaming (core.py lines 764-766)."""
        manager = LLMManager(use_subprocess=True)
        manager.pool = MagicMock()

        # Mock pool.acquire to yield a worker that raises exception on send_streaming_command
        worker = MagicMock()
        worker.send_streaming_command.side_effect = Exception("Stream fail")

        # Setup context manager for pool.acquire
        manager.pool.acquire.return_value.__enter__.return_value = worker

        with pytest.raises(GenerationError, match="Streaming error: Stream fail"):
            # Consume iterator to trigger execution
            list(manager._generate_subprocess_streaming([], 10, 0.7))


class TestBuildCommand:
    """Tests for _build_command helper."""

    def test_build_command_basic(self, tmp_path):
        """Test basic command building."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))
            manager.model_path = Path("/tmp/model.gguf")
            manager.model_config = {"n_ctx": 2048}

            command = manager._build_command(
                "generate",
                [{"role": "user", "content": "Hello"}],
                max_tokens=100,
                temperature=0.7,
                stream=False,
            )

        assert command["operation"] == "generate"
        assert command["max_tokens"] == 100
        assert command["temperature"] == 0.7
        assert command["stream"] is False


# ============================================
# Unload Tests
# ============================================


class TestUnloadModel:
    """Tests for unload_model()."""

    def test_unload_direct_model(self, tmp_path):
        """Test unloading direct model."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
            manager.model = Mock()
            manager.model_path = Path("/tmp/model.gguf")
            manager.model_name = "test"

            manager.unload_model()

        assert manager.model is None
        assert manager.model_path is None

    def test_unload_torch_cleanup(self, tmp_path):
        """Test torch CUDA cleanup on unload."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            with patch("llm_manager.core.TORCH_AVAILABLE", True):
                mock_torch = Mock()
                mock_torch.cuda.is_available.return_value = True
                with patch("llm_manager.core.torch", mock_torch):
                    manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
                    manager.model = Mock()

                    manager.unload_model()

            mock_torch.cuda.empty_cache.assert_called_once()


# ============================================
# Is Loaded Tests
# ============================================


class TestIsLoaded:
    """Tests for is_loaded()."""

    def test_not_loaded_initially(self, tmp_path):
        """Test not loaded initially."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))
            assert manager.is_loaded() is False

    def test_loaded_direct(self, tmp_path):
        """Test loaded with direct model."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
            manager.model = Mock()
            assert manager.is_loaded() is True

    def test_loaded_subprocess(self, tmp_path):
        """Test loaded with subprocess."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), use_subprocess=True)
            manager.model_path = Path("/tmp/model.gguf")

            mock_worker = Mock()
            mock_worker.is_alive.return_value = True
            manager.worker = mock_worker

            assert manager.is_loaded() is True

    def test_is_loaded_complex_logic(self):
        """Cover complex is_loaded logic (core.py line 500 etc)."""
        manager = LLMManager(use_subprocess=True)
        manager.model_path = "path"
        manager.worker = None
        manager.async_worker = None

        # path set but no workers -> False
        assert manager.is_loaded() is False

        # sync worker alive -> True
        manager.worker = MagicMock()
        manager.worker.is_alive.return_value = True
        assert manager.is_loaded() is True

        # async worker alive -> True
        manager.worker = None
        manager.async_worker = MagicMock()
        manager.async_worker.is_alive.return_value = True
        assert manager.is_loaded() is True


# ============================================
# Context Stats Tests
# ============================================


class TestContextStats:
    """Tests for get_context_stats()."""

    def test_stats_not_loaded(self, tmp_path):
        """Test stats when not loaded."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))
            stats = manager.get_context_stats()

        assert stats.loaded is False

    def test_stats_loaded(self, tmp_path):
        """Test stats when loaded."""
        mock_registry_instance = Mock()
        mock_registry_instance.get_max_context.return_value = 32768

        with patch("llm_manager.core.ModelRegistry") as mock_registry_class:
            mock_registry_class.return_value = mock_registry_instance
            manager = LLMManager(models_dir=str(tmp_path), use_subprocess=True)

            with patch.object(manager, "is_loaded", return_value=True):
                manager.model_name = "test"
                manager.model_path = Path("/tmp/model.gguf")
                manager.model_config = {
                    "n_ctx": 4096,
                    "n_batch": 512,
                    "n_ubatch": 256,
                    "flash_attn": True,
                }
                manager._last_used_tokens = 1000
                manager._conversation_type = ConversationType.CHAT

                stats = manager.get_context_stats()

        assert stats.loaded is True
        assert stats.model_name == "test"
        assert stats.allocated_context == 4096
        assert stats.model_name == "test"
        assert stats.allocated_context == 4096
        assert stats.used_tokens == 1000

    def test_print_context_stats(self, tmp_path):
        """Cover print_context_stats."""
        manager = LLMManager(models_dir=str(tmp_path))
        with patch("builtins.print") as mock_print:
            manager.print_context_stats()

    def test_core_get_context_stats_edge_cases(self, tmp_path):
        """Cover get_context_stats edge cases."""
        manager = LLMManager(models_dir=str(tmp_path))
        stats = manager.get_context_stats()
        assert stats.utilization_percent == 0.0

        manager.registry = Mock()
        manager.registry.get_max_context.return_value = 16384

        with patch.object(manager, "is_loaded", return_value=True):
            manager.model_name = "test"
            manager.model_config = {"n_ctx": 0}
            manager._last_used_tokens = 0

            stats = manager.get_context_stats()
            assert stats.utilization_percent == 0.0


# ============================================
# Token Estimation Tests
# ============================================


class TestEstimateTokens:
    """Tests for estimate_tokens()."""

    def test_estimate_heuristic(self, tmp_path):
        """Test heuristic estimation."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            estimate = manager.estimate_tokens(
                [{"role": "user", "content": "Hello world"}], use_heuristic=True
            )

        assert estimate.total_tokens > 0

    def test_estimate_accurate_fallback(self, tmp_path):
        """Test accurate estimation falls back to heuristic."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            estimate = manager.estimate_tokens(
                [{"role": "user", "content": "Hello world"}], use_heuristic=False
            )

        assert estimate.total_tokens > 0


# ============================================
# VRAM Tests
# ============================================


class TestVRAM:
    """Tests for VRAM detection."""

    def test_vram_torch_available(self, tmp_path):
        """Test VRAM detection with torch."""
        with patch("llm_manager.core.TORCH_AVAILABLE", True):
            mock_torch = Mock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.mem_get_info.return_value = (8e9, 16e9)

            with patch("llm_manager.core.torch", mock_torch):
                with patch("llm_manager.core.ModelRegistry") as mock_registry:
                    mock_registry.return_value = None
                    manager = LLMManager(models_dir=str(tmp_path))
                    vram = manager._get_vram_gb()

        assert vram == 8.0

    def test_vram_torch_not_available(self, tmp_path):
        """Test VRAM detection without torch."""
        with patch("llm_manager.core.TORCH_AVAILABLE", False):
            with patch("llm_manager.core.ModelRegistry") as mock_registry:
                mock_registry.return_value = None
                manager = LLMManager(models_dir=str(tmp_path))
                vram = manager._get_vram_gb()

        assert vram == 0.0

    def test_vram_cuda_not_available(self, tmp_path):
        """Test VRAM when CUDA not available."""
        with patch("llm_manager.core.TORCH_AVAILABLE", True):
            mock_torch = Mock()
            mock_torch.cuda.is_available.return_value = False

            with patch("llm_manager.core.torch", mock_torch):
                with patch("llm_manager.core.ModelRegistry") as mock_registry:
                    mock_registry.return_value = None
                    manager = LLMManager(models_dir=str(tmp_path))
                    vram = manager._get_vram_gb()

        assert vram == 0.0

    def test_vram_exception(self, tmp_path):
        """Cover VRAM detection exception."""
        with patch("llm_manager.core.TORCH_AVAILABLE", True):
            mock_torch = Mock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.mem_get_info.side_effect = Exception("CUDA Error")

            with patch("llm_manager.core.torch", mock_torch):
                with patch("llm_manager.core.ModelRegistry") as mock_registry:
                    mock_registry.return_value = None
                    manager = LLMManager(models_dir=str(tmp_path))
                    vram = manager._get_vram_gb()

        assert vram == 0.0


# ============================================
# Cleanup Tests
# ============================================


class TestCleanup:
    """Tests for cleanup methods."""

    def test_cleanup_sync(self, tmp_path):
        """Test synchronous cleanup."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), pool_size=2)

            mock_pool = Mock()
            manager.pool = mock_pool

            manager.cleanup()

        mock_pool.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_async(self, tmp_path):
        """Test asynchronous cleanup."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), pool_size=2)

            mock_async_pool = Mock()
            mock_async_pool.shutdown = AsyncMock()
            manager.async_pool = mock_async_pool

            await manager.async_cleanup()

        mock_async_pool.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_core_async_cleanup_error(self, tmp_path):
        """Cover async cleanup error handling."""
        manager = LLMManager(models_dir=str(tmp_path))
        manager.worker_pool = Mock()
        manager.worker_pool.shutdown = AsyncMock(side_effect=Exception("Pool error"))
        await manager.async_cleanup()  # Should not raise

    def test_cleanup_error_logging(self, tmp_path, caplog):
        """Test cleanup error logging."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), pool_size=2)

            mock_pool = Mock()
            mock_pool.shutdown.side_effect = Exception("Shutdown error")
            manager.pool = mock_pool

            with caplog.at_level(logging.ERROR):
                manager.cleanup()

            assert "cleanup error" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_async_cleanup_error_logging(self, tmp_path, caplog):
        """Test async cleanup error logging."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), pool_size=2)

            mock_async_pool = Mock()
            mock_async_pool.shutdown = AsyncMock(side_effect=Exception("Async shutdown error"))
            manager.async_pool = mock_async_pool

            with caplog.at_level(logging.ERROR):
                await manager.async_cleanup()

            assert "async cleanup error" in caplog.text.lower()

    def test_del_exception_handling(self, tmp_path):
        """Test __del__ exception handling."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            with patch.object(manager, "cleanup", side_effect=Exception("Cleanup failed")):
                manager.__del__()

    def test_core_cleanup_logging_debug(self, tmp_path):
        """Cover extensive cleanup logging."""
        manager = LLMManager(models_dir=str(tmp_path))
        manager.pool = Mock()
        manager.async_pool = Mock()
        manager.worker = Mock()
        manager.async_worker = Mock()
        manager.cleanup()


# ============================================
# Repr Tests
# ============================================


class TestRepr:
    """Tests for __repr__."""

    def test_repr_not_loaded(self, tmp_path):
        """Test repr when not loaded."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))

            r = repr(manager)

        assert "none" in r

    def test_repr_loaded(self, tmp_path):
        """Test repr when loaded."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path))
            manager.model_name = "test-model"

            r = repr(manager)

        assert "test-model" in r


# ============================================
# Context Manager Tests
# ============================================


class TestContextManagerProtocol:
    """Tests for context manager protocol."""

    def test_sync_context_manager(self, tmp_path):
        """Test synchronous context manager."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None

            with patch.object(LLMManager, "cleanup") as mock_cleanup:
                with LLMManager(models_dir=str(tmp_path)) as manager:
                    pass

                mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, tmp_path):
        """Test asynchronous context manager."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None

            with patch.object(
                LLMManager, "async_cleanup", new_callable=lambda: AsyncMock()
            ) as mock_cleanup:
                async with LLMManager(models_dir=str(tmp_path)) as manager:
                    pass

                mock_cleanup.assert_called_once()


# ============================================
# Async Generation Tests
# ============================================


class TestAsyncGenerate:
    """Tests for async generation."""

    @pytest.mark.asyncio
    async def test_generate_async(self, tmp_path):
        """Test async generation."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), use_subprocess=True)
            manager.model_path = model_file
            manager.model_name = "test"
            manager.model_config = {"n_ctx": 2048}

            mock_worker = Mock()
            mock_worker.is_alive = Mock(return_value=True)
            mock_worker.send_command = AsyncMock(
                return_value={
                    "success": True,
                    "response": {"choices": [{"message": {"content": "Hi!"}}]},
                }
            )
            manager.async_worker = mock_worker

            result = await manager.generate_async([{"role": "user", "content": "Hello"}])

        assert result["choices"][0]["message"]["content"] == "Hi!"

    @pytest.mark.asyncio
    async def test_generate_subprocess_async_pool(self, tmp_path):
        """Test async generation with pool."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_manager.core.ModelRegistry") as mock_registry:
            mock_registry.return_value = None
            manager = LLMManager(models_dir=str(tmp_path), pool_size=2)
            manager.model_path = model_file
            manager.model_name = "test"
            manager.model_config = {"n_ctx": 2048}

            mock_worker = Mock()
            mock_worker.send_command = AsyncMock(
                return_value={
                    "success": True,
                    "response": {"choices": [{"message": {"content": "Hi!"}}]},
                }
            )

            mock_pool = Mock()
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_worker)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
            manager.async_pool = mock_pool

            result = await manager._generate_subprocess_async(
                [{"role": "user", "content": "Hello"}], max_tokens=100, temperature=0.7
            )

        assert result["choices"][0]["message"]["content"] == "Hi!"

    @pytest.mark.asyncio
    async def test_core_generate_async_coverage(self, tmp_path):
        """Cover async generation error paths in core.py."""
        manager = LLMManager(models_dir=str(tmp_path))
        manager.async_worker = AsyncMock()

        # Test 1: Response success=False
        manager.async_worker.send_command.return_value = {"success": False, "error": "Fail"}
        msgs = [{"role": "user", "content": "hi"}]

        with pytest.raises(GenerationError) as e:
            await manager._generate_subprocess_async(msgs, 100, 0.7)
        assert "Async subprocess generation failed" in str(e.value)

        # Test 2: Exception during send_command
        manager.async_worker.send_command.side_effect = Exception("Boom")
        with pytest.raises(GenerationError):
            await manager._generate_subprocess_async(msgs, 100, 0.7)

    @pytest.mark.asyncio
    async def test_core_generate_streaming_async_error(self, tmp_path):
        """Cover async streaming error paths."""
        manager = LLMManager(models_dir=str(tmp_path))
        manager.async_worker = AsyncMock()
        manager.async_pool = None

        # Mock generator raising exception
        async def fail_gen(*args, **kwargs):
            raise Exception("Stream Boom")
            yield "chunk"  # unreachable

        manager.async_worker.send_streaming_command_gen = fail_gen
        msgs = [{"role": "user", "content": "hi"}]

        gen = manager._generate_subprocess_streaming_async(msgs, 100, 0.7)
        with pytest.raises(GenerationError):
            async for _ in gen:
                pass

    @pytest.mark.asyncio
    async def test_core_generate_streaming_async_pool(self, tmp_path):
        """Cover async streaming with pool."""
        manager = LLMManager(models_dir=str(tmp_path))
        manager.async_pool = Mock()
        worker = AsyncMock()

        async def stream_gen(*args, **kwargs):
            yield "chunk1"

        worker.send_streaming_command_gen = stream_gen

        # Mock acquire context manager correctly
        acquire_ctx = AsyncMock()
        acquire_ctx.__aenter__.return_value = worker
        manager.async_pool.acquire.return_value = acquire_ctx

        msgs = [{"role": "user", "content": "hi"}]
        chunks = []
        async for c in manager._generate_subprocess_streaming_async(msgs, 100, 0.7):
            chunks.append(c)
        assert chunks == ["chunk1"]


# ============================================
# Import Warning Tests
# ============================================


class TestImportWarnings:
    """Test import warning lines."""

    def test_llama_cpp_import_warning(self, caplog):
        """Test warning when llama_cpp not available."""
        import importlib
        import sys

        import llm_manager.core

        # simulated import error
        with patch.dict(sys.modules, {"llama_cpp": None}):
            importlib.reload(llm_manager.core)
            assert llm_manager.core.LLAMA_CPP_AVAILABLE is False
            # Check for the warning emitted at module level
            # Note: logging at module level might be captured depending on configuration
            # but we at least verify the flag is False

        # Restore
        importlib.reload(llm_manager.core)

    def test_torch_import_warning(self, caplog):
        """Test torch import handling."""
        import importlib
        import sys

        import llm_manager.core

        # simulated import error
        with patch.dict(sys.modules, {"torch": None}):
            importlib.reload(llm_manager.core)
            assert llm_manager.core.TORCH_AVAILABLE is False

        # Restore
        importlib.reload(llm_manager.core)


class TestSyncLoadModel:
    """Tests for sync load_model dispatch."""

    def test_load_model_direct(self, tmp_path):
        """Test load_model calls _load_direct."""
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
        with patch.object(manager, "_load_direct", return_value=True) as mock_load:
            with patch.object(
                manager,
                "_prepare_load_config",
                return_value=(Path("model"), {"n_ctx": 2048, "n_batch": 512, "n_gpu_layers": 0}),
            ):
                manager.load_model("model")
                mock_load.assert_called_once()

    def test_load_model_subprocess(self, tmp_path):
        """Test load_model calls _load_subprocess."""
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=True)
        with patch.object(manager, "_load_subprocess", return_value=True) as mock_load:
            with patch.object(
                manager,
                "_prepare_load_config",
                return_value=(Path("model"), {"n_ctx": 2048, "n_batch": 512, "n_gpu_layers": 0}),
            ):
                manager.load_model("model")
                mock_load.assert_called_once()

    def test_prepare_load_config_registry_error(self, tmp_path, caplog):
        """Cover registry.get exception in _prepare_load_config."""
        with patch("llm_manager.core.ModelRegistry") as mock_registry_cls:
            mock_registry = Mock()
            mock_registry_cls.return_value = mock_registry
            manager = LLMManager(models_dir=str(tmp_path))

            # Simulate exception during registry get
            mock_registry.get.side_effect = Exception("Registry Error")
            manager.registry = mock_registry

            # Setup necessary mocks to bypass other checks
            with (
                patch.object(manager, "_resolve_model_path", return_value=Path("model")),
                patch("llm_manager.core.validate_model_path"),
                patch("llm_manager.core.logger") as mock_logger,
            ):
                manager._prepare_load_config("model")

                # Check logger
                mock_logger.warning.assert_called()
                assert "Failed to get metadata" in mock_logger.warning.call_args[0][0]


class TestScanModelsIntegration:
    """Integration tests for scan_models (lines 936-953)."""

    def test_scan_models_integration(self, tmp_path):
        """Test scan_models method with actual scanner."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Create a proper fake GGUF file with metadata
        import struct

        from llm_manager.scanner import GGUFConstants

        gguf_file = models_dir / "test-model.gguf"

        # Build minimal valid GGUF file
        data = b"GGUF"
        data += struct.pack("<I", 3)  # Version 3
        data += struct.pack("<Q", 0)  # tensor_count
        data += struct.pack("<Q", 2)  # metadata_count

        # Add architecture
        key = b"general.architecture"
        data += struct.pack("<Q", len(key)) + key
        data += struct.pack("<I", GGUFConstants.STRING)
        value = b"llama"
        data += struct.pack("<Q", len(value)) + value

        # Add size label
        key = b"general.size_label"
        data += struct.pack("<Q", len(key)) + key
        data += struct.pack("<I", GGUFConstants.STRING)
        value = b"7B"
        data += struct.pack("<Q", len(value)) + value

        gguf_file.write_bytes(data)

        manager = LLMManager(models_dir=str(models_dir), enable_registry=False)

        results = manager.scan_models(test_context=False)

        # Should find the model and return results
        assert "models_found" in results
        assert results["models_found"] == 1

    def test_scan_models_with_registry_reload(self, tmp_path):
        """Test scan_models reloads registry after scan."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Create a proper fake GGUF file with metadata
        import struct

        from llm_manager.scanner import GGUFConstants

        gguf_file = models_dir / "test-model.gguf"

        data = b"GGUF"
        data += struct.pack("<I", 3)
        data += struct.pack("<Q", 0)
        data += struct.pack("<Q", 2)

        key = b"general.architecture"
        data += struct.pack("<Q", len(key)) + key
        data += struct.pack("<I", GGUFConstants.STRING)
        value = b"llama"
        data += struct.pack("<Q", len(value)) + value

        key = b"general.size_label"
        data += struct.pack("<Q", len(key)) + key
        data += struct.pack("<I", GGUFConstants.STRING)
        value = b"7B"
        data += struct.pack("<Q", len(value)) + value

        gguf_file.write_bytes(data)

        # Pre-create registry file
        registry_file = models_dir / "models.json"
        registry_file.write_text("{}")

        manager = LLMManager(models_dir=str(models_dir))

        with patch.object(manager.registry, "load") as mock_reload:
            results = manager.scan_models(test_context=False)

            # Registry should be reloaded after scan
            mock_reload.assert_called_once()
            assert results["models_found"] == 1

    def test_scan_models_async_integration(self, tmp_path):
        """Test scan_models_async wraps scan_models correctly."""
        import struct

        from llm_manager.scanner import GGUFConstants

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Create a proper fake GGUF file with metadata
        gguf_file = models_dir / "test-model.gguf"

        data = b"GGUF"
        data += struct.pack("<I", 3)
        data += struct.pack("<Q", 0)
        data += struct.pack("<Q", 2)

        key = b"general.architecture"
        data += struct.pack("<Q", len(key)) + key
        data += struct.pack("<I", GGUFConstants.STRING)
        value = b"llama"
        data += struct.pack("<Q", len(value)) + value

        key = b"general.size_label"
        data += struct.pack("<Q", len(key)) + key
        data += struct.pack("<I", GGUFConstants.STRING)
        value = b"7B"
        data += struct.pack("<Q", len(value)) + value

        gguf_file.write_bytes(data)

        manager = LLMManager(models_dir=str(models_dir), enable_registry=False)

        async def run_test():
            results = await manager.scan_models_async(test_context=False)
            return results

        results = asyncio.run(run_test())

        assert "models_found" in results
        assert results["models_found"] == 1
