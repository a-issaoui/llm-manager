"""Pytest configuration and shared fixtures."""

import gc
import warnings

import pytest

# Filter out AsyncMock RuntimeWarning - it's a known issue with unittest.mock
# when AsyncMock return values are garbage collected without being explicitly awaited
warnings.filterwarnings(
    "ignore",
    message="coroutine 'AsyncMockMixin._execute_mock_call' was never awaited",
    category=RuntimeWarning,
)


@pytest.fixture(scope="session", autouse=True)
def cleanup_async_mock_warnings():
    """Session-scoped fixture to clean up AsyncMock warnings at exit."""
    yield
    # Force garbage collection at end of session to trigger warnings before exit
    gc.collect()


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create a temporary models directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from llm_manager.server import create_app

    app = create_app()
    return TestClient(app)
