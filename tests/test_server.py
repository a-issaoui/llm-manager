"""Tests for OpenAI-compatible REST API server."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

# Skip all tests if server dependencies not installed
pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from llm_manager.server import create_app, LLMServer
from llm_manager.schemas.openai import (
    ChatMessage,
    ChatCompletionRequest,
    ModelInfo,
    HealthStatus
)


class TestServerBasic:
    """Test basic server functionality."""
    
    def test_app_creation(self):
        """Test FastAPI app can be created."""
        app = create_app(models_dir="./models")
        assert app.title == "LLM Manager API"
        assert app.version == "1.0.0"
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        app = create_app(models_dir="./models")
        client = TestClient(app)
        
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "LLM Manager API"
        assert "/docs" in data["documentation"]
    
    def test_health_endpoint_no_model(self, tmp_path):
        """Test health endpoint when no model loaded."""
        app = create_app(models_dir=str(tmp_path))
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False
        assert data["current_model"] is None
        assert "uptime_seconds" in data


class TestChatCompletions:
    """Test chat completions endpoint."""
    
    def test_chat_completions_success(self, tmp_path):
        """Test successful chat completion."""
        from llm_manager.server.dependencies import get_llm_manager
        from llm_manager.server.routes import chat
        
        # Create mock manager
        manager = Mock()
        manager.is_loaded.return_value = True
        manager.model_path = Mock()
        manager.model_path.stem = "test-model"
        
        mock_response = {
            "choices": [{
                "message": {"content": "Hello! How can I help you?"}
            }]
        }
        manager.generate_async = AsyncMock(return_value=mock_response)
        
        # Create app with dependency overrides
        app = create_app(models_dir=str(tmp_path))
        app.dependency_overrides[get_llm_manager] = lambda: manager
        
        # Patch get_or_load_model in the chat module
        async def mock_get_or_load(mgr, name):
            return manager
        
        with patch.object(chat, 'get_or_load_model', mock_get_or_load):
            client = TestClient(app)
            
            response = client.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "temperature": 0.7,
                "max_tokens": 100
            })
            
            assert response.status_code == 200, f"Response: {response.text}"
            data = response.json()
            assert data["object"] == "chat.completion"
            assert len(data["choices"]) == 1
            assert "usage" in data
    
    def test_chat_completions_streaming(self, tmp_path):
        """Test streaming chat completion."""
        from llm_manager.server.dependencies import get_llm_manager
        from llm_manager.server.routes import chat
        
        manager = Mock()
        manager.is_loaded.return_value = True
        manager.model_path = Mock()
        manager.model_path.stem = "test-model"
        
        mock_response = {
            "choices": [{
                "message": {"content": "Hello world"}
            }]
        }
        manager.generate_async = AsyncMock(return_value=mock_response)
        
        app = create_app(models_dir=str(tmp_path))
        app.dependency_overrides[get_llm_manager] = lambda: manager
        
        # Patch get_or_load_model in the chat module
        async def mock_get_or_load(mgr, name):
            return manager
        
        with patch.object(chat, 'get_or_load_model', mock_get_or_load):
            client = TestClient(app)
            
            response = client.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True
            })
            
            assert response.status_code == 200, f"Response: {response.text}"
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
            
            # Check stream contains data
            content = response.content.decode()
            assert "data:" in content
            assert "[DONE]" in content


class TestModelsEndpoint:
    """Test models listing endpoint."""
    
    def test_list_models(self, tmp_path):
        """Test listing available models."""
        from llm_manager.server.dependencies import get_llm_manager
        
        # Create mock manager with registry
        # ModelRegistry.list_models() returns List[str]
        manager = Mock()
        manager.models_dir = tmp_path
        
        mock_metadata = Mock()
        mock_metadata.specs.architecture = "llama"
        
        registry = Mock()
        registry.list_models.return_value = ["test-model.gguf"]
        registry.get.return_value = mock_metadata
        manager.registry = registry
        
        app = create_app(models_dir=str(tmp_path))
        app.dependency_overrides[get_llm_manager] = lambda: manager
        
        client = TestClient(app)
        
        response = client.get("/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model.gguf"
    
    def test_get_specific_model(self, tmp_path):
        """Test getting a specific model."""
        from llm_manager.server.dependencies import get_llm_manager
        
        manager = Mock()
        manager.models_dir = tmp_path
        
        # ModelRegistry.get() returns ModelMetadata
        mock_metadata = Mock()
        mock_metadata.specs.architecture = "llama"
        
        registry = Mock()
        registry.get.return_value = mock_metadata
        manager.registry = registry
        
        app = create_app(models_dir=str(tmp_path))
        app.dependency_overrides[get_llm_manager] = lambda: manager
        
        client = TestClient(app)
        
        response = client.get("/v1/models/test-model")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-model"


class TestSchemas:
    """Test OpenAI schema models."""
    
    def test_chat_message_creation(self):
        """Test ChatMessage schema."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        
        data = msg.model_dump()
        assert data["role"] == "user"
        assert data["content"] == "Hello"
    
    def test_chat_completion_request(self):
        """Test ChatCompletionRequest schema."""
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello")
            ],
            temperature=0.5,
            max_tokens=100
        )
        
        assert request.model == "gpt-4"
        assert len(request.messages) == 2
        assert request.temperature == 0.5
        assert request.max_tokens == 100
        assert request.stream is False  # Default
    
    def test_default_temperature(self):
        """Test default temperature value."""
        request = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")]
        )
        assert request.temperature == 0.7


class TestLLMServer:
    """Test LLMServer class."""
    
    def test_server_initialization(self):
        """Test server can be initialized."""
        server = LLMServer(
            models_dir="./models",
            port=8000,
            host="127.0.0.1"
        )
        
        assert server.models_dir == "./models"
        assert server.port == 8000
        assert server.host == "127.0.0.1"
    
    def test_create_app(self):
        """Test server creates app correctly."""
        server = LLMServer(models_dir="./models")
        app = server.create_app()
        
        assert app is not None
        assert app.title == "LLM Manager API"


class TestLegacyCompletions:
    """Test legacy completions endpoint."""
    
    def test_legacy_completion(self, tmp_path):
        """Test legacy text completion."""
        from llm_manager.server.dependencies import get_llm_manager
        from llm_manager.server.routes import completions
        
        # Create mock manager
        manager = Mock()
        manager.is_loaded.return_value = True
        manager.model_path = Mock()
        manager.model_path.stem = "test-model"
        
        mock_response = {
            "choices": [{
                "message": {"content": "This is a completion"}
            }]
        }
        manager.generate_async = AsyncMock(return_value=mock_response)
        
        app = create_app(models_dir=str(tmp_path))
        app.dependency_overrides[get_llm_manager] = lambda: manager
        
        # Patch get_or_load_model in the completions module
        async def mock_get_or_load(mgr, name):
            return manager
        
        with patch.object(completions, 'get_or_load_model', mock_get_or_load):
            client = TestClient(app)
            
            response = client.post("/v1/completions", json={
                "model": "test-model",
                "prompt": "Once upon a time",
                "max_tokens": 50
            })
            
            assert response.status_code == 200, f"Response: {response.text}"
            data = response.json()
            assert data["object"] == "text_completion"
            assert len(data["choices"]) == 1


class TestAdminEndpoints:
    """Test admin endpoints."""
    
    def test_server_info(self, tmp_path):
        """Test server info endpoint."""
        app = create_app(models_dir=str(tmp_path))
        client = TestClient(app)
        
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "llm-manager"
        assert data["supports_streaming"] is True
    
    def test_readiness_check(self, tmp_path):
        """Test readiness endpoint."""
        app = create_app(models_dir=str(tmp_path))
        client = TestClient(app)
        
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
    
    def test_liveness_check(self, tmp_path):
        """Test liveness endpoint."""
        app = create_app(models_dir=str(tmp_path))
        client = TestClient(app)
        
        response = client.get("/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers(self, tmp_path):
        """Test CORS headers are present on actual requests."""
        app = create_app(models_dir=str(tmp_path))
        client = TestClient(app)
        
        # Test CORS on actual endpoint with Origin header
        response = client.get("/health", headers={
            "Origin": "http://localhost:3000"
        })
        
        assert response.status_code == 200
        # CORS headers should be present when Origin is provided
        assert "access-control-allow-origin" in response.headers or "access-control-allow-credentials" in response.headers
