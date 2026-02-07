#!/usr/bin/env python3
"""Tests for FastAPI app factory and lifespan."""

import pytest

pytest.importorskip("fastapi")



class TestCreateApp:
    """Test create_app function."""

    def test_create_app_basic(self):
        """Test creating app with default settings."""
        from llm_manager.server import create_app

        app = create_app()

        assert app is not None
        assert app.title == "LLM Manager API"

    def test_create_app_with_custom_config(self):
        """Test creating app with custom config."""
        from llm_manager.server import create_app

        app = create_app(
            models_dir="./custom_models", api_key="test-key", default_model="test-model"
        )

        assert app is not None

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "LLM Manager API"
        assert "version" in data
        assert "documentation" in data

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint exists."""
        response = client.get("/metrics")

        # Should return 200 or 401 (if auth required)
        assert response.status_code in [200, 401]


class TestMiddleware:
    """Test middleware functionality."""

    def test_rate_limit_middleware_skips_health(self, client):
        """Test rate limiting skips health endpoint."""
        # Make many requests to health endpoint
        for _ in range(70):  # Over the 60/min limit
            response = client.get("/health")
            assert response.status_code == 200

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
        )

        assert "access-control-allow-origin" in response.headers


class TestErrorHandlers:
    """Test error handling."""

    def test_not_found(self, client):
        """Test 404 handling."""
        response = client.get("/nonexistent-endpoint")

        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test 405 handling."""
        response = client.delete("/health")

        assert response.status_code == 405
