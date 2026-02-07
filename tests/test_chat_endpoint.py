#!/usr/bin/env python3
"""Tests for chat completions endpoint."""

import pytest

pytest.importorskip("fastapi")

from llm_manager.schemas.openai import ChatCompletionRequest, ChatMessage


class TestChatValidation:
    """Test chat request validation."""

    def test_valid_request(self):
        """Test valid chat completion request."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello"),
            ],
            temperature=0.7,
            max_tokens=100,
        )

        assert request.model == "test-model"
        assert len(request.messages) == 2
        assert request.temperature == 0.7

    def test_temperature_range(self):
        """Test temperature must be 0-2."""
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="test", messages=[ChatMessage(role="user", content="Hi")], temperature=3.0
            )

    def test_max_tokens_positive(self):
        """Test max_tokens must be positive."""
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="test", messages=[ChatMessage(role="user", content="Hi")], max_tokens=0
            )

    def test_logit_bias_validation(self):
        """Test logit_bias value range."""
        # Valid
        request = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            logit_bias={"123": 50.0},
        )
        assert request.logit_bias == {"123": 50.0}

        # Invalid - too high
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="Hi")],
                logit_bias={"123": 150.0},
            )

    def test_n_parameter_validation(self):
        """Test n parameter only accepts 1."""
        # Valid (default)
        request = ChatCompletionRequest(
            model="test", messages=[ChatMessage(role="user", content="Hi")]
        )
        assert request.n == 1

        # Invalid
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="test", messages=[ChatMessage(role="user", content="Hi")], n=3
            )

    def test_presence_penalty_range(self):
        """Test presence_penalty must be -2 to 2."""
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="Hi")],
                presence_penalty=3.0,
            )

    def test_frequency_penalty_range(self):
        """Test frequency_penalty must be -2 to 2."""
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="Hi")],
                frequency_penalty=-3.0,
            )

    def test_top_p_range(self):
        """Test top_p must be 0-1."""
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="test", messages=[ChatMessage(role="user", content="Hi")], top_p=1.5
            )

    def test_stream_options_with_stream(self):
        """Test stream_options can be set with stream=True."""
        request = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            stream=True,
            stream_options={"include_usage": True},
        )
        assert request.stream is True
        assert request.stream_options.include_usage is True

    def test_stream_options_without_stream(self):
        """Test stream_options without stream."""
        # Currently allowed - stream_options is stored but only used if stream=True
        request = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            stream=False,
            stream_options={"include_usage": True},
        )
        assert request.stream is False
        assert request.stream_options is not None


class TestChatEndpointIntegration:
    """Integration tests for chat endpoint."""

    def test_chat_endpoint_basic(self, client):
        """Test chat endpoint exists and handles requests."""
        # Without configured models, should return 404
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]},
        )
        # Status code depends on model availability - just verify endpoint exists
        assert response.status_code in [200, 404, 422]

    def test_chat_endpoint_missing_messages(self, client):
        """Test chat without messages."""
        response = client.post("/v1/chat/completions", json={"model": "test"})
        # Should be 422 validation error
        assert response.status_code in [400, 422]

    def test_chat_endpoint_invalid_json(self, client):
        """Test chat with invalid JSON."""
        response = client.post(
            "/v1/chat/completions",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        # Should be 422 validation error
        assert response.status_code in [400, 422]
