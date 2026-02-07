#!/usr/bin/env python3
"""
Example 03: OpenAI-Compatible HTTP Server

Starts an HTTP server with OpenAI-compatible API and makes requests to it.
This demonstrates the full server-client workflow.
"""

import json
import logging
import threading
import time
from pathlib import Path

import requests

from llm_manager.server import LLMServer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def wait_for_server(timeout: int = 30) -> bool:
    """Wait for server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
    return False


def test_health_endpoint() -> None:
    """Test health endpoint."""
    logger.info("Testing /health endpoint...")
    
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    response.raise_for_status()
    data = response.json()
    
    print("\n" + "="*70)
    print("HEALTH CHECK:")
    print("="*70)
    print(json.dumps(data, indent=2))
    print("="*70)


def test_list_models() -> None:
    """Test models listing endpoint."""
    logger.info("Testing /v1/models endpoint...")
    
    response = requests.get(f"{BASE_URL}/v1/models", timeout=10)
    response.raise_for_status()
    data = response.json()
    
    print("\n" + "="*70)
    print("AVAILABLE MODELS:")
    print("="*70)
    for model in data.get("data", []):
        print(f"  - {model['id']} ({model.get('owned_by', 'unknown')})")
    print("="*70)


def test_get_specific_model() -> None:
    """Test getting a specific model."""
    logger.info("Testing /v1/models/{model_id} endpoint...")
    
    # First list models to get one
    response = requests.get(f"{BASE_URL}/v1/models", timeout=10)
    data = response.json()
    
    if data.get("data"):
        model_id = data["data"][0]["id"]
        response = requests.get(f"{BASE_URL}/v1/models/{model_id}", timeout=10)
        response.raise_for_status()
        model = response.json()
        
        print("\n" + "="*70)
        print(f"MODEL DETAILS: {model_id}")
        print("="*70)
        print(f"  ID: {model.get('id')}")
        print(f"  Owned by: {model.get('owned_by')}")
        print(f"  Context window: {model.get('context_window', 'N/A')}")
        print("="*70)


def main():
    """Run OpenAI server example."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    logger.info(f"Starting OpenAI-compatible server example")
    logger.info(f"Models directory: {models_dir}")
    
    # Create server instance (don't preload model to speed up startup)
    server = LLMServer(
        models_dir=str(models_dir),
        host=SERVER_HOST,
        port=SERVER_PORT,
    )
    
    # Start server in background thread
    logger.info("Starting server in background thread...")
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()
    
    try:
        # Wait for server to be ready
        logger.info("Waiting for server to be ready...")
        if not wait_for_server(timeout=30):
            logger.error("Server failed to start within timeout!")
            return 1
        
        logger.info("Server is ready!")
        
        # Run tests
        test_health_endpoint()
        test_list_models()
        test_get_specific_model()
        
        logger.info("All tests passed!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
        
    finally:
        # Cleanup
        logger.info("Shutting down server...")
        server.stop()
        server_thread.join(timeout=5)
        logger.info("Server stopped")
    
    return 0


if __name__ == "__main__":
    exit(main())
