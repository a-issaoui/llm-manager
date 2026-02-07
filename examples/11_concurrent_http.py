#!/usr/bin/env python3
"""
Example 11: Concurrent HTTP Requests

Demonstrates making multiple concurrent requests to the LLM server.
Note: This example requires a model to be loaded first via the server.
"""

import asyncio
import logging
import threading
import time
from pathlib import Path

import aiohttp

from llm_manager.server import LLMServer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8005  # Use different port to avoid conflicts
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


async def wait_for_server(timeout: int = 30) -> bool:
    """Wait for server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{BASE_URL}/health", timeout=2) as resp:
                    if resp.status == 200:
                        return True
        except Exception:
            await asyncio.sleep(0.5)
    return False


async def test_health(session: aiohttp.ClientSession) -> dict:
    """Test health endpoint."""
    async with session.get(f"{BASE_URL}/health", timeout=10) as resp:
        return {"endpoint": "health", "status": resp.status}


async def test_models(session: aiohttp.ClientSession) -> dict:
    """Test models endpoint."""
    async with session.get(f"{BASE_URL}/v1/models", timeout=10) as resp:
        data = await resp.json()
        return {
            "endpoint": "models",
            "status": resp.status,
            "count": len(data.get("data", []))
        }


async def run_concurrent_tests() -> list[dict]:
    """Run multiple endpoint tests concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            test_health(session),
            test_models(session),
        ]
        # Run same tests multiple times concurrently
        for i in range(5):
            tasks.extend([
                test_health(session),
                test_models(session),
            ])
        return await asyncio.gather(*tasks)


async def main():
    """Run concurrent HTTP example."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    logger.info("Starting concurrent HTTP requests example")
    
    # Create server instance (no model preloading for faster startup)
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
        # Wait for server
        logger.info("Waiting for server...")
        if not await wait_for_server(timeout=30):
            logger.error("Server failed to start")
            return 1
        
        logger.info("Server ready!")
        
        print("\n" + "="*70)
        print("CONCURRENT ENDPOINT TESTS")
        print("="*70)
        
        start_time = time.time()
        results = await run_concurrent_tests()
        total_time = time.time() - start_time
        
        # Display results
        success_count = sum(1 for r in results if r.get("status") == 200)
        
        print(f"\nTotal requests: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(results) - success_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average: {total_time/len(results):.3f}s per request")
        print(f"Throughput: {len(results)/total_time:.1f} requests/sec")
        
        # Show unique endpoints tested
        endpoints = {}
        for r in results:
            ep = r['endpoint']
            if ep not in endpoints:
                endpoints[ep] = {'count': 0, 'status': r['status']}
            endpoints[ep]['count'] += 1
        
        print(f"\nEndpoints tested:")
        for ep, info in endpoints.items():
            print(f"  - {ep}: {info['count']} requests (status: {info['status']})")
        
        print("="*70)
        
        if success_count == len(results):
            logger.info("All concurrent tests passed!")
        else:
            logger.warning(f"Some tests failed: {len(results) - success_count}")
        
    finally:
        logger.info("Shutting down server...")
        server.stop()
        server_thread.join(timeout=5)
        logger.info("Server stopped")
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
