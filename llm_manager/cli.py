"""CLI entry point for LLM Manager.

Usage:
    # Start server - basic usage
    llm-manager --models-dir ./models

    # With default model
    llm-manager --models-dir ./models --model qwen2.5-7b

    # With authentication
    llm-manager --api-key my-secret-key

    # Remote access (not just localhost)
    llm-manager --host 0.0.0.0 --port 8080

    # Using environment variables
    export LLM_MODELS_DIR=./models
    export LLM_DEFAULT_MODEL=qwen2.5-7b
    llm-manager
"""

import argparse
import logging
import os
import sys

# Configure logging before anything else
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(
        description="LLM Manager OpenAI-compatible API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  LLM_MODELS_DIR     Default models directory (default: ./models)
  LLM_DEFAULT_MODEL  Model to load on startup
  LLM_API_KEY        Optional API key for authentication

Examples:
  %(prog)s --models-dir ./models
  %(prog)s --models-dir ./models --model qwen2.5-7b --port 8000
  %(prog)s --host 0.0.0.0 --port 8080 --api-key secret123
        """,
    )

    parser.add_argument(
        "--models-dir",
        "-d",
        default=os.getenv("LLM_MODELS_DIR", "./models"),
        help="Directory containing GGUF models (default: ./models or LLM_MODELS_DIR)",
    )

    parser.add_argument(
        "--model",
        "-m",
        default=os.getenv("LLM_DEFAULT_MODEL"),
        help="Default model to load on startup",
    )

    parser.add_argument(
        "--host",
        "-H",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1, use 0.0.0.0 for remote access)",
    )

    parser.add_argument(
        "--port", "-p", type=int, default=8000, help="Port to bind to (default: 8000)"
    )

    parser.add_argument(
        "--api-key",
        "-k",
        default=os.getenv("LLM_API_KEY"),
        help="Optional API key for authentication",
    )

    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)",
    )

    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (development only)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, for local use keep at 1)",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=4096,
        help="Default context size for models (default: 4096)",
    )

    parser.add_argument(
        "--gpu-layers",
        "-ngl",
        type=int,
        default=-1,
        help="Number of GPU layers to use (-1 for all, default: -1)",
    )

    args = parser.parse_args()

    # Load config (from file or defaults)
    try:
        from .config import get_config, load_config

        config = load_config()  # Load from standard paths
        if not config:
            config = get_config()  # Use defaults
    except Exception as e:
        logger.warning(f"Failed to load config: {e}, using defaults")
        from .config import get_config

        config = get_config()

    # Override config with CLI args
    models_dir = args.models_dir or config.models.dir
    host = args.host or config.server.host
    port = args.port or config.server.port
    api_key = args.api_key or config.server.api_key
    log_level = args.log_level or config.logging.level.lower()

    # Validate models directory
    if not os.path.isdir(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        print(f"Error: Models directory not found: {models_dir}")
        print("Create it or specify with --models-dir")
        sys.exit(1)

    # Set log level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))

    # Import here to avoid slow startup for --help
    try:
        from .server import LLMServer
    except ImportError as e:
        logger.error(f"Failed to import server module: {e}")
        print(f"Error: {e}")
        print("Make sure you have installed: pip install llm-manager[server]")
        sys.exit(1)

    # Create and start server with config
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Server will bind to: {host}:{port}")

    server = LLMServer(
        models_dir=models_dir,
        host=host,
        port=port,
        api_key=api_key,
        log_level=log_level,
        n_gpu_layers=args.gpu_layers,
        n_ctx=args.context_size,
        config=config,
    )

    try:
        server.start(default_model=args.model, reload=args.reload, workers=args.workers)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        print("\nServer stopped.")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
