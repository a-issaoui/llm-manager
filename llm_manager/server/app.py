"""FastAPI application factory for OpenAI-compatible API server."""

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from ..config import Config, get_config
from .dependencies import _server_config, configure_server, shutdown_manager
from .rate_limiter import get_rate_limiter
from .routes import chat_router, completions_router, health_router, metrics_router, models_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting LLM Manager server...")

    # Configure from environment if not already configured
    if not _server_config:
        configure_server()

    # Pre-load default model if specified
    default_model = _server_config.get("default_model")
    if default_model:
        from .dependencies import get_llm_manager, get_or_load_model

        try:
            manager = get_llm_manager()
            await get_or_load_model(manager, default_model)
            logger.info(f"Pre-loaded default model: {default_model}")
        except Exception as e:
            logger.warning(f"Failed to pre-load default model: {e}")

    yield

    # Shutdown
    logger.info("Shutting down LLM Manager server...")
    shutdown_manager()


def create_app(
    models_dir: str | None = None,
    api_key: str | None = None,
    default_model: str | None = None,
    cors_origins: list[str] | None = None,
    config: Config | None = None,
    **manager_kwargs: Any,
) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        models_dir: Directory containing GGUF models
        api_key: Optional API key for authentication
        default_model: Default model to load on startup
        cors_origins: List of allowed CORS origins
        config: Optional Config object (uses get_config() if not provided)
        **manager_kwargs: Additional kwargs for LLMManager

    Returns:
        Configured FastAPI application
    """
    # Load config if not provided
    if config is None:
        config = get_config()

    # Get server config
    server_config = config.server

    # Override with explicit args if provided
    models_dir = models_dir or config.models.dir
    api_key = api_key or server_config.api_key
    cors_origins = cors_origins or server_config.cors_origins

    # Configure server settings
    configure_server(
        models_dir=models_dir, api_key=api_key, default_model=default_model, **manager_kwargs
    )

    # Create FastAPI app
    app = FastAPI(
        title="LLM Manager API",
        description="OpenAI-compatible API for local LLM inference",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if server_config.enable_docs else None,
        redoc_url="/redoc" if server_config.enable_docs else None,
        openapi_url="/openapi.json" if server_config.enable_docs else None,
    )

    # Add middleware
    # CORS
    if cors_origins is None:
        cors_origins = server_config.cors_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Rate limiting middleware
    rate_limiter = get_rate_limiter(requests_per_minute=60)

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next: Callable[[Request], Any]) -> Any:
        """Apply rate limiting to API endpoints."""
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Get client identifier (IP or forwarded IP)
        if request.client:
            client_id = request.headers.get("x-forwarded-for", request.client.host)
        else:
            client_id = "unknown"

        allowed, retry_after = rate_limiter.is_allowed(client_id)

        if not allowed:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={
                    "error": {
                        "message": f"Rate limit exceeded. Retry after {retry_after} seconds.",
                        "type": "rate_limit_exceeded",
                        "code": "rate_limit_exceeded",
                    }
                },
            )

        return await call_next(request)

    # Include routers
    app.include_router(chat_router)
    app.include_router(models_router)
    app.include_router(completions_router)
    app.include_router(health_router)
    app.include_router(metrics_router)

    # Exception handlers
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "internal_error"}},
        )

    # Root endpoint
    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint with API information."""
        return {
            "name": "LLM Manager API",
            "version": "1.0.0",
            "description": "OpenAI-compatible API for local LLM inference",
            "documentation": "/docs",
            "health": "/health",
        }

    return app


class LLMServer:
    """High-level server interface.

    Example:
        server = LLMServer(models_dir="./models", port=8000)
        server.start()
        # Or with default model pre-loaded:
        server.start(default_model="qwen2.5-7b")
        # Or use config from file:
        server = LLMServer.from_config()
    """

    def __init__(
        self,
        models_dir: str | None = None,
        port: int | None = None,
        host: str | None = None,
        api_key: str | None = None,
        log_level: str | None = None,
        config: Config | None = None,
        **kwargs: Any,
    ):
        """Initialize server.

        Args:
            models_dir: Directory with GGUF models (overrides config)
            port: HTTP port (overrides config)
            host: Bind address (overrides config)
            api_key: Optional API key (overrides config)
            log_level: Logging level (overrides config)
            config: Config object (uses get_config() if not provided)
            **kwargs: Additional config for LLMManager
        """
        # Load config if not provided
        self.config = config or get_config()
        server_config = self.config.server

        # Use explicit args or config values
        self.models_dir = models_dir or self.config.models.dir
        self.port = port or server_config.port
        self.host = host or server_config.host
        self.api_key = api_key or server_config.api_key
        _log_level = getattr(server_config, "log_level", "info")
        self.log_level = log_level or _log_level
        self.kwargs = kwargs
        self.app: FastAPI | None = None
        self._server: Any | None = None

    @classmethod
    def from_config(
        cls, config_path: str | None = None, profile: str | None = None
    ) -> "LLMServer":
        """Create LLMServer from config file.

        Args:
            config_path: Path to config file (optional)
            profile: Configuration profile (optional)

        Returns:
            Configured LLMServer instance
        """
        from ..config import load_config

        config = load_config(config_path, profile)
        return cls(config=config)

    def create_app(self, default_model: str | None = None) -> FastAPI:
        """Create FastAPI application."""
        self.app = create_app(
            models_dir=self.models_dir,
            api_key=self.api_key,
            default_model=default_model,
            config=self.config,
            **self.kwargs,
        )
        return self.app

    def start(
        self, default_model: str | None = None, reload: bool = False, workers: int = 1
    ) -> None:
        """Start the server (blocking).

        Args:
            default_model: Model to load on startup
            reload: Enable auto-reload (dev only)
            workers: Number of worker processes (1 for local use)
        """
        import uvicorn

        # Ensure app is created and capture local reference for Mypy
        app = self.app or self.create_app(default_model=default_model)

        logger.info(f"Starting server on {self.host}:{self.port}")

        uvicorn.run(
            app,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            reload=reload,
            workers=workers if not reload else 1,
        )

    async def start_async(self, default_model: str | None = None) -> None:
        """Start server asynchronously (non-blocking).

        Useful for embedding in other applications.
        """
        import uvicorn

        # Ensure app is created and capture local reference
        app = self.app or self.create_app(default_model=default_model)

        config = uvicorn.Config(app, host=self.host, port=self.port, log_level=self.log_level)
        self._server = uvicorn.Server(config)
        await self._server.serve()

    def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.should_exit = True
