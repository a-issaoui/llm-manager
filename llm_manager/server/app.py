"""FastAPI application factory for OpenAI-compatible API server."""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from ..config import Config, get_config, ServerConfig
from .dependencies import configure_server, shutdown_manager, _server_config
from .routes import chat_router, models_router, completions_router, health_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    models_dir: str = None,
    api_key: str = None,
    default_model: str = None,
    cors_origins: list = None,
    config: Optional[Config] = None,
    **manager_kwargs
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
        models_dir=models_dir,
        api_key=api_key,
        default_model=default_model,
        **manager_kwargs
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="LLM Manager API",
        description="OpenAI-compatible API for local LLM inference",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if server_config.enable_docs else None,
        redoc_url="/redoc" if server_config.enable_docs else None,
        openapi_url="/openapi.json" if server_config.enable_docs else None
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
    
    # Include routers
    app.include_router(chat_router)
    app.include_router(models_router)
    app.include_router(completions_router)
    app.include_router(health_router)
    
    # Exception handlers
    @app.exception_handler(Exception)
    async def generic_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "internal_error"
                }
            }
        )
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "LLM Manager API",
            "version": "1.0.0",
            "description": "OpenAI-compatible API for local LLM inference",
            "documentation": "/docs",
            "health": "/health"
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
        models_dir: str = None,
        port: int = None,
        host: str = None,
        api_key: str = None,
        log_level: str = None,
        config: Optional[Config] = None,
        **kwargs
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
        self.log_level = log_level or server_config.log_level if hasattr(server_config, 'log_level') else "info"
        self.kwargs = kwargs
        self.app = None
        self._server = None
    
    @classmethod
    def from_config(cls, config_path: str = None, profile: str = None) -> 'LLMServer':
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
    
    def create_app(self, default_model: str = None) -> FastAPI:
        """Create FastAPI application."""
        self.app = create_app(
            models_dir=self.models_dir,
            api_key=self.api_key,
            default_model=default_model,
            config=self.config,
            **self.kwargs
        )
        return self.app
    
    def start(
        self,
        default_model: str = None,
        reload: bool = False,
        workers: int = 1
    ):
        """Start the server (blocking).
        
        Args:
            default_model: Model to load on startup
            reload: Enable auto-reload (dev only)
            workers: Number of worker processes (1 for local use)
        """
        import uvicorn
        
        if self.app is None:
            self.create_app(default_model=default_model)
        
        logger.info(f"Starting server on {self.host}:{self.port}")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            reload=reload,
            workers=workers if not reload else 1
        )
    
    async def start_async(
        self,
        default_model: str = None
    ):
        """Start server asynchronously (non-blocking).
        
        Useful for embedding in other applications.
        """
        import uvicorn
        
        if self.app is None:
            self.create_app(default_model=default_model)
        
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level=self.log_level
        )
        self._server = uvicorn.Server(config)
        await self._server.serve()
    
    def stop(self):
        """Stop the server."""
        if self._server:
            self._server.should_exit = True
