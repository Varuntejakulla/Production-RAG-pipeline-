# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.exceptions import register_exception_handlers
from app.api.routes import ingest, query, health
from app.api.dependencies import get_cache_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle.
    - Setup logging
    - Initialize Qdrant collection
    - Warm up embedding model (avoids first-request latency)
    """
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    # Ensure Qdrant collection exists
    from app.core.vector_store import VectorStore
    vs = VectorStore()
    vs.ensure_collection()

    # Warm up embedder (loads model into memory once)
    from app.core.embedder import Embedder
    _ = Embedder()

    yield  # app runs here

    # Cleanup (close connections, etc.)
    await get_cache_service().close()


def create_app() -> FastAPI:
    """Application factory pattern — makes testing easier."""
    settings = get_settings()
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — allow frontend origin (adjust for production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # restrict to your frontend domain in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(query.router)

    register_exception_handlers(app)
    return app


app = create_app()

