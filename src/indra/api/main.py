"""Application factory and entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import IndraConfig
from .routes import advisory, forecast, health, xai
from .service import NowcastService
from .state import load_state

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load every component once, tolerating the absence of each."""
    state = load_state()
    app.state.service_state = state
    app.state.nowcast_service = NowcastService(state)

    logger.info("indra %s ready=%s on %s", __version__, state.ready, state.device)
    try:
        yield
    finally:
        # Nothing here holds an external connection that must be closed:
        # chromadb is file-backed and the Gemini client is created per call.
        # The model is released with the process.
        app.state.nowcast_service = None
        app.state.service_state = None


def create_app(config: IndraConfig | None = None) -> FastAPI:
    """Build the application."""
    settings = config or _load_api_settings()

    app = FastAPI(
        title="Indra Nowcast API",
        version=__version__,
        description=(
            "Hyper-localised precipitation nowcasting over India: ensemble "
            "forecasts, district advisories and model explanations."
        ),
        root_path=settings.inference.api.root_path,
        lifespan=lifespan,
    )

    api = settings.inference.api
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(api.cors_allow_origins),
        allow_credentials=api.cors_allow_credentials,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # Routers first. See the module docstring: a static mount at "/" claims
    # every path below it.
    app.include_router(health.router)
    app.include_router(forecast.router)
    app.include_router(advisory.router)
    app.include_router(xai.router)

    _mount_frontend(app)
    return app


def _load_api_settings() -> IndraConfig:
    from ..config import load_config

    return load_config()


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built frontend at ``/`` when it exists."""
    from .state import FRONTEND_DIST

    if not FRONTEND_DIST.is_dir():
        logger.info(
            "Frontend build directory (%s) not found; running in headless API " "mode.",
            Path("frontend") / "dist",
        )
        return

    # html=True serves index.html for unknown paths, which a single-page
    # application needs so a deep link does not 404 on reload. It applies only
    # below this mount, so the API routes registered above are unaffected.
    app.mount(
        "/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend"
    )
    logger.info("serving frontend from %s", FRONTEND_DIST)


#: Module-level application, for ``uvicorn indra.api.main:app``.
app = create_app()


__all__ = ["app", "create_app", "lifespan"]
