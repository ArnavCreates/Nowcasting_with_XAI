"""``GET /healthz`` -- liveness, readiness, and what is missing."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ... import __version__
from ..schemas import ComponentHealth, HealthResponse
from ..service import NowcastService
from .dependencies import ServiceDep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Service readiness and loaded configuration",
)
async def healthz(service: NowcastService = ServiceDep) -> HealthResponse:
    state = service.state
    config = state.config
    grid = state.grid

    return HealthResponse(
        status="ok",
        ready=state.ready,
        version=__version__,
        model_name=config.model.fusion.name,
        valid_time_default=config.inference.run.default_valid_time,
        record_start=config.inference.run.record_start,
        record_end=config.inference.run.record_end,
        domain_bounds=service.domain_bounds(),
        grid_shape=(grid.height, grid.width),
        resolution_deg=grid.resolution_deg,
        lead_frames=config.inference.lead_times.frames,
        lead_interval_minutes=config.inference.lead_times.interval_minutes,
        ensemble_members=config.inference.ensemble.members,
        components=[
            ComponentHealth(
                name=status.name,
                available=status.available,
                path=status.path,
                detail=status.detail,
            )
            for status in state.status_list()
        ],
    )


__all__ = ["router"]
