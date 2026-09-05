"""Shared route plumbing: service access, and error translation."""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import Depends, HTTPException, Request, status

from ..service import (
    NowcastBundle,
    NowcastService,
    OutOfRecordError,
    WindowUnavailableError,
)
from ..state import ComponentUnavailable, ServiceState

logger = logging.getLogger(__name__)


def get_service(request: Request) -> NowcastService:
    """The service instance built during lifespan startup."""
    service = getattr(request.app.state, "nowcast_service", None)
    if service is None:  # pragma: no cover - only if lifespan did not run
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The service is still starting.",
        )
    return service


def get_state(request: Request) -> ServiceState:
    service = get_service(request)
    return service.state


async def resolve_bundle(
    valid_time: datetime | None,
    service: NowcastService,
) -> NowcastBundle:
    """Validate a requested valid time and return its nowcast."""
    try:
        moment = service.validate_valid_time(valid_time)
    except OutOfRecordError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    try:
        return await service.get_nowcast(moment)
    except ComponentUnavailable as exc:
        # The loader's own instructional message, verbatim.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.detail
        ) from exc
    except WindowUnavailableError as exc:
        # The service is healthy; this particular epoch has no usable inputs.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                "The nowcast did not finish within the configured request "
                "timeout. The first request for a valid time computes the "
                "full ensemble; subsequent requests are served from cache."
            ),
        ) from exc


def unavailable(exc: ComponentUnavailable) -> HTTPException:
    """Render a missing component as 503 with its own guidance."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.detail
    )


ServiceDep = Depends(get_service)
StateDep = Depends(get_state)


__all__ = [
    "ServiceDep",
    "StateDep",
    "get_service",
    "get_state",
    "resolve_bundle",
    "unavailable",
]
