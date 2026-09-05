"""``GET /api/forecast/point`` -- the map's point-and-click."""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status

from ...advisory.geospatial import DistrictGrid
from ..schemas import PointForecast
from ..service import NowcastService
from .dependencies import ServiceDep, resolve_bundle

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/forecast", tags=["forecast"])


def _districts_at(districts: DistrictGrid | None, row: int, col: int) -> list[str]:
    """District labels covering a cell, or none when boundaries are absent."""
    if districts is None:
        return []
    try:
        return [
            f"{district}, {state}"
            for state, district in districts.districts_at(row, col)
        ]
    except Exception as exc:
        logger.warning("district lookup failed at (%d, %d): %s", row, col, exc)
        return []


@router.get(
    "/point",
    response_model=PointForecast,
    summary="Forecast for the grid cell nearest a coordinate",
    responses={
        422: {
            "description": "Coordinate outside the domain, or valid time outside the record"
        },
        503: {"description": "Model weights or climatology unavailable"},
    },
)
async def point_forecast(
    lat: float = Query(..., description="Latitude in degrees north."),
    lon: float = Query(..., description="Longitude in degrees east."),
    valid_time: datetime | None = Query(
        None,
        description=(
            "Nowcast epoch in UTC. Defaults to the configured demonstration "
            "time. Must lie within the fine-tuning record."
        ),
    ),
    service: NowcastService = ServiceDep,
) -> PointForecast:
    bundle = await resolve_bundle(valid_time, service)

    try:
        row, col = service.state.grid.latlon_to_indices(lat, lon)
    except ValueError as exc:
        # Outside the domain. 422 rather than 404: the request is well-formed
        # and simply not answerable, and no retry will change that. Clamping
        # to the nearest coastal cell would answer a different question and
        # present it as this one.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    cell_lat, cell_lon = service.state.grid.indices_to_latlon(row, col)
    series = service.point_series(bundle, row, col)
    onset_index, expiry_index = service.point_window(bundle, row, col)

    probabilities = series["probability"]
    peak_index = int(max(range(len(probabilities)), key=probabilities.__getitem__))

    return PointForecast(
        valid_time=bundle.valid_time,
        requested_latitude=lat,
        requested_longitude=lon,
        cell_latitude=cell_lat,
        cell_longitude=cell_lon,
        row=row,
        col=col,
        lead_times=list(bundle.lead_times),
        precipitation_mean_mm_h=series["mean"],
        precipitation_max_mm_h=series["max"],
        exceedance_probability=probabilities,
        threshold_mm_h=bundle.threshold_mm_h,
        peak_probability=probabilities[peak_index],
        peak_lead_index=peak_index,
        onset_time=(
            bundle.lead_times[onset_index] if onset_index is not None else None
        ),
        expiry_time=(
            bundle.lead_times[expiry_index] if expiry_index is not None else None
        ),
        districts=_districts_at(service.state.districts, row, col),
    )


__all__ = ["router"]
