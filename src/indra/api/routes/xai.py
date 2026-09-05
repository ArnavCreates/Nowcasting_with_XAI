"""``GET /api/xai/explanation`` -- why the model forecast what it did."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from ..schemas import DriverOut, EncodedArray, EvidenceFrameOut, ExplanationResponse
from ..service import NowcastService, encode_png
from ..state import ComponentUnavailable
from .dependencies import ServiceDep, resolve_bundle, unavailable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/xai", tags=["xai"])


@router.get(
    "/explanation",
    response_model=ExplanationResponse,
    summary="Attribution, attention maps and evidence frames",
    responses={
        404: {"description": "Named district not present in the boundary layer"},
        422: {"description": "Valid time outside the record"},
        503: {"description": "Model, climatology or district boundaries unavailable"},
    },
)
async def explanation(
    valid_time: datetime | None = Query(
        None, description="Nowcast epoch in UTC. Defaults to the configured time."
    ),
    state: str | None = Query(
        None, description="State name, to scope the explanation to one district."
    ),
    district: str | None = Query(None, description="District name. Requires `state`."),
    service: NowcastService = ServiceDep,
) -> ExplanationResponse:
    if bool(state) != bool(district):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "state and district must be supplied together: a district name "
                "alone is ambiguous, since several Indian states share one."
            ),
        )

    bundle = await resolve_bundle(valid_time, service)

    try:
        report = await service.get_explanation(bundle, state, district)
    except ComponentUnavailable as exc:
        raise unavailable(exc) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ValueError as exc:
        # Raised when the window itself was rejected, or the region covers no
        # attention cell. Both are answerable questions with no answer.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    colormap = service.state.config.inference.xai.attribution_maps.colormap
    bounds = service.domain_bounds()

    def raster(field: Any, native: int | None = None) -> EncodedArray:
        return encode_png(
            field, colormap=colormap, bounds=bounds, native_resolution=native
        )

    return ExplanationResponse(
        valid_time=bundle.valid_time,
        state=state,
        district=district,
        evidence_frames=[
            EvidenceFrameOut(
                timestamp=frame.timestamp,
                lookback_index=frame.lookback_index,
                attention_share=frame.attention_share,
                relative_to_uniform=frame.relative_to_uniform,
            )
            for frame in report.evidence_frames
        ],
        drivers=[
            DriverOut(
                channel=driver["channel"],
                signed_attribution=driver["signed_attribution"],
                magnitude=driver["magnitude"],
                direction=driver["direction"],
            )
            for driver in report.drivers(
                service.state.config.inference.xai.attribution_maps.top_k_drivers
            )
        ],
        attribution_map=raster(report.attribution.spatial_map),
        # Summed over the frame axis: one map of where attention concentrated
        # across the whole window. The per-frame detail is in evidence_frames,
        # which is the form a forecaster can actually read.
        attention_encoder_map=raster(
            report.maps.encoder_relative.sum(axis=0), report.maps.native_resolution
        ),
        attention_decoder_map=raster(
            report.maps.decoder_share.sum(axis=0), report.maps.native_resolution
        ),
        provenance=report.provenance(),
        caveats=list(report.caveats),
        excluded_channels=list(report.excluded_channels),
    )


__all__ = ["router"]
