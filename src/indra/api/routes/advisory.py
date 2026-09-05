"""``GET /api/advisories/districts`` -- who is affected, and what to tell them."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Query

from ..schemas import DistrictAdvisory, DistrictAdvisoryList
from ..service import NowcastService
from ..state import ComponentUnavailable
from .dependencies import ServiceDep, resolve_bundle, unavailable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/advisories", tags=["advisories"])


@router.get(
    "/districts",
    response_model=DistrictAdvisoryList,
    summary="Qualifying districts and their CAP advisories",
    responses={
        422: {"description": "Valid time outside the record"},
        503: {"description": "Model, climatology or district boundaries unavailable"},
    },
)
async def district_advisories(
    valid_time: datetime | None = Query(
        None, description="Nowcast epoch in UTC. Defaults to the configured time."
    ),
    limit: int | None = Query(
        None,
        gt=0,
        description=(
            "Cap the number of districts returned, most severe first. Omitted, "
            "every qualifying district is returned."
        ),
    ),
    service: NowcastService = ServiceDep,
) -> DistrictAdvisoryList:
    bundle = await resolve_bundle(valid_time, service)

    try:
        advisories = await service.get_advisories(bundle)
    except ComponentUnavailable as exc:
        raise unavailable(exc) from exc

    interval = service.state.config.inference.lead_times.interval_minutes

    def lead_time(index: int) -> datetime:
        return bundle.valid_time + timedelta(minutes=interval * (index + 1))

    entries: list[DistrictAdvisory] = []
    for impact, alert, grounded, reason in zip(
        advisories.impacts,
        advisories.alerts,
        advisories.grounded,
        advisories.reasons,
        strict=False,
    ):
        entries.append(
            DistrictAdvisory(
                state=impact.state,
                district=impact.district,
                peak_probability=impact.peak_probability,
                peak_lead_index=impact.peak_lead_index,
                onset_time=lead_time(impact.onset_lead_index),
                expiry_time=lead_time(impact.expiry_lead_index),
                affected_area_km2=impact.affected_area_km2,
                affected_fraction=impact.affected_fraction,
                district_area_km2=impact.district_area_km2,
                severity_band=impact.severity,
                peak_intensity_mm_h=impact.peak_intensity_mm_h,
                cap_alert=alert,
                grounded_in_ndma=grounded,
                advisory_unavailable_reason=reason,
            )
        )

    if limit is not None:
        # Already ordered by peak probability, so a limit takes the most
        # severe rather than an arbitrary slice.
        entries = entries[:limit]

    return DistrictAdvisoryList(
        valid_time=bundle.valid_time,
        threshold_mm_h=bundle.threshold_mm_h,
        districts_evaluated=advisories.districts_evaluated,
        advisories=entries,
    )


__all__ = ["router"]
