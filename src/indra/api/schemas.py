"""Response contracts for the nowcast API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    """Base for every response model."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# ---------------------------------------------------------------------------
# Encoded rasters
# ---------------------------------------------------------------------------


class EncodedArray(APIModel):
    """A 2-D field as a base64 PNG, with the scale needed to read it back."""

    #: ``data:image/png;base64,...`` payload, ready for an ``<img>`` src.
    png_base64: str
    #: ``(rows, cols)`` of the source field, before encoding.
    shape: tuple[int, int]
    #: Value the darkest end of the colour map represents.
    min_value: float
    #: Value the brightest end represents. Equal to ``min_value`` for a
    #: constant field, which the client must handle rather than divide by zero.
    max_value: float
    colormap: str
    #: ``[south, west, north, east]`` in degrees, the order Leaflet's
    #: ``ImageOverlay`` takes. The image is already north-up.
    bounds: tuple[float, float, float, float]
    #: Native resolution of the underlying field, when the encoded map was
    #: resampled to get here. Attention is computed at 48 and reported at 96;
    #: stating so keeps interpolated cells from being read as resolved detail.
    native_resolution: int | None = None


# ---------------------------------------------------------------------------
# Point forecast
# ---------------------------------------------------------------------------


class PointForecast(APIModel):
    """The forecast for one grid cell."""

    valid_time: datetime
    #: The coordinate that was asked about.
    requested_latitude: float
    requested_longitude: float
    #: The centre of the cell that answered. A cell is roughly 9 km across, so
    #: this is not the same point and is not presented as one.
    cell_latitude: float
    cell_longitude: float
    row: int
    col: int

    lead_times: list[datetime]
    #: Ensemble mean rain rate per lead frame.
    precipitation_mean_mm_h: list[float]
    #: Ensemble maximum, so a client can show the range rather than only the
    #: central estimate. A mean of 4 mm/h with a maximum of 40 is a different
    #: situation from a mean of 4 with a maximum of 5.
    precipitation_max_mm_h: list[float]
    #: Fraction of members exceeding the heavy-rain threshold, per lead frame.
    exceedance_probability: list[float]

    threshold_mm_h: float
    peak_probability: float
    peak_lead_index: int
    #: Null when the threshold is never crossed at this cell. That is an
    #: ordinary answer -- most cells, most of the time -- and not an error.
    onset_time: datetime | None = None
    expiry_time: datetime | None = None
    #: Districts containing this cell, as ``"District, State"``. A boundary
    #: cell belongs to more than one.
    districts: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# District advisories
# ---------------------------------------------------------------------------


class DistrictAdvisory(APIModel):
    """One qualified district: the impact, and the alert composed for it."""

    state: str
    district: str
    peak_probability: float
    peak_lead_index: int
    onset_time: datetime
    expiry_time: datetime
    affected_area_km2: float
    affected_fraction: float
    district_area_km2: float
    severity_band: str | None
    peak_intensity_mm_h: float | None

    #: The CAP 1.2 alert, in protocol element names. Null when generation was
    #: unavailable or its output was rejected -- the impact is still reported,
    #: because the forecast stands whether or not an advisory could be written
    #: for it.
    cap_alert: dict[str, Any] | None = None
    #: False when the advisory rests on generic guidance rather than retrieved
    #: NDMA text. Surfaced here as well as inside the CAP parameters so a
    #: client cannot miss it.
    grounded_in_ndma: bool | None = None
    advisory_unavailable_reason: str | None = None


class DistrictAdvisoryList(APIModel):
    valid_time: datetime
    threshold_mm_h: float
    #: Districts examined, against those that qualified. A response listing
    #: three districts means something different when 700 were checked than
    #: when 4 were.
    districts_evaluated: int
    advisories: list[DistrictAdvisory]


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


class EvidenceFrameOut(APIModel):
    """One input frame the forecast leaned on."""

    timestamp: datetime
    lookback_index: int
    attention_share: float
    #: Share divided by ``1/T``. One is indifference. Without this a share of
    #: 0.08 across thirteen frames reads as a finding when it is the opposite.
    relative_to_uniform: float


class DriverOut(APIModel):
    """One named channel's contribution."""

    channel: str
    signed_attribution: float
    magnitude: float
    #: ``"increased"`` or ``"decreased"``, spelled out rather than left to the
    #: sign of a float.
    direction: str


class ExplanationResponse(APIModel):
    """Why the model forecast what it did."""

    valid_time: datetime
    #: Present when the explanation was scoped to one district.
    state: str | None = None
    district: str | None = None

    evidence_frames: list[EvidenceFrameOut]
    drivers: list[DriverOut]

    attribution_map: EncodedArray
    attention_encoder_map: EncodedArray
    attention_decoder_map: EncodedArray

    #: Every setting the maps depend on: relaxation width, member count, seed,
    #: step count, convergence delta. A heatmap without these is a picture.
    provenance: dict[str, Any]
    #: Conditions that weaken this explanation, in plain language: unconverged
    #: attribution, near-uniform evidence, channels zero by construction.
    caveats: list[str] = Field(default_factory=list)
    #: Channels whose attribution is structurally zero rather than measured.
    excluded_channels: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class ComponentHealth(APIModel):
    """Readiness of one external artefact this service depends on."""

    name: str
    available: bool
    #: Where it was looked for, so an operator can check the mount rather than
    #: guess at it.
    path: str | None = None
    #: The instructional message from the loader that failed, verbatim. These
    #: already say what to run; repeating them here means the fix reaches
    #: whoever is looking at the health endpoint.
    detail: str | None = None


class HealthResponse(APIModel):
    """Liveness, readiness, and what is missing."""

    status: str
    #: True only when every component a forecast needs is present. The service
    #: answers this endpoint either way -- a degraded service that explains
    #: itself is more useful than one that would not start.
    ready: bool
    version: str
    model_name: str
    valid_time_default: datetime
    record_start: datetime
    record_end: datetime
    #: Domain corners as ``[south, west, north, east]``, so a client can bound
    #: its map without a second request.
    domain_bounds: tuple[float, float, float, float]
    grid_shape: tuple[int, int]
    resolution_deg: float
    lead_frames: int
    lead_interval_minutes: int
    ensemble_members: int
    components: list[ComponentHealth]


__all__ = [
    "APIModel",
    "ComponentHealth",
    "DistrictAdvisory",
    "DistrictAdvisoryList",
    "DriverOut",
    "EncodedArray",
    "EvidenceFrameOut",
    "ExplanationResponse",
    "HealthResponse",
    "PointForecast",
]
