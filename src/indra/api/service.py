"""Computes a nowcast once per valid time and serves it to every endpoint."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt

from ..advisory.geospatial import DistrictImpact, extract_impacts
from .schemas import EncodedArray
from .state import (
    ADVISORY_MODEL,
    CHECKPOINTS,
    CLIMATOLOGY,
    DISTRICTS,
    NDMA,
    ServiceState,
)

logger = logging.getLogger(__name__)


class OutOfRecordError(ValueError):
    """A requested valid time lies outside the fine-tuning record."""


class WindowUnavailableError(RuntimeError):
    """The input window for a valid time could not be assembled."""


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NowcastBundle:
    """One computed nowcast, shared by every endpoint that reads it."""

    valid_time: datetime
    lead_times: tuple[datetime, ...]
    #: ``(T, H, W)`` fraction of members at or above the heavy threshold.
    probability: npt.NDArray[np.float32]
    #: ``(T, H, W)`` ensemble mean and maximum rain rate, mm h-1.
    mean_mm_h: npt.NDArray[np.float32]
    max_mm_h: npt.NDArray[np.float32]
    threshold_mm_h: float
    seed: int
    members: int
    #: Retained so the explanation endpoint does not rebuild the window --
    #: which would cost a full Stage 1 to 2 pass and, worse, could differ from
    #: the window the forecast was actually made from.
    window: Any
    computed_at: datetime


@dataclass(frozen=True)
class AdvisoryBundle:
    impacts: tuple[DistrictImpact, ...]
    #: Parallel to ``impacts``. ``None`` where generation was unavailable or
    #: its output was rejected.
    alerts: tuple[Any | None, ...]
    grounded: tuple[bool | None, ...]
    reasons: tuple[str | None, ...]
    districts_evaluated: int


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class TTLCache:
    """Bounded cache with an age limit, oldest evicted first."""

    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self._entries: OrderedDict[Any, tuple[float, Any]] = OrderedDict()
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        stored_at, value = entry
        if self.ttl_seconds and (time.monotonic() - stored_at) > self.ttl_seconds:
            del self._entries[key]
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: Any, value: Any) -> None:
        self._entries[key] = (time.monotonic(), value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        looked_up = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / looked_up) if looked_up else None,
        }


# ---------------------------------------------------------------------------
# Raster encoding
# ---------------------------------------------------------------------------


def encode_png(
    field: npt.NDArray[np.floating],
    *,
    colormap: str,
    bounds: tuple[float, float, float, float],
    native_resolution: int | None = None,
) -> EncodedArray:
    """Colour-map a 2-D field into a base64 PNG for a web map overlay."""
    from matplotlib import colormaps
    from PIL import Image

    array = np.asarray(field, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D field; got {array.shape}")

    finite = np.isfinite(array)
    low = float(array[finite].min()) if finite.any() else 0.0
    high = float(array[finite].max()) if finite.any() else 0.0

    span = high - low
    normalised = (array - low) / span if span > 0 else np.zeros_like(array)
    normalised = np.clip(np.nan_to_num(normalised, nan=0.0), 0.0, 1.0)

    rgba = colormaps[colormap](np.flipud(normalised), bytes=True)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")

    return EncodedArray(
        png_base64=f"data:image/png;base64,{payload}",
        shape=(int(array.shape[0]), int(array.shape[1])),
        min_value=low,
        max_value=high,
        colormap=colormap,
        bounds=bounds,
        native_resolution=native_resolution,
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class NowcastService:
    """Computes, caches and slices nowcasts for the route layer."""

    def __init__(self, state: ServiceState) -> None:
        self.state = state
        cache_config = state.config.inference.api.cache
        self.cache = TTLCache(
            max_entries=cache_config.max_entries if cache_config.enabled else 1,
            ttl_seconds=cache_config.ttl_seconds if cache_config.enabled else 0.0,
        )
        self.explanations = TTLCache(
            max_entries=max(cache_config.max_entries // 4, 1),
            ttl_seconds=cache_config.ttl_seconds,
        )
        self.advisories = TTLCache(
            max_entries=max(cache_config.max_entries // 4, 1),
            ttl_seconds=cache_config.ttl_seconds,
        )
        self._locks: dict[Any, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    # ------------------------------------------------------------ utilities
    @property
    def default_valid_time(self) -> datetime:
        return self.state.config.inference.run.default_valid_time

    @property
    def threshold_mm_h(self) -> float:
        return self.state.config.inference.thresholds.precipitation_mm_h["heavy"]

    def validate_valid_time(self, valid_time: datetime | None) -> datetime:
        """Resolve and bound-check a requested valid time."""
        run = self.state.config.inference.run
        moment = valid_time or run.default_valid_time
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)

        if run.allow_out_of_record_time:
            return moment
        if not (run.record_start <= moment <= run.record_end):
            raise OutOfRecordError(
                f"{moment.isoformat()} lies outside the fine-tuning record "
                f"{run.record_start.isoformat()} to {run.record_end.isoformat()}. "
                "The model has not seen this period and will not extrapolate "
                "into it."
            )
        return moment

    async def _lock_for(self, key: Any) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            # Bounded so a long-lived process does not accumulate one lock per
            # valid time ever requested. Only unheld locks are dropped.
            if len(self._locks) > 4 * self.cache.max_entries:
                for stale in [k for k, v in self._locks.items() if not v.locked()][:16]:
                    if stale != key:
                        del self._locks[stale]
            return lock

    # ------------------------------------------------------------- forecast
    def _compute_nowcast(self, valid_time: datetime) -> NowcastBundle:
        """Blocking. Assemble the window, run the ensemble, derive the fields."""
        import torch

        from ..datasets.nowcast import build_input_window

        config = self.state.config
        window = build_input_window(
            config.ingestion, config.preprocessing, valid_time, self.state.stats
        )
        if not window.accepted:
            raise WindowUnavailableError(
                f"the input window for {valid_time.isoformat()} could not be "
                f"assembled: {window.rejection_reason}"
            )

        x = window.to_torch(device=self.state.device, add_batch=True)
        ensemble = config.inference.ensemble
        forecast = self.state.model.predict_ensemble(
            x,
            seed=ensemble.seed,
            members=ensemble.members,
            reductions=("mean", "max"),
            return_members=True,
        )

        threshold = self.threshold_mm_h
        # The counted probability, not the surrogate the attribution path
        # differentiates. This is what the API reports.
        probability = (forecast.members >= threshold).to(torch.float32).mean(dim=0)

        def flatten(tensor: Any) -> npt.NDArray[np.float32]:
            # (B, T, C, H, W) -> (T, H, W); one window, one channel.
            return tensor[0, :, 0].detach().cpu().numpy().astype(np.float32)

        interval = config.inference.lead_times.interval_minutes
        lead_times = tuple(
            valid_time + timedelta(minutes=interval * (index + 1))
            for index in range(config.inference.lead_times.frames)
        )

        return NowcastBundle(
            valid_time=valid_time,
            lead_times=lead_times,
            probability=flatten(probability),
            mean_mm_h=flatten(forecast.reductions["mean"]),
            max_mm_h=flatten(forecast.reductions["max"]),
            threshold_mm_h=threshold,
            seed=ensemble.seed,
            members=ensemble.members,
            window=window,
            computed_at=datetime.now(tz=UTC),
        )

    async def get_nowcast(self, valid_time: datetime) -> NowcastBundle:
        """Cached nowcast for a valid time, computed at most once concurrently."""
        self.state.require(CHECKPOINTS, CLIMATOLOGY)

        cached = self.cache.get(valid_time)
        if cached is not None:
            return cached

        lock = await self._lock_for(valid_time)
        async with lock:
            # Re-check: whoever held the lock first has usually filled it.
            cached = self.cache.get(valid_time)
            if cached is not None:
                return cached

            started = time.monotonic()
            bundle = await asyncio.to_thread(self._compute_nowcast, valid_time)
            logger.info(
                "nowcast for %s computed in %.1f s",
                valid_time.isoformat(),
                time.monotonic() - started,
            )
            self.cache.put(valid_time, bundle)
            return bundle

    # ---------------------------------------------------------------- point
    def point_series(
        self, bundle: NowcastBundle, row: int, col: int
    ) -> dict[str, list[float]]:
        """The three per-lead series at one cell. Pure indexing."""
        return {
            "mean": bundle.mean_mm_h[:, row, col].astype(float).tolist(),
            "max": bundle.max_mm_h[:, row, col].astype(float).tolist(),
            "probability": bundle.probability[:, row, col].astype(float).tolist(),
        }

    def point_window(
        self, bundle: NowcastBundle, row: int, col: int
    ) -> tuple[int | None, int | None]:
        """First and last lead indices where this cell is a reportable exceedance."""
        floor = self.state.config.inference.thresholds.min_reported_probability
        crossed = np.flatnonzero(bundle.probability[:, row, col] >= floor)
        if crossed.size == 0:
            return None, None
        return int(crossed[0]), int(crossed[-1])

    # ------------------------------------------------------------ districts
    def _compute_advisories(self, bundle: NowcastBundle) -> AdvisoryBundle:
        """Blocking. Impacts, then one advisory per qualifying district."""
        from ..advisory.cap import AdvisoryRequest, compose_advisory
        from ..advisory.retrieval import (
            RetrievalContext,
            RetrievalResult,
            build_profiles,
        )

        config = self.state.config
        impacts = extract_impacts(
            bundle.probability,
            self.state.districts,
            config.inference.geospatial,
            config.inference.thresholds,
            intensity_mm_h=bundle.max_mm_h,
        )

        alerts: list[Any | None] = []
        grounded: list[bool | None] = []
        reasons: list[str | None] = []

        # Both halves must be present: the corpus supplies the grounding and
        # the model composes from it, and either alone produces nothing.
        corpus = self.state.components.get(NDMA)
        generator = self.state.components.get(ADVISORY_MODEL)
        blocked = next(
            (c for c in (corpus, generator) if c is None or not c.available), None
        )
        can_advise = blocked is None

        profiles = {}
        if can_advise and impacts:
            profiles = build_profiles(
                bundle.window,
                self.state.districts,
                config.preprocessing,
                config.inference.advisory.taxonomy,
                self.state.stats,
            )

        for impact in impacts:
            if not can_advise:
                alerts.append(None)
                grounded.append(None)
                # Which of the two is missing, in its own words: "the corpus
                # is absent" and "the model is absent" need different fixes.
                reasons.append(
                    blocked.detail if blocked is not None else "advisory disabled"
                )
                continue

            profile = profiles.get((impact.state, impact.district))
            if profile is None:
                alerts.append(None)
                grounded.append(None)
                reasons.append("no terrain profile for this district")
                continue

            context = RetrievalContext.from_impact(
                impact, profile, config.inference.advisory.taxonomy
            )
            retrieval: RetrievalResult = self.state.retriever.retrieve(
                context, impact, top_k=config.inference.advisory.vector_store.top_k
            )
            request = AdvisoryRequest(
                impact=impact,
                retrieval=retrieval,
                valid_time=bundle.valid_time,
                lead_interval_minutes=config.inference.lead_times.interval_minutes,
                # No drivers: attribution costs 128 relay evaluations against
                # the forecast's 8, and running it per district would cost an
                # order of magnitude more than the nowcast itself.
                drivers=(),
            )
            alert = compose_advisory(
                request,
                self.state.advisory_generator,
                config.inference.advisory.cap,
            )
            alerts.append(alert.to_cap_dict() if alert is not None else None)
            # is_official, not "did retrieval return anything". A bootstrap
            # corpus grounds an advisory in something, and that something is
            # not NDMA guidance.
            grounded.append(retrieval.is_official)
            reasons.append(None if alert is not None else "advisory generation failed")

        return AdvisoryBundle(
            impacts=tuple(impacts),
            alerts=tuple(alerts),
            grounded=tuple(grounded),
            reasons=tuple(reasons),
            districts_evaluated=len(self.state.districts),
        )

    async def get_advisories(self, bundle: NowcastBundle) -> AdvisoryBundle:
        self.state.require(DISTRICTS)

        cached = self.advisories.get(bundle.valid_time)
        if cached is not None:
            return cached

        lock = await self._lock_for(("advisories", bundle.valid_time))
        async with lock:
            cached = self.advisories.get(bundle.valid_time)
            if cached is not None:
                return cached
            result = await asyncio.to_thread(self._compute_advisories, bundle)
            self.advisories.put(bundle.valid_time, result)
            return result

    # ------------------------------------------------------------------ xai
    def _district_region_mask(self, state_name: str, district: str) -> Any:
        """A full-resolution boolean mask for one district, for scoped XAI."""
        import torch

        districts = self.state.districts
        try:
            index = districts.keys.index((state_name, district))
        except ValueError as exc:
            raise KeyError(f"no district {district!r} in {state_name!r}") from exc

        flat = np.zeros(districts.grid.height * districts.grid.width, dtype=bool)
        flat[districts.cells_of(index)] = True
        return torch.from_numpy(
            flat.reshape(districts.grid.height, districts.grid.width)
        )

    def _compute_explanation(
        self, bundle: NowcastBundle, state_name: str | None, district: str | None
    ) -> Any:
        """Blocking. 128 relay evaluations at the configured settings."""
        from ..xai.report import build_report

        config = self.state.config
        region = None
        if state_name and district:
            region = self._district_region_mask(state_name, district)

        return build_report(
            self.state.model,
            bundle.window,
            self.state.baseline,
            config.inference.xai,
            threshold_mm_h=bundle.threshold_mm_h,
            seed=bundle.seed,
            served_members=bundle.members,
            region=region,
            device=self.state.device,
        )

    async def get_explanation(
        self,
        bundle: NowcastBundle,
        state_name: str | None = None,
        district: str | None = None,
    ) -> Any:
        self.state.require(CHECKPOINTS, CLIMATOLOGY)
        if state_name and district:
            self.state.require(DISTRICTS)

        key = (bundle.valid_time, state_name, district)
        cached = self.explanations.get(key)
        if cached is not None:
            return cached

        lock = await self._lock_for(("xai", key))
        async with lock:
            cached = self.explanations.get(key)
            if cached is not None:
                return cached

            started = time.monotonic()
            report = await asyncio.to_thread(
                self._compute_explanation, bundle, state_name, district
            )
            logger.info(
                "explanation for %s computed in %.1f s",
                bundle.valid_time.isoformat(),
                time.monotonic() - started,
            )
            self.explanations.put(key, report)
            return report

    # -------------------------------------------------------------- helpers
    def domain_bounds(self) -> tuple[float, float, float, float]:
        """``[south, west, north, east]``, the order a web map expects."""
        west, south, east, north = self.state.grid.outer_bounds
        return (south, west, north, east)

    def cache_stats(self) -> dict[str, Any]:
        return {
            "nowcast": self.cache.stats(),
            "advisories": self.advisories.stats(),
            "explanations": self.explanations.stats(),
        }


__all__ = [
    "AdvisoryBundle",
    "NowcastBundle",
    "NowcastService",
    "OutOfRecordError",
    "TTLCache",
    "WindowUnavailableError",
    "encode_png",
]
