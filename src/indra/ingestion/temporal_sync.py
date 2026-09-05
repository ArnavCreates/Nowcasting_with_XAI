"""Temporal synchronisation — assembling the T = 13 lookback window."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import xarray as xr

from ..config import IngestionConfig, QualityControlConfig
from ..types import (
    ATTR_QC_FLAG,
    QCFlag,
    SourceStream,
    SyncedWindow,
    masked_like,
)
from . import imd_surface, imdaa, insat3d, static_priors
from . import qc as qc_module

logger = logging.getLogger(__name__)

#: Channel whose motion field is used to advect every satellite channel.
#: See the module docstring for why TIR-1 rather than WV.
FLOW_REFERENCE_CHANNEL = "insat_tir1"

#: Streams whose gaps are reconstructed by dense optical flow. The others fall
#: back to hold/linear, because advecting a model analysis or a daily rainfall
#: total along an estimated motion field is not physically meaningful -- there
#: is no coherent advecting feature to track in a daily accumulation.
FLOW_STREAMS: frozenset[str] = frozenset({SourceStream.INSAT.value})


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def build_timestamps(valid_time: datetime, config: IngestionConfig) -> list[datetime]:
    """The T nominal slot times, oldest first, ending at ``valid_time``."""
    step = timedelta(minutes=config.temporal.interval_minutes)
    return [valid_time + offset * step for offset in config.temporal.lookback_indices]


def _within_tolerance(
    actual: datetime | None, nominal: datetime, tolerance_minutes: int
) -> bool:
    """Whether an observation may be admitted into a slot."""
    if actual is None:
        # No parseable acquisition time. Accepted, because the filename match
        # already located it for this slot and rejecting on a naming quirk
        # would discard usable data.
        return True
    return abs((actual - nominal).total_seconds()) <= tolerance_minutes * 60


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------


def _modal_shape(frames: list[xr.Dataset], variable: str) -> tuple[int, ...] | None:
    """The shape most frames agree on for a variable."""
    shapes: dict[tuple[int, ...], int] = {}
    for frame in frames:
        if variable not in frame.data_vars:
            continue
        shape = tuple(frame[variable].shape)
        if 0 in shape:
            continue
        shapes[shape] = shapes.get(shape, 0) + 1
    if not shapes:
        return None
    return max(shapes.items(), key=lambda kv: kv[1])[0]


def _stack_frames(
    frames: list[xr.Dataset],
    timestamps: list[datetime],
    stream: str,
) -> xr.Dataset | None:
    """Stack per-slot datasets along a new ``time`` axis."""
    if not frames:
        return None

    variables = sorted({str(v) for frame in frames for v in frame.data_vars})
    if not variables:
        return None

    data_vars: dict[str, Any] = {}
    template: xr.Dataset | None = next(
        (
            f
            for f in frames
            if f.data_vars and all(0 not in tuple(f[v].shape) for v in f.data_vars)
        ),
        None,
    )

    for name in variables:
        shape = _modal_shape(frames, name)
        if shape is None:
            logger.error(
                "%s: variable %s has no readable frame in the window", stream, name
            )
            continue

        stacked = []
        for frame in frames:
            if name in frame.data_vars and tuple(frame[name].shape) == shape:
                stacked.append(np.asarray(frame[name].values, dtype=np.float32))
            else:
                stacked.append(masked_like(shape))

        dims = (
            "time",
            *tuple(
                template[name].dims
                if template is not None and name in template.data_vars
                else tuple(f"dim_{i}" for i in range(len(shape)))
            ),
        )
        attrs = next((dict(f[name].attrs) for f in frames if name in f.data_vars), {})
        data_vars[name] = (dims, np.stack(stacked, axis=0), attrs)

    if not data_vars:
        return None

    coords: dict[str, Any] = {
        "time": (
            "time",
            np.array(
                [t.replace(tzinfo=None) for t in timestamps], dtype="datetime64[ns]"
            ),
            {"standard_name": "time"},
        )
    }
    if template is not None:
        geo = _stack_geolocation(frames, template, stream)
        coords.update(geo)
        for coord in ("level", "x", "y"):
            if coord in template.coords:
                coords[coord] = (
                    template[coord].dims,
                    template[coord].values,
                    dict(template[coord].attrs),
                )

    attrs = dict(template.attrs) if template is not None else {}
    attrs["stream"] = stream
    attrs["sequence_length"] = len(timestamps)
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def _stack_geolocation(
    frames: list[xr.Dataset], template: xr.Dataset, stream: str
) -> dict[str, Any]:
    """Carry geolocation into the stack, per frame when it actually varies."""
    coords: dict[str, Any] = {}
    if "latitude" not in template.coords or "longitude" not in template.coords:
        return coords

    reference_shape = tuple(template["latitude"].shape)
    if template["latitude"].ndim != 2:
        # 1-D separable axes belong to a regular grid, which parallax never
        # touches; carry them through unchanged.
        for coord in ("latitude", "longitude"):
            coords[coord] = (
                template[coord].dims,
                template[coord].values,
                dict(template[coord].attrs),
            )
        return coords

    usable = [
        f
        for f in frames
        if "latitude" in f.coords
        and "longitude" in f.coords
        and tuple(f["latitude"].shape) == reference_shape
    ]
    if len(usable) != len(frames):
        logger.info(
            "%s: %d of %d frames carry usable geolocation; using the shared "
            "grid from the reference frame",
            stream,
            len(usable),
            len(frames),
        )
        for coord in ("latitude", "longitude"):
            coords[coord] = (
                ("y", "x"),
                np.asarray(template[coord].values, dtype=np.float64),
                dict(template[coord].attrs),
            )
        return coords

    lat_stack = np.stack(
        [np.asarray(f["latitude"].values, dtype=np.float64) for f in usable]
    )
    lon_stack = np.stack(
        [np.asarray(f["longitude"].values, dtype=np.float64) for f in usable]
    )

    varies = not (
        np.allclose(lat_stack, lat_stack[0], equal_nan=True)
        and np.allclose(lon_stack, lon_stack[0], equal_nan=True)
    )

    if varies:
        spread_km = float(np.nanmax(np.abs(lat_stack - lat_stack[0])) * 111.32)
        logger.info(
            "%s: geolocation varies across the window (max %.2f km between "
            "frames); stacking per-frame coordinates",
            stream,
            spread_km,
        )
        coords["latitude"] = (
            ("time", "y", "x"),
            lat_stack,
            dict(template["latitude"].attrs),
        )
        coords["longitude"] = (
            ("time", "y", "x"),
            lon_stack,
            dict(template["longitude"].attrs),
        )
    else:
        coords["latitude"] = (
            ("y", "x"),
            lat_stack[0],
            dict(template["latitude"].attrs),
        )
        coords["longitude"] = (
            ("y", "x"),
            lon_stack[0],
            dict(template["longitude"].attrs),
        )
    return coords


def _frame_flags(frames: list[xr.Dataset]) -> list[QCFlag]:
    return [QCFlag(int(f.attrs.get(ATTR_QC_FLAG, 0))) for f in frames]


# ---------------------------------------------------------------------------
# Per-stream collection
# ---------------------------------------------------------------------------


def _collect_insat(
    config: IngestionConfig,
    qc_config: QualityControlConfig,
    timestamps: list[datetime],
) -> tuple[list[xr.Dataset], list[QCFlag]]:
    source = config.sources.insat
    tolerance = config.temporal.alignment_tolerance_minutes
    lut_interpolation = qc_config.radiometric_calibration.lut_interpolation

    frames: list[xr.Dataset] = []
    for nominal in timestamps:
        path = insat3d.locate_granule(source, nominal)
        if path is not None:
            acquired = insat3d.parse_granule_time(path)
            if not _within_tolerance(acquired, nominal, tolerance):
                logger.warning(
                    "granule %s acquired at %s is outside the %d-minute "
                    "tolerance for slot %s; treating the slot as missing",
                    path.name,
                    acquired.isoformat() if acquired else "unknown",
                    tolerance,
                    nominal.isoformat(),
                )
                path = None

        frame = (
            insat3d.read_granule(path, source, nominal, lut_interpolation)
            if path is not None
            else insat3d.read_for_time(source, nominal, lut_interpolation)
        )
        frames.append(
            qc_module.apply_frame_qc(
                frame, qc_config, SourceStream.INSAT, ctt_variable="insat_ctt"
            )
        )
    return frames, _frame_flags(frames)


def _collect_imdaa(
    config: IngestionConfig,
    qc_config: QualityControlConfig,
    timestamps: list[datetime],
) -> tuple[list[xr.Dataset], list[QCFlag]]:
    source = config.sources.imdaa
    # A whole native interval, because the reader deliberately snaps back to
    # the analysis at or before the slot rather than interpolating between two.
    tolerance = source.native_interval_minutes

    frames = [
        qc_module.apply_frame_qc(
            imdaa.read_for_time(source, nominal, tolerance),
            qc_config,
            SourceStream.IMDAA,
        )
        for nominal in timestamps
    ]
    return frames, _frame_flags(frames)


def _collect_surface(
    config: IngestionConfig,
    qc_config: QualityControlConfig,
    timestamps: list[datetime],
) -> tuple[dict[str, list[xr.Dataset]], list[QCFlag]]:
    source = config.sources.imd_surface
    per_variable: dict[str, list[xr.Dataset]] = {name: [] for name in source.variables}
    combined: list[QCFlag] = []

    for nominal in timestamps:
        # Tolerance is the coarsest native cadence present, so a 3-hourly AWS
        # field is not rejected for failing to land on a 30-minute slot.
        tolerance = max(
            (spec.native_interval_minutes for spec in source.variables.values()),
            default=180,
        )
        datasets = imd_surface.read_for_time(source, nominal, tolerance)
        flags = QCFlag.OK
        for name, dataset in datasets.items():
            checked = qc_module.apply_frame_qc(
                dataset, qc_config, SourceStream.IMD_SURFACE
            )
            per_variable.setdefault(name, []).append(checked)
            flags |= QCFlag(int(checked.attrs.get(ATTR_QC_FLAG, 0)))
        combined.append(flags)

    return per_variable, combined


# ---------------------------------------------------------------------------
# Gap filling
# ---------------------------------------------------------------------------


def _reference_index(dataset: xr.Dataset, preferred: str) -> int:
    """Position of the channel that drives the motion field."""
    names = list(dataset.data_vars)
    if preferred in names:
        return names.index(preferred)
    logger.warning(
        "flow reference channel %r absent; falling back to %r",
        preferred,
        names[0] if names else "none",
    )
    return 0


def _fill_stream(
    dataset: xr.Dataset | None,
    flags: list[QCFlag],
    qc_config: QualityControlConfig,
    stream: str,
) -> tuple[xr.Dataset | None, list[QCFlag], dict[str, Any]]:
    """Reconstruct missing frames in one stacked stream."""
    if dataset is None:
        return None, flags, {"assessment": "stream absent"}

    observed = np.array([f.is_observed for f in flags], dtype=bool)
    if observed.all():
        return dataset, flags, {"assessment": "complete", "n_filled": 0}

    names = list(dataset.data_vars)
    if not names:
        return dataset, flags, {"assessment": "no variables"}

    # Stack the variables into (T, C, H, W). Variables with extra axes -- the
    # IMDAA vertical coordinate -- are folded into the channel axis and
    # restored afterwards, so one flow field still governs the whole column.
    blocks: list[np.ndarray] = []
    layout: list[tuple[str, tuple[int, ...], int]] = []
    for name in names:
        values = np.asarray(dataset[name].values, dtype=np.float32)
        leading, trailing = values.shape[0], values.shape[-2:]
        middle = values.shape[1:-2]
        n_planes = int(np.prod(middle)) if middle else 1
        blocks.append(values.reshape(leading, n_planes, *trailing))
        layout.append((name, middle, n_planes))

    stack = np.concatenate(blocks, axis=1)

    use_flow = stream in FLOW_STREAMS
    config = qc_config.gap_filling
    if not use_flow and config.strategy.value == "optical_flow":
        # Advecting a model analysis or a daily accumulation along a motion
        # field estimated from it is not meaningful; there is no coherent
        # feature to track. Those streams take the spline/hold path.
        config = config.model_copy(update={"strategy": type(config.strategy).SPLINE})

    reference = _reference_index(dataset, FLOW_REFERENCE_CHANNEL) if use_flow else 0
    filled, new_flags, report = qc_module.gapfill.fill_stack(
        stack, observed, config, reference_channel=reference
    )
    report["stream"] = stream
    report["flow_reference"] = (
        FLOW_REFERENCE_CHANNEL if use_flow else "n/a (hold/spline)"
    )

    out = dataset.copy(deep=True)
    offset = 0
    for name, _middle, n_planes in layout:
        block = filled[:, offset : offset + n_planes]
        offset += n_planes
        out[name] = (
            dataset[name].dims,
            block.reshape(dataset[name].shape).astype(np.float32),
            dict(dataset[name].attrs),
        )

    merged = [old | new for old, new in zip(flags, new_flags, strict=False)]
    return out, merged, report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def synchronise(
    config: IngestionConfig,
    qc_config: QualityControlConfig,
    valid_time: datetime,
) -> SyncedWindow:
    """Assemble one quality-controlled lookback window."""
    timestamps = build_timestamps(valid_time, config)
    logger.info(
        "synchronising %d slots, %s .. %s",
        len(timestamps),
        timestamps[0].isoformat(),
        timestamps[-1].isoformat(),
    )

    flags: dict[str, tuple[QCFlag, ...]] = {}
    observed: dict[str, np.ndarray] = {}
    reports: dict[str, dict[str, Any]] = {}

    # -- satellite ---------------------------------------------------------
    satellite: xr.Dataset | None = None
    if config.sources.insat.enabled:
        frames, frame_flags = _collect_insat(config, qc_config, timestamps)
        satellite = _stack_frames(frames, timestamps, SourceStream.INSAT.value)
        satellite, frame_flags, report = _fill_stream(
            satellite, frame_flags, qc_config, SourceStream.INSAT.value
        )
        flags[SourceStream.INSAT.value] = tuple(frame_flags)
        observed[SourceStream.INSAT.value] = np.array(
            [f.is_observed for f in frame_flags], dtype=bool
        )
        reports[SourceStream.INSAT.value] = report

    # -- reanalysis --------------------------------------------------------
    nwp: xr.Dataset | None = None
    if config.sources.imdaa.enabled:
        frames, frame_flags = _collect_imdaa(config, qc_config, timestamps)
        nwp = _stack_frames(frames, timestamps, SourceStream.IMDAA.value)
        nwp, frame_flags, report = _fill_stream(
            nwp, frame_flags, qc_config, SourceStream.IMDAA.value
        )
        flags[SourceStream.IMDAA.value] = tuple(frame_flags)
        observed[SourceStream.IMDAA.value] = np.array(
            [f.is_observed for f in frame_flags], dtype=bool
        )
        reports[SourceStream.IMDAA.value] = report

    # -- surface -----------------------------------------------------------
    surface: dict[str, xr.Dataset] = {}
    if config.sources.imd_surface.enabled:
        per_variable, frame_flags = _collect_surface(config, qc_config, timestamps)
        for name, frames in per_variable.items():
            stacked = _stack_frames(frames, timestamps, SourceStream.IMD_SURFACE.value)
            if stacked is not None:
                surface[name] = stacked
        flags[SourceStream.IMD_SURFACE.value] = tuple(frame_flags)
        observed[SourceStream.IMD_SURFACE.value] = np.array(
            [f.is_observed for f in frame_flags], dtype=bool
        )

    # -- static ------------------------------------------------------------
    static: dict[str, xr.Dataset] = {}
    if config.sources.static_priors.enabled:
        # No time axis: read once from the process cache and broadcast in
        # Stage 2 rather than stored T times.
        static = static_priors.read_all(config.sources.static_priors)
        static_flag = static_priors.combined_flags(static)
        flags[SourceStream.STATIC_PRIORS.value] = (static_flag,)
        observed[SourceStream.STATIC_PRIORS.value] = np.array(
            [static_flag.is_observed], dtype=bool
        )

    # -- acceptance --------------------------------------------------------
    # A window is rejected on the dynamic streams only. Static priors are
    # time-invariant, so a missing DEM is a deployment fault rather than a
    # reason to discard this particular window.
    accepted = True
    rejection: str | None = None
    for stream in (SourceStream.INSAT.value, SourceStream.IMDAA.value):
        if stream not in observed:
            continue
        ok, reason = qc_module.gapfill.assess(observed[stream], qc_config.gap_filling)
        if not ok:
            accepted = False
            rejection = f"{stream}: {reason}"
            logger.warning("window %s rejected — %s", valid_time.isoformat(), rejection)
            break

    window = SyncedWindow(
        valid_time=valid_time,
        timestamps=tuple(timestamps),
        lookback_indices=tuple(config.temporal.lookback_indices),
        interval_minutes=config.temporal.interval_minutes,
        satellite=satellite,
        nwp=nwp,
        surface=surface,
        static=static,
        flags=flags,
        observed=observed,
        gapfill=reports,
        accepted=accepted,
        rejection_reason=rejection,
    )

    logger.info(
        "window %s assembled: %s",
        valid_time.isoformat(),
        ", ".join(
            f"{name} {window.observed_fraction(name) * 100:.0f}% observed"
            for name in window.flags
        ),
    )
    return window


__all__ = [
    "FLOW_REFERENCE_CHANNEL",
    "FLOW_STREAMS",
    "build_timestamps",
    "synchronise",
]
