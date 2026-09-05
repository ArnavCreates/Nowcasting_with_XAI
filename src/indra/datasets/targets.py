"""Assemble the HEM ground truth: native granules to a model-grid target."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import xarray as xr

from ..config import IngestionConfig, PreprocessingConfig, Resampling
from ..ingestion import hem, insat3d
from ..ingestion.qc import parallax
from ..preprocessing.reprojection import (
    build_target_area,
    radius_of_influence_m,
    resample_swath,
)
from ..types import (
    ATTR_QC_FLAG,
    FloatArray,
    QCFlag,
    TargetWindow,
    masked_like,
)

logger = logging.getLogger(__name__)

#: Conditions that are survivable for an input frame and fatal for a target.
#:
#: ``GEOLOCATION_MISSING`` is the whole set. An input frame without coordinates
#: can be dropped and reconstructed from its neighbours; a target frame without
#: coordinates cannot be placed on the model grid at all, and there is no
#: reconstruction path to fall back on. An unplaceable target is a hole, and
#: holes are what this module exists to refuse.
_FATAL_FOR_TARGET = QCFlag.GEOLOCATION_MISSING

#: The variable ``qc/parallax`` reads cloud-top temperature from.
_CTT_VARIABLE = "insat_ctt"


# ---------------------------------------------------------------------------
# Parallax
# ---------------------------------------------------------------------------


def _attach_cloud_top(
    frame: xr.Dataset,
    insat_frame: xr.Dataset,
    target_var: str,
    ctt_variable: str = _CTT_VARIABLE,
) -> xr.Dataset | None:
    """Copy the CTT field onto a HEM frame so parallax can be computed."""
    if ctt_variable not in insat_frame.data_vars:
        logger.warning(
            "INSAT frame carries no %s; parallax cannot be computed for the " "target",
            ctt_variable,
        )
        return None

    ctt = np.asarray(insat_frame[ctt_variable].values, dtype=np.float32)
    hem_shape = frame[target_var].shape

    if ctt.shape != hem_shape:
        logger.error(
            "CTT grid %s does not match the HEM grid %s; refusing to correct "
            "parallax against a misaligned cloud field",
            ctt.shape,
            hem_shape,
        )
        return None

    out = frame.copy()
    out[ctt_variable] = (
        frame[target_var].dims,
        ctt,
        dict(insat_frame[ctt_variable].attrs),
    )
    return out


def _correct_parallax(
    frame: xr.Dataset,
    ingestion: IngestionConfig,
    moment: datetime,
    target_var: str,
) -> tuple[xr.Dataset, bool]:
    """Apply parallax to one target frame. Returns the frame and whether it worked."""
    config = ingestion.quality_control.parallax_correction
    if not config.enabled:
        # Inputs are uncorrected too, so the geometry is consistent, which is
        # the property that actually matters.
        return frame, True

    insat_frame = insat3d.read_for_time(ingestion.sources.insat, moment)
    insat_flags = QCFlag(int(insat_frame.attrs.get(ATTR_QC_FLAG, 0)))
    if not insat_flags.is_usable:
        logger.warning(
            "no usable INSAT granule at %s; the target frame cannot be "
            "parallax-corrected",
            moment.isoformat(),
        )
        return frame, False

    merged = _attach_cloud_top(frame, insat_frame, target_var)
    if merged is None:
        return frame, False

    corrected = parallax.apply(merged, config, ctt_variable=_CTT_VARIABLE)
    if corrected.attrs.get("parallax_corrected") != 1:
        return frame, False

    # The borrowed cloud field has served its purpose. Dropping it keeps the
    # frame single-variable, so nothing downstream can mistake a target window
    # for something that carries satellite channels.
    return corrected.drop_vars(_CTT_VARIABLE, errors="ignore"), True


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _resample_frame(
    frame: xr.Dataset,
    variable: str,
    target_area: Any,
) -> FloatArray | None:
    """One native-grid frame onto the model grid, by nearest neighbour."""
    if "latitude" not in frame.coords or "longitude" not in frame.coords:
        return None

    lat = np.asarray(frame["latitude"].values, dtype=np.float64)
    lon = np.asarray(frame["longitude"].values, dtype=np.float64)
    field = np.asarray(frame[variable].values, dtype=np.float32)

    if lat.shape != field.shape or lon.shape != field.shape:
        logger.error(
            "target geolocation %s does not match the field %s",
            lat.shape,
            field.shape,
        )
        return None

    # Measured from the source grid rather than configured, so it stays
    # correct if the product's native resolution ever changes.
    radius = radius_of_influence_m(lat, lon)
    return resample_swath(field, lat, lon, target_area, Resampling.NEAREST, radius)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _rejected(
    valid_time: datetime,
    timestamps: tuple[datetime, ...],
    ingestion: IngestionConfig,
    preprocessing: PreprocessingConfig,
    reason: str,
    flags: tuple[QCFlag, ...] = (),
) -> TargetWindow:
    """A correctly-shaped, entirely masked window carrying its rejection."""
    grid = preprocessing.target_grid
    n_lead = len(ingestion.temporal.lead_indices)
    shape = (n_lead, 1, grid.height, grid.width)
    logger.info("target window %s rejected: %s", valid_time.isoformat(), reason)
    return TargetWindow(
        valid_time=valid_time,
        timestamps=timestamps,
        lead_indices=tuple(ingestion.temporal.lead_indices),
        interval_minutes=ingestion.temporal.interval_minutes,
        rain_rate_mm_h=masked_like(shape),
        validity=np.zeros(shape, dtype=np.bool_),
        flags=flags,
        observed=tuple(False for _ in range(n_lead)),
        grid=_grid_dict(preprocessing),
        accepted=False,
        rejection_reason=reason,
    )


def _grid_dict(preprocessing: PreprocessingConfig) -> dict[str, float]:
    grid = preprocessing.target_grid
    return {
        "lat_min": grid.lat_min,
        "lat_max": grid.lat_max,
        "lon_min": grid.lon_min,
        "lon_max": grid.lon_max,
        "resolution_deg": grid.resolution_deg,
        "height": float(grid.height),
        "width": float(grid.width),
    }


def build_target_window(
    ingestion: IngestionConfig,
    preprocessing: PreprocessingConfig,
    valid_time: datetime,
) -> TargetWindow:
    """Read, correct, resample and stack the ground truth for one nowcast time."""
    target = ingestion.targets.hem
    timestamps = tuple(hem.lead_times(valid_time, ingestion.temporal))

    if not target.enabled:
        return _rejected(
            valid_time,
            timestamps,
            ingestion,
            preprocessing,
            "the HEM target stream is disabled in configuration",
        )

    frames = hem.read_lead_sequence(target, ingestion.temporal, valid_time)
    flags = hem.frame_flags(frames)

    accepted, reason = hem.sequence_status(frames)
    if not accepted:
        return _rejected(
            valid_time,
            timestamps,
            ingestion,
            preprocessing,
            reason or "target sequence unusable",
            flags,
        )

    for index, flag in enumerate(flags, start=1):
        if flag & _FATAL_FOR_TARGET:
            return _rejected(
                valid_time,
                timestamps,
                ingestion,
                preprocessing,
                f"lead frame t+{index} has no usable geolocation "
                f"({', '.join((flag & _FATAL_FOR_TARGET).describe())}); "
                "it cannot be placed on the model grid and targets are never "
                "reconstructed",
                flags,
            )

    target_area = build_target_area(preprocessing.target_grid)
    variable = target.variable.name
    grid = preprocessing.target_grid

    resampled: list[FloatArray] = []
    corrected_flags: list[QCFlag] = []

    for index, (frame, moment) in enumerate(
        zip(frames, timestamps, strict=False), start=1
    ):
        frame, ok = _correct_parallax(frame, ingestion, moment, variable)
        if not ok:
            return _rejected(
                valid_time,
                timestamps,
                ingestion,
                preprocessing,
                f"lead frame t+{index} could not be parallax-corrected. The "
                "inputs are corrected, so an uncorrected target would train "
                "the model on a systematic displacement between what it sees "
                "and what it is scored against",
                flags,
            )

        field = _resample_frame(frame, variable, target_area)
        if field is None:
            return _rejected(
                valid_time,
                timestamps,
                ingestion,
                preprocessing,
                f"lead frame t+{index} could not be resampled onto the model " "grid",
                flags,
            )
        resampled.append(field)
        corrected_flags.append(QCFlag(int(frame.attrs.get(ATTR_QC_FLAG, 0))))

    # (T_out, H, W) -> (T_out, 1, H, W). The channel axis is explicit rather
    # than implied so the array pairs directly with the generator's
    # (B, T_out, 1, H, W) output and nothing has to guess where to unsqueeze.
    stacked = np.stack(resampled).astype(np.float32)[:, np.newaxis, :, :]
    validity = np.isfinite(stacked)

    if stacked.shape != (len(timestamps), 1, grid.height, grid.width):
        return _rejected(
            valid_time,
            timestamps,
            ingestion,
            preprocessing,
            f"assembled target is {stacked.shape}, expected "
            f"{(len(timestamps), 1, grid.height, grid.width)}",
            flags,
        )

    window = TargetWindow(
        valid_time=valid_time,
        timestamps=timestamps,
        lead_indices=tuple(ingestion.temporal.lead_indices),
        interval_minutes=ingestion.temporal.interval_minutes,
        rain_rate_mm_h=stacked,
        validity=validity,
        units=target.variable.units,
        flags=tuple(corrected_flags),
        observed=hem.frame_observed(frames),
        grid=_grid_dict(preprocessing),
        accepted=True,
    )

    logger.info(
        "target window %s assembled: %.1f%% of cells carry a retrieval",
        valid_time.isoformat(),
        100.0 * window.valid_fraction,
    )
    return window


__all__ = ["build_target_window"]
