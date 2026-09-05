"""INSAT-3D / 3DR Hydro Estimator Method (HEM) target reader."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import h5py
import numpy as np
import xarray as xr

from ..config import HemTargetSource, TemporalConfig
from ..types import (
    ATTR_CALIBRATION,
    ATTR_GRANULE_PATH,
    ATTR_QC_FLAG,
    ATTR_QC_FLAG_NAMES,
    ATTR_SOURCE_STREAM,
    ATTR_VALID_TIME,
    Calibration,
    QCFlag,
    masked_like,
)
from ._hdf5 import mask_out_of_range, read_geolocation, read_scaled

logger = logging.getLogger(__name__)

#: Stream label for the target, deliberately *not* a ``SourceStream`` member.
#: ``SourceStream`` enumerates the four model inputs, and
#: ``replay.policy.required_streams`` validates against it. Adding HEM there
#: would make it a spellable value for the replay gate, which checks that the
#: *inputs* were genuinely observed -- requiring the target in that gate is a
#: category error, and the schema should not permit writing one.
TARGET_STREAM: Final[str] = "hem"

# MOSDAC L2B naming, e.g.
#   3DIMG_23SEP2017_0600_L2B_HEM_V01R00.h5      (INSAT-3D)
#   3RIMG_23SEP2017_0600_L2B_HEM_V01R00.h5      (INSAT-3DR)
# Only the instrument prefix and the timestamp are load-bearing; the version
# and revision suffixes vary between reprocessings of the same slot.
_GRANULE_RE = re.compile(
    r"^(?P<sat>3[A-Z])IMG_"
    r"(?P<day>\d{2})(?P<mon>[A-Z]{3})(?P<year>\d{4})_"
    r"(?P<hour>\d{2})(?P<minute>\d{2})",
    re.IGNORECASE,
)

_MONTHS: dict[str, int] = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

_MONTH_ABBR: tuple[str, ...] = (
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
)


# ---------------------------------------------------------------------------
# Filename and timestamp handling
# ---------------------------------------------------------------------------


def parse_granule_time(filename: str | Path) -> datetime | None:
    """Extract the nominal scan start time from a MOSDAC granule name."""
    match = _GRANULE_RE.match(Path(filename).name)
    if match is None:
        return None
    month = _MONTHS.get(match.group("mon").upper())
    if month is None:
        return None
    try:
        return datetime(
            year=int(match.group("year")),
            month=month,
            day=int(match.group("day")),
            hour=int(match.group("hour")),
            minute=int(match.group("minute")),
            tzinfo=UTC,
        )
    except ValueError:
        # Syntactically well-formed but impossible, e.g. 31FEB.
        return None


def build_granule_glob(target: HemTargetSource, valid_time: datetime) -> str:
    """Render the configured filename pattern for one timestamp."""
    return target.filename_pattern.format(
        day=valid_time.day,
        month_abbr=_MONTH_ABBR[valid_time.month - 1],
        year=valid_time.year,
        hour=valid_time.hour,
        minute=valid_time.minute,
    )


def locate_granule(target: HemTargetSource, valid_time: datetime) -> Path | None:
    """Find the HEM granule for a nominal timestamp, or ``None`` if absent."""
    root = Path(target.root)
    pattern = build_granule_glob(target, valid_time)
    try:
        matches = sorted(root.glob(pattern))
    except OSError as exc:
        logger.warning("cannot list %s for %s: %s", root, valid_time.isoformat(), exc)
        return None

    if not matches:
        return None
    if len(matches) > 1:
        # Reprocessed granules coexist with the originals; for MOSDAC names the
        # lexicographically last is the later version.
        logger.info(
            "%d HEM granules match %s; using %s",
            len(matches),
            pattern,
            matches[-1].name,
        )
    return matches[-1]


def lead_times(valid_time: datetime, temporal: TemporalConfig) -> list[datetime]:
    """The forecast timestamps this window must be verified against."""
    if valid_time.tzinfo is None:
        raise ValueError(
            f"valid_time {valid_time!r} is naive; lead times must be anchored "
            "to UTC or the granule lookup will resolve the wrong slot"
        )
    step = timedelta(minutes=temporal.interval_minutes)
    return [valid_time + index * step for index in temporal.lead_indices]


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def _dataset_attrs(
    target: HemTargetSource,
    valid_time: datetime,
    flags: QCFlag,
    granule_path: Path | None,
) -> dict[str, Any]:
    return {
        ATTR_SOURCE_STREAM: TARGET_STREAM,
        ATTR_VALID_TIME: valid_time.isoformat(),
        ATTR_GRANULE_PATH: str(granule_path) if granule_path else "",
        ATTR_QC_FLAG: int(flags),
        ATTR_QC_FLAG_NAMES: ",".join(flags.describe()),
        "subsatellite_lon": target.geolocation.subsatellite_lon,
        "satellite_altitude_km": target.geolocation.satellite_altitude_km,
        "role": "target",
    }


def _empty_frame(
    target: HemTargetSource,
    valid_time: datetime,
    flags: QCFlag,
    shape: tuple[int, int] | None = None,
    granule_path: Path | None = None,
) -> xr.Dataset:
    """An all-missing frame for a granule that could not be read."""
    dims = shape if shape is not None else target.native_shape
    variable = target.variable
    return xr.Dataset(
        data_vars={
            variable.name: (
                ("y", "x"),
                masked_like(dims),
                {
                    "units": variable.units,
                    "long_name": variable.description,
                    ATTR_CALIBRATION: Calibration.L2_PRODUCT.value,
                    ATTR_QC_FLAG: int(flags),
                    ATTR_QC_FLAG_NAMES: ",".join(flags.describe()),
                },
            )
        },
        attrs=_dataset_attrs(target, valid_time, flags, granule_path),
    )


def read_granule(
    path: str | Path,
    target: HemTargetSource,
    valid_time: datetime,
    expected_shape: tuple[int, int] | None = None,
) -> xr.Dataset:
    """Read one HEM granule into a native-grid dataset."""
    path = Path(path)
    variable = target.variable
    if expected_shape is None:
        expected_shape = target.native_shape

    try:
        handle = h5py.File(path, "r")
    except (OSError, RuntimeError) as exc:
        logger.warning("cannot open HEM granule %s: %s", path, exc)
        return _empty_frame(
            target, valid_time, QCFlag.CORRUPT_OR_MISSING, expected_shape, path
        )

    flags = QCFlag.OK
    try:
        if variable.path not in handle:
            logger.warning(
                "HEM granule %s has no dataset at %s", path.name, variable.path
            )
            return _empty_frame(
                target, valid_time, QCFlag.VARIABLE_ABSENT, expected_shape, path
            )

        field = np.squeeze(read_scaled(handle[variable.path])).astype(np.float32)

        if field.ndim != 2:
            logger.warning(
                "HEM field in %s is %d-dimensional after squeezing; expected 2",
                path.name,
                field.ndim,
            )
            return _empty_frame(
                target, valid_time, QCFlag.CORRUPT_OR_MISSING, expected_shape, path
            )

        if field.shape != tuple(target.native_shape):
            # The declared shape is the domain subset this pipeline ordered. A
            # different one means the granule covers a different area, which is
            # what PARTIAL_COVERAGE denotes -- not absence.
            logger.warning(
                "HEM granule %s is %s, but %s was declared; treating as partial "
                "coverage",
                path.name,
                field.shape,
                tuple(target.native_shape),
            )
            flags |= QCFlag.PARTIAL_COVERAGE

        # Sentinel fill declared in configuration, distinct from any
        # _FillValue attribute already honoured during unpacking. Some MOSDAC
        # releases carry one and not the other.
        field = np.where(field == np.float32(variable.fill_value), np.nan, field)

        field, out_of_range = mask_out_of_range(field, variable.valid_range)
        if out_of_range:
            flags |= QCFlag.OUT_OF_PHYSICAL_RANGE

        lat, lon, geo_flags = read_geolocation(
            handle, target.geolocation, field.shape, label="HEM granule"
        )
        flags |= geo_flags

        # Recorded, not performed. Correction needs the viewing geometry that
        # exists only on the native grid, and for HEM it is mandatory: the
        # retrieval reports rain at the observed cloud-top pixel, not at the
        # ground location beneath it.
        flags |= QCFlag.PARALLAX_UNCORRECTED

        if not np.isfinite(field).any():
            # Readable and entirely empty. Not a fatal flag: the frame is
            # genuinely what the instrument reported, and the coverage gate in
            # the dataset layer decides whether a window with no valid target
            # cells is worth training on.
            logger.info("HEM granule %s carries no finite rain rates", path.name)

        coords: dict[str, Any] = {}
        if lat is not None and lon is not None:
            coords["latitude"] = (
                ("y", "x"),
                lat,
                {"units": "degrees_north", "standard_name": "latitude"},
            )
            coords["longitude"] = (
                ("y", "x"),
                lon,
                {"units": "degrees_east", "standard_name": "longitude"},
            )

        dataset = xr.Dataset(
            data_vars={
                variable.name: (
                    ("y", "x"),
                    field,
                    {
                        "units": variable.units,
                        "long_name": variable.description,
                        "valid_range": list(variable.valid_range),
                        # No lookup table is involved: HEM arrives as a
                        # retrieved geophysical field, not as packed counts.
                        ATTR_CALIBRATION: Calibration.L2_PRODUCT.value,
                        ATTR_QC_FLAG: int(flags),
                        ATTR_QC_FLAG_NAMES: ",".join(flags.describe()),
                    },
                )
            },
            coords=coords,
            attrs={
                **_dataset_attrs(target, valid_time, flags, path),
                "scan_start_time": (
                    t.isoformat() if (t := parse_granule_time(path)) else ""
                ),
                "grid": "native_curvilinear_geostationary",
            },
        )

        if flags is not QCFlag.OK:
            logger.info(
                "HEM granule %s read with flags: %s",
                path.name,
                ", ".join(flags.describe()),
            )
        return dataset

    except Exception as exc:
        logger.exception("unhandled error reading HEM granule %s: %s", path, exc)
        return _empty_frame(
            target,
            valid_time,
            flags | QCFlag.CORRUPT_OR_MISSING,
            expected_shape,
            path,
        )
    finally:
        handle.close()


def read_for_time(
    target: HemTargetSource,
    valid_time: datetime,
    expected_shape: tuple[int, int] | None = None,
) -> xr.Dataset:
    """Locate and read the HEM granule for one forecast timestamp."""
    if expected_shape is None:
        expected_shape = target.native_shape
    path = locate_granule(target, valid_time)
    if path is None:
        logger.warning(
            "no HEM granule for %s under %s",
            valid_time.isoformat(),
            target.root,
        )
        return _empty_frame(target, valid_time, QCFlag.MISSING_FILE, expected_shape)
    return read_granule(path, target, valid_time, expected_shape)


# ---------------------------------------------------------------------------
# The forward sequence
# ---------------------------------------------------------------------------


def read_lead_sequence(
    target: HemTargetSource,
    temporal: TemporalConfig,
    valid_time: datetime,
    expected_shape: tuple[int, int] | None = None,
) -> list[xr.Dataset]:
    """Read every forecast frame for one nowcast time, in lead order."""
    if not target.enabled:
        raise ValueError(
            "the HEM target is disabled in configuration; there is no ground "
            "truth to read and no training sample can be built"
        )

    frames = [
        read_for_time(target, moment, expected_shape)
        for moment in lead_times(valid_time, temporal)
    ]

    accepted, reason = sequence_status(frames)
    if not accepted:
        logger.info(
            "target sequence for t0=%s rejected: %s", valid_time.isoformat(), reason
        )
    return frames


def frame_flags(frames: list[xr.Dataset]) -> tuple[QCFlag, ...]:
    """Per-frame quality-control flags, in lead order."""
    return tuple(QCFlag(int(frame.attrs.get(ATTR_QC_FLAG, 0))) for frame in frames)


def frame_observed(frames: list[xr.Dataset]) -> tuple[bool, ...]:
    """Per-frame observation mask, in lead order."""
    return tuple(flag.is_observed for flag in frame_flags(frames))


def sequence_status(frames: list[xr.Dataset]) -> tuple[bool, str | None]:
    """Whether a target sequence may be used, and why not when it may not."""
    if not frames:
        return False, "no target frames were read"

    flags = frame_flags(frames)
    for index, flag in enumerate(flags, start=1):
        if not flag.is_usable:
            return False, (
                f"lead frame t+{index} is unusable ({', '.join(flag.describe())}); "
                "targets are never gap-filled, so the window is discarded"
            )
        if not flag.is_observed:
            # Unreachable through this reader, which never reconstructs
            # anything. Checked because a target that reached here
            # reconstructed by some future path would otherwise be verified
            # against silently, and that failure is undetectable downstream.
            return False, (
                f"lead frame t+{index} is reconstructed rather than observed "
                f"({', '.join(flag.describe())}); a target must be evidence"
            )
    return True, None


__all__ = [
    "TARGET_STREAM",
    "build_granule_glob",
    "frame_flags",
    "frame_observed",
    "lead_times",
    "locate_granule",
    "parse_granule_time",
    "read_for_time",
    "read_granule",
    "read_lead_sequence",
    "sequence_status",
]
