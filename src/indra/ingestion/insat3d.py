"""INSAT-3D / 3DS Level-1C and Level-2B granule reader."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr

from ..config import InsatDataset, InsatSource
from ..types import (
    ATTR_CALIBRATION,
    ATTR_GRANULE_PATH,
    ATTR_QC_FLAG,
    ATTR_QC_FLAG_NAMES,
    ATTR_SOURCE_STREAM,
    ATTR_VALID_TIME,
    Calibration,
    FloatArray,
    QCFlag,
    SourceStream,
    masked_like,
)
from ._hdf5 import mask_out_of_range, read_geolocation, read_scaled

logger = logging.getLogger(__name__)

# MOSDAC granule naming, e.g.
#   3DIMG_23SEP2017_0600_L1C_ASIA_MER.h5      (INSAT-3D)
#   3SIMG_23SEP2017_0600_L1C_ASIA_MER.h5      (INSAT-3DS)
# Product code, sector and projection suffix vary by request; only the
# instrument prefix and the timestamp are load-bearing here.
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

#: INSAT imager channels are 10-bit, so counts span 0..1023 and a
#: count-indexed lookup table has exactly this many entries.
_COUNT_LEVELS = 1024

#: Beyond this fraction of a frame sitting at the top of the dynamic range,
#: the scene is treated as detector saturation rather than a very cold cloud.
_SATURATION_FRACTION = 0.98


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
        # A syntactically well-formed but impossible date, e.g. 31FEB.
        return None


def build_granule_glob(source: InsatSource, valid_time: datetime) -> str:
    """Render the configured filename pattern for one timestamp."""
    return source.filename_pattern.format(
        day=valid_time.day,
        month_abbr=_MONTH_ABBR[valid_time.month - 1],
        year=valid_time.year,
        hour=valid_time.hour,
        minute=valid_time.minute,
    )


def locate_granule(source: InsatSource, valid_time: datetime) -> Path | None:
    """Find the granule for a nominal timestamp, or ``None`` if absent."""
    root = Path(source.root)
    pattern = build_granule_glob(source, valid_time)
    try:
        matches = sorted(root.glob(pattern))
    except OSError as exc:
        logger.warning("cannot list %s for %s: %s", root, valid_time.isoformat(), exc)
        return None

    if not matches:
        return None
    if len(matches) > 1:
        # Reprocessed granules coexist with the originals. Sorting puts the
        # lexicographically last first, which for MOSDAC names is the later
        # processing version.
        logger.info(
            "%d granules match %s; using %s",
            len(matches),
            pattern,
            matches[-1].name,
        )
    return matches[-1]


# ---------------------------------------------------------------------------
# Calibration
#
# Attribute reading, unpacking, range masking and geolocation live in
# ``_hdf5.py``: ``hem.py`` reads the same MOSDAC conventions, and two copies of
# unpacking logic diverging by one line is a failure this pipeline could not
# detect. What remains here is specific to the imager -- the count-to-
# temperature lookup table and detector saturation, neither of which applies
# to a Level-2 retrieval.
# ---------------------------------------------------------------------------


def _apply_calibration_lut(
    counts: np.ndarray,
    lut: np.ndarray,
    interpolation: str,
) -> tuple[FloatArray, QCFlag]:
    """Map instrument counts to brightness temperature via the granule's LUT."""
    flags = QCFlag.OK
    table = np.asarray(lut, dtype=np.float32).ravel()
    if table.size == 0:
        return masked_like(counts.shape), QCFlag.CALIBRATION_FAILED

    idx = np.asarray(counts)
    valid = np.isfinite(idx)

    if table.size == _COUNT_LEVELS and interpolation == "nearest":
        gather = np.clip(
            np.nan_to_num(idx, nan=0.0).astype(np.int64), 0, table.size - 1
        )
        out = table[gather].astype(np.float32)
        out[~valid] = np.nan
        out[valid & ((idx < 0) | (idx > table.size - 1))] = np.nan
    else:
        # Linear interpolation across the table's own index axis. Handles both
        # the full 1024-entry table and any subsampled variant.
        positions = np.linspace(0.0, float(_COUNT_LEVELS - 1), num=table.size)
        out = np.interp(
            np.where(valid, idx, np.nan).astype(np.float64),
            positions,
            table.astype(np.float64),
            left=np.nan,
            right=np.nan,
        ).astype(np.float32)

    if not np.isfinite(out).any():
        flags |= QCFlag.CALIBRATION_FAILED
    return out, flags


def _detect_saturation(field: FloatArray, valid_range: tuple[float, float]) -> bool:
    """True when the frame is dominated by top-of-scale values."""
    finite = np.isfinite(field)
    if not finite.any():
        return False
    hi = valid_range[1]
    at_top = np.count_nonzero(field[finite] >= hi)
    return at_top / max(np.count_nonzero(finite), 1) >= _SATURATION_FRACTION


# ---------------------------------------------------------------------------
# Channel reading
# ---------------------------------------------------------------------------


def _read_channel(
    handle: h5py.File,
    name: str,
    spec: InsatDataset,
    lut_interpolation: str,
) -> tuple[FloatArray | None, QCFlag, Calibration]:
    """Read and calibrate one configured channel."""
    flags = QCFlag.OK

    if spec.path not in handle:
        logger.warning("dataset %s (%s) absent from granule", spec.path, name)
        return None, QCFlag.VARIABLE_ABSENT, Calibration.COUNTS

    node = handle[spec.path]
    if not isinstance(node, h5py.Dataset):
        logger.warning("%s is not an HDF5 dataset", spec.path)
        return None, QCFlag.VARIABLE_ABSENT, Calibration.COUNTS

    raw = read_scaled(node)
    # Imager products carry a leading singleton time axis in some MOSDAC
    # releases and not in others.
    raw = np.squeeze(raw)

    # -- calibration -------------------------------------------------------
    if spec.calibration_lut:
        if spec.calibration_lut in handle:
            lut_node = handle[spec.calibration_lut]
            field, lut_flags = _apply_calibration_lut(
                raw, np.asarray(lut_node[...]), lut_interpolation
            )
            flags |= lut_flags
            calibration = (
                Calibration.BRIGHTNESS_TEMPERATURE_K
                if not (lut_flags & QCFlag.CALIBRATION_FAILED)
                else Calibration.COUNTS
            )
        else:
            # Without the table the field stays in counts. Flagged rather than
            # guessed: there is no defensible way to invent a calibration.
            logger.warning(
                "calibration LUT %s absent for %s; leaving field in raw counts",
                spec.calibration_lut,
                name,
            )
            field = raw
            flags |= QCFlag.CALIBRATION_FAILED
            calibration = Calibration.COUNTS
    else:
        # Level-2 geophysical product: already in physical units.
        field = raw
        calibration = Calibration.L2_PRODUCT

    # -- explicit fill sentinel -------------------------------------------
    if np.isfinite(spec.fill_value):
        field = np.where(np.isclose(field, spec.fill_value), np.nan, field).astype(
            np.float32
        )

    # -- plausibility ------------------------------------------------------
    if calibration is not Calibration.COUNTS:
        if _detect_saturation(field, spec.valid_range):
            flags |= QCFlag.SATURATED
        field, clipped = mask_out_of_range(field, spec.valid_range)
        if clipped:
            flags |= QCFlag.OUT_OF_PHYSICAL_RANGE

    if not np.isfinite(field).any():
        flags |= QCFlag.CORRUPT_OR_MISSING

    return field.astype(np.float32, copy=False), flags, calibration


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def _empty_dataset(
    source: InsatSource,
    valid_time: datetime,
    flags: QCFlag,
    shape: tuple[int, int] | None,
    granule_path: Path | None = None,
) -> xr.Dataset:
    """An all-missing dataset for a granule that could not be read."""
    # Fall back to the configured native shape rather than a zero-sized
    # array. A masked frame must have the same dimensions as an observed one,
    # or the gap filler has nothing to write into and the sequence cannot be
    # stacked.
    dims = shape if shape is not None else source.native_shape
    data_vars = {
        name: (
            ("y", "x"),
            masked_like(dims),
            {
                "units": spec.units,
                "long_name": spec.description,
                ATTR_CALIBRATION: Calibration.COUNTS.value,
                ATTR_QC_FLAG: int(flags),
            },
        )
        for name, spec in source.datasets.items()
    }
    return xr.Dataset(
        data_vars=data_vars,
        attrs={
            ATTR_SOURCE_STREAM: SourceStream.INSAT.value,
            ATTR_VALID_TIME: valid_time.isoformat(),
            ATTR_GRANULE_PATH: str(granule_path) if granule_path else "",
            ATTR_QC_FLAG: int(flags),
            ATTR_QC_FLAG_NAMES: ",".join(flags.describe()),
            "subsatellite_lon": source.geolocation.subsatellite_lon,
            "satellite_altitude_km": source.geolocation.satellite_altitude_km,
        },
    )


def read_granule(
    path: str | Path,
    source: InsatSource,
    valid_time: datetime,
    lut_interpolation: str = "linear",
    expected_shape: tuple[int, int] | None = None,
) -> xr.Dataset:
    """Read one INSAT granule into an ``xarray.Dataset``."""
    if expected_shape is None:
        expected_shape = source.native_shape
    path = Path(path)
    flags = QCFlag.OK

    if not path.exists():
        logger.warning("granule not found: %s", path)
        return _empty_dataset(
            source, valid_time, QCFlag.MISSING_FILE, expected_shape, path
        )

    try:
        handle = h5py.File(path, "r")
    except (OSError, KeyError) as exc:
        # Truncated downloads and unreadable superblocks both land here. This
        # is the routine case the log-and-mask policy exists for.
        logger.error("cannot open granule %s: %s", path, exc)
        return _empty_dataset(
            source, valid_time, QCFlag.CORRUPT_OR_MISSING, expected_shape, path
        )

    try:
        channels: dict[str, FloatArray] = {}
        calibrations: dict[str, Calibration] = {}
        channel_flags: dict[str, QCFlag] = {}
        shape: tuple[int, ...] | None = None

        for name, spec in source.datasets.items():
            field, ch_flags, calibration = _read_channel(
                handle, name, spec, lut_interpolation
            )
            channel_flags[name] = ch_flags
            calibrations[name] = calibration
            flags |= ch_flags
            if field is not None:
                channels[name] = field
                if shape is None:
                    shape = field.shape

        if shape is None:
            logger.error("no configured channel could be read from %s", path)
            return _empty_dataset(
                source,
                valid_time,
                flags | QCFlag.CORRUPT_OR_MISSING,
                expected_shape,
                path,
            )

        if len(shape) != 2:
            logger.error("unexpected imagery rank %s in %s", shape, path)
            return _empty_dataset(
                source,
                valid_time,
                flags | QCFlag.CORRUPT_OR_MISSING,
                expected_shape,
                path,
            )

        # Channels absent from this granule become all-missing arrays of the
        # granule's own shape, so the dataset is rectangular regardless.
        for name in source.datasets:
            if name not in channels:
                channels[name] = masked_like(shape)

        # Channels that read but disagree on shape cannot be co-registered.
        for name, field in list(channels.items()):
            if field.shape != shape:
                logger.error(
                    "channel %s has shape %s, expected %s; masking it",
                    name,
                    field.shape,
                    shape,
                )
                channels[name] = masked_like(shape)
                channel_flags[name] = (
                    channel_flags.get(name, QCFlag.OK) | QCFlag.CORRUPT_OR_MISSING
                )
                flags |= QCFlag.CORRUPT_OR_MISSING

        lat, lon, geo_flags = read_geolocation(
            handle, source.geolocation, shape, label="INSAT granule"
        )
        flags |= geo_flags

        # Recorded, not performed: correction needs the viewing geometry that
        # is only available while the data is still on its native grid, and it
        # is applied in the quality-control stage.
        flags |= QCFlag.PARALLAX_UNCORRECTED

        data_vars = {
            name: (
                ("y", "x"),
                field,
                {
                    "units": source.datasets[name].units,
                    "long_name": source.datasets[name].description,
                    "valid_range": list(source.datasets[name].valid_range),
                    ATTR_CALIBRATION: calibrations.get(name, Calibration.COUNTS).value,
                    ATTR_QC_FLAG: int(channel_flags.get(name, QCFlag.OK)),
                    ATTR_QC_FLAG_NAMES: ",".join(
                        channel_flags.get(name, QCFlag.OK).describe()
                    ),
                },
            )
            for name, field in channels.items()
        }

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
            data_vars=data_vars,
            coords=coords,
            attrs={
                ATTR_SOURCE_STREAM: SourceStream.INSAT.value,
                ATTR_VALID_TIME: valid_time.isoformat(),
                ATTR_GRANULE_PATH: str(path),
                ATTR_QC_FLAG: int(flags),
                ATTR_QC_FLAG_NAMES: ",".join(flags.describe()),
                "scan_start_time": (
                    t.isoformat() if (t := parse_granule_time(path)) else ""
                ),
                "subsatellite_lon": source.geolocation.subsatellite_lon,
                "satellite_altitude_km": source.geolocation.satellite_altitude_km,
                "grid": "native_curvilinear_geostationary",
            },
        )

        if flags is not QCFlag.OK:
            logger.info(
                "granule %s read with flags: %s",
                path.name,
                ", ".join(flags.describe()),
            )
        return dataset

    except Exception as exc:
        # Anything unanticipated inside the granule still must not abort a
        # forecast cycle.
        logger.exception("unhandled error reading %s: %s", path, exc)
        return _empty_dataset(
            source,
            valid_time,
            flags | QCFlag.CORRUPT_OR_MISSING,
            expected_shape,
            path,
        )
    finally:
        handle.close()


def read_for_time(
    source: InsatSource,
    valid_time: datetime,
    lut_interpolation: str = "linear",
    expected_shape: tuple[int, int] | None = None,
) -> xr.Dataset:
    """Locate and read the granule for a nominal timestamp."""
    if expected_shape is None:
        expected_shape = source.native_shape
    path = locate_granule(source, valid_time)
    if path is None:
        logger.warning(
            "no INSAT granule for %s under %s",
            valid_time.isoformat(),
            source.root,
        )
        return _empty_dataset(source, valid_time, QCFlag.MISSING_FILE, expected_shape)
    return read_granule(path, source, valid_time, lut_interpolation, expected_shape)


__all__ = [
    "build_granule_glob",
    "locate_granule",
    "parse_granule_time",
    "read_for_time",
    "read_granule",
]
