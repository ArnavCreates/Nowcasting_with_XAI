"""IMDAA / NCUM reanalysis reader (GRIB2 via cfgrib, or NetCDF4)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ..config import ImdaaSource
from ..types import (
    ATTR_GRANULE_PATH,
    ATTR_QC_FLAG,
    ATTR_QC_FLAG_NAMES,
    ATTR_SOURCE_STREAM,
    ATTR_VALID_TIME,
    FloatArray,
    QCFlag,
    SourceStream,
    masked_like,
)

logger = logging.getLogger(__name__)

#: GRIB short names that denote geopotential height in gpm. Distinct from
#: ``z``, which in GRIB2 is geopotential in m2 s-2.
_HEIGHT_SHORT_NAMES = frozenset({"gh"})

#: cfgrib exposes the native projection through this attribute.
_GRID_TYPE_ATTR = "GRIB_gridType"

#: Grid types cfgrib reports for a regular latitude/longitude mesh. Anything
#: else is treated as projected and left for the reprojection stage.
_GEOGRAPHIC_GRID_TYPES = frozenset({"regular_ll", "regular_gg"})


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _pattern_is_per_level(pattern: str) -> bool:
    """True when each file holds a single variable at a single level."""
    return "{level}" in pattern


def build_path(
    source: ImdaaSource,
    valid_time: datetime,
    variable: str,
    level: int | None = None,
) -> Path:
    """Render the configured filename pattern for one variable (and level)."""
    fields: dict[str, Any] = {
        "variable": variable,
        "year": valid_time.year,
        "month": valid_time.month,
        "day": valid_time.day,
        "hour": valid_time.hour,
    }
    if level is not None:
        fields["level"] = level
    return Path(source.root) / source.filename_pattern.format(**fields)


def _nominal_source_time(source: ImdaaSource, valid_time: datetime) -> datetime:
    """Snap a target timestamp back to the reanalysis' own cadence."""
    step = timedelta(minutes=source.native_interval_minutes)
    if step <= timedelta(0):
        return valid_time
    epoch = valid_time.replace(minute=0, second=0, microsecond=0)
    while epoch + step <= valid_time:
        epoch += step
    return epoch


# ---------------------------------------------------------------------------
# File opening
# ---------------------------------------------------------------------------


def _open_dataset(path: Path, source: ImdaaSource) -> xr.Dataset | None:
    """Open one IMDAA file, or return ``None`` if it cannot be read."""
    if not path.exists():
        logger.warning("IMDAA file not found: %s", path)
        return None

    try:
        if source.format == "grib2":
            backend_kwargs = {
                "filter_by_keys": {
                    "typeOfLevel": source.backend_kwargs.filter_by_keys.typeOfLevel
                },
                # Empty string disables the .idx sidecar cfgrib would otherwise
                # try to write beside the data, which fails read-only.
                "indexpath": source.backend_kwargs.indexpath,
            }
            return xr.open_dataset(
                path,
                engine=source.engine,
                backend_kwargs=backend_kwargs,
                decode_timedelta=False,
            )
        return xr.open_dataset(path, engine="netcdf4", decode_timedelta=False)
    except Exception as exc:
        logger.error("cannot open IMDAA file %s: %s", path, exc)
        return None


def _find_variable(dataset: xr.Dataset, short_name: str) -> str | None:
    """Locate a variable by GRIB short name, then by exact key."""
    for name, array in dataset.data_vars.items():
        if str(array.attrs.get("GRIB_shortName", "")) == short_name:
            return str(name)
    if short_name in dataset.data_vars:
        return short_name
    return None


def _select_time(
    array: xr.DataArray, target: datetime, tolerance_minutes: int
) -> tuple[xr.DataArray, datetime | None, QCFlag]:
    """Select the timestep nearest ``target`` within tolerance."""
    time_dim = next(
        (d for d in ("time", "valid_time", "forecast_time") if d in array.dims), None
    )
    if time_dim is None:
        # A single-timestep file; nothing to select.
        return array, None, QCFlag.OK

    try:
        selected = array.sel(
            {time_dim: np.datetime64(target.replace(tzinfo=None), "ns")},
            method="nearest",
            tolerance=np.timedelta64(tolerance_minutes, "m"),
        )
    except KeyError:
        logger.warning(
            "no IMDAA timestep within %d min of %s",
            tolerance_minutes,
            target.isoformat(),
        )
        return array.isel({time_dim: 0}), None, QCFlag.MISSING_FILE

    actual_raw = selected[time_dim].values
    actual = (
        datetime.fromisoformat(
            str(np.datetime_as_string(actual_raw, unit="s"))
        ).replace(tzinfo=UTC)
        if actual_raw is not None
        else None
    )
    return selected, actual, QCFlag.OK


def _select_level(
    array: xr.DataArray, level: int
) -> tuple[xr.DataArray | None, QCFlag]:
    """Extract one isobaric level, or report it absent."""
    level_dim = next(
        (d for d in ("isobaricInhPa", "level", "plev", "pressure") if d in array.dims),
        None,
    )
    if level_dim is None:
        # A per-level file has already been sliced by the distributor; verify
        # the scalar coordinate agrees rather than trusting the filename.
        for coord in ("isobaricInhPa", "level", "plev"):
            if coord in array.coords:
                found = int(np.asarray(array[coord].values).reshape(-1)[0])
                if found != level:
                    logger.error(
                        "file declares level %d but %d was requested", found, level
                    )
                    return None, QCFlag.VARIABLE_ABSENT
        return array, QCFlag.OK

    available = [int(v) for v in np.asarray(array[level_dim].values).ravel()]
    if level not in available:
        logger.warning("level %d hPa absent; file holds %s", level, available)
        return None, QCFlag.VARIABLE_ABSENT
    return array.sel({level_dim: level}), QCFlag.OK


def _to_yx(array: xr.DataArray) -> FloatArray | None:
    """Reduce a selected slice to a plain 2-D ``(y, x)`` float32 field."""
    values = np.squeeze(np.asarray(array.values))
    if values.ndim != 2:
        logger.error("expected a 2-D field after selection, got shape %s", values.shape)
        return None
    return values.astype(np.float32, copy=False)


def _extract_geolocation(array: xr.DataArray) -> dict[str, Any]:
    """Pull latitude/longitude coordinates in whatever form the file carries."""
    coords: dict[str, Any] = {}
    lat_name = next((n for n in ("latitude", "lat") if n in array.coords), None)
    lon_name = next((n for n in ("longitude", "lon") if n in array.coords), None)
    if lat_name is None or lon_name is None:
        return coords

    lat = np.asarray(array[lat_name].values, dtype=np.float64)
    lon = np.asarray(array[lon_name].values, dtype=np.float64)

    if lat.ndim == 1 and lon.ndim == 1:
        coords["latitude"] = (
            ("y",),
            lat,
            {"units": "degrees_north", "standard_name": "latitude"},
        )
        coords["longitude"] = (
            ("x",),
            lon,
            {"units": "degrees_east", "standard_name": "longitude"},
        )
    elif lat.ndim == 2 and lon.ndim == 2:
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
    return coords


def _detect_crs(array: xr.DataArray, source: ImdaaSource) -> tuple[str, str]:
    """Report the native grid type and the CRS to assume for it."""
    grid_type = str(array.attrs.get(_GRID_TYPE_ATTR, "")) or "unknown"
    if grid_type in _GEOGRAPHIC_GRID_TYPES:
        return grid_type, "EPSG:4326"
    if grid_type == "unknown":
        return grid_type, source.native_crs
    # Projected. The reprojection stage reads the full GRIB projection
    # attributes to build the exact CRS; naming a specific EPSG code here
    # would be a guess.
    return grid_type, "projected:see_grib_attributes"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _empty_dataset(
    source: ImdaaSource,
    valid_time: datetime,
    flags: QCFlag,
    expected_shape: tuple[int, int] | None,
    paths: list[Path] | None = None,
) -> xr.Dataset:
    """An all-missing dataset covering every configured variable and level."""
    levels = source.pressure_levels_hpa
    # Configured native shape, not a zero-sized array: a masked level must
    # stack against observed ones.
    ny, nx = expected_shape if expected_shape is not None else source.native_shape
    data_vars = {
        key: (
            ("level", "y", "x"),
            masked_like((len(levels), ny, nx)),
            {
                "units": spec.units,
                "long_name": spec.description,
                "GRIB_shortName": spec.short_name,
                ATTR_QC_FLAG: int(flags),
            },
        )
        for key, spec in source.variables.items()
    }
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "level": (
                "level",
                np.asarray(levels, dtype=np.int32),
                {"units": "hPa", "positive": "down", "standard_name": "air_pressure"},
            )
        },
        attrs={
            ATTR_SOURCE_STREAM: SourceStream.IMDAA.value,
            ATTR_VALID_TIME: valid_time.isoformat(),
            ATTR_GRANULE_PATH: ";".join(str(p) for p in (paths or [])),
            ATTR_QC_FLAG: int(flags),
            ATTR_QC_FLAG_NAMES: ",".join(flags.describe()),
            "native_crs": source.native_crs,
            "grid_type": "unknown",
        },
    )


def read_for_time(
    source: ImdaaSource,
    valid_time: datetime,
    tolerance_minutes: int = 30,
    expected_shape: tuple[int, int] | None = None,
) -> xr.Dataset:
    """Read all configured variables and levels for one nominal timestamp."""
    if expected_shape is None:
        expected_shape = source.native_shape
    levels = source.pressure_levels_hpa
    per_level = _pattern_is_per_level(source.filename_pattern)
    source_time = _nominal_source_time(source, valid_time)

    flags = QCFlag.OK
    fields: dict[str, list[FloatArray | None]] = {k: [] for k in source.variables}
    used_paths: list[Path] = []
    geo_coords: dict[str, Any] = {}
    grid_type, crs = "unknown", source.native_crs
    actual_time: datetime | None = None
    shape: tuple[int, int] | None = None

    # Consolidated files are opened once and reused across every level, rather
    # than reopened per level: cfgrib decodes the whole message index on open,
    # and repeating that twenty times dominates the read.
    open_cache: dict[Path, xr.Dataset | None] = {}

    def _get(path: Path) -> xr.Dataset | None:
        if path not in open_cache:
            open_cache[path] = _open_dataset(path, source)
        return open_cache[path]

    try:
        for key, spec in source.variables.items():
            for level in levels:
                path = build_path(
                    source, source_time, key, level if per_level else None
                )
                dataset = _get(path)
                if dataset is None:
                    flags |= QCFlag.MISSING_FILE
                    fields[key].append(None)
                    continue

                if path not in used_paths:
                    used_paths.append(path)

                var_name = _find_variable(dataset, spec.short_name)
                if var_name is None:
                    logger.warning(
                        "variable %s (short name %s) absent from %s",
                        key,
                        spec.short_name,
                        path.name,
                    )
                    flags |= QCFlag.VARIABLE_ABSENT
                    fields[key].append(None)
                    continue

                array = dataset[var_name]

                selected, t_actual, t_flags = _select_time(
                    array, source_time, tolerance_minutes
                )
                flags |= t_flags
                if t_actual is not None and actual_time is None:
                    actual_time = t_actual

                at_level, lvl_flags = _select_level(selected, level)
                flags |= lvl_flags
                if at_level is None:
                    fields[key].append(None)
                    continue

                field = _to_yx(at_level)
                if field is None:
                    flags |= QCFlag.CORRUPT_OR_MISSING
                    fields[key].append(None)
                    continue

                if shape is None:
                    shape = (int(field.shape[0]), int(field.shape[1]))
                    geo_coords = _extract_geolocation(at_level)
                    grid_type, crs = _detect_crs(at_level, source)
                elif field.shape != shape:
                    # Variables that disagree on grid cannot be co-registered
                    # into one tensor, and silently broadcasting them would
                    # misalign the vertical profile.
                    logger.error(
                        "%s at %d hPa has shape %s, expected %s; masking",
                        key,
                        level,
                        field.shape,
                        shape,
                    )
                    flags |= QCFlag.CORRUPT_OR_MISSING
                    fields[key].append(None)
                    continue

                fields[key].append(field)

        if shape is None:
            logger.error(
                "no IMDAA variable could be read for %s", valid_time.isoformat()
            )
            return _empty_dataset(
                source,
                valid_time,
                flags | QCFlag.CORRUPT_OR_MISSING,
                expected_shape,
                used_paths,
            )

        ny, nx = shape
        data_vars: dict[str, Any] = {}
        for key, spec in source.variables.items():
            stack = np.stack(
                [f if f is not None else masked_like((ny, nx)) for f in fields[key]],
                axis=0,
            ).astype(np.float32, copy=False)

            attrs: dict[str, Any] = {
                "units": spec.units,
                "long_name": spec.description,
                "GRIB_shortName": spec.short_name,
                ATTR_QC_FLAG: int(flags),
            }
            if spec.short_name in _HEIGHT_SHORT_NAMES:
                # Recorded explicitly because geopotential height (gpm) and
                # geopotential (m2 s-2) differ by g, and a mix-up produces
                # thicknesses that are wrong yet entirely plausible.
                attrs["note"] = "geopotential HEIGHT in gpm, not geopotential in m2 s-2"
            data_vars[key] = (("level", "y", "x"), stack, attrs)

        coords: dict[str, Any] = {
            "level": (
                "level",
                np.asarray(levels, dtype=np.int32),
                {
                    "units": "hPa",
                    "positive": "down",
                    "standard_name": "air_pressure",
                    "note": "vertical axis retained; flattened to channel names "
                    "in preprocessing/tensor_assembly.py",
                },
            )
        }
        coords.update(geo_coords)

        dataset = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                ATTR_SOURCE_STREAM: SourceStream.IMDAA.value,
                ATTR_VALID_TIME: valid_time.isoformat(),
                ATTR_GRANULE_PATH: ";".join(str(p) for p in used_paths),
                ATTR_QC_FLAG: int(flags),
                ATTR_QC_FLAG_NAMES: ",".join(flags.describe()),
                # The analysis actually used, which may precede valid_time by
                # up to one native interval. Kept distinct so preprocessing
                # knows how far it is interpolating.
                "source_time": (actual_time or source_time).isoformat(),
                "native_interval_minutes": source.native_interval_minutes,
                "native_crs": crs,
                "grid_type": grid_type,
                "engine": source.engine if source.format == "grib2" else "netcdf4",
            },
        )

        if flags is not QCFlag.OK:
            logger.info(
                "IMDAA %s read with flags: %s",
                valid_time.isoformat(),
                ", ".join(flags.describe()),
            )
        return dataset

    except Exception as exc:
        logger.exception(
            "unhandled error reading IMDAA for %s: %s", valid_time.isoformat(), exc
        )
        return _empty_dataset(
            source,
            valid_time,
            flags | QCFlag.CORRUPT_OR_MISSING,
            expected_shape,
            used_paths,
        )
    finally:
        for handle in open_cache.values():
            if handle is not None:
                handle.close()


__all__ = [
    "build_path",
    "read_for_time",
]
