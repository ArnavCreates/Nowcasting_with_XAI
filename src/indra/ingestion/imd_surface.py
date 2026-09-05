"""IMD gauge and AWS surface observation reader."""

from __future__ import annotations

import calendar
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ..config import ImdNativeGrid, ImdSurfaceSource, ImdSurfaceVariable
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

#: IMD distributes ``.grd`` products little-endian. Reading them with the
#: host's native order on a big-endian machine yields values around 1e-38 or
#: 1e38 -- not a crash, just numbers that are wrong by many orders of
#: magnitude. Pinning the byte order removes the possibility.
_GRD_DTYPES: dict[str, str] = {
    "float32": "<f4",
    "float64": "<f8",
    "int16": "<i2",
    "int32": "<i4",
}

#: Tolerance when checking that a declared grid's spacing matches its extents
#: and cell count. 1e-6 deg is far below any real geolocation error.
_DEG_EPS = 1e-6


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def _axes(grid: ImdNativeGrid) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude axes of a declared IMD grid."""
    lat = np.linspace(grid.lat_min, grid.lat_max, grid.n_lat, dtype=np.float64)
    lon = np.linspace(grid.lon_min, grid.lon_max, grid.n_lon, dtype=np.float64)
    return lat, lon


def _check_grid_spacing(grid: ImdNativeGrid, label: str) -> None:
    """Warn when a declared grid's spacing contradicts its extents and count."""
    d_lat = (grid.lat_max - grid.lat_min) / max(grid.n_lat - 1, 1)
    d_lon = (grid.lon_max - grid.lon_min) / max(grid.n_lon - 1, 1)
    for axis, spacing in (("latitude", d_lat), ("longitude", d_lon)):
        if abs(spacing - grid.resolution_deg) > _DEG_EPS:
            logger.warning(
                "%s: declared %s spacing %.8f deg disagrees with "
                "resolution_deg %.8f; the configured grid may not describe "
                "this product",
                label,
                axis,
                spacing,
                grid.resolution_deg,
            )


def _days_in_year(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


# ---------------------------------------------------------------------------
# Binary .grd reader
# ---------------------------------------------------------------------------


def _validate_grd_size(
    path: Path, grid: ImdNativeGrid, year: int
) -> tuple[int, QCFlag]:
    """Confirm the file is exactly the size the declared grid implies."""
    dtype = _GRD_DTYPES.get(grid.dtype)
    if dtype is None:
        logger.error("unsupported .grd dtype %r for %s", grid.dtype, path.name)
        return 0, QCFlag.CORRUPT_OR_MISSING

    itemsize = np.dtype(dtype).itemsize
    field_bytes = grid.n_lat * grid.n_lon * itemsize
    expected_days = _days_in_year(year)
    expected_bytes = expected_days * field_bytes

    try:
        actual_bytes = path.stat().st_size
    except OSError as exc:
        logger.error("cannot stat %s: %s", path, exc)
        return 0, QCFlag.CORRUPT_OR_MISSING

    if actual_bytes == expected_bytes:
        return expected_days, QCFlag.OK

    implied = actual_bytes / field_bytes if field_bytes else 0.0
    logger.error(
        "%s is %d bytes; the declared %dx%d %s grid implies %d days x %d bytes "
        "= %d bytes for %d. The file holds %.3f fields, so either the grid or "
        "the year is wrong. Refusing to read rather than return misaligned "
        "rainfall.",
        path.name,
        actual_bytes,
        grid.n_lat,
        grid.n_lon,
        grid.dtype,
        expected_days,
        field_bytes,
        expected_bytes,
        year,
        implied,
    )
    return 0, QCFlag.CORRUPT_OR_MISSING


def _read_grd_day(
    path: Path, grid: ImdNativeGrid, target: date
) -> tuple[FloatArray | None, QCFlag]:
    """Read a single day's field out of a year-long ``.grd`` file."""
    n_days, flags = _validate_grd_size(path, grid, target.year)
    if flags is not QCFlag.OK:
        return None, flags

    day_index = (target - date(target.year, 1, 1)).days
    if not 0 <= day_index < n_days:
        logger.error(
            "day index %d out of range for %d (%d days)",
            day_index,
            target.year,
            n_days,
        )
        return None, QCFlag.MISSING_FILE

    dtype = _GRD_DTYPES[grid.dtype]
    try:
        memmap = np.memmap(
            path, dtype=dtype, mode="r", shape=(n_days, grid.n_lat, grid.n_lon)
        )
        field = np.array(memmap[day_index], dtype=np.float32)
        del memmap
    except (OSError, ValueError) as exc:
        logger.error("cannot memory-map %s: %s", path, exc)
        return None, QCFlag.CORRUPT_OR_MISSING

    # Fill sentinel first: -999.0 is not a measurement and must not survive
    # into any statistic computed downstream.
    field = np.where(np.isclose(field, grid.fill_value), np.nan, field).astype(
        np.float32
    )

    if not np.isfinite(field).any():
        logger.warning("%s day %d is entirely fill", path.name, day_index)
        return field, QCFlag.CORRUPT_OR_MISSING
    return field, QCFlag.OK


# ---------------------------------------------------------------------------
# NetCDF4 reader
# ---------------------------------------------------------------------------


def _open_netcdf(path: Path) -> xr.Dataset | None:
    if not path.exists():
        logger.warning("IMD NetCDF file not found: %s", path)
        return None
    try:
        return xr.open_dataset(path, engine="netcdf4", decode_timedelta=False)
    except Exception as exc:
        logger.error("cannot open %s: %s", path, exc)
        return None


def _select_time(
    array: xr.DataArray, target: datetime, tolerance_minutes: int
) -> tuple[xr.DataArray, datetime | None, QCFlag]:
    """Select the timestep nearest ``target``, within tolerance."""
    time_dim = next(
        (d for d in ("time", "TIME", "valid_time") if d in array.dims), None
    )
    if time_dim is None:
        return array, None, QCFlag.OK

    try:
        selected = array.sel(
            {time_dim: np.datetime64(target.replace(tzinfo=None), "ns")},
            method="nearest",
            tolerance=np.timedelta64(tolerance_minutes, "m"),
        )
    except KeyError:
        logger.warning(
            "no IMD timestep within %d min of %s",
            tolerance_minutes,
            target.isoformat(),
        )
        return array.isel({time_dim: 0}), None, QCFlag.MISSING_FILE

    raw = selected[time_dim].values
    actual = datetime.fromisoformat(str(np.datetime_as_string(raw, unit="s"))).replace(
        tzinfo=UTC
    )
    return selected, actual, QCFlag.OK


def _field_2d(array: xr.DataArray) -> FloatArray | None:
    values = np.squeeze(np.asarray(array.values))
    if values.ndim != 2:
        logger.error("expected a 2-D field after selection, got %s", values.shape)
        return None
    return values.astype(np.float32, copy=False)


def _netcdf_axes(array: xr.DataArray) -> dict[str, Any]:
    """Latitude/longitude coordinates as the file carries them."""
    coords: dict[str, Any] = {}
    lat_name = next(
        (n for n in ("latitude", "lat", "LATITUDE") if n in array.coords), None
    )
    lon_name = next(
        (n for n in ("longitude", "lon", "LONGITUDE") if n in array.coords), None
    )
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


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def _wrap(
    name: str,
    field: FloatArray,
    spec: ImdSurfaceVariable,
    valid_time: datetime,
    flags: QCFlag,
    path: Path | None,
    coords: dict[str, Any] | None = None,
    source_time: datetime | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Package one field into a single-variable Dataset on its native grid."""
    attrs: dict[str, Any] = {
        "units": spec.units,
        "long_name": spec.description or name,
        ATTR_QC_FLAG: int(flags),
        # Carried per variable, not per dataset: these products have different
        # cadences, and preprocessing's `hold` policy needs to know how stale
        # each field is at the moment it is used.
        "native_interval_minutes": spec.native_interval_minutes,
    }
    if extra_attrs:
        attrs.update(extra_attrs)

    return xr.Dataset(
        data_vars={name: (("y", "x"), field, attrs)},
        coords=coords or {},
        attrs={
            ATTR_SOURCE_STREAM: SourceStream.IMD_SURFACE.value,
            ATTR_VALID_TIME: valid_time.isoformat(),
            ATTR_GRANULE_PATH: str(path) if path else "",
            ATTR_QC_FLAG: int(flags),
            ATTR_QC_FLAG_NAMES: ",".join(flags.describe()),
            "source_time": (source_time or valid_time).isoformat(),
            "native_interval_minutes": spec.native_interval_minutes,
            "grid": "native",
        },
    )


def _empty(
    name: str,
    spec: ImdSurfaceVariable,
    valid_time: datetime,
    flags: QCFlag,
    path: Path | None = None,
) -> xr.Dataset:
    """An all-missing dataset for a variable that could not be read."""
    if spec.native_grid is not None:
        shape = (spec.native_grid.n_lat, spec.native_grid.n_lon)
        lat, lon = _axes(spec.native_grid)
        coords: dict[str, Any] = {
            "latitude": (("y",), lat, {"units": "degrees_north"}),
            "longitude": (("x",), lon, {"units": "degrees_east"}),
        }
    else:
        shape = (0, 0)
        coords = {}
    return _wrap(name, masked_like(shape), spec, valid_time, flags, path, coords)


# ---------------------------------------------------------------------------
# Per-variable readers
# ---------------------------------------------------------------------------


def _read_binary_variable(
    name: str,
    spec: ImdSurfaceVariable,
    root: Path,
    valid_time: datetime,
) -> xr.Dataset:
    """Read one day out of a headerless yearly ``.grd`` archive."""
    grid = spec.native_grid
    if grid is None:
        # Cannot happen: the config model requires native_grid for binary_grd.
        return _empty(name, spec, valid_time, QCFlag.CORRUPT_OR_MISSING)

    _check_grid_spacing(grid, f"{name} ({spec.filename_pattern})")
    path = root / spec.filename_pattern.format(
        year=valid_time.year, month=valid_time.month, day=valid_time.day
    )
    if not path.exists():
        logger.warning("IMD .grd archive not found: %s", path)
        return _empty(name, spec, valid_time, QCFlag.MISSING_FILE, path)

    field, flags = _read_grd_day(path, grid, valid_time.date())
    if field is None:
        return _empty(name, spec, valid_time, flags, path)

    lat, lon = _axes(grid)
    coords = {
        "latitude": (
            ("y",),
            lat,
            {"units": "degrees_north", "standard_name": "latitude"},
        ),
        "longitude": (
            ("x",),
            lon,
            {"units": "degrees_east", "standard_name": "longitude"},
        ),
    }
    # The day this field actually covers, at midnight. Distinct from
    # valid_time, which may be any slot within that day.
    day_start = datetime(valid_time.year, valid_time.month, valid_time.day, tzinfo=UTC)
    return _wrap(
        name,
        field,
        spec,
        valid_time,
        flags,
        path,
        coords,
        source_time=day_start,
        extra_attrs={
            # CF convention. This is an accumulated total over the day, not an
            # instantaneous rate: treating it as mm/h would overstate intensity
            # by a factor of 24.
            "cell_methods": "time: sum over 1 day",
            "accumulation_period_hours": 24,
        },
    )


def _read_netcdf_variable(
    name: str,
    spec: ImdSurfaceVariable,
    root: Path,
    valid_time: datetime,
    tolerance_minutes: int,
) -> xr.Dataset:
    """Read one NetCDF4 surface variable, deriving wind speed where configured."""
    path = root / spec.filename_pattern.format(
        year=valid_time.year, month=valid_time.month, day=valid_time.day
    )
    dataset = _open_netcdf(path)
    if dataset is None:
        return _empty(name, spec, valid_time, QCFlag.MISSING_FILE, path)

    try:
        flags = QCFlag.OK

        if spec.derive == "magnitude":
            # Wind speed from its components. Reading only one and treating it
            # as speed would understate every south-westerly monsoon flow,
            # where the meridional component is far from negligible.
            if spec.u_variable not in dataset or spec.v_variable not in dataset:
                logger.warning(
                    "wind components %s / %s absent from %s",
                    spec.u_variable,
                    spec.v_variable,
                    path.name,
                )
                return _empty(name, spec, valid_time, QCFlag.VARIABLE_ABSENT, path)

            u_sel, actual, u_flags = _select_time(
                dataset[spec.u_variable], valid_time, tolerance_minutes
            )
            v_sel, _, v_flags = _select_time(
                dataset[spec.v_variable], valid_time, tolerance_minutes
            )
            flags |= u_flags | v_flags

            u = _field_2d(u_sel)
            v = _field_2d(v_sel)
            if u is None or v is None:
                return _empty(
                    name, spec, valid_time, flags | QCFlag.CORRUPT_OR_MISSING, path
                )
            if u.shape != v.shape:
                logger.error(
                    "wind components disagree on shape: %s vs %s", u.shape, v.shape
                )
                return _empty(
                    name, spec, valid_time, flags | QCFlag.CORRUPT_OR_MISSING, path
                )

            # hypot rather than sqrt(u*u + v*v): it avoids intermediate
            # overflow and propagates NaN from either component, so a missing
            # u does not silently become a valid speed of |v|.
            field = np.hypot(u, v).astype(np.float32)
            coords = _netcdf_axes(u_sel)
            extra = {
                "derived_from": f"hypot({spec.u_variable}, {spec.v_variable})",
                "standard_name": "wind_speed",
            }
        else:
            if spec.variable is None or spec.variable not in dataset:
                logger.warning("variable %s absent from %s", spec.variable, path.name)
                return _empty(name, spec, valid_time, QCFlag.VARIABLE_ABSENT, path)

            selected, actual, t_flags = _select_time(
                dataset[spec.variable], valid_time, tolerance_minutes
            )
            flags |= t_flags
            extracted_field = _field_2d(selected)
            if extracted_field is None:
                return _empty(
                    name, spec, valid_time, flags | QCFlag.CORRUPT_OR_MISSING, path
                )
            field = extracted_field
            coords = _netcdf_axes(selected)
            extra = {"source_variable": spec.variable}

        if not np.isfinite(field).any():
            flags |= QCFlag.CORRUPT_OR_MISSING

        return _wrap(
            name,
            field,
            spec,
            valid_time,
            flags,
            path,
            coords,
            source_time=actual,
            extra_attrs=extra,
        )

    except Exception as exc:
        logger.exception("unhandled error reading %s from %s: %s", name, path, exc)
        return _empty(name, spec, valid_time, QCFlag.CORRUPT_OR_MISSING, path)
    finally:
        dataset.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def read_for_time(
    source: ImdSurfaceSource,
    valid_time: datetime,
    tolerance_minutes: int = 180,
) -> dict[str, xr.Dataset]:
    """Read every configured IMD surface variable for one nominal timestamp."""
    root = Path(source.root)
    out: dict[str, xr.Dataset] = {}

    for name, spec in source.variables.items():
        try:
            if spec.format == "binary_grd":
                out[name] = _read_binary_variable(name, spec, root, valid_time)
            else:
                out[name] = _read_netcdf_variable(
                    name, spec, root, valid_time, tolerance_minutes
                )
        except Exception as exc:
            logger.exception("unhandled error reading IMD variable %s: %s", name, exc)
            out[name] = _empty(name, spec, valid_time, QCFlag.CORRUPT_OR_MISSING)

        flags = QCFlag(int(out[name].attrs[ATTR_QC_FLAG]))
        if flags is not QCFlag.OK:
            logger.info(
                "IMD %s at %s read with flags: %s",
                name,
                valid_time.isoformat(),
                ", ".join(flags.describe()),
            )

    return out


def combined_flags(datasets: dict[str, xr.Dataset]) -> QCFlag:
    """Union of the quality-control flags across the returned variables."""
    flags = QCFlag.OK
    for dataset in datasets.values():
        flags |= QCFlag(int(dataset.attrs.get(ATTR_QC_FLAG, 0)))
    return flags


__all__ = [
    "combined_flags",
    "read_for_time",
]
