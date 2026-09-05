"""Static geophysical prior reader — DEM, LULC and soil classification."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ..config import StaticLayer, StaticPriorsSource
from ..types import (
    ATTR_GRANULE_PATH,
    ATTR_QC_FLAG,
    ATTR_QC_FLAG_NAMES,
    ATTR_SOURCE_STREAM,
    QCFlag,
    SourceStream,
    masked_like,
)

logger = logging.getLogger(__name__)

#: Cache of decoded rasters, keyed by (resolved path, overview level).
#: Guarded because the FastAPI service may read priors from several worker
#: threads while serving concurrent nowcast requests.
_CACHE: dict[tuple[str, int | None], xr.Dataset] = {}
_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Raster access
# ---------------------------------------------------------------------------


def _open_raster(path: Path, overview_level: int | None) -> Any:
    """Open a raster, honouring the requested overview level."""
    import rasterio

    if overview_level is None:
        return rasterio.open(path), None, 1

    # Probe the available pyramid before asking for a level, since GDAL's
    # behaviour on an out-of-range OVERVIEW_LEVEL is not uniform across
    # drivers.
    try:
        with rasterio.open(path) as probe:
            factors = probe.overviews(1)
    except Exception as exc:
        logger.warning("cannot inspect overviews of %s: %s", path.name, exc)
        factors = []

    if not factors:
        logger.warning(
            "%s has no overview pyramid; reading base resolution. Build one "
            "with `gdaladdo` to avoid loading the full array.",
            path.name,
        )
        return rasterio.open(path), None, 1

    if overview_level >= len(factors):
        logger.warning(
            "%s: overview level %d requested but only %d exist (factors %s); "
            "using the coarsest available",
            path.name,
            overview_level,
            len(factors),
            factors,
        )
        overview_level = len(factors) - 1

    decimation = factors[overview_level]
    logger.info(
        "%s: reading overview level %d (%dx decimation)",
        path.name,
        overview_level,
        decimation,
    )
    return (
        rasterio.open(path, overview_level=overview_level),
        overview_level,
        decimation,
    )


def _axes_from_transform(
    transform: Any, height: int, width: int
) -> dict[str, Any] | None:
    """Build 1-D coordinate axes from a north-up affine transform."""
    # Affine is (a, b, c, d, e, f): b and d are the rotation terms.
    if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
        return None

    # Cell centres, offset half a pixel from the origin corner. Using the
    # corner instead shifts the whole raster by half a cell.
    x = transform.c + transform.a * (np.arange(width, dtype=np.float64) + 0.5)
    y = transform.f + transform.e * (np.arange(height, dtype=np.float64) + 0.5)
    return {
        "x": ("x", x, {"long_name": "projection x coordinate"}),
        "y": ("y", y, {"long_name": "projection y coordinate"}),
    }


def _read_layer(name: str, spec: StaticLayer, root: Path) -> xr.Dataset:
    """Read one static raster into a single-variable Dataset on its native grid."""
    path = root / spec.path

    if not path.exists():
        logger.error("static prior not found: %s", path)
        return _empty(name, spec, QCFlag.MISSING_FILE, path)

    handle = None
    try:
        handle, level_used, decimation = _open_raster(path, spec.overview_level)

        if handle.count < 1:
            logger.error("%s contains no raster bands", path.name)
            return _empty(name, spec, QCFlag.VARIABLE_ABSENT, path)

        band = handle.read(1)
        field = band.astype(np.float32, copy=True)

        # Nodata from the raster's own tag takes precedence: it reflects how
        # the file was actually written, while the configured fill_value is
        # only what we expected it to be.
        nodata = handle.nodata
        if nodata is None:
            nodata = spec.fill_value
            logger.debug(
                "%s declares no nodata tag; using the configured fill_value %s",
                path.name,
                nodata,
            )
        if nodata is not None and np.isfinite(nodata):
            field = np.where(np.isclose(field, nodata), np.nan, field).astype(
                np.float32
            )

        flags = QCFlag.OK
        if not np.isfinite(field).any():
            logger.error("%s is entirely nodata", path.name)
            flags |= QCFlag.CORRUPT_OR_MISSING

        if spec.is_categorical:
            # A class raster read through an averaging pyramid would arrive
            # with fractional indices. Configuration forbids that combination,
            # but the check is cheap and the failure is otherwise invisible.
            finite = field[np.isfinite(field)]
            if finite.size and not np.allclose(finite, np.round(finite)):
                logger.error(
                    "%s carries non-integer class indices, which means it was "
                    "read through an averaging overview. The mask is not "
                    "usable as a classification.",
                    path.name,
                )
                flags |= QCFlag.CORRUPT_OR_MISSING

        coords = _axes_from_transform(handle.transform, handle.height, handle.width)
        if coords is None:
            logger.info(
                "%s has a rotated transform; recording it without separable axes",
                path.name,
            )
            coords = {}

        crs = str(handle.crs) if handle.crs else ""
        if not crs:
            logger.warning(
                "%s declares no CRS; reprojection cannot place it and will "
                "skip this layer",
                path.name,
            )
            flags |= QCFlag.GEOLOCATION_MISSING

        attrs: dict[str, Any] = {
            "units": spec.units,
            "long_name": spec.description,
            # Carried forward so reprojection uses the method this layer
            # requires rather than re-deriving it. A categorical raster warped
            # bilinearly produces class 4.7, which denotes nothing, and the
            # result still looks like a plausible raster.
            "resampling": spec.resampling.value,
            "is_categorical": int(spec.is_categorical),
            ATTR_QC_FLAG: int(flags),
        }
        if spec.is_categorical:
            finite = field[np.isfinite(field)]
            attrs["class_values"] = (
                sorted(int(v) for v in np.unique(finite)) if finite.size else []
            )

        dataset = xr.Dataset(
            data_vars={name: (("y", "x"), field, attrs)},
            coords=coords,
            attrs={
                ATTR_SOURCE_STREAM: SourceStream.STATIC_PRIORS.value,
                ATTR_GRANULE_PATH: str(path),
                ATTR_QC_FLAG: int(flags),
                ATTR_QC_FLAG_NAMES: ",".join(flags.describe()),
                "native_crs": crs,
                # Serialised because an Affine object does not survive a
                # NetCDF or zarr round-trip. Reprojection rebuilds it.
                "native_transform": list(handle.transform)[:6],
                "native_shape": [int(handle.height), int(handle.width)],
                "overview_level": -1 if level_used is None else int(level_used),
                "overview_decimation": int(decimation),
                "nodata": float(nodata) if nodata is not None else float("nan"),
                "time_invariant": 1,
            },
        )

        if flags is not QCFlag.OK:
            logger.info(
                "static prior %s read with flags: %s",
                name,
                ", ".join(flags.describe()),
            )
        return dataset

    except ImportError:
        # rasterio is a hard dependency of the reader but not of the package's
        # lighter modules; say so plainly rather than surfacing a stray
        # ModuleNotFoundError from deep in the call stack.
        logger.error(
            "rasterio is required to read static priors; install the GIS "
            "extras from requirements.txt"
        )
        return _empty(name, spec, QCFlag.CORRUPT_OR_MISSING, path)
    except Exception as exc:
        logger.exception("cannot read static prior %s: %s", path, exc)
        return _empty(name, spec, QCFlag.CORRUPT_OR_MISSING, path)
    finally:
        if handle is not None:
            handle.close()


def _empty(
    name: str, spec: StaticLayer, flags: QCFlag, path: Path | None = None
) -> xr.Dataset:
    """An all-missing dataset for a static layer that could not be read."""
    return xr.Dataset(
        data_vars={
            name: (
                ("y", "x"),
                masked_like((0, 0)),
                {
                    "units": spec.units,
                    "long_name": spec.description,
                    "resampling": spec.resampling.value,
                    "is_categorical": int(spec.is_categorical),
                    ATTR_QC_FLAG: int(flags),
                },
            )
        },
        attrs={
            ATTR_SOURCE_STREAM: SourceStream.STATIC_PRIORS.value,
            ATTR_GRANULE_PATH: str(path) if path else "",
            ATTR_QC_FLAG: int(flags),
            ATTR_QC_FLAG_NAMES: ",".join(flags.describe()),
            "native_crs": "",
            "time_invariant": 1,
        },
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def read_layer(name: str, spec: StaticLayer, root: Path | str) -> xr.Dataset:
    """Read one static layer, using the process cache."""
    root = Path(root)
    key = (str(root / spec.path), spec.overview_level)

    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            # Shallow copy: the arrays are shared, but a caller mutating attrs
            # on its copy cannot corrupt what the next caller receives.
            return cached.copy(deep=False)

    dataset = _read_layer(name, spec, root)

    with _CACHE_LOCK:
        _CACHE[key] = dataset
    return dataset.copy(deep=False)


def read_all(source: StaticPriorsSource) -> dict[str, xr.Dataset]:
    """Read every configured static prior."""
    root = Path(source.root)
    out: dict[str, xr.Dataset] = {}
    for name, spec in source.layers.items():
        out[name] = read_layer(name, spec, root)
    return out


def combined_flags(datasets: dict[str, xr.Dataset]) -> QCFlag:
    """Union of the quality-control flags across the static layers."""
    flags = QCFlag.OK
    for dataset in datasets.values():
        flags |= QCFlag(int(dataset.attrs.get(ATTR_QC_FLAG, 0)))
    return flags


def clear_cache() -> None:
    """Drop the cached rasters."""
    with _CACHE_LOCK:
        _CACHE.clear()
    logger.info("static prior cache cleared")


def cache_info() -> dict[str, Any]:
    """Which layers are currently cached, for diagnostics."""
    with _CACHE_LOCK:
        return {
            "entries": len(_CACHE),
            "keys": [{"path": path, "overview_level": level} for path, level in _CACHE],
        }


__all__ = [
    "cache_info",
    "clear_cache",
    "combined_flags",
    "read_all",
    "read_layer",
]
