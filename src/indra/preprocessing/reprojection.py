"""Spatial harmonisation onto the common 384 x 384 EPSG:4326 domain."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np
import xarray as xr

from ..config import PreprocessingConfig, Resampling, TargetGrid
from ..types import (
    ATTR_QC_FLAG,
    ATTR_QC_FLAG_NAMES,
    FloatArray,
    QCFlag,
    SyncedWindow,
    masked_like,
)

logger = logging.getLogger(__name__)

#: Multiple of the source pixel size used as the neighbour search radius.
#: 1.5 covers the diagonal of a source cell (~1.41) with a little margin, so
#: every target cell finds a source sample without reaching into the next one
#: but one.
_RADIUS_PIXELS = 1.5

#: Metres per degree of latitude. Used only to convert a native grid spacing
#: expressed in degrees into the metres pyresample expects.
_M_PER_DEG = 111_320.0

#: Neighbours considered by the bilinear swath resampler.
_BILINEAR_NEIGHBOURS = 32


# ---------------------------------------------------------------------------
# Target geometry
# ---------------------------------------------------------------------------


def build_target_area(grid: TargetGrid) -> Any:
    """The pyresample ``AreaDefinition`` for the model domain."""
    from pyresample.geometry import AreaDefinition

    half = grid.resolution_deg / 2.0
    area_extent = (
        grid.lon_min - half,  # lower-left x
        grid.lat_min - half,  # lower-left y
        grid.lon_max + half,  # upper-right x
        grid.lat_max + half,  # upper-right y
    )
    return AreaDefinition(
        area_id="indra_india_domain",
        description="Indra nowcasting domain, 384x384 at 1/12 degree",
        proj_id="epsg4326",
        projection={"proj": "longlat", "datum": "WGS84", "no_defs": True},
        width=grid.width,
        height=grid.height,
        area_extent=area_extent,
    )


def _as_2d_geolocation(
    dataset: xr.Dataset, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Latitude/longitude as 2-D arrays matching a field's shape."""
    if "latitude" in dataset.coords and "longitude" in dataset.coords:
        lat = np.asarray(dataset["latitude"].values, dtype=np.float64)
        lon = np.asarray(dataset["longitude"].values, dtype=np.float64)
        if lat.ndim == 2 and lon.ndim == 2:
            return lat, lon
        if lat.ndim == 1 and lon.ndim == 1:
            if (lat.size, lon.size) != shape:
                logger.error(
                    "geolocation axes %s x %s do not match field shape %s",
                    lat.size,
                    lon.size,
                    shape,
                )
                return None
            lon2d, lat2d = np.meshgrid(lon, lat)
            return lat2d, lon2d
    return None


# ---------------------------------------------------------------------------
# Swath resampling
# ---------------------------------------------------------------------------


def radius_of_influence_m(
    lat: np.ndarray, lon: np.ndarray, fallback_deg: float = 0.1
) -> float:
    """Neighbour search radius, in metres, from the source grid's own spacing."""
    try:
        if lat.ndim == 2 and lat.shape[0] > 1:
            d_lat = np.abs(np.diff(lat[:, lat.shape[1] // 2]))
            spacing = float(np.nanmedian(d_lat[np.isfinite(d_lat) & (d_lat > 0)]))
        else:
            spacing = fallback_deg
    except (ValueError, IndexError):
        spacing = fallback_deg

    if not np.isfinite(spacing) or spacing <= 0:
        spacing = fallback_deg
    return float(spacing * _M_PER_DEG * _RADIUS_PIXELS)


def resample_swath(
    field: FloatArray,
    lat: np.ndarray,
    lon: np.ndarray,
    target_area: Any,
    method: Resampling,
    radius_m: float,
) -> FloatArray:
    """Resample one 2-D field from scattered source points onto the target."""
    from pyresample.geometry import SwathDefinition

    finite = np.isfinite(lat) & np.isfinite(lon)
    if not finite.any():
        logger.error("no finite geolocation; cannot resample")
        return masked_like((target_area.height, target_area.width))

    # Off-disc pixels carry NaN coordinates, which pyresample cannot index.
    # They are replaced with an out-of-domain sentinel so the KD-tree simply
    # never selects them, rather than being dropped, which would break the
    # correspondence between the coordinate arrays and the field.
    safe_lat = np.where(finite, lat, 1.0e6)
    safe_lon = np.where(finite, lon, 1.0e6)
    swath = SwathDefinition(lons=safe_lon, lats=safe_lat)

    data = np.where(finite, field, np.nan).astype(np.float64)

    try:
        if method is Resampling.NEAREST:
            from pyresample.kd_tree import resample_nearest

            out = resample_nearest(
                swath,
                data,
                target_area,
                radius_of_influence=radius_m,
                fill_value=None,  # returns a masked array
            )
        else:
            from pyresample.bilinear import resample_bilinear

            out = resample_bilinear(
                data,
                swath,
                target_area,
                radius=radius_m,
                neighbours=_BILINEAR_NEIGHBOURS,
                nprocs=1,
                fill_value=None,
                reduce_data=True,
            )
    except Exception as exc:
        # Bilinear needs enough valid neighbours; a sparse or heavily masked
        # frame can leave it without a solvable system. Nearest always has an
        # answer, so it is the fallback rather than losing the frame.
        logger.warning(
            "%s swath resampling failed (%s); falling back to nearest",
            method.value,
            exc,
        )
        try:
            from pyresample.kd_tree import resample_nearest

            out = resample_nearest(
                swath,
                data,
                target_area,
                radius_of_influence=radius_m,
                fill_value=None,
            )
        except Exception as inner:
            logger.error("nearest fallback also failed: %s", inner)
            return masked_like((target_area.height, target_area.width))

    result = np.ma.filled(np.ma.asarray(out), np.nan).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Raster reprojection (static priors)
# ---------------------------------------------------------------------------


def reproject_raster(
    field: FloatArray,
    src_crs: str,
    src_transform: list[float],
    grid: TargetGrid,
    method: Resampling,
) -> FloatArray:
    """Warp a georeferenced raster onto the target grid."""
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling as RioResampling
    from rasterio.warp import reproject

    resampling_map = {
        Resampling.NEAREST: RioResampling.nearest,
        Resampling.BILINEAR: RioResampling.bilinear,
        Resampling.BICUBIC: RioResampling.cubic,
    }

    # Target transform in cell-edge convention, which is what an affine
    # describes; the grid is specified by centres.
    half = grid.resolution_deg / 2.0
    dst_transform = Affine(
        grid.resolution_deg,
        0.0,
        grid.lon_min - half,
        0.0,
        -grid.resolution_deg,
        grid.lat_max + half,
    )
    destination = np.full((grid.height, grid.width), np.nan, dtype=np.float32)

    try:
        reproject(
            source=np.asarray(field, dtype=np.float32),
            destination=destination,
            src_transform=Affine(*src_transform),
            src_crs=rasterio.crs.CRS.from_string(src_crs),
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=rasterio.crs.CRS.from_epsg(4326),
            dst_nodata=np.nan,
            resampling=resampling_map.get(method, RioResampling.bilinear),
        )
    except Exception as exc:
        logger.error("raster reprojection failed: %s", exc)
        return masked_like((grid.height, grid.width))

    # rasterio writes rows north-to-south; the domain is defined south-to-north
    # so that latitude ascends with the row index, matching every other stream.
    return destination[::-1].astype(np.float32)


# ---------------------------------------------------------------------------
# Per-stream drivers
# ---------------------------------------------------------------------------


def _method_for(kind: str, config: PreprocessingConfig) -> Resampling:
    mapping = config.reprojection.resampling_by_kind
    return {
        "continuous": mapping.continuous,
        "categorical": mapping.categorical,
        "precipitation": mapping.precipitation,
    }.get(kind, mapping.continuous)


def _target_coords(grid: TargetGrid) -> dict[str, Any]:
    lat = np.round(
        grid.lat_min + grid.resolution_deg * (np.arange(grid.height) + 0.0), 6
    )
    lon = np.round(
        grid.lon_min + grid.resolution_deg * (np.arange(grid.width) + 0.0), 6
    )
    return {
        "latitude": ("y", lat, {"units": "degrees_north", "standard_name": "latitude"}),
        "longitude": (
            "x",
            lon,
            {"units": "degrees_east", "standard_name": "longitude"},
        ),
    }


def reproject_timeseries(
    dataset: xr.Dataset | None,
    config: PreprocessingConfig,
    target_area: Any,
    kind: str = "continuous",
    stream: str = "",
) -> xr.Dataset | None:
    """Reproject every variable of a stacked stream, frame by frame."""
    if dataset is None:
        return None

    grid = config.target_grid
    method = _method_for(kind, config)
    out_vars: dict[str, Any] = {}

    # Per-frame coordinates present themselves as (time, y, x).
    per_frame_geo = (
        "latitude" in dataset.coords
        and dataset["latitude"].ndim == 3
        and "time" in dataset["latitude"].dims
    )

    for name in dataset.data_vars:
        array = dataset[name]
        values = np.asarray(array.values, dtype=np.float32)

        # Everything but the trailing two axes is treated as a stack of
        # frames, so the IMDAA vertical coordinate is carried through without
        # a special case.
        spatial = values.shape[-2:]
        leading = values.shape[:-2]
        flat = values.reshape(-1, *spatial)

        resampled = np.empty((flat.shape[0], grid.height, grid.width), dtype=np.float32)

        shared_geo = None
        if not per_frame_geo:
            shared_geo = _as_2d_geolocation(dataset, (int(spatial[0]), int(spatial[1])))
            if shared_geo is None:
                logger.error(
                    "%s/%s has no usable geolocation; emitting masked field",
                    stream,
                    name,
                )
                resampled[:] = np.nan
                out_vars[str(name)] = (
                    ("time", *tuple(array.dims[1:-2]), "y", "x"),
                    resampled.reshape(*leading, grid.height, grid.width),
                    dict(array.attrs),
                )
                continue
            radius = radius_of_influence_m(shared_geo[0], shared_geo[1])

        leading[0] if leading else 1
        planes_per_frame = int(np.prod(leading[1:])) if len(leading) > 1 else 1

        for index in range(flat.shape[0]):
            if per_frame_geo:
                frame = index // max(planes_per_frame, 1)
                lat = np.asarray(
                    dataset["latitude"].isel(time=frame).values, dtype=np.float64
                )
                lon = np.asarray(
                    dataset["longitude"].isel(time=frame).values, dtype=np.float64
                )
                radius = radius_of_influence_m(lat, lon)
            else:
                assert shared_geo is not None
                lat, lon = shared_geo

            resampled[index] = resample_swath(
                flat[index], lat, lon, target_area, method, radius
            )

        name_str = str(name)
        out_vars[name_str] = (
            (*tuple(array.dims[:-2]), "y", "x"),
            resampled.reshape(*leading, grid.height, grid.width),
            {
                **dict(array.attrs),
                "resampling": method.value,
                "regridded_from": stream,
            },
        )

    coords: dict[str, Any] = _target_coords(grid)
    if "time" in dataset.coords:
        coords["time"] = ("time", dataset["time"].values, dict(dataset["time"].attrs))
    if "level" in dataset.coords:
        coords["level"] = (
            "level",
            dataset["level"].values,
            dict(dataset["level"].attrs),
        )

    attrs = dict(dataset.attrs)
    attrs.update(
        {
            "grid": "target_384x384_epsg4326",
            "target_crs": grid.crs,
            "resampling_method": method.value,
            "per_frame_geolocation": int(per_frame_geo),
        }
    )
    return xr.Dataset(data_vars=out_vars, coords=coords, attrs=attrs)


def reproject_static(
    layers: dict[str, xr.Dataset], config: PreprocessingConfig
) -> dict[str, xr.Dataset]:
    """Warp the static priors onto the target grid."""
    grid = config.target_grid
    out: dict[str, xr.Dataset] = {}

    for name, dataset in layers.items():
        variables = list(dataset.data_vars)
        if not variables:
            continue
        array = dataset[variables[0]]
        field = np.asarray(array.values, dtype=np.float32)
        flags = QCFlag(int(dataset.attrs.get(ATTR_QC_FLAG, 0)))

        crs = str(dataset.attrs.get("native_crs", ""))
        transform = dataset.attrs.get("native_transform")

        if field.size == 0 or not crs or not transform:
            logger.error(
                "static layer %s cannot be reprojected (crs=%r, transform=%s, "
                "shape=%s); emitting masked field",
                name,
                crs,
                bool(transform),
                field.shape,
            )
            warped = masked_like((grid.height, grid.width))
            flags |= QCFlag.GEOLOCATION_MISSING
        else:
            declared = str(array.attrs.get("resampling", "bilinear"))
            method = Resampling(declared)
            warped = reproject_raster(field, crs, list(transform), grid, method)

            if int(array.attrs.get("is_categorical", 0)):
                # Guard the invariant rather than trusting it. A class index
                # that arrived fractional means an interpolating method was
                # used somewhere upstream, and the mask is no longer a
                # classification.
                finite = warped[np.isfinite(warped)]
                if finite.size and not np.allclose(finite, np.round(finite)):
                    logger.error(
                        "%s holds fractional class indices after reprojection; "
                        "an interpolating method was applied to a categorical "
                        "raster",
                        name,
                    )
                    flags |= QCFlag.CORRUPT_OR_MISSING

        out[name] = xr.Dataset(
            data_vars={
                name: (("y", "x"), warped, {**dict(array.attrs), "regridded": 1})
            },
            coords=_target_coords(grid),
            attrs={
                **dict(dataset.attrs),
                "grid": "target_384x384_epsg4326",
                "target_crs": grid.crs,
                ATTR_QC_FLAG: int(flags),
                ATTR_QC_FLAG_NAMES: ",".join(flags.describe()),
            },
        )
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def reproject_window(window: SyncedWindow, config: PreprocessingConfig) -> SyncedWindow:
    """Harmonise every stream in a window onto the target grid."""
    target_area = build_target_area(config.target_grid)
    logger.info(
        "reprojecting window %s onto %dx%d at %.6f deg",
        window.valid_time.isoformat(),
        config.target_grid.height,
        config.target_grid.width,
        config.target_grid.resolution_deg,
    )

    satellite = reproject_timeseries(
        window.satellite, config, target_area, "continuous", "insat"
    )
    nwp = reproject_timeseries(window.nwp, config, target_area, "continuous", "imdaa")

    surface: dict[str, xr.Dataset] = {}
    for name, dataset in window.surface.items():
        # Precipitation has its own entry in resampling_by_kind, since it is
        # bounded below at zero and a resampler that can undershoot would
        # produce negative rainfall.
        kind = "precipitation" if "precip" in name else "continuous"
        result = reproject_timeseries(dataset, config, target_area, kind, f"imd:{name}")
        if result is not None:
            surface[name] = result

    static = reproject_static(window.static, config)

    return replace(window, satellite=satellite, nwp=nwp, surface=surface, static=static)


__all__ = [
    "build_target_area",
    "radius_of_influence_m",
    "reproject_raster",
    "reproject_static",
    "reproject_timeseries",
    "reproject_window",
    "resample_swath",
]
