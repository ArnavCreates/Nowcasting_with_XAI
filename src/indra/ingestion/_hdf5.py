"""Low-level HDF5 reading shared by the MOSDAC granule readers."""

from __future__ import annotations

import logging
from typing import Any

import h5py
import numpy as np

from ..config import InsatGeolocation
from ..types import FloatArray, QCFlag

logger = logging.getLogger(__name__)


def attribute(dataset: h5py.Dataset, name: str) -> Any | None:
    """Read an HDF5 attribute, decoding bytes and unwrapping 1-element arrays."""
    if name not in dataset.attrs:
        return None
    value = dataset.attrs[name]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0]
    return value


def read_scaled(dataset: h5py.Dataset) -> FloatArray:
    """Read a dataset and apply its stored packing."""
    raw = np.asarray(dataset[...])
    fill = attribute(dataset, "_FillValue")
    if fill is None:
        fill = attribute(dataset, "fill_value")

    out = raw.astype(np.float32, copy=True)
    if fill is not None:
        out[raw == fill] = np.nan

    scale = attribute(dataset, "scale_factor")
    offset = attribute(dataset, "add_offset")
    if scale is not None:
        out *= np.float32(scale)
    if offset is not None:
        out += np.float32(offset)
    return out


def mask_out_of_range(
    field: FloatArray, valid_range: tuple[float, float]
) -> tuple[FloatArray, bool]:
    """Mask values outside a physical range, returning whether any were masked."""
    lo, hi = valid_range
    finite = np.isfinite(field)
    bad = finite & ((field < lo) | (field > hi))
    if not bad.any():
        return field, False
    out = field.copy()
    out[bad] = np.nan
    return out, True


def read_geolocation(
    handle: h5py.File,
    geo: InsatGeolocation,
    shape: tuple[int, ...],
    label: str = "granule",
) -> tuple[np.ndarray | None, np.ndarray | None, QCFlag]:
    """Read the curvilinear latitude/longitude arrays."""
    if geo.latitude_dataset not in handle or geo.longitude_dataset not in handle:
        logger.warning("geolocation arrays absent from %s", label)
        return None, None, QCFlag.GEOLOCATION_MISSING

    lat = np.squeeze(read_scaled(handle[geo.latitude_dataset])).astype(np.float64)
    lon = np.squeeze(read_scaled(handle[geo.longitude_dataset])).astype(np.float64)

    if lat.shape != shape or lon.shape != shape:
        logger.warning(
            "%s geolocation shape %s / %s does not match the data %s",
            label,
            lat.shape,
            lon.shape,
            shape,
        )
        return None, None, QCFlag.GEOLOCATION_MISSING

    # Off-disc pixels are stored as out-of-range sentinels in some releases.
    lat = np.where((lat >= -90.0) & (lat <= 90.0), lat, np.nan)
    lon = np.where((lon >= -180.0) & (lon <= 180.0), lon, np.nan)
    return lat, lon, QCFlag.OK


__all__ = [
    "attribute",
    "mask_out_of_range",
    "read_geolocation",
    "read_scaled",
]
