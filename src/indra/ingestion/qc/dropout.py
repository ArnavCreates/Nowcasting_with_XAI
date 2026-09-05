"""Scan-line dropout detection for geostationary imagery."""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from ...config import ScanlineDropout
from ...types import ATTR_QC_FLAG, ATTR_QC_FLAG_NAMES, FloatArray, QCFlag

logger = logging.getLogger(__name__)


def _runs_of_true(flags: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs as ``(start, length)`` pairs."""
    if not flags.any():
        return []
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(edges[i]), int(edges[i + 1] - edges[i])) for i in range(0, len(edges), 2)
    ]


def detect_rows(
    field: FloatArray,
    config: ScanlineDropout,
    valid_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Identify corrupt scan lines in one 2-D frame."""
    n_rows = field.shape[0]
    finite = np.isfinite(field)

    # -- fully missing rows --------------------------------------------------
    # A single missing row is ordinary and is left to the gap filler. Only a
    # run longer than the configured tolerance is treated as a dropout, since
    # that is what distinguishes a transmission fault from sparse bad pixels.
    empty_rows = ~finite.any(axis=1)
    fill_bad = np.zeros(n_rows, dtype=bool)
    for start, length in _runs_of_true(empty_rows):
        if length > config.max_consecutive_fill_rows:
            fill_bad[start : start + length] = True

    # -- stalled detector ----------------------------------------------------
    # Computed only over rows with enough valid samples to have a meaningful
    # variance; a row with two surviving pixels is trivially near-constant and
    # would be flagged for the wrong reason.
    counts = finite.sum(axis=1)
    enough = counts >= max(8, field.shape[1] // 64)
    constant_bad = np.zeros(n_rows, dtype=bool)
    if enough.any():
        with np.errstate(invalid="ignore"):
            spread = np.nanstd(np.where(finite, field, np.nan), axis=1)
        constant_bad = (
            enough & np.isfinite(spread) & (spread < config.row_constant_tolerance)
        )

    # -- saturation ----------------------------------------------------------
    saturated_bad = np.zeros(n_rows, dtype=bool)
    if valid_range is not None:
        ceiling = valid_range[1]
        at_ceiling = finite & (field >= ceiling)
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = at_ceiling.sum(axis=1) / np.maximum(counts, 1)
        saturated_bad = enough & (fraction >= config.saturation_fraction_threshold)

    bad = fill_bad | constant_bad | saturated_bad
    return bad, {
        "fill_rows": int(fill_bad.sum()),
        "constant_rows": int(constant_bad.sum()),
        "saturated_rows": int(saturated_bad.sum()),
        "total_bad_rows": int(bad.sum()),
    }


def apply(
    dataset: xr.Dataset,
    config: ScanlineDropout,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Detect and mask corrupt scan lines across the imagery variables."""
    if not config.enabled:
        return dataset

    out = dataset.copy(deep=True)
    flags = QCFlag(int(out.attrs.get(ATTR_QC_FLAG, 0)))
    names = variables if variables is not None else [str(v) for v in out.data_vars]
    any_bad = False

    for name in names:
        if name not in out.data_vars:
            continue
        array = out[name]
        if array.ndim != 2:
            # Dropout is a property of a single sweep; a stacked or
            # multi-level variable is not scan-line structured.
            continue

        field = np.asarray(array.values, dtype=np.float32)
        valid_range = array.attrs.get("valid_range")
        rng = (
            (float(valid_range[0]), float(valid_range[1]))
            if isinstance(valid_range, list | tuple) and len(valid_range) == 2
            else None
        )

        bad, counts = detect_rows(field, config, rng)
        if not bad.any():
            continue

        any_bad = True
        masked = field.copy()
        masked[bad, :] = np.nan

        attrs = dict(array.attrs)
        var_flags = QCFlag(int(attrs.get(ATTR_QC_FLAG, 0))) | QCFlag.SCANLINE_DROPOUT
        if counts["saturated_rows"]:
            var_flags |= QCFlag.SATURATED
        attrs[ATTR_QC_FLAG] = int(var_flags)
        attrs.update({f"dropout_{k}": v for k, v in counts.items()})

        out[name] = (array.dims, masked, attrs)
        logger.info(
            "%s: masked %d scan lines (%d fill, %d constant, %d saturated) " "of %d",
            name,
            counts["total_bad_rows"],
            counts["fill_rows"],
            counts["constant_rows"],
            counts["saturated_rows"],
            field.shape[0],
        )

    if any_bad:
        flags |= QCFlag.SCANLINE_DROPOUT
        out.attrs[ATTR_QC_FLAG] = int(flags)
        out.attrs[ATTR_QC_FLAG_NAMES] = ",".join(flags.describe())
    return out


__all__ = ["apply", "detect_rows"]
