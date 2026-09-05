"""Physical boundary enforcement."""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from ...config import PhysicalBounds
from ...types import ATTR_QC_FLAG, ATTR_QC_FLAG_NAMES, QCFlag

logger = logging.getLogger(__name__)

#: Below this fraction of affected cells, an out-of-range population is
#: treated as isolated bad pixels and logged at debug level. Above it, the
#: field is likely systematically wrong — a wrong scale factor, a wrong
#: calibration table — and that deserves a warning.
_SYSTEMATIC_FRACTION = 0.01


def apply_to_field(
    field: np.ndarray,
    limits: tuple[float, float],
    action: str,
    label: str = "",
) -> tuple[np.ndarray, int]:
    """Enforce limits on one array. Returns ``(field, n_affected)``."""
    lo, hi = limits
    finite = np.isfinite(field)
    offending = finite & ((field < lo) | (field > hi))
    n = int(np.count_nonzero(offending))
    if n == 0:
        return field, 0

    out = field.astype(np.float32, copy=True)
    if action == "clip":
        out = np.where(finite, np.clip(out, lo, hi), out).astype(np.float32)
    else:
        out[offending] = np.nan

    fraction = n / max(int(np.count_nonzero(finite)), 1)
    message = "%s: %d cells (%.3f%%) outside [%g, %g]; %s"
    args = (
        label or "field",
        n,
        fraction * 100,
        lo,
        hi,
        "clipped" if action == "clip" else "masked",
    )
    if fraction >= _SYSTEMATIC_FRACTION:
        # A systematic excursion is rarely a handful of bad detectors. It
        # usually means the field is wrong as a whole, and masking it silently
        # would hide that.
        logger.warning(
            message + " — this fraction suggests a calibration or "
            "scaling fault rather than isolated bad pixels",
            *args,
        )
    else:
        logger.debug(message, *args)
    return out, n


def apply(
    dataset: xr.Dataset,
    config: PhysicalBounds,
    variable_map: dict[str, str] | None = None,
) -> xr.Dataset:
    """Enforce physical bounds across every variable that has them declared."""
    limits = config.bounds()
    action = config.action
    out = dataset.copy(deep=True)
    flags = QCFlag(int(out.attrs.get(ATTR_QC_FLAG, 0)))
    touched = 0

    for name in list(out.data_vars):
        key = (variable_map or {}).get(str(name), str(name))
        if key not in limits:
            continue

        field = np.asarray(out[name].values, dtype=np.float32)
        corrected, n = apply_to_field(field, limits[key], action, label=str(name))
        if n == 0:
            continue

        out[name] = (out[name].dims, corrected, dict(out[name].attrs))
        var_flags = QCFlag(int(out[name].attrs.get(ATTR_QC_FLAG, 0)))
        var_flags |= QCFlag.OUT_OF_PHYSICAL_RANGE
        out[name].attrs[ATTR_QC_FLAG] = int(var_flags)
        out[name].attrs["out_of_range_cells"] = n
        out[name].attrs["physical_bounds"] = list(limits[key])
        out[name].attrs["bounds_action"] = action
        touched += n

    if touched:
        flags |= QCFlag.OUT_OF_PHYSICAL_RANGE
        out.attrs[ATTR_QC_FLAG] = int(flags)
        out.attrs[ATTR_QC_FLAG_NAMES] = ",".join(flags.describe())
    return out


__all__ = ["apply", "apply_to_field"]
