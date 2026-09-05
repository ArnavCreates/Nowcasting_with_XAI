"""Quality control for ingested observations."""

from __future__ import annotations

import logging

import xarray as xr

from ...config import QualityControlConfig
from ...types import ATTR_QC_FLAG, ATTR_SOURCE_STREAM, QCFlag, SourceStream
from . import bounds, dropout, gapfill, parallax

logger = logging.getLogger(__name__)


def apply_frame_qc(
    dataset: xr.Dataset,
    config: QualityControlConfig,
    stream: SourceStream | str | None = None,
    ctt_variable: str = "insat_ctt",
    variable_map: dict[str, str] | None = None,
) -> xr.Dataset:
    """Run the per-frame quality-control chain on one reader output."""
    if stream is None:
        stream = dataset.attrs.get(ATTR_SOURCE_STREAM, "")
    stream_value = stream.value if isinstance(stream, SourceStream) else str(stream)
    is_satellite = stream_value == SourceStream.INSAT.value

    out = dataset

    if is_satellite and config.scanline_dropout.enabled:
        try:
            out = dropout.apply(out, config.scanline_dropout)
        except Exception as exc:
            logger.exception("scan-line dropout detection failed: %s", exc)

    try:
        out = bounds.apply(out, config.physical_bounds, variable_map)
    except Exception as exc:
        logger.exception("physical bounds enforcement failed: %s", exc)

    if is_satellite and config.parallax_correction.enabled:
        try:
            out = parallax.apply(out, config.parallax_correction, ctt_variable)
        except Exception as exc:
            logger.exception("parallax correction failed: %s", exc)

    flags = QCFlag(int(out.attrs.get(ATTR_QC_FLAG, 0)))
    if flags is not QCFlag.OK:
        logger.debug(
            "%s frame quality control complete: %s",
            stream_value or "unknown",
            ", ".join(flags.describe()),
        )
    return out


__all__ = [
    "apply_frame_qc",
    "bounds",
    "dropout",
    "gapfill",
    "parallax",
]
