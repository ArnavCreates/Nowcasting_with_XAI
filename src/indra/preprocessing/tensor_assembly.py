"""5D tensor assembly — the last step before PyTorch."""

from __future__ import annotations

import logging
import re

import numpy as np
import xarray as xr

from ..config import IngestionConfig, PreprocessingConfig
from ..types import AssembledWindow, SyncedWindow

logger = logging.getLogger(__name__)

#: Channel names ending in a pressure level, e.g. ``u850`` -> (``u``, 850).
_LEVEL_SUFFIX = re.compile(r"^(?P<variable>[a-z_]+?)(?P<level>\d{2,4})$")


# ---------------------------------------------------------------------------
# Channel resolution
# ---------------------------------------------------------------------------


def split_level_channel(
    channel: str, variables: set[str], levels: set[int]
) -> tuple[str, int] | None:
    """Split ``u850`` into ``("u", 850)``."""
    match = _LEVEL_SUFFIX.match(channel)
    if match is None:
        return None
    variable = match.group("variable")
    level = int(match.group("level"))
    if variable in variables and level in levels:
        return variable, level
    return None


def _select_level(array: xr.DataArray, level: int) -> np.ndarray | None:
    """Extract one pressure level by coordinate value, never by position."""
    if "level" not in array.dims:
        return None
    coordinates = [int(v) for v in np.asarray(array["level"].values).ravel()]
    if level not in coordinates:
        logger.error(
            "level %d hPa absent from the stacked NWP variable; it holds %s",
            level,
            coordinates,
        )
        return None
    return np.asarray(array.sel(level=level).values, dtype=np.float32)


def _from_dataset(dataset: xr.Dataset | None, name: str) -> np.ndarray | None:
    if dataset is None or name not in dataset.data_vars:
        return None
    return np.asarray(dataset[name].values, dtype=np.float32)


def gather_channel(
    channel: str,
    window: SyncedWindow,
    ingestion: IngestionConfig,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, str]:
    """Fetch one named channel as ``(T, H, W)``."""
    n_time, height, width = shape

    field = _from_dataset(window.satellite, channel)
    if field is not None:
        return field, "insat"

    parsed = split_level_channel(
        channel,
        set(ingestion.sources.imdaa.variables),
        set(ingestion.sources.imdaa.pressure_levels_hpa),
    )
    if parsed is not None and window.nwp is not None:
        variable, level = parsed
        if variable in window.nwp.data_vars:
            selected = _select_level(window.nwp[variable], level)
            if selected is not None:
                return selected, f"imdaa:{variable}@{level}hPa"

    if channel in window.surface:
        field = _from_dataset(window.surface[channel], channel)
        if field is not None:
            return field, f"imd:{channel}"

    if channel in window.static:
        field = _from_dataset(window.static[channel], channel)
        if field is not None:
            # Time-invariant: broadcast rather than stored T times. The
            # broadcast is a view until the stack copies it.
            if field.ndim == 2:
                return np.broadcast_to(field, (n_time, *field.shape)), "static"
            return field, "static"

    logger.error("channel %r could not be resolved from any stream", channel)
    return np.full((n_time, height, width), np.nan, dtype=np.float32), "absent"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _verify_channel_order(assembled: list[str], config: PreprocessingConfig) -> None:
    """Fail if the assembled order deviates from the declared one."""
    declared = config.channels.names
    if assembled != declared:
        mismatches = [
            f"index {i}: assembled {a!r} vs declared {d!r}"
            for i, (a, d) in enumerate(zip(assembled, declared, strict=False))
            if a != d
        ]
        raise ValueError(
            "assembled channel order does not match the declared order.\n"
            + "\n".join(mismatches[:10])
            + (f"\n...and {len(mismatches) - 10} more" if len(mismatches) > 10 else "")
        )


def assemble(
    window: SyncedWindow,
    config: PreprocessingConfig,
    ingestion: IngestionConfig,
) -> AssembledWindow:
    """Build the ``(T, C, H, W)`` tensor and its validity mask."""
    tensor_spec = config.tensor
    grid = config.target_grid
    n_time = tensor_spec.sequence_length
    height, width = grid.height, grid.width
    shape = (n_time, height, width)

    declared = config.channels.names
    static_names = set(config.channels.static.names)

    fields: list[np.ndarray] = []
    names: list[str] = []
    provenance: dict[str, str] = {}
    coverage: dict[str, float] = {}

    for channel in declared:
        field, source = gather_channel(channel, window, ingestion, shape)

        if field.shape != shape:
            # A stream that arrived on the wrong grid cannot be co-registered
            # with the others; masking it is preferable to broadcasting it
            # into a misaligned block.
            logger.error(
                "channel %s has shape %s, expected %s; masking it",
                channel,
                field.shape,
                shape,
            )
            field = np.full(shape, np.nan, dtype=np.float32)
            source = f"{source}:shape_mismatch"

        finite = np.isfinite(field)
        coverage[channel] = float(np.count_nonzero(finite)) / finite.size
        if coverage[channel] == 0.0:
            logger.error(
                "channel %s (%s) is entirely missing across the window",
                channel,
                source,
            )
        elif coverage[channel] < 0.5:
            logger.warning(
                "channel %s (%s) is only %.1f%% populated",
                channel,
                source,
                coverage[channel] * 100,
            )

        fields.append(field)
        names.append(channel)
        provenance[channel] = source

    _verify_channel_order(names, config)

    tensor = np.stack(fields, axis=1).astype(np.float32)  # (T, C, H, W)

    # -- validity mask -----------------------------------------------------
    # Built from the DYNAMIC channels only. A DEM is nodata over the sea, and
    # folding statics in here would mark every ocean cell invalid for every
    # frame -- masking out the Bay of Bengal and the Arabian Sea, which is
    # where much of the monsoon's precipitation is.
    dynamic_indices = [i for i, name in enumerate(names) if name not in static_names]
    if dynamic_indices:
        validity = np.all(
            np.isfinite(tensor[:, dynamic_indices]), axis=1, keepdims=True
        )
    else:
        validity = np.ones((n_time, 1, height, width), dtype=bool)

    # -- fill ---------------------------------------------------------------
    # Recorded first, replaced second. The model sees a finite number; the
    # loss consults the mask and ignores those cells. Reversing the order
    # would make the fill indistinguishable from a measurement.
    n_filled = int(np.count_nonzero(~np.isfinite(tensor)))
    tensor = np.nan_to_num(
        tensor,
        nan=tensor_spec.nan_fill_value,
        posinf=tensor_spec.nan_fill_value,
        neginf=tensor_spec.nan_fill_value,
    ).astype(np.float32)

    expected = tensor_spec.shape_without_batch
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"assembled tensor is {tuple(tensor.shape)} but the configuration "
            f"declares {expected}"
        )

    valid_fraction = float(np.count_nonzero(validity)) / validity.size
    logger.info(
        "assembled %s for %s: %.1f%% of cells valid, %d filled",
        tuple(tensor.shape),
        window.valid_time.isoformat(),
        valid_fraction * 100,
        n_filled,
    )

    empty = [name for name, c in coverage.items() if c == 0.0]
    if empty:
        logger.error(
            "%d channels are entirely absent: %s. Training on these teaches "
            "the model that the fill value is a measurement.",
            len(empty),
            empty,
        )

    return AssembledWindow(
        valid_time=window.valid_time,
        timestamps=window.timestamps,
        tensor=tensor,
        validity=validity if tensor_spec.emit_validity_mask else np.ones_like(validity),
        channel_names=tuple(names),
        flags=dict(window.flags),
        observed=dict(window.observed),
        grid={
            "lat_min": grid.lat_min,
            "lat_max": grid.lat_max,
            "lon_min": grid.lon_min,
            "lon_max": grid.lon_max,
            "resolution_deg": grid.resolution_deg,
            "height": float(height),
            "width": float(width),
        },
        channel_coverage=coverage,
        accepted=window.accepted,
        rejection_reason=window.rejection_reason,
    )


def channel_provenance(
    config: PreprocessingConfig, ingestion: IngestionConfig
) -> dict[str, str]:
    """Which stream each declared channel is expected to come from."""
    variables = set(ingestion.sources.imdaa.variables)
    levels = set(ingestion.sources.imdaa.pressure_levels_hpa)
    satellite = set(config.channels.satellite.names)
    surface = set(config.channels.surface.names)
    static = set(config.channels.static.names)

    out: dict[str, str] = {}
    for channel in config.channels.names:
        if channel in satellite:
            out[channel] = "insat"
        elif split_level_channel(channel, variables, levels) is not None:
            out[channel] = "imdaa"
        elif channel in surface:
            out[channel] = "imd_surface"
        elif channel in static:
            out[channel] = "static_priors"
        else:
            out[channel] = "unmapped"
    return out


__all__ = [
    "assemble",
    "channel_provenance",
    "gather_channel",
    "split_level_channel",
]
