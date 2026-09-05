"""Climatological normalisation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ..config import (
    ChannelOverride,
    FieldKind,
    NormalizationConfig,
    NormalizationMethod,
    PreprocessingConfig,
)
from ..types import FloatArray, SyncedWindow

logger = logging.getLogger(__name__)


class MissingClimatologyError(RuntimeError):
    """Raised when the climatology file or a channel's statistics are absent."""


@dataclass(frozen=True)
class ChannelStats:
    """Climatological moments for one channel."""

    mean: float
    std: float
    minimum: float
    maximum: float
    count: int | None = None
    source: str | None = None

    @property
    def span(self) -> float:
        return self.maximum - self.minimum


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_statistics(
    config: NormalizationConfig, root: Path | None = None
) -> dict[str, ChannelStats]:
    """Read and validate the climatology file."""
    path = config.resolved_stats_path(root) if root else config.resolved_stats_path()

    if not path.exists():
        raise MissingClimatologyError(
            f"climatology statistics not found at {path}.\n"
            "Fetch them with 'bash scripts/fetch_data.sh', or mount the "
            "volume that holds them. They are the real moments of the "
            f"{config.reference_period.start}..{config.reference_period.end} "
            "record; deriving substitutes from the current batch would leak "
            "across the temporal split and make validation meaningless."
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MissingClimatologyError(f"cannot read {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise MissingClimatologyError(f"{path} must contain a JSON object")

    required = set(config.stats_schema.required_keys)
    stats: dict[str, ChannelStats] = {}

    for channel, entry in raw.items():
        if channel.startswith("_"):  # metadata blocks
            continue
        if not isinstance(entry, dict):
            raise MissingClimatologyError(
                f"{path}: entry for {channel!r} is not an object"
            )
        missing = required - set(entry)
        if missing:
            raise MissingClimatologyError(
                f"{path}: {channel!r} is missing required keys {sorted(missing)}"
            )
        stats[channel] = ChannelStats(
            mean=float(entry["mean"]),
            std=float(entry["std"]),
            minimum=float(entry["min"]),
            maximum=float(entry["max"]),
            count=int(entry["count"]) if "count" in entry else None,
            source=entry.get("source"),
        )

    logger.info("loaded climatology for %d channels from %s", len(stats), path)
    return stats


def resolve_stats(
    channel: str, stats: dict[str, ChannelStats], config: NormalizationConfig
) -> ChannelStats:
    """Look up a channel's statistics, or fail loudly."""
    if channel in stats:
        return stats[channel]

    if channel in config.fallback_moments:
        mean, std = config.fallback_moments[channel]
        lo, hi = config.fallback_bounds.get(channel, (mean - 3 * std, mean + 3 * std))
        logger.warning(
            "channel %s uses configured fallback moments rather than " "climatology",
            channel,
        )
        return ChannelStats(
            mean=mean, std=std, minimum=lo, maximum=hi, source="configured_fallback"
        )

    raise MissingClimatologyError(
        f"no climatological statistics for channel {channel!r}, and no "
        "fallback is configured for it. Add it to the statistics file rather "
        "than substituting values."
    )


# ---------------------------------------------------------------------------
# Scalar transforms
# ---------------------------------------------------------------------------


def apply_transform(field: FloatArray, transform: str | None) -> FloatArray:
    """Pre-scaling transform, applied before any standardisation."""
    if transform is None:
        return field
    if transform == "log1p":
        return np.log1p(np.clip(field, 0.0, None)).astype(np.float32)
    if transform == "sqrt":
        return np.sqrt(np.clip(field, 0.0, None)).astype(np.float32)
    raise ValueError(f"unknown transform {transform!r}")


def invert_transform(field: FloatArray, transform: str | None) -> FloatArray:
    if transform is None:
        return field
    if transform == "log1p":
        return np.expm1(field).astype(np.float32)
    if transform == "sqrt":
        return np.square(field).astype(np.float32)
    raise ValueError(f"unknown transform {transform!r}")


def normalise_array(
    field: FloatArray,
    stats: ChannelStats,
    method: NormalizationMethod,
    config: NormalizationConfig,
    transform: str | None = None,
) -> FloatArray:
    """Scale one array into network units."""
    out = apply_transform(np.asarray(field, dtype=np.float32), transform)

    if method is NormalizationMethod.NONE:
        return out

    if method is NormalizationMethod.ZSCORE:
        # A near-zero spread would divide the field into infinities. That
        # happens for a channel that was constant over the climatology period,
        # which is a data problem, but it must not become a NaN cascade.
        std = max(stats.std, config.min_std)
        if stats.std < config.min_std:
            logger.warning(
                "climatological std %.3g is below min_std %.3g; clamping",
                stats.std,
                config.min_std,
            )
        out = (out - stats.mean) / std
        # Clip only the tails. The extremes are the events of interest, so
        # this is set generously and exists to stop a single artefact
        # dominating a batch, not to tame real weather.
        return np.clip(out, -config.clip_to_sigma, config.clip_to_sigma).astype(
            np.float32
        )

    span = stats.span
    if abs(span) < config.epsilon:
        logger.warning(
            "climatological range is degenerate (%.3g); returning zeros", span
        )
        return np.zeros_like(out)
    return ((out - stats.minimum) / span).astype(np.float32)


def denormalise_array(
    field: FloatArray,
    stats: ChannelStats,
    method: NormalizationMethod,
    config: NormalizationConfig,
    transform: str | None = None,
) -> FloatArray:
    """Return network units to physical units."""
    out = np.asarray(field, dtype=np.float32)
    if method is NormalizationMethod.ZSCORE:
        out = out * max(stats.std, config.min_std) + stats.mean
    elif method is NormalizationMethod.MINMAX:
        out = out * stats.span + stats.minimum
    return invert_transform(out, transform)


# ---------------------------------------------------------------------------
# Channel policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelPolicy:
    """How one named channel is scaled."""

    name: str
    method: NormalizationMethod
    kind: FieldKind
    transform: str | None = None


def build_policies(config: PreprocessingConfig) -> dict[str, ChannelPolicy]:
    """Resolve every channel's treatment from the group defaults and overrides."""
    policies: dict[str, ChannelPolicy] = {}

    for _group_name, group in config.channels.groups.items():
        default_method = group.normalization or NormalizationMethod.ZSCORE
        default_kind = group.kind or FieldKind.CONTINUOUS

        for name in group.names:
            override: ChannelOverride | None = group.per_channel_overrides.get(name)
            method = default_method
            kind = default_kind
            transform: str | None = None

            if override is not None:
                method = override.normalization or method
                kind = override.kind or kind
                transform = override.transform

            # A class index is an identifier, not a magnitude. Scaling it
            # would make class 3 numerically "between" 2 and 4, which it is
            # not, and the model would learn an ordering that does not exist.
            if kind is FieldKind.CATEGORICAL:
                method = NormalizationMethod.NONE
                transform = None

            policies[name] = ChannelPolicy(name, method, kind, transform)

    return policies


# ---------------------------------------------------------------------------
# Dataset application
# ---------------------------------------------------------------------------


def normalise_dataset(
    dataset: xr.Dataset | None,
    policies: dict[str, ChannelPolicy],
    stats: dict[str, ChannelStats],
    config: NormalizationConfig,
    level_expand: bool = False,
) -> xr.Dataset | None:
    """Normalise every variable in a dataset against its own statistics."""
    if dataset is None:
        return dataset

    out = dataset.copy(deep=True)

    for name in list(out.data_vars):
        array = out[name]
        values = np.asarray(array.values, dtype=np.float32)

        if level_expand and "level" in array.dims:
            axis = array.dims.index("level")
            levels = [int(v) for v in np.asarray(out["level"].values).ravel()]
            normalised = np.empty_like(values)

            for index, level in enumerate(levels):
                channel = f"{name}{level}"
                policy = policies.get(channel)
                if policy is None:
                    raise MissingClimatologyError(
                        f"no channel policy for {channel!r}; the vertical "
                        "expansion of the NWP variables does not match the "
                        "declared channel names"
                    )
                selector: list[Any] = [slice(None)] * values.ndim
                selector[axis] = index
                normalised[tuple(selector)] = normalise_array(
                    values[tuple(selector)],
                    resolve_stats(channel, stats, config),
                    policy.method,
                    config,
                    policy.transform,
                )

            out[name] = (
                array.dims,
                normalised,
                {**dict(array.attrs), "normalization": "per_level", "normalized": 1},
            )
            continue

        policy = policies.get(str(name))
        if policy is None:
            logger.debug("no policy for variable %s; leaving it unscaled", name)
            continue

        channel_stats = resolve_stats(str(name), stats, config)
        out[name] = (
            array.dims,
            normalise_array(
                values, channel_stats, policy.method, config, policy.transform
            ),
            {
                **dict(array.attrs),
                "normalization": policy.method.value,
                "transform": policy.transform or "none",
                "climatology_mean": channel_stats.mean,
                "climatology_std": channel_stats.std,
                "climatology_min": channel_stats.minimum,
                "climatology_max": channel_stats.maximum,
                "normalized": 1,
            },
        )

    out.attrs["normalized"] = 1
    out.attrs["climatology_period"] = (
        f"{config.reference_period.start}..{config.reference_period.end}"
    )
    return out


def normalise_window(
    window: SyncedWindow,
    config: PreprocessingConfig,
    stats: dict[str, ChannelStats] | None = None,
) -> SyncedWindow:
    """Normalise every stream in a reprojected window."""
    if stats is None:
        stats = load_statistics(config.normalization)

    policies = build_policies(config)
    norm = config.normalization

    satellite = normalise_dataset(window.satellite, policies, stats, norm)
    nwp = normalise_dataset(window.nwp, policies, stats, norm, level_expand=True)
    surface = {
        name: ds
        for name, dataset in window.surface.items()
        if (ds := normalise_dataset(dataset, policies, stats, norm)) is not None
    }
    static = {
        name: ds
        for name, dataset in window.static.items()
        if (ds := normalise_dataset(dataset, policies, stats, norm)) is not None
    }

    logger.info(
        "normalised window %s against the %s climatology",
        window.valid_time.isoformat(),
        f"{norm.reference_period.start}..{norm.reference_period.end}",
    )
    return replace(window, satellite=satellite, nwp=nwp, surface=surface, static=static)


__all__ = [
    "ChannelPolicy",
    "ChannelStats",
    "MissingClimatologyError",
    "apply_transform",
    "build_policies",
    "denormalise_array",
    "invert_transform",
    "load_statistics",
    "normalise_array",
    "normalise_dataset",
    "normalise_window",
    "resolve_stats",
]
