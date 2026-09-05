"""The climatological baseline that attribution is measured against."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..config import FieldKind, NormalizationMethod, PreprocessingConfig
from ..preprocessing.normalization import (
    ChannelPolicy,
    ChannelStats,
    build_policies,
    load_statistics,
    normalise_array,
    resolve_stats,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelBaseline:
    """The reference state for one named channel, in both unit systems."""

    name: str
    method: NormalizationMethod
    kind: FieldKind
    transform: str | None
    #: Climatological mean in physical units -- K, m/s, %, mm h-1, m.
    physical_mean: float
    #: The same quantity in network units. NaN for a passthrough channel,
    #: which has no fixed baseline value at all.
    normalised_value: float
    #: True when the baseline is the input itself, so attribution is zero by
    #: construction. See the module docstring on categorical channels.
    passthrough: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "channel": self.name,
            "method": self.method.value,
            "transform": self.transform,
            "physical_mean": round(self.physical_mean, 4),
            "normalised_value": (
                None
                if self.passthrough or not np.isfinite(self.normalised_value)
                else round(float(self.normalised_value), 6)
            ),
            "passthrough": self.passthrough,
        }


@dataclass(frozen=True)
class ClimatologicalBaseline:
    """Per-channel reference state, expandable to a full input tensor."""

    channels: tuple[ChannelBaseline, ...]

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    @property
    def values(self) -> npt.NDArray[np.float32]:
        """``(C,)`` normalised baseline; passthrough entries hold NaN."""
        return np.asarray(
            [channel.normalised_value for channel in self.channels], dtype=np.float32
        )

    @property
    def passthrough_mask(self) -> npt.NDArray[np.bool_]:
        """``(C,)`` True where the baseline is the input's own value."""
        return np.asarray(
            [channel.passthrough for channel in self.channels], dtype=np.bool_
        )

    def as_tensor(self, reference: Any) -> Any:
        """Baseline shaped like ``reference``, an input tensor."""
        import torch

        if reference.ndim not in (4, 5):
            raise ValueError(
                f"expected (T, C, H, W) or (B, T, C, H, W); got "
                f"{tuple(reference.shape)}"
            )
        channel_axis = 2 if reference.ndim == 5 else 1
        n_channels = reference.shape[channel_axis]
        if n_channels != len(self.channels):
            raise ValueError(
                f"baseline covers {len(self.channels)} channels but the input "
                f"has {n_channels}. Channel order is the contract between the "
                "tensor and these names; a mismatch means they describe "
                "different things."
            )

        shape = [1] * reference.ndim
        shape[channel_axis] = n_channels

        values = torch.as_tensor(
            np.nan_to_num(self.values, nan=0.0),
            dtype=reference.dtype,
            device=reference.device,
        ).view(shape)
        baseline = values.expand_as(reference)

        if not self.passthrough_mask.any():
            return baseline

        keep = torch.as_tensor(
            self.passthrough_mask, dtype=torch.bool, device=reference.device
        ).view(shape)
        # Where the channel is passthrough, the baseline *is* the input, so
        # the path integral over it is identically zero.
        return torch.where(keep, reference, baseline)

    def describe(self) -> list[dict[str, Any]]:
        return [channel.describe() for channel in self.channels]

    def summary(self) -> dict[str, Any]:
        """Counts an operator needs to read a baseline as sane rather than broken."""
        zeros = sum(
            1
            for channel in self.channels
            if not channel.passthrough and channel.normalised_value == 0.0
        )
        return {
            "channels": len(self.channels),
            "zero_valued": zeros,
            "passthrough": int(self.passthrough_mask.sum()),
            "non_zero": len(self.channels) - zeros - int(self.passthrough_mask.sum()),
        }


def _normalised_scalar(
    value: float,
    stats: ChannelStats,
    policy: ChannelPolicy,
    config: PreprocessingConfig,
) -> float:
    """Push one physical scalar through the input normalisation pipeline."""
    array = np.asarray([value], dtype=np.float32)
    out = normalise_array(
        array, stats, policy.method, config.normalization, policy.transform
    )
    return float(out[0])


def build_climatological_baseline(
    config: PreprocessingConfig,
    stats: dict[str, ChannelStats] | None = None,
) -> ClimatologicalBaseline:
    """Assemble the baseline for every channel, in tensor order."""
    if stats is None:
        stats = load_statistics(config.normalization)

    policies = build_policies(config)
    baselines: list[ChannelBaseline] = []

    for name in config.channels.names:
        policy = policies.get(name)
        if policy is None:
            raise KeyError(
                f"channel {name!r} appears in the channel order but has no "
                "normalisation policy; the two are built from the same "
                "configuration and cannot disagree"
            )

        channel_stats = resolve_stats(name, stats, config.normalization)
        categorical = policy.kind is FieldKind.CATEGORICAL

        baselines.append(
            ChannelBaseline(
                name=name,
                method=policy.method,
                kind=policy.kind,
                transform=policy.transform,
                physical_mean=channel_stats.mean,
                normalised_value=(
                    float("nan")
                    if categorical
                    else _normalised_scalar(
                        channel_stats.mean, channel_stats, policy, config
                    )
                ),
                passthrough=categorical,
            )
        )

    baseline = ClimatologicalBaseline(channels=tuple(baselines))
    logger.info("climatological baseline: %s", baseline.summary())
    return baseline


__all__ = [
    "ChannelBaseline",
    "ClimatologicalBaseline",
    "build_climatological_baseline",
]
