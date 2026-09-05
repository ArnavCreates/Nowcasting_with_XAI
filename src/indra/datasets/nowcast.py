"""Pairs an input window with its ground truth for the trainer."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from ..config import IngestionConfig, PreprocessingConfig
from ..ingestion.temporal_sync import build_timestamps, synchronise
from ..preprocessing.normalization import (
    ChannelStats,
    load_statistics,
    normalise_window,
)
from ..preprocessing.reprojection import reproject_window
from ..preprocessing.tensor_assembly import assemble
from ..types import AssembledWindow, TargetWindow, masked_like
from .cache import WindowCache
from .targets import build_target_window

logger = logging.getLogger(__name__)

Split = Literal["train", "validation"]


# ---------------------------------------------------------------------------
# Input assembly
# ---------------------------------------------------------------------------


def _rejected_input(
    ingestion: IngestionConfig,
    preprocessing: PreprocessingConfig,
    valid_time: datetime,
    reason: str,
) -> AssembledWindow:
    """A correctly shaped, entirely masked input window carrying its rejection."""
    grid = preprocessing.target_grid
    shape = (
        preprocessing.tensor.sequence_length,
        preprocessing.channels.count,
        grid.height,
        grid.width,
    )
    mask_shape = (shape[0], 1, grid.height, grid.width)
    return AssembledWindow(
        valid_time=valid_time,
        timestamps=tuple(build_timestamps(valid_time, ingestion)),
        tensor=masked_like(shape),
        validity=np.zeros(mask_shape, dtype=np.bool_),
        channel_names=tuple(preprocessing.channels.names),
        accepted=False,
        rejection_reason=reason,
    )


def build_input_window(
    ingestion: IngestionConfig,
    preprocessing: PreprocessingConfig,
    valid_time: datetime,
    stats: dict[str, ChannelStats],
) -> AssembledWindow:
    """Run the Stage 1 to Stage 2 chain for one nowcast time."""
    try:
        synced = synchronise(ingestion, ingestion.quality_control, valid_time)
        if not synced.accepted:
            # Short-circuit before the expensive half. Reprojecting and
            # normalising a window that is already disqualified would cost a
            # full 384 x 384 resampling of thirty channels for a result
            # nothing will consume.
            return _rejected_input(
                ingestion,
                preprocessing,
                valid_time,
                synced.rejection_reason or "lookback window rejected in ingestion",
            )

        reprojected = reproject_window(synced, preprocessing)
        normalised = normalise_window(reprojected, preprocessing, stats)
        return assemble(normalised, preprocessing, ingestion)

    except Exception as exc:
        logger.exception(
            "unhandled error building the input window for %s: %s",
            valid_time.isoformat(),
            exc,
        )
        return _rejected_input(
            ingestion,
            preprocessing,
            valid_time,
            f"input assembly raised {type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NowcastSample:
    """One training example, still in numpy."""

    valid_time: datetime
    input_window: AssembledWindow
    target_window: TargetWindow

    @property
    def accepted(self) -> bool:
        return self.input_window.accepted and self.target_window.accepted

    @property
    def rejection_reason(self) -> str | None:
        if self.input_window.accepted and self.target_window.accepted:
            return None
        reasons = [
            f"input: {self.input_window.rejection_reason}"
            if not self.input_window.accepted
            else None,
            f"target: {self.target_window.rejection_reason}"
            if not self.target_window.accepted
            else None,
        ]
        return "; ".join(r for r in reasons if r)

    @property
    def x(self) -> npt.NDArray[np.float32]:
        """``(T, C, H, W)`` normalised inputs."""
        return self.input_window.tensor

    @property
    def input_validity(self) -> npt.NDArray[np.bool_]:
        return self.input_window.validity

    @property
    def target_mm_h(self) -> npt.NDArray[np.float32]:
        """``(T_out, 1, H, W)`` rain rate in physical units."""
        return self.target_window.rain_rate_mm_h

    @property
    def target_validity(self) -> npt.NDArray[np.bool_]:
        return self.target_window.validity

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self.input_window.channel_names


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


@dataclass
class NowcastBatch:
    """A collated batch, plus the samples the trainer needs in the parent."""

    x: torch.Tensor  # (B, T, C, H, W)
    target_mm_h: torch.Tensor  # (B, T_out, 1, H, W)
    target_validity: torch.Tensor  # (B, T_out, 1, H, W), bool
    input_validity: torch.Tensor  # (B, T, 1, H, W), bool
    valid_times: tuple[datetime, ...]
    channel_names: tuple[str, ...]
    samples: tuple[NowcastSample, ...]
    #: Samples discarded during collation. A rising count is a degrading
    #: archive, and it is only visible if someone logs it.
    dropped: int = 0

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def pin_memory(self) -> NowcastBatch:
        """Pin the tensors for a faster host-to-device copy."""
        return NowcastBatch(
            x=self.x.pin_memory(),
            target_mm_h=self.target_mm_h.pin_memory(),
            target_validity=self.target_validity.pin_memory(),
            input_validity=self.input_validity.pin_memory(),
            valid_times=self.valid_times,
            channel_names=self.channel_names,
            samples=self.samples,
            dropped=self.dropped,
        )

    def to(
        self, device: str | torch.device, non_blocking: bool = False
    ) -> NowcastBatch:
        """Move the tensors, leaving ``samples`` on the host."""
        return NowcastBatch(
            x=self.x.to(device, non_blocking=non_blocking),
            target_mm_h=self.target_mm_h.to(device, non_blocking=non_blocking),
            target_validity=self.target_validity.to(device, non_blocking=non_blocking),
            input_validity=self.input_validity.to(device, non_blocking=non_blocking),
            valid_times=self.valid_times,
            channel_names=self.channel_names,
            samples=self.samples,
            dropped=self.dropped,
        )


def collate_nowcast(samples: Sequence[NowcastSample]) -> NowcastBatch | None:
    """Stack accepted samples; return ``None`` when none survive."""
    kept = [sample for sample in samples if sample.accepted]
    dropped = len(samples) - len(kept)

    if not kept:
        if samples:
            logger.warning(
                "all %d samples in this batch were rejected: %s",
                len(samples),
                "; ".join(
                    f"{s.valid_time.isoformat()} ({s.rejection_reason})"
                    for s in samples[:3]
                ),
            )
        return None

    if dropped:
        logger.info("dropped %d rejected sample(s) from a batch", dropped)

    names = kept[0].channel_names
    for sample in kept[1:]:
        # Channel order is the contract between the tensor, the XAI labels and
        # the API schema. Two samples disagreeing means two different
        # configurations reached one batch, most likely through a stale cache
        # entry, and stacking them would silently mix them.
        if sample.channel_names != names:
            raise ValueError(
                f"sample {sample.valid_time.isoformat()} has a different "
                "channel order from the rest of its batch"
            )

    def stack(arrays: list[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(np.stack(arrays)))

    return NowcastBatch(
        x=stack([s.x for s in kept]),
        target_mm_h=stack([s.target_mm_h for s in kept]),
        target_validity=stack([s.target_validity for s in kept]),
        input_validity=stack([s.input_validity for s in kept]),
        valid_times=tuple(s.valid_time for s in kept),
        channel_names=names,
        samples=tuple(kept),
        dropped=dropped,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class NowcastDataset(Dataset):
    """Nowcast windows over one side of the chronological split."""

    def __init__(
        self,
        valid_times: Sequence[datetime],
        ingestion: IngestionConfig,
        preprocessing: PreprocessingConfig,
        *,
        split: Split,
        cache: WindowCache | None = None,
        stats: dict[str, ChannelStats] | None = None,
    ) -> None:
        if not valid_times:
            raise ValueError(
                f"the {split} split holds no valid times; there is nothing to "
                "iterate"
            )

        self.valid_times = tuple(valid_times)
        self.ingestion = ingestion
        self.preprocessing = preprocessing
        self.split = split
        self.cache = cache

        # Eagerly, and once. load_statistics re-reads its JSON on every call,
        # and MissingClimatologyError has to surface here rather than in a
        # worker part-way through an epoch -- by which point the run has spent
        # hours to discover a file was absent at startup.
        self.stats = (
            stats if stats is not None else load_statistics(preprocessing.normalization)
        )

        logger.info(
            "%s dataset: %d windows, %s to %s",
            split,
            len(self.valid_times),
            self.valid_times[0].isoformat(),
            self.valid_times[-1].isoformat(),
        )

    def __len__(self) -> int:
        return len(self.valid_times)

    def __getitem__(self, index: int) -> NowcastSample:
        """Build or fetch one sample."""
        valid_time = self.valid_times[index]

        # Target first. It is the cached half by default, so a window that is
        # going to be rejected is usually discovered from a metadata-only
        # cache entry of about a kilobyte, and the 219 MiB input path is never
        # entered at all.
        target = self._target(valid_time)
        if not target.accepted:
            return NowcastSample(
                valid_time=valid_time,
                input_window=_rejected_input(
                    self.ingestion,
                    self.preprocessing,
                    valid_time,
                    "not built: the target for this window was rejected",
                ),
                target_window=target,
            )

        return NowcastSample(
            valid_time=valid_time,
            input_window=self._input(valid_time),
            target_window=target,
        )

    # ------------------------------------------------------------- internals
    def _target(self, valid_time: datetime) -> TargetWindow:
        if self.cache is not None:
            cached = self.cache.load_target(valid_time)
            if cached is not None:
                return cached

        window = build_target_window(self.ingestion, self.preprocessing, valid_time)
        if self.cache is not None:
            self.cache.store_target(window)
        return window

    def _input(self, valid_time: datetime) -> AssembledWindow:
        if self.cache is not None:
            cached = self.cache.load_input(valid_time)
            if cached is not None:
                return cached

        window = build_input_window(
            self.ingestion, self.preprocessing, valid_time, self.stats
        )
        if self.cache is not None:
            self.cache.store_input(window)
        return window

    def describe(self) -> dict[str, Any]:
        """What this dataset covers, for the training log."""
        return {
            "split": self.split,
            "windows": len(self.valid_times),
            "first": self.valid_times[0].isoformat(),
            "last": self.valid_times[-1].isoformat(),
            "channels": list(self.preprocessing.channels.names),
            "cache": self.cache.stats() if self.cache is not None else None,
        }


__all__ = [
    "NowcastBatch",
    "NowcastDataset",
    "NowcastSample",
    "Split",
    "build_input_window",
    "collate_nowcast",
]
