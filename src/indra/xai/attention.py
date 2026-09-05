"""Earthformer attention: where the model looked, and when."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from ..config import AttributionMapConfig, EvidenceFramesConfig
from ..models.fusion import IndraFusion

logger = logging.getLogger(__name__)

#: ``evidence_frames.rank_by`` values this module implements.
_RANKINGS = ("decoder_cross_attention",)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_attention(model: IndraFusion, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run the backbone alone and return its two attention tensors."""
    if x.ndim != 5:
        raise ValueError(f"expected (B, T, C, H, W); got {tuple(x.shape)}")

    model.eval()
    _, attention = model.earthformer(x, return_attention=True)

    missing = [key for key in ("spatial", "temporal") if key not in attention]
    if missing:
        raise RuntimeError(
            f"the backbone returned no {missing} attention. Both are produced "
            "only when return_attention is set, and the decoder's requires "
            "cross_attention to be enabled in the model configuration."
        )
    return attention


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionMaps:
    """Both attention fields, resampled to the reporting grid."""

    #: ``(T, R, R)`` per-frame relative attention within the encoder volume.
    #: Normalised per cuboid, not per domain -- see the module docstring.
    encoder_relative: npt.NDArray[np.float32]
    #: ``(T, R, R)`` per-frame share of the decoder's cross-attention. Sums to
    #: 1 across the frame axis at every cell.
    decoder_share: npt.NDArray[np.float32]
    timestamps: tuple[datetime, ...]
    lookback_indices: tuple[int, ...]
    native_resolution: int
    resolution: int
    normalize: str

    @property
    def sequence_length(self) -> int:
        return int(self.decoder_share.shape[0])

    def summary(self) -> dict[str, Any]:
        return {
            "resolution": self.resolution,
            "native_resolution": self.native_resolution,
            "upsampled": self.resolution != self.native_resolution,
            "normalize": self.normalize,
            "frames": self.sequence_length,
            "encoder_map_is_cuboid_relative": True,
        }


def _resample(field: torch.Tensor, resolution: int) -> torch.Tensor:
    """``(T, h, w)`` to ``(T, R, R)``."""
    if field.shape[-1] == resolution and field.shape[-2] == resolution:
        return field
    return F.interpolate(
        field.unsqueeze(0),
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _normalise(field: torch.Tensor, method: str) -> torch.Tensor:
    """Scale a map for display without changing what it ranks."""
    if method == "per_sample_max":
        peak = field.abs().amax()
        # A uniformly zero map divided by its own maximum is NaN, which would
        # render as a hole rather than as the flat field it is.
        return field if peak <= 0 else field / peak
    if method == "none":
        return field
    raise ValueError(
        f"unknown attribution map normalisation {method!r}; expected "
        "'per_sample_max' or 'none'"
    )


def build_attention_maps(
    attention: dict[str, torch.Tensor],
    config: AttributionMapConfig,
    timestamps: Sequence[datetime],
    lookback_indices: Sequence[int],
    batch_index: int = 0,
) -> AttentionMaps:
    """Resample and normalise both attention fields for the report."""
    spatial = attention["spatial"][batch_index].detach().float()
    temporal = attention["temporal"][batch_index].detach().float()

    if spatial.shape[0] != len(timestamps):
        raise ValueError(
            f"attention covers {spatial.shape[0]} frames but "
            f"{len(timestamps)} timestamps were given; the labels would name "
            "the wrong moments"
        )

    native = int(spatial.shape[-1])
    resolution = config.output_resolution
    if resolution > native:
        logger.info(
            "attention maps upsampled from %d to %d; the extra cells are "
            "interpolation, not resolved structure",
            native,
            resolution,
        )

    encoder = _normalise(_resample(spatial, resolution), config.normalize)
    # The decoder map is deliberately *not* normalised: its values are already
    # shares that sum to one across frames, and rescaling them by a maximum
    # would destroy exactly the property that makes them interpretable.
    decoder = _resample(temporal, resolution)

    return AttentionMaps(
        encoder_relative=encoder.cpu().numpy().astype(np.float32),
        decoder_share=decoder.cpu().numpy().astype(np.float32),
        timestamps=tuple(timestamps),
        lookback_indices=tuple(lookback_indices),
        native_resolution=native,
        resolution=resolution,
        normalize=config.normalize,
    )


# ---------------------------------------------------------------------------
# Evidence frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceFrame:
    """One input frame the forecast leaned on, named by when it was observed."""

    #: Position in the lookback window, 0 oldest.
    index: int
    #: Offset from t0, e.g. -5.
    lookback_index: int
    timestamp: datetime
    #: Share of the decoder's cross-attention, within the ranking region.
    attention_share: float
    #: The share divided by 1/T. One means indifference; this is the number
    #: that decides whether the ranking says anything.
    relative_to_uniform: float

    def describe(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "lookback_index": self.lookback_index,
            "attention_share": round(self.attention_share, 5),
            "relative_to_uniform": round(self.relative_to_uniform, 3),
        }


def _region_weights(
    region: torch.Tensor | None, shape: tuple[int, int], device: Any
) -> torch.Tensor | None:
    """Reduce a full-resolution alert mask onto the attention grid."""
    if region is None:
        return None

    mask = region.detach().to(device=device, dtype=torch.float32)
    while mask.ndim > 2:
        mask = mask.amax(dim=0)
    if mask.ndim != 2:
        raise ValueError(f"region must reduce to 2-D; got {tuple(region.shape)}")

    weights = F.adaptive_avg_pool2d(mask.unsqueeze(0).unsqueeze(0), shape)
    weights = weights.squeeze(0).squeeze(0)
    if float(weights.sum()) <= 0:
        raise ValueError(
            "the alert region does not overlap any attention cell. Ranking "
            "evidence frames over an empty region would return whichever "
            "frame happened to be first."
        )
    return weights


def rank_evidence_frames(
    attention: dict[str, torch.Tensor],
    config: EvidenceFramesConfig,
    timestamps: Sequence[datetime],
    lookback_indices: Sequence[int],
    region: torch.Tensor | None = None,
    batch_index: int = 0,
) -> list[EvidenceFrame]:
    """The ``top_k`` input frames the decoder consulted most."""
    if not config.enabled:
        return []
    if config.rank_by not in _RANKINGS:
        raise ValueError(
            f"evidence_frames.rank_by is {config.rank_by!r}, which this module "
            f"does not implement; available: {list(_RANKINGS)}"
        )

    temporal = attention["temporal"][batch_index].detach().float()
    n_frames = int(temporal.shape[0])
    if n_frames != len(timestamps) or n_frames != len(lookback_indices):
        raise ValueError(
            f"attention covers {n_frames} frames but {len(timestamps)} "
            f"timestamps and {len(lookback_indices)} offsets were given"
        )

    weights = _region_weights(
        region, (int(temporal.shape[-2]), int(temporal.shape[-1])), temporal.device
    )
    if weights is None:
        per_frame = temporal.flatten(1).mean(dim=1)
    else:
        per_frame = (temporal * weights).flatten(1).sum(dim=1) / weights.sum()

    total = float(per_frame.sum())
    if total <= 0:
        raise ValueError(
            "the decoder's cross-attention is empty over this region; there is "
            "nothing to rank"
        )
    # Renormalised because the region weighting is a weighted average over
    # cells whose individual distributions each sum to one; floating-point
    # drift aside, this restores an exact share.
    shares = (per_frame / total).cpu().numpy().astype(np.float64)
    uniform = 1.0 / n_frames

    order = np.argsort(-shares)[: config.top_k]
    frames = [
        EvidenceFrame(
            index=int(index),
            lookback_index=int(lookback_indices[int(index)]),
            timestamp=timestamps[int(index)],
            attention_share=float(shares[int(index)]),
            relative_to_uniform=float(shares[int(index)] / uniform),
        )
        for index in order
    ]

    leader = frames[0].relative_to_uniform if frames else 0.0
    if leader < 1.2:
        # Not an error. A flat distribution is a real answer -- the model drew
        # on the whole window rather than on a few frames -- and reporting it
        # as evidence without saying so would overstate it.
        logger.info(
            "evidence frames are close to uniform (top frame %.2fx); the "
            "forecast drew on the window broadly rather than on specific "
            "moments",
            leader,
        )
    return frames


__all__ = [
    "AttentionMaps",
    "EvidenceFrame",
    "build_attention_maps",
    "extract_attention",
    "rank_evidence_frames",
]
