"""Composes integrated gradients, attention maps and evidence frames into one report."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from ..config import XAIConfig
from ..models.fusion import IndraFusion
from ..types import AssembledWindow
from .attention import (
    AttentionMaps,
    EvidenceFrame,
    build_attention_maps,
    extract_attention,
    rank_evidence_frames,
)
from .attribution import IntegratedGradientsResult, integrated_gradients
from .baselines import ClimatologicalBaseline

logger = logging.getLogger(__name__)

#: Convergence residual, as a fraction of total attribution, above which the
#: integrated gradients are reported as unconverged rather than read.
_CONVERGENCE_TOLERANCE = 0.05

#: Ratio to uniform below which an evidence ranking is called flat.
_UNIFORM_TOLERANCE = 1.2


@dataclass(frozen=True)
class XAIReport:
    """One nowcast, explained."""

    valid_time: datetime
    generated_at: datetime

    attribution: IntegratedGradientsResult
    maps: AttentionMaps
    evidence_frames: tuple[EvidenceFrame, ...]

    threshold_mm_h: float
    #: True when attribution and evidence were restricted to an alert polygon
    #: rather than averaged over the domain.
    region_scoped: bool
    #: Members in the served ensemble, against ``attribution.members``. The two
    #: differ by design and the difference must be visible.
    served_members: int
    #: Channels whose attribution is zero by construction, not by measurement.
    excluded_channels: tuple[str, ...]
    #: Conditions that weaken this explanation, in plain language.
    caveats: tuple[str, ...] = ()

    # ------------------------------------------------------------- accessors
    def provenance(self) -> dict[str, Any]:
        """Everything needed to reproduce and to audit this explanation."""
        return {
            "valid_time": self.valid_time.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "threshold_mm_h": self.threshold_mm_h,
            "relaxation_width_mm_h": self.attribution.temperature_mm_h,
            "attributed_members": self.attribution.members,
            "served_members": self.served_members,
            "seed": self.attribution.seed,
            "n_steps": self.attribution.n_steps,
            "convergence_delta": round(self.attribution.convergence_delta, 6),
            "target": self.attribution.target,
            "region_scoped": self.region_scoped,
            "attention_native_resolution": self.maps.native_resolution,
            "attention_resolution": self.maps.resolution,
        }

    def drivers(self, k: int | None = None) -> list[dict[str, Any]]:
        """Ranked channel drivers, named and signed."""
        top = self.attribution.top_drivers(k or 5)
        return [
            {
                "channel": name,
                "signed_attribution": round(score, 6),
                "magnitude": round(self.attribution.channel_magnitude[name], 6),
                "direction": "increased" if score >= 0 else "decreased",
            }
            for name, score in top
        ]

    def arrays(self) -> dict[str, npt.NDArray[np.float32]]:
        """Every raw array, keyed. The serving layer's single entry point."""
        return {
            "attribution_spatial": self.attribution.spatial_map,
            "attribution_per_frame": self.attribution.frame_magnitude,
            "attention_encoder_relative": self.maps.encoder_relative,
            "attention_decoder_share": self.maps.decoder_share,
        }

    def summary(self) -> dict[str, Any]:
        """The array-free view: safe to log, to cache, or to return as metadata."""
        return {
            "provenance": self.provenance(),
            "drivers": self.drivers(),
            "evidence_frames": [frame.describe() for frame in self.evidence_frames],
            "attention": self.maps.summary(),
            "excluded_channels": list(self.excluded_channels),
            "caveats": list(self.caveats),
            "array_keys": sorted(self.arrays()),
        }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _lookback_indices(
    timestamps: Sequence[datetime], valid_time: datetime
) -> tuple[int, ...]:
    """Frame offsets from ``t0``, derived from the timestamps themselves."""
    if len(timestamps) < 2:
        raise ValueError("need at least two timestamps to infer the cadence")

    deltas = {int((b - a).total_seconds()) for a, b in itertools.pairwise(timestamps)}
    if len(deltas) != 1:
        raise ValueError(
            f"timestamps are not evenly spaced ({sorted(deltas)} s); this "
            "window was not synchronised onto the model cadence and its "
            "frames cannot be labelled with regular offsets"
        )

    step = deltas.pop()
    if step <= 0:
        raise ValueError("timestamps must ascend")
    return tuple(
        int(round((t - valid_time).total_seconds() / step)) for t in timestamps
    )


def _caveats(
    attribution: IntegratedGradientsResult,
    evidence: Sequence[EvidenceFrame],
    excluded: Sequence[str],
    attributed_members: int,
    served_members: int,
) -> tuple[str, ...]:
    """Conditions that weaken the explanation, stated in the report itself."""
    notes: list[str] = []

    total = abs(sum(attribution.channel_scores.values()))
    if attribution.convergence_delta > _CONVERGENCE_TOLERANCE * max(total, 1e-6):
        notes.append(
            f"Integrated gradients did not converge: residual "
            f"{attribution.convergence_delta:.3g} against a total attribution "
            f"of {total:.3g}. The map is an approximation rather than a "
            f"decomposition; raise n_steps above {attribution.n_steps}."
        )

    if evidence and evidence[0].relative_to_uniform < _UNIFORM_TOLERANCE:
        notes.append(
            f"Evidence frames are close to uniform (top frame "
            f"{evidence[0].relative_to_uniform:.2f}x). The forecast drew on "
            "the input window broadly rather than on particular moments, so "
            "the ranking should not be read as identifying a trigger."
        )

    if excluded:
        notes.append(
            f"{', '.join(excluded)} score exactly zero by construction, not by "
            "measurement: they are categorical priors held fixed along the "
            "integration path. Their influence was not assessed."
        )

    if attributed_members < served_members:
        notes.append(
            f"The explanation covers {attributed_members} of the "
            f"{served_members} members whose exceedance probability was "
            "reported."
        )

    return tuple(notes)


def build_report(
    model: IndraFusion,
    window: AssembledWindow,
    baseline: ClimatologicalBaseline,
    config: XAIConfig,
    *,
    threshold_mm_h: float,
    seed: int,
    served_members: int,
    region: torch.Tensor | None = None,
    device: str | torch.device | None = None,
) -> XAIReport:
    """Run the whole explanation path for one window."""
    if not window.accepted:
        raise ValueError(f"cannot explain a rejected window: {window.rejection_reason}")

    x = window.to_torch(device=str(device) if device else None, add_batch=True)
    lookbacks = _lookback_indices(window.timestamps, window.valid_time)

    logger.info(
        "explaining %s: %d-frame window, threshold %.1f mm/h, seed %d%s",
        window.valid_time.isoformat(),
        len(window.timestamps),
        threshold_mm_h,
        seed,
        ", region-scoped" if region is not None else "",
    )

    attribution = integrated_gradients(
        model,
        x,
        baseline,
        config.integrated_gradients,
        threshold_mm_h=threshold_mm_h,
        seed=seed,
        channel_names=window.channel_names,
        region=region,
        output_resolution=config.attribution_maps.output_resolution,
    )

    attention = extract_attention(model, x)
    maps = build_attention_maps(
        attention,
        config.attribution_maps,
        timestamps=window.timestamps,
        lookback_indices=lookbacks,
    )
    evidence = rank_evidence_frames(
        attention,
        config.evidence_frames,
        timestamps=window.timestamps,
        lookback_indices=lookbacks,
        region=region,
    )

    excluded = tuple(
        channel.name for channel in baseline.channels if channel.passthrough
    )
    caveats = _caveats(
        attribution,
        evidence,
        excluded,
        config.integrated_gradients.members,
        served_members,
    )
    for note in caveats:
        logger.info("caveat: %s", note)

    return XAIReport(
        valid_time=window.valid_time,
        generated_at=datetime.now(tz=UTC),
        attribution=attribution,
        maps=maps,
        evidence_frames=tuple(evidence),
        threshold_mm_h=threshold_mm_h,
        region_scoped=region is not None,
        served_members=served_members,
        excluded_channels=excluded,
        caveats=caveats,
    )


__all__ = ["XAIReport", "build_report"]
