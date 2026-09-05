"""Integrated Gradients over the input window."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from ..config import IntegratedGradientsConfig
from ..models.fusion import IndraFusion
from .baselines import ClimatologicalBaseline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Differentiable targets
# ---------------------------------------------------------------------------


def soft_exceedance_probability(
    samples: torch.Tensor,
    *,
    threshold: float,
    temperature: float,
    mask: torch.Tensor | None = None,
    region: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable stand-in for the fraction of members over a threshold."""
    if temperature <= 0:
        raise ValueError(
            f"temperature must be positive; got {temperature}. At zero the "
            "surrogate is the step function it exists to replace and the "
            "gradient disappears."
        )

    probability = torch.sigmoid((samples - threshold) / temperature)
    probability = probability.mean(dim=0)  # over members -> (B, T, C, H, W)

    selector = None
    if mask is not None:
        selector = mask.to(dtype=torch.bool).expand_as(probability)
    if region is not None:
        region_b = region.to(dtype=torch.bool).expand_as(probability)
        selector = region_b if selector is None else (selector & region_b)

    if selector is None:
        return probability.flatten(1).mean(dim=1)

    counts = selector.flatten(1).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError(
            "no valid cells inside the attribution region; the target is "
            "undefined. Returning zero here would be indistinguishable from a "
            "forecast of no rain."
        )
    totals = (probability * selector).flatten(1).sum(dim=1)
    return totals / counts


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntegratedGradientsResult:
    """Attribution, aggregated, with the settings that produced it."""

    #: Signed total attribution per channel, in tensor order. Sign is
    #: meaningful: negative means the channel pushed the probability *down*.
    channel_scores: dict[str, float]
    #: Absolute attribution mass per channel. Ranking uses this, because a
    #: channel that pushed hard in both directions across the domain is
    #: influential even when its signed total is near zero.
    channel_magnitude: dict[str, float]
    #: ``(T,)`` absolute attribution per input frame, oldest first.
    frame_magnitude: npt.NDArray[np.float32]
    #: ``(H', W')`` absolute attribution summed over channels and time.
    spatial_map: npt.NDArray[np.float32]
    channel_names: tuple[str, ...]

    target: str
    threshold_mm_h: float
    #: The relaxation width. A map cannot be read without it.
    temperature_mm_h: float
    members: int
    n_steps: int
    seed: int
    #: Captum's completeness residual: the gap between the summed attribution
    #: and ``F(input) - F(baseline)``. Large means ``n_steps`` was too small
    #: for this sample, and the map should be distrusted rather than read.
    convergence_delta: float

    def top_drivers(self, k: int = 5) -> list[tuple[str, float]]:
        """The ``k`` channels with the greatest attribution mass, with signs."""
        ranked = sorted(
            self.channel_magnitude.items(), key=lambda item: item[1], reverse=True
        )
        return [(name, self.channel_scores[name]) for name, _ in ranked[:k]]

    def summary(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "threshold_mm_h": self.threshold_mm_h,
            "relaxation_width_mm_h": self.temperature_mm_h,
            "members": self.members,
            "n_steps": self.n_steps,
            "seed": self.seed,
            "convergence_delta": round(self.convergence_delta, 6),
            "top_drivers": [
                {"channel": name, "signed_attribution": round(score, 6)}
                for name, score in self.top_drivers()
            ],
        }


# ---------------------------------------------------------------------------
# The wrapped relay
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _parameters_frozen(model: torch.nn.Module) -> Iterator[None]:
    """Disable parameter gradients for the duration of an attribution."""
    previous = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for parameter, _ in previous:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, state in previous:
            parameter.requires_grad_(state)


class _RelayProbability(torch.nn.Module):
    """The relay reduced to one differentiable scalar per sample."""

    def __init__(
        self,
        model: IndraFusion,
        noise: torch.Tensor,
        *,
        threshold: float,
        temperature: float,
        mask: torch.Tensor | None,
        region: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.model = model
        # (N, B, C, R, R), drawn once. Registered as a buffer so a device move
        # carries it along with the model.
        self.register_buffer("noise", noise, persistent=False)
        self.threshold = threshold
        self.temperature = temperature
        self.mask = mask
        self.region = region
        self.base_batch = noise.shape[1]

    def _noise_for(self, batch: int, index: int) -> torch.Tensor:
        """One member's noise, tiled to Captum's expanded batch."""
        member = self.noise[index]
        if batch == self.base_batch:
            return member
        if batch % self.base_batch:
            raise ValueError(
                f"Captum passed a batch of {batch}, which is not a multiple of "
                f"the {self.base_batch} the noise was drawn for; the noise "
                "cannot be aligned to the interpolation steps"
            )
        return member.repeat(batch // self.base_batch, 1, 1, 1)

    def _tile(self, tensor: torch.Tensor | None, batch: int) -> torch.Tensor | None:
        """Align a per-sample tensor to Captum's expanded batch."""
        if tensor is None or tensor.shape[0] == 1 or batch == self.base_batch:
            return tensor
        repeats = [batch // self.base_batch] + [1] * (tensor.ndim - 1)
        return tensor.repeat(*repeats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone and bridge once, as in the trainer: the conditioning does
        # not depend on the noise, and recomputing it per member would
        # multiply the dominant cost of every interpolation step.
        latent = self.model.earthformer(x)
        conditioning = self.model.adapter(latent)

        batch = x.shape[0]
        members = [
            self.model.generator(conditioning, self._noise_for(batch, index))
            for index in range(self.noise.shape[0])
        ]
        samples = torch.stack(members)

        mask = self._tile(self.mask, batch)
        region = self._tile(self.region, batch)

        return soft_exceedance_probability(
            samples,
            threshold=self.threshold,
            temperature=self.temperature,
            mask=mask,
            region=region,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _pool(field: torch.Tensor, resolution: int) -> npt.NDArray[np.float32]:
    """Reduce an ``(H, W)`` map to ``(resolution, resolution)`` by summing."""
    height = field.shape[-1]
    if height % resolution:
        raise ValueError(
            f"attribution map resolution {resolution} does not divide the "
            f"{height}-cell grid"
        )
    factor = height // resolution
    pooled = F.avg_pool2d(field.unsqueeze(0).unsqueeze(0), factor) * (factor**2)
    return pooled.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def integrated_gradients(
    model: IndraFusion,
    x: torch.Tensor,
    baseline: ClimatologicalBaseline,
    config: IntegratedGradientsConfig,
    *,
    threshold_mm_h: float,
    seed: int,
    channel_names: tuple[str, ...],
    mask: torch.Tensor | None = None,
    region: torch.Tensor | None = None,
    output_resolution: int = 96,
) -> IntegratedGradientsResult:
    """Attribute the heavy-rain probability to the input window."""
    from captum.attr import IntegratedGradients

    if x.ndim != 5 or x.shape[0] != 1:
        raise ValueError(
            f"expected a single window shaped (1, T, C, H, W); got "
            f"{tuple(x.shape)}. Attribution explains one issued nowcast."
        )
    if len(channel_names) != x.shape[2]:
        raise ValueError(
            f"{len(channel_names)} channel names for {x.shape[2]} channels"
        )
    if baseline.channel_names != channel_names:
        raise ValueError(
            "the baseline's channel order differs from the window's. They come "
            "from one configuration and disagreeing means one of them is stale."
        )

    model.eval()
    x = x.detach().requires_grad_(True)
    baseline_tensor = baseline.as_tensor(x)

    # Drawn once, from the forecast's own seed, and held constant along the
    # entire integration path.
    noise = model.draw_noise(
        members=config.members,
        seed=seed,
        batch=x.shape[0],
        device=x.device,
        dtype=x.dtype,
    )

    wrapped = _RelayProbability(
        model,
        noise,
        threshold=threshold_mm_h,
        temperature=config.surrogate_temperature_mm_h,
        mask=mask,
        region=region,
    )

    logger.info(
        "integrated gradients: %d steps x %d members, T=%.2f mm/h, seed %d",
        config.n_steps,
        config.members,
        config.surrogate_temperature_mm_h,
        seed,
    )

    with _parameters_frozen(model):
        explainer = IntegratedGradients(wrapped)
        attributions, delta = explainer.attribute(
            inputs=x,
            baselines=baseline_tensor,
            n_steps=config.n_steps,
            internal_batch_size=config.internal_batch_size,
            return_convergence_delta=True,
        )

    attributions = attributions.detach()[0]  # (T, C, H, W)
    magnitude = attributions.abs()

    channel_scores = {
        name: float(attributions[:, index].sum().item())
        for index, name in enumerate(channel_names)
    }
    channel_magnitude = {
        name: float(magnitude[:, index].sum().item())
        for index, name in enumerate(channel_names)
    }
    frame_magnitude = (
        magnitude.sum(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float32)
    )
    spatial_map = _pool(magnitude.sum(dim=(0, 1)), output_resolution)

    convergence = float(delta.abs().max().item())
    # Completeness is an axiom of the method, not a nicety: the attributions
    # must sum to the change in the target. Both quantities are in the
    # target's units -- probability -- so the residual is compared against the
    # total signed attribution rather than against attribution *magnitude*,
    # which is a different quantity and would make the test meaningless.
    total_signed = abs(sum(channel_scores.values()))
    if convergence > 0.05 * max(total_signed, 1e-6):
        logger.warning(
            "integrated gradients convergence delta is %.4g against a total "
            "attribution of %.4g; raise n_steps above %d before reading this "
            "map as a decomposition",
            convergence,
            total_signed,
            config.n_steps,
        )

    return IntegratedGradientsResult(
        channel_scores=channel_scores,
        channel_magnitude=channel_magnitude,
        frame_magnitude=frame_magnitude,
        spatial_map=spatial_map,
        channel_names=tuple(channel_names),
        target=config.target,
        threshold_mm_h=threshold_mm_h,
        temperature_mm_h=config.surrogate_temperature_mm_h,
        members=config.members,
        n_steps=config.n_steps,
        seed=seed,
        convergence_delta=convergence,
    )


__all__ = [
    "IntegratedGradientsResult",
    "integrated_gradients",
    "soft_exceedance_probability",
]
