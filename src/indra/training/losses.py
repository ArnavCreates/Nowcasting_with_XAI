"""Adversarial objectives for the fusion relay."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossWeights:
    """Relative weights of the terms."""

    #: Per-critic weights. Named rather than positional, matching the dict
    #: ``DGMRDiscriminators.forward`` returns.
    adversarial_spatial: float = 1.0
    adversarial_temporal: float = 1.0

    #: lambda in the paper's generator objective. Not a free choice: at 20 the
    #: regularisation dominates early training and hands the critics a
    #: roughly-positioned field to sharpen, rather than asking them to solve
    #: placement and texture simultaneously.
    grid_cell: float = 20.0

    #: Weight on the Earthformer's deterministic head. A neutral default, not
    #: a tuned value -- the training configuration is expected to set it.
    auxiliary: float = 1.0

    #: Saturation point of w(y), in mm h-1. Above this a cell's weight stops
    #: growing, so one extreme pixel cannot dominate the batch gradient.
    grid_cell_max_weight: float = 24.0

    def __post_init__(self) -> None:
        negative = {
            spec.name: getattr(self, spec.name)
            for spec in fields(self)
            if getattr(self, spec.name) < 0
        }
        if negative:
            raise ValueError(
                f"loss weights must be non-negative; got {negative}. A negative "
                "weight inverts the term it scales, which trains the model "
                "toward exactly what that term was added to prevent."
            )
        if self.grid_cell_max_weight <= 0:
            raise ValueError(
                f"grid_cell_max_weight must be positive; got "
                f"{self.grid_cell_max_weight}. At zero every cell weighs "
                "nothing and R vanishes silently."
            )

    def critic_weight(self, name: str) -> float:
        """Weight for a critic, by the name the discriminator container uses."""
        table = {
            "spatial": self.adversarial_spatial,
            "temporal": self.adversarial_temporal,
        }
        if name not in table:
            raise KeyError(
                f"no weight configured for critic '{name}'; known critics are "
                f"{sorted(table)}. An unrecognised critic must not be silently "
                "dropped from the objective or weighted at an assumed 1.0."
            )
        return table[name]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class LossBreakdown:
    """The scalar to backpropagate, plus every term that went into it."""

    total: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)

    def detached(self) -> dict[str, float]:
        """Python floats for the logger. One device sync, deliberately."""
        return {
            name: float(value.detach().cpu()) for name, value in self.components.items()
        }

    def summary(self) -> str:
        return ", ".join(
            f"{name}={value:.4f}" for name, value in sorted(self.detached().items())
        )


# ---------------------------------------------------------------------------
# Masked reduction
# ---------------------------------------------------------------------------


def _masked_mean(
    values: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean over valid cells only."""
    if mask is None:
        return values.mean()

    if mask.dtype != torch.bool:
        raise TypeError(
            f"mask must be boolean; got {mask.dtype}. A float mask would be "
            "multiplied in as a weight rather than used as a selector, which "
            "silently changes the loss for every partially-valid cell."
        )

    expanded = mask.expand_as(values)
    count = expanded.sum()
    if count == 0:
        raise ValueError(
            "no valid cells in this batch; the loss is undefined. Returning "
            "zero here would be indistinguishable from a perfect forecast."
        )
    return (values * expanded).sum() / count


# ---------------------------------------------------------------------------
# Adversarial terms
# ---------------------------------------------------------------------------


def hinge_discriminator(
    real_scores: Mapping[str, torch.Tensor],
    fake_scores: Mapping[str, torch.Tensor],
    weights: LossWeights,
) -> LossBreakdown:
    """Hinge loss for the critics: ``relu(1 - real) + relu(1 + fake)``."""
    if not real_scores:
        raise ValueError(
            "no critic scores supplied; with both discriminators disabled the "
            "generator has no adversarial signal at all"
        )
    if set(real_scores) != set(fake_scores):
        raise ValueError(
            f"critics disagree between real {sorted(real_scores)} and generated "
            f"{sorted(fake_scores)}; both passes must score the same critics"
        )

    components: dict[str, torch.Tensor] = {}
    total: torch.Tensor | None = None

    for name in sorted(real_scores):
        real_term = F.relu(1.0 - real_scores[name]).mean()
        fake_term = F.relu(1.0 + fake_scores[name]).mean()
        critic_loss = real_term + fake_term

        components[f"d_{name}_real"] = real_term
        components[f"d_{name}_fake"] = fake_term
        components[f"d_{name}"] = critic_loss

        scaled = weights.critic_weight(name) * critic_loss
        total = scaled if total is None else total + scaled

    assert total is not None  # guaranteed by the emptiness check above
    components["d_total"] = total
    return LossBreakdown(total=total, components=components)


def hinge_generator(
    fake_scores: Mapping[str, torch.Tensor], weights: LossWeights
) -> LossBreakdown:
    """Generator's adversarial term: ``-mean(score)`` per critic."""
    if not fake_scores:
        raise ValueError(
            "no critic scores supplied; the generator would train on the "
            "regularisation alone and regress to the blurred conditional mean"
        )

    components: dict[str, torch.Tensor] = {}
    total: torch.Tensor | None = None

    for name in sorted(fake_scores):
        critic_loss = -fake_scores[name].mean()
        components[f"g_{name}"] = critic_loss

        scaled = weights.critic_weight(name) * critic_loss
        total = scaled if total is None else total + scaled

    assert total is not None
    components["g_adversarial"] = total
    return LossBreakdown(total=total, components=components)


# ---------------------------------------------------------------------------
# Grid-cell regularisation
# ---------------------------------------------------------------------------


def grid_cell_weights(
    target_mm_h: torch.Tensor, max_weight: float = 24.0
) -> torch.Tensor:
    """``w(y) = clip(y, 0, max_weight)`` over a rain-rate field."""
    if max_weight <= 0:
        raise ValueError(f"max_weight must be positive; got {max_weight}")
    return target_mm_h.clamp(min=0.0, max=max_weight)


def grid_cell_regularizer(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted L1 between the **ensemble mean** and the observation."""
    if samples.ndim != 6:
        raise ValueError(
            f"expected samples (N, B, T, C, H, W); got {tuple(samples.shape)}"
        )
    if samples.shape[1:] != target.shape:
        raise ValueError(
            f"samples are {tuple(samples.shape[1:])} per member but the target "
            f"is {tuple(target.shape)}"
        )
    if samples.shape[0] < 1:
        raise ValueError("at least one sample is needed to form an ensemble mean")

    ensemble_mean = samples.mean(dim=0)
    error = (ensemble_mean - target).abs() * weights
    return _masked_mean(error, mask)


def auxiliary_anchor(
    auxiliary: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unweighted masked L1 on the Earthformer's deterministic head."""
    if auxiliary.shape != target.shape:
        raise ValueError(
            f"auxiliary estimate is {tuple(auxiliary.shape)} but the target is "
            f"{tuple(target.shape)}"
        )
    return _masked_mean((auxiliary - target).abs(), mask)


# ---------------------------------------------------------------------------
# Composed objectives
# ---------------------------------------------------------------------------


def generator_objective(
    fake_scores: Mapping[str, torch.Tensor],
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: LossWeights,
    grid_weights: torch.Tensor,
    mask: torch.Tensor | None = None,
    auxiliary: torch.Tensor | None = None,
) -> LossBreakdown:
    """Full generator objective: adversarial + lambda*R + auxiliary anchor."""
    breakdown = hinge_generator(fake_scores, weights)
    components = dict(breakdown.components)

    regularisation = grid_cell_regularizer(
        samples, target, weights=grid_weights, mask=mask
    )
    components["g_grid_cell"] = regularisation
    total = breakdown.total + weights.grid_cell * regularisation

    if auxiliary is not None:
        anchor = auxiliary_anchor(auxiliary, target, mask)
        components["g_auxiliary"] = anchor
        total = total + weights.auxiliary * anchor

    components["g_total"] = total
    return LossBreakdown(total=total, components=components)


def discriminator_objective(
    real_scores: Mapping[str, torch.Tensor],
    fake_scores: Mapping[str, torch.Tensor],
    weights: LossWeights,
) -> LossBreakdown:
    """Full critic objective."""
    return hinge_discriminator(real_scores, fake_scores, weights)


__all__ = [
    "LossBreakdown",
    "LossWeights",
    "auxiliary_anchor",
    "discriminator_objective",
    "generator_objective",
    "grid_cell_regularizer",
    "grid_cell_weights",
    "hinge_discriminator",
    "hinge_generator",
]
