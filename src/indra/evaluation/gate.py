"""Promotion gate: decides whether a retrained checkpoint replaces the incumbent."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .metrics import higher_is_better


@dataclass(frozen=True)
class GateCriteria:
    """Thresholds a candidate must clear."""

    #: Metric deciding promotion, e.g. ``csi_heavy``.
    monitor: str = "csi_heavy"
    #: Required improvement. Zero would promote on floating-point noise.
    min_delta: float = 0.005
    #: Metrics that must not get worse, and by how much they may move.
    guarded: tuple[str, ...] = ("far_heavy", "crps")
    max_regression: float = 0.02
    #: Observed events the validation pass must have contained.
    min_events: int = 50


@dataclass(frozen=True)
class GateDecision:
    """The outcome, and every reason behind it."""

    promote: bool
    monitor: str
    candidate: float | None
    incumbent: float | None
    delta: float | None
    reasons: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "monitor": self.monitor,
            "candidate": self.candidate,
            "incumbent": self.incumbent,
            "delta": self.delta,
            "reasons": list(self.reasons),
        }


def _improvement(metric: str, candidate: float, incumbent: float) -> float:
    """Signed improvement, positive when the candidate is better."""
    return candidate - incumbent if higher_is_better(metric) else incumbent - candidate


def evaluate_gate(
    candidate: dict[str, float],
    incumbent: dict[str, float] | None,
    criteria: GateCriteria | None = None,
) -> GateDecision:
    """Compare two metric sets and decide."""
    criteria = criteria or GateCriteria()
    reasons: list[str] = []

    value = candidate.get(criteria.monitor)
    if value is None or math.isnan(value):
        return GateDecision(
            promote=False,
            monitor=criteria.monitor,
            candidate=value,
            incumbent=None if incumbent is None else incumbent.get(criteria.monitor),
            delta=None,
            reasons=(
                f"{criteria.monitor} is undefined for the candidate; a "
                "validation pass with no observed events cannot promote",
            ),
        )

    events = int(candidate.get("events_observed", 0))
    if events < criteria.min_events:
        reasons.append(
            f"validated over {events} observed events, below the {criteria.min_events} "
            "required; a score from that few events is noise"
        )

    if incumbent is None:
        promote = not reasons
        if promote:
            reasons.append("no incumbent; promoting the first model")
        return GateDecision(
            promote=promote,
            monitor=criteria.monitor,
            candidate=value,
            incumbent=None,
            delta=None,
            reasons=tuple(reasons),
        )

    previous = incumbent.get(criteria.monitor)
    if previous is None or math.isnan(previous):
        # An incumbent whose score is unknown cannot be defended, but it also
        # cannot be shown to be worse. Promote on the candidate's own merits.
        reasons.append(
            f"incumbent {criteria.monitor} is unknown; judging the candidate alone"
        )
        return GateDecision(
            promote=not any("below the" in r for r in reasons),
            monitor=criteria.monitor,
            candidate=value,
            incumbent=previous,
            delta=None,
            reasons=tuple(reasons),
        )

    delta = _improvement(criteria.monitor, value, previous)
    if delta < criteria.min_delta:
        reasons.append(
            f"{criteria.monitor} moved {delta:+.4f}, short of the "
            f"{criteria.min_delta:+.4f} required"
        )

    for metric in criteria.guarded:
        if metric == criteria.monitor:
            continue
        new, old = candidate.get(metric), incumbent.get(metric)
        if new is None or old is None or math.isnan(new) or math.isnan(old):
            continue
        regression = -_improvement(metric, new, old)
        if regression > criteria.max_regression:
            reasons.append(
                f"{metric} regressed by {regression:.4f}, beyond the "
                f"{criteria.max_regression:.4f} allowed"
            )

    promote = not reasons
    if promote:
        reasons.append(
            f"{criteria.monitor} improved {delta:+.4f} over {events} events "
            "with no guarded regression"
        )

    return GateDecision(
        promote=promote,
        monitor=criteria.monitor,
        candidate=value,
        incumbent=previous,
        delta=delta,
        reasons=tuple(reasons),
    )


__all__ = ["GateCriteria", "GateDecision", "evaluate_gate"]
