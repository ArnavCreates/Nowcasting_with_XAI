"""CSI, POD, FAR, MAE and CRPS for an ensemble precipitation nowcast."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

#: Returned when a metric's denominator is zero. See the module docstring.
UNDEFINED = float("nan")

#: Metric names this module can compute, as they appear in
#: ``validation.metrics``. Threshold-dependent names gain a band suffix --
#: "csi" becomes "csi_heavy" -- matching what ``checkpointing.monitor``
#: accepts.
CONTINGENCY_METRICS: tuple[str, ...] = ("csi", "pod", "far")
SCALAR_METRICS: tuple[str, ...] = ("crps", "mae_mm_h")
SUPPORTED_METRICS: tuple[str, ...] = CONTINGENCY_METRICS + SCALAR_METRICS


def _check_shapes(
    samples: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
) -> None:
    if samples.ndim != 6:
        raise ValueError(
            f"samples must be (N, B, T, C, H, W); got {tuple(samples.shape)}"
        )
    if target.ndim != 5:
        raise ValueError(f"target must be (B, T, C, H, W); got {tuple(target.shape)}")
    if samples.shape[1:] != target.shape:
        raise ValueError(
            f"samples {tuple(samples.shape)} do not match target "
            f"{tuple(target.shape)} on the batch and field axes"
        )
    if mask is not None and mask.shape[:2] != target.shape[:2]:
        raise ValueError(
            f"mask {tuple(mask.shape)} does not match target "
            f"{tuple(target.shape)} on the batch and time axes"
        )


def _valid(mask: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    """Broadcast the validity mask, or an all-true mask when none is given."""
    if mask is None:
        return torch.ones_like(like, dtype=torch.bool)
    return mask.to(dtype=torch.bool).expand_as(like)


# ---------------------------------------------------------------------------
# Contingency table
# ---------------------------------------------------------------------------


@dataclass
class ContingencyTable:
    """Per-member event counts at one intensity threshold."""

    hits: torch.Tensor
    misses: torch.Tensor
    false_alarms: torch.Tensor
    correct_negatives: torch.Tensor
    threshold: float

    def __add__(self, other: ContingencyTable) -> ContingencyTable:
        if abs(self.threshold - other.threshold) > 1e-12:
            raise ValueError(
                f"cannot pool tables at different thresholds: "
                f"{self.threshold} and {other.threshold}"
            )
        return ContingencyTable(
            hits=self.hits + other.hits,
            misses=self.misses + other.misses,
            false_alarms=self.false_alarms + other.false_alarms,
            correct_negatives=self.correct_negatives + other.correct_negatives,
            threshold=self.threshold,
        )

    def _ratio(self, numerator: torch.Tensor, denominator: torch.Tensor) -> float:
        """Per-member ratio, averaged over members that are defined."""
        defined = denominator > 0
        if not bool(defined.any()):
            return UNDEFINED
        values = numerator[defined].to(torch.float64) / denominator[defined].to(
            torch.float64
        )
        return float(values.mean().item())

    @property
    def csi(self) -> float:
        """Critical Success Index: hits / (hits + misses + false alarms)."""
        return self._ratio(self.hits, self.hits + self.misses + self.false_alarms)

    @property
    def pod(self) -> float:
        """Probability of Detection: hits / (hits + misses)."""
        return self._ratio(self.hits, self.hits + self.misses)

    @property
    def far(self) -> float:
        """False Alarm Ratio: false alarms / (hits + false alarms)."""
        return self._ratio(self.false_alarms, self.hits + self.false_alarms)

    @property
    def events_observed(self) -> int:
        """Observed events pooled across members, for reporting confidence."""
        return int((self.hits + self.misses).sum().item())


def contingency_table(
    samples: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    mask: torch.Tensor | None = None,
) -> ContingencyTable:
    """Count hits, misses and false alarms per member at one threshold."""
    _check_shapes(samples, target, mask)

    observed = target >= threshold
    forecast = samples >= threshold
    valid = _valid(mask, target)

    # Everything below broadcasts (B, T, C, H, W) against
    # (N, B, T, C, H, W); the reduction leaves the member axis alone.
    reduce_over = tuple(range(1, samples.ndim))
    # Both already carry the validity mask, so no term below needs to re-apply
    # it: an invalid cell is neither observed-and-wet nor observed-and-clear,
    # and so falls out of every count.
    observed_b = (observed & valid).unsqueeze(0)
    clear_b = (~observed & valid).unsqueeze(0)

    hits = (forecast & observed_b).sum(dim=reduce_over)
    misses = (~forecast & observed_b).sum(dim=reduce_over)
    false_alarms = (forecast & clear_b).sum(dim=reduce_over)
    correct_negatives = (~forecast & clear_b).sum(dim=reduce_over)

    return ContingencyTable(
        hits=hits,
        misses=misses,
        false_alarms=false_alarms,
        correct_negatives=correct_negatives,
        threshold=float(threshold),
    )


# ---------------------------------------------------------------------------
# Continuous scores
# ---------------------------------------------------------------------------


def masked_absolute_error(
    samples: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[float, int]:
    """Mean absolute error of the **ensemble mean**, and the cell count."""
    _check_shapes(samples, target, mask)

    ensemble_mean = samples.mean(dim=0)
    valid = _valid(mask, target)
    error = (ensemble_mean - target).abs()

    total = float(error[valid].to(torch.float64).sum().item())
    count = int(valid.sum().item())
    return total, count


def crps_ensemble(
    samples: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[float, int]:
    """Continuous Ranked Probability Score, non-parametric estimator."""
    _check_shapes(samples, target, mask)

    n_members = samples.shape[0]
    valid = _valid(mask, target)
    valid_b = valid.unsqueeze(0)

    accuracy = (samples - target.unsqueeze(0)).abs()
    accuracy_sum = accuracy.masked_fill(~valid_b, 0.0).sum(dim=0).to(torch.float64)

    spread_sum = torch.zeros_like(target, dtype=torch.float64)
    for index in range(n_members):
        differences = (samples[index].unsqueeze(0) - samples).abs()
        spread_sum += (
            differences.masked_fill(~valid_b, 0.0).sum(dim=0).to(torch.float64)
        )

    per_cell = accuracy_sum / n_members - spread_sum / (2 * n_members**2)

    total = float(per_cell[valid].sum().item())
    count = int(valid.sum().item())
    return total, count


# ---------------------------------------------------------------------------
# Accumulation across a validation pass
# ---------------------------------------------------------------------------


@dataclass
class MetricAccumulator:
    """Pools a whole validation pass before forming any ratio."""

    threshold: float
    threshold_name: str = "heavy"
    metrics: Sequence[str] = field(default_factory=lambda: list(SUPPORTED_METRICS))

    _table: ContingencyTable | None = field(default=None, init=False, repr=False)
    _mae_sum: float = field(default=0.0, init=False, repr=False)
    _mae_count: int = field(default=0, init=False, repr=False)
    _crps_sum: float = field(default=0.0, init=False, repr=False)
    _crps_count: int = field(default=0, init=False, repr=False)
    _windows: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        unknown = [name for name in self.metrics if name not in SUPPORTED_METRICS]
        if unknown:
            raise ValueError(
                f"unsupported metrics {unknown}; this module computes "
                f"{list(SUPPORTED_METRICS)}"
            )
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive; got {self.threshold}")

    @property
    def _wants_contingency(self) -> bool:
        return any(name in self.metrics for name in CONTINGENCY_METRICS)

    def reset(self) -> None:
        self._table = None
        self._mae_sum = self._crps_sum = 0.0
        self._mae_count = self._crps_count = 0
        self._windows = 0

    @torch.no_grad()
    def update(
        self,
        samples: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Fold one batch in. Detached: verification never backpropagates."""
        samples = samples.detach()
        target = target.detach()
        if mask is not None:
            mask = mask.detach()

        if self._wants_contingency:
            table = contingency_table(samples, target, self.threshold, mask)
            self._table = table if self._table is None else self._table + table

        if "mae_mm_h" in self.metrics:
            total, count = masked_absolute_error(samples, target, mask)
            self._mae_sum += total
            self._mae_count += count

        if "crps" in self.metrics:
            total, count = crps_ensemble(samples, target, mask)
            self._crps_sum += total
            self._crps_count += count

        self._windows += int(target.shape[0])

    def compute(self) -> dict[str, float]:
        """Final scores, keyed as ``validation.metrics`` names them."""
        results: dict[str, float] = {}

        if self._wants_contingency:
            if self._table is None:
                for name in CONTINGENCY_METRICS:
                    if name in self.metrics:
                        results[f"{name}_{self.threshold_name}"] = UNDEFINED
            else:
                available = {
                    "csi": self._table.csi,
                    "pod": self._table.pod,
                    "far": self._table.far,
                }
                for name in CONTINGENCY_METRICS:
                    if name in self.metrics:
                        results[f"{name}_{self.threshold_name}"] = available[name]
                results["events_observed"] = float(self._table.events_observed)

        if "mae_mm_h" in self.metrics:
            results["mae_mm_h"] = (
                self._mae_sum / self._mae_count if self._mae_count else UNDEFINED
            )
        if "crps" in self.metrics:
            results["crps"] = (
                self._crps_sum / self._crps_count if self._crps_count else UNDEFINED
            )

        results["windows"] = float(self._windows)

        undefined = [
            name
            for name, value in results.items()
            if isinstance(value, float) and math.isnan(value)
        ]
        if undefined:
            logger.warning(
                "undefined after %d windows: %s. The event may not have "
                "occurred anywhere in this validation set.",
                self._windows,
                ", ".join(sorted(undefined)),
            )
        return results


def evaluate(
    samples: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    threshold: float,
    threshold_name: str = "heavy",
    metrics: Iterable[str] = SUPPORTED_METRICS,
) -> dict[str, float]:
    """Score a single batch."""
    accumulator = MetricAccumulator(
        threshold=threshold,
        threshold_name=threshold_name,
        metrics=list(metrics),
    )
    accumulator.update(samples, target, mask)
    return accumulator.compute()


def higher_is_better(metric: str) -> bool:
    """Whether a larger value of a metric is an improvement."""
    base = metric.split("_")[0]
    if base in ("far", "crps", "mae"):
        return False
    if base in ("csi", "pod"):
        return True
    raise KeyError(f"unknown metric {metric!r}; cannot say which direction is better")


__all__ = [
    "CONTINGENCY_METRICS",
    "SCALAR_METRICS",
    "SUPPORTED_METRICS",
    "UNDEFINED",
    "ContingencyTable",
    "MetricAccumulator",
    "contingency_table",
    "crps_ensemble",
    "evaluate",
    "higher_is_better",
    "masked_absolute_error",
]
