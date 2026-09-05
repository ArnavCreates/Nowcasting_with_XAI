"""Bounded reservoir of extreme convective events, replayed to resist forgetting."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import AssembledWindow, QCFlag, SourceStream

logger = logging.getLogger(__name__)

#: Bumped when a field changes meaning. A buffer written by an older schema is
#: refused rather than silently reinterpreted.
SCHEMA_VERSION = 1

_HEADER_KIND = "header"
_RECORD_KIND = "record"


def _as_utc(moment: datetime) -> datetime:
    """Normalise to timezone-aware UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetentionPolicy:
    """The gate and the ranking rule."""

    # -- gate --------------------------------------------------------------
    #: How many of the most recent frames the observation gate covers.
    gate_frames: int = 4

    #: Streams that must be genuinely observed across the gate frames. IMDAA
    #: is excluded: it is a reanalysis product, so "observed" does not mean
    #: the same thing, and the static priors have no time axis at all.
    required_streams: tuple[str, ...] = (
        SourceStream.INSAT.value,
        SourceStream.IMD_SURFACE.value,
    )

    #: Any of these on a gate frame disqualifies the window outright.
    #:
    #: This is a separate test from ``is_observed``, and the distinction is the
    #: whole point: a saturated detector produces a frame that is *genuinely
    #: observed* and simultaneously worthless. ``is_observed`` would pass it,
    #: the intensity score would love it, and the buffer would fill with
    #: instrument failures ranked as record rainfall.
    disqualifying_flags: QCFlag = (
        QCFlag.SATURATED
        | QCFlag.SCANLINE_DROPOUT
        | QCFlag.CALIBRATION_FAILED
        | QCFlag.PARTIAL_COVERAGE
    )

    #: Minimum share of valid cells across the gate frames.
    min_valid_fraction: float = 0.85
    #: Minimum share of valid cells across the target frames. An event whose
    #: target is half-missing is mostly masked out by the loss, so replaying
    #: it teaches almost nothing while costing a full rehydration.
    min_target_valid_fraction: float = 0.85
    #: The score is derived from this channel, so its own coverage is gated
    #: separately -- a window can look healthy overall while precipitation
    #: specifically is half absent.
    precip_channel: str = "imd_precip"
    min_precip_coverage: float = 0.85

    # -- score -------------------------------------------------------------
    #: mm h-1. IMD "heavy rainfall", matching
    #: ``configs/inference/nowcast.yaml`` thresholds.precipitation_mm_h.heavy.
    heavy_threshold_mm_h: float = 15.0

    # -- admission ---------------------------------------------------------
    admission_quantile: float = 0.90
    #: Absolute floor used before enough scores exist to estimate a quantile.
    #: A conservative default, not a climatological statistic; the
    #: training configuration is expected to set it.
    min_exceedance_area: float = 0.001
    #: Bounded sample of scores from which the quantile is estimated.
    score_history_size: int = 4096
    #: Below this many observed scores the quantile is meaningless and the
    #: absolute floor is used instead.
    min_history: int = 256

    # -- diversity ---------------------------------------------------------
    #: Two windows closer than this in time are treated as the same event.
    #: Without it a single slow-moving monsoon depression contributes a window
    #: every thirty minutes for a day and monopolises the reservoir.
    min_separation_minutes: int = 180

    def __post_init__(self) -> None:
        if self.gate_frames < 1:
            raise ValueError(
                f"gate_frames must be at least 1; got {self.gate_frames}. A "
                "zero-frame gate admits every window regardless of quality."
            )
        for name in (
            "min_valid_fraction",
            "min_target_valid_fraction",
            "min_precip_coverage",
            "min_exceedance_area",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]; got {value}")
        if not 0.0 < self.admission_quantile < 1.0:
            raise ValueError(
                f"admission_quantile must lie strictly in (0, 1); got "
                f"{self.admission_quantile}"
            )
        if self.heavy_threshold_mm_h <= 0:
            raise ValueError(
                f"heavy_threshold_mm_h must be positive; got "
                f"{self.heavy_threshold_mm_h}"
            )
        if self.score_history_size < 1 or self.min_history < 1:
            raise ValueError("score history sizes must be positive")
        if self.min_separation_minutes < 0:
            raise ValueError("min_separation_minutes cannot be negative")
        if not self.required_streams:
            logger.warning(
                "no required_streams: the observation gate will not check "
                "whether any frame was genuinely observed, and gap-filled "
                "frames will be archived as extreme events"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "gate_frames": self.gate_frames,
            "required_streams": list(self.required_streams),
            "disqualifying_flags": int(self.disqualifying_flags),
            "min_valid_fraction": self.min_valid_fraction,
            "min_target_valid_fraction": self.min_target_valid_fraction,
            "precip_channel": self.precip_channel,
            "min_precip_coverage": self.min_precip_coverage,
            "heavy_threshold_mm_h": self.heavy_threshold_mm_h,
            "admission_quantile": self.admission_quantile,
            "min_exceedance_area": self.min_exceedance_area,
            "score_history_size": self.score_history_size,
            "min_history": self.min_history,
            "min_separation_minutes": self.min_separation_minutes,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RetentionPolicy:
        payload = dict(payload)
        payload["required_streams"] = tuple(payload["required_streams"])
        payload["disqualifying_flags"] = QCFlag(payload["disqualifying_flags"])
        return cls(**payload)


# ---------------------------------------------------------------------------
# Records and decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayRecord:
    """One archived event."""

    valid_time: datetime
    score: float
    reservoir_key: float
    metrics: dict[str, float] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": _RECORD_KIND,
            "valid_time": _as_utc(self.valid_time).isoformat(),
            "score": self.score,
            "reservoir_key": self.reservoir_key,
            "metrics": self.metrics,
            "gate": self.gate,
            "created_at": _as_utc(self.created_at).isoformat(),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ReplayRecord:
        return cls(
            valid_time=_as_utc(datetime.fromisoformat(payload["valid_time"])),
            score=float(payload["score"]),
            reservoir_key=float(payload["reservoir_key"]),
            metrics=dict(payload.get("metrics", {})),
            gate=dict(payload.get("gate", {})),
            created_at=_as_utc(datetime.fromisoformat(payload["created_at"])),
        )


@dataclass(frozen=True)
class ReplayDecision:
    """Why a window was or was not archived."""

    admitted: bool
    code: str
    reason: str
    score: float | None = None
    record: ReplayRecord | None = None
    #: False when the window failed the gate, so no score was computed and
    #: none was contributed to the quantile estimate.
    gate_passed: bool = False


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------


class ExperienceReplayBuffer:
    """Bounded weighted reservoir of extreme convective events."""

    def __init__(
        self,
        capacity: int,
        policy: RetentionPolicy | None = None,
        seed: int = 0,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1; got {capacity}")

        self.capacity = capacity
        self.policy = policy or RetentionPolicy()
        self.seed = seed

        # Explicit, seeded, and never the module-level ``random`` -- the
        # reservoir lottery has to be reproducible independently of whatever
        # else in the process draws random numbers.
        self._rng = random.Random(seed)

        # Records are kept in a plain list. Capacity is in the hundreds, so an
        # O(n) scan for the minimum key or a temporal neighbour is irrelevant
        # beside the 219 MiB rehydration it guards, and a list survives the
        # arbitrary removals the decorrelation guard performs -- which a heap
        # does not.
        self._records: list[ReplayRecord] = []

        self._score_history: list[float] = []
        self._scores_seen = 0
        self._decisions: Counter[str] = Counter()

    # ----------------------------------------------------------- container
    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ReplayRecord]:
        return iter(self._records)

    @property
    def records(self) -> tuple[ReplayRecord, ...]:
        return tuple(self._records)

    @property
    def is_full(self) -> bool:
        return len(self._records) >= self.capacity

    # ---------------------------------------------------------------- gate
    def _gate(
        self,
        window: AssembledWindow,
        target_validity: npt.NDArray[np.bool_],
    ) -> ReplayDecision | dict[str, Any]:
        """Run the admission gate. Returns a rejection, or the gate metrics."""
        policy = self.policy

        if not window.accepted:
            return ReplayDecision(
                False,
                "rejected_window_not_accepted",
                f"window was rejected upstream: {window.rejection_reason}",
            )

        n_time = window.sequence_length
        gate_slice = slice(max(n_time - policy.gate_frames, 0), n_time)
        gate_indices = range(gate_slice.start, gate_slice.stop)

        for stream in policy.required_streams:
            flags = window.flags.get(stream)
            if flags is None:
                return ReplayDecision(
                    False,
                    "rejected_stream_absent",
                    f"required stream {stream!r} carries no QC flags",
                )
            if len(flags) < n_time:
                return ReplayDecision(
                    False,
                    "rejected_flags_truncated",
                    f"stream {stream!r} has {len(flags)} flags for {n_time} frames",
                )

            for index in gate_indices:
                flag = flags[index]
                # is_observed, not is_usable. A gap-filled frame is legitimate
                # model input and is exactly what must not be archived as an
                # extreme event -- the buffer would be teaching the model to
                # reproduce the optical flow interpolator.
                if not flag.is_observed:
                    return ReplayDecision(
                        False,
                        "rejected_not_observed",
                        f"{stream!r} frame {index} is reconstructed, not "
                        f"observed: {flag.describe()}",
                    )
                overlap = flag & policy.disqualifying_flags
                if overlap:
                    return ReplayDecision(
                        False,
                        "rejected_disqualifying_flag",
                        f"{stream!r} frame {index} carries "
                        f"{overlap.describe()}, which mimics extreme "
                        "intensity without being it",
                    )

        validity = window.validity
        if validity.size == 0:
            return ReplayDecision(
                False, "rejected_no_validity", "window carries no validity mask"
            )
        gate_validity = float(np.mean(validity[gate_slice]))
        if gate_validity < policy.min_valid_fraction:
            return ReplayDecision(
                False,
                "rejected_low_validity",
                f"gate-frame validity {gate_validity:.3f} is below "
                f"{policy.min_valid_fraction}",
            )

        precip_coverage = window.channel_coverage.get(policy.precip_channel)
        if precip_coverage is None:
            return ReplayDecision(
                False,
                "rejected_precip_coverage_unknown",
                f"no coverage recorded for channel {policy.precip_channel!r}",
            )
        if precip_coverage < policy.min_precip_coverage:
            return ReplayDecision(
                False,
                "rejected_low_precip_coverage",
                f"{policy.precip_channel} coverage {precip_coverage:.3f} is "
                f"below {policy.min_precip_coverage}; the score would be "
                "computed from mostly-absent data",
            )

        target_valid_fraction = float(np.mean(target_validity))
        if target_valid_fraction < policy.min_target_valid_fraction:
            return ReplayDecision(
                False,
                "rejected_low_target_validity",
                f"target validity {target_valid_fraction:.3f} is below "
                f"{policy.min_target_valid_fraction}; most of the replayed "
                "sample would be masked out of the loss",
            )

        return {
            "validity_fraction": round(gate_validity, 4),
            "target_validity_fraction": round(target_valid_fraction, 4),
            "precip_coverage": round(float(precip_coverage), 4),
            "gate_frames": [int(i) for i in gate_indices],
            "observed_streams": list(policy.required_streams),
        }

    # --------------------------------------------------------------- score
    def _score(
        self,
        target_mm_h: npt.NDArray[np.float32],
        target_validity: npt.NDArray[np.bool_],
    ) -> dict[str, float]:
        """Peak exceedance area over the forecast horizon, plus diagnostics."""
        threshold = self.policy.heavy_threshold_mm_h
        n_time = target_mm_h.shape[0]

        peak_area = 0.0
        peak_frame = -1
        for index in range(n_time):
            valid = target_validity[index]
            n_valid = int(np.count_nonzero(valid))
            if n_valid == 0:
                continue
            field_ = target_mm_h[index]
            exceeding = int(np.count_nonzero(valid & (field_ >= threshold)))
            area = exceeding / n_valid
            if area > peak_area:
                peak_area, peak_frame = area, index

        valid_values = target_mm_h[target_validity]
        if valid_values.size:
            p999 = float(np.percentile(valid_values, 99.9))
            maximum = float(valid_values.max())
        else:
            p999 = maximum = 0.0

        return {
            "peak_exceedance_area": round(peak_area, 6),
            "peak_frame": float(peak_frame),
            # Recorded for the manifest, never used for ranking: a maximum is
            # exactly the statistic a single artifact pixel can dictate.
            "p999_mm_h": round(p999, 3),
            "max_mm_h": round(maximum, 3),
        }

    # ------------------------------------------------------ score history
    def _observe_score(self, score: float) -> None:
        """Algorithm R over the score stream."""
        self._scores_seen += 1
        if len(self._score_history) < self.policy.score_history_size:
            self._score_history.append(score)
            return
        index = self._rng.randrange(self._scores_seen)
        if index < len(self._score_history):
            self._score_history[index] = score

    def admission_threshold(self) -> tuple[float, str]:
        """Current admission floor, and which rule produced it."""
        if len(self._score_history) < self.policy.min_history:
            return self.policy.min_exceedance_area, "cold_start_floor"
        estimate = float(
            np.quantile(
                np.asarray(self._score_history, dtype=np.float64),
                self.policy.admission_quantile,
            )
        )
        return estimate, "streaming_quantile"

    # ----------------------------------------------------------- reservoir
    def _reservoir_key(self, score: float) -> float:
        """A-Res key in log space: ``log(u) / w``."""
        u = self._rng.random()
        while u <= 0.0:  # log(0) is undefined; a zero draw is astronomically rare
            u = self._rng.random()
        return math.log(u) / score

    # -------------------------------------------------------------- public
    def consider(
        self,
        window: AssembledWindow,
        target_mm_h: npt.NDArray[np.float32],
        target_validity: npt.NDArray[np.bool_],
    ) -> ReplayDecision:
        """Offer one window to the buffer."""
        if target_mm_h.ndim != 4 or target_validity.ndim != 4:
            raise ValueError(
                f"target arrays must be (T, 1, H, W); got "
                f"{target_mm_h.shape} and {target_validity.shape}"
            )
        if target_mm_h.shape != target_validity.shape:
            raise ValueError(
                f"target {target_mm_h.shape} and its validity "
                f"{target_validity.shape} must have the same shape"
            )
        if target_validity.dtype != np.bool_:
            raise TypeError(
                f"target_validity must be boolean; got {target_validity.dtype}"
            )

        outcome = self._gate(window, target_validity)
        if isinstance(outcome, ReplayDecision):
            return self._record_decision(outcome)
        gate_metrics = outcome

        metrics = self._score(target_mm_h, target_validity)
        score = metrics["peak_exceedance_area"]
        self._observe_score(score)

        if score <= 0.0:
            return self._record_decision(
                ReplayDecision(
                    False,
                    "rejected_zero_score",
                    "no valid cell reached the heavy-rain threshold",
                    score=score,
                    gate_passed=True,
                )
            )

        threshold, rule = self.admission_threshold()
        if score < threshold:
            return self._record_decision(
                ReplayDecision(
                    False,
                    "rejected_below_threshold",
                    f"score {score:.5f} is below the {rule} threshold "
                    f"{threshold:.5f}",
                    score=score,
                    gate_passed=True,
                )
            )

        valid_time = _as_utc(window.valid_time)
        separation = timedelta(minutes=self.policy.min_separation_minutes)
        neighbours = [
            record
            for record in self._records
            if abs(_as_utc(record.valid_time) - valid_time) < separation
        ]
        if neighbours:
            strongest = max(neighbours, key=lambda record: record.score)
            if score <= strongest.score:
                return self._record_decision(
                    ReplayDecision(
                        False,
                        "rejected_temporal_duplicate",
                        f"within {self.policy.min_separation_minutes} min of "
                        f"{strongest.valid_time.isoformat()}, which scores "
                        f"{strongest.score:.5f}",
                        score=score,
                        gate_passed=True,
                    )
                )
            # The candidate supersedes the whole cluster, not just the weakest
            # member: they are all the same storm.
            for record in neighbours:
                self._records.remove(record)
            logger.debug(
                "%s supersedes %d neighbouring window(s) of the same event",
                valid_time.isoformat(),
                len(neighbours),
            )

        candidate = ReplayRecord(
            valid_time=valid_time,
            score=score,
            reservoir_key=self._reservoir_key(score),
            metrics=metrics,
            gate=gate_metrics,
        )

        if len(self._records) < self.capacity:
            self._records.append(candidate)
            return self._record_decision(
                ReplayDecision(
                    True,
                    "admitted",
                    f"archived with score {score:.5f}",
                    score=score,
                    record=candidate,
                    gate_passed=True,
                )
            )

        weakest = min(self._records, key=lambda record: record.reservoir_key)
        if candidate.reservoir_key <= weakest.reservoir_key:
            return self._record_decision(
                ReplayDecision(
                    False,
                    "rejected_reservoir_key",
                    f"reservoir key {candidate.reservoir_key:.4f} does not "
                    f"beat the weakest held key {weakest.reservoir_key:.4f}",
                    score=score,
                    gate_passed=True,
                )
            )

        self._records.remove(weakest)
        self._records.append(candidate)
        return self._record_decision(
            ReplayDecision(
                True,
                "admitted_evicting",
                f"archived with score {score:.5f}, evicting "
                f"{weakest.valid_time.isoformat()}",
                score=score,
                record=candidate,
                gate_passed=True,
            )
        )

    def _record_decision(self, decision: ReplayDecision) -> ReplayDecision:
        self._decisions[decision.code] += 1
        return decision

    def sample(self, n: int, rng: random.Random | None = None) -> list[ReplayRecord]:
        """Draw ``n`` events uniformly without replacement."""
        if n < 0:
            raise ValueError(f"n must be non-negative; got {n}")
        if not self._records:
            return []
        if n >= len(self._records):
            # Early in a run the buffer is legitimately smaller than the
            # request; returning what exists is correct, and the caller sizes
            # the batch from the result rather than assuming n.
            return list(self._records)
        return (rng or self._rng).sample(self._records, n)

    def stats(self) -> dict[str, Any]:
        """Counts and thresholds, for the training log."""
        threshold, rule = self.admission_threshold()
        times = sorted(_as_utc(record.valid_time) for record in self._records)
        scores = [record.score for record in self._records]
        return {
            "size": len(self._records),
            "capacity": self.capacity,
            "scores_seen": self._scores_seen,
            "score_history": len(self._score_history),
            "admission_threshold": threshold,
            "admission_rule": rule,
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "earliest": times[0].isoformat() if times else None,
            "latest": times[-1].isoformat() if times else None,
            "decisions": dict(self._decisions),
        }

    # -------------------------------------------------------- persistence
    def save(self, path: Path | str) -> Path:
        """Write the manifest as JSON Lines, atomically."""
        target = Path(path)
        state = self._rng.getstate()
        header = {
            "kind": _HEADER_KIND,
            "schema_version": SCHEMA_VERSION,
            "capacity": self.capacity,
            "seed": self.seed,
            "policy": self.policy.to_json(),
            "scores_seen": self._scores_seen,
            "score_history": self._score_history,
            "decisions": dict(self._decisions),
            "rng_state": [state[0], list(state[1]), state[2]],
            "written_at": datetime.now(tz=UTC).isoformat(),
        }

        lines = [json.dumps(header)]
        lines.extend(
            json.dumps(record.to_json())
            for record in sorted(
                self._records, key=lambda record: _as_utc(record.valid_time)
            )
        )

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, target)

        logger.info("wrote %d replay records to %s", len(self._records), target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> ExperienceReplayBuffer:
        """Rebuild a buffer from its manifest, including its sampling state."""
        source = Path(path)
        lines = [
            line for line in source.read_text(encoding="utf-8").splitlines() if line
        ]
        if not lines:
            raise ValueError(f"{source} is empty; it carries no header")

        header = json.loads(lines[0])
        if header.get("kind") != _HEADER_KIND:
            raise ValueError(
                f"{source} does not start with a header line; it was not "
                "written by this buffer"
            )
        version = header.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{source} uses schema version {version}, this build writes "
                f"{SCHEMA_VERSION}. Refusing to reinterpret fields whose "
                "meaning may have changed."
            )

        buffer = cls(
            capacity=int(header["capacity"]),
            policy=RetentionPolicy.from_json(header["policy"]),
            seed=int(header["seed"]),
        )
        buffer._scores_seen = int(header.get("scores_seen", 0))
        buffer._score_history = [float(v) for v in header.get("score_history", [])]
        buffer._decisions = Counter(header.get("decisions", {}))

        raw_state = header.get("rng_state")
        if raw_state is not None:
            # json gives lists where Random.setstate demands tuples.
            buffer._rng.setstate((raw_state[0], tuple(raw_state[1]), raw_state[2]))

        for line in lines[1:]:
            payload = json.loads(line)
            if payload.get("kind") != _RECORD_KIND:
                raise ValueError(f"unexpected line kind {payload.get('kind')!r}")
            buffer._records.append(ReplayRecord.from_json(payload))

        if len(buffer._records) > buffer.capacity:
            raise ValueError(
                f"{source} holds {len(buffer._records)} records but declares "
                f"capacity {buffer.capacity}"
            )

        logger.info("loaded %d replay records from %s", len(buffer._records), source)
        return buffer


__all__ = [
    "SCHEMA_VERSION",
    "ExperienceReplayBuffer",
    "ReplayDecision",
    "ReplayRecord",
    "RetentionPolicy",
]
