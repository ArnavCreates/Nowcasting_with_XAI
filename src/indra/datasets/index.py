"""Season-aware enumeration of nowcast windows, and the leakage-safe split."""

from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ..config import ModelConfig, SeasonsConfig, SplitConfig, TrainingConfig

logger = logging.getLogger(__name__)

#: Minutes in a day. A cadence that does not divide this would drift relative
#: to the clock, so successive days would carry different slot times.
_MINUTES_PER_DAY = 24 * 60


def _as_utc(moment: datetime) -> datetime:
    """Normalise to timezone-aware UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonRun:
    """One contiguous stretch of months within one calendar year."""

    year: int
    months: tuple[int, ...]
    start: datetime
    end: datetime

    @property
    def label(self) -> str:
        first, last = self.months[0], self.months[-1]
        span = f"{calendar.month_abbr[first]}" + (
            "" if first == last else f"-{calendar.month_abbr[last]}"
        )
        return f"{span} {self.year}"

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment <= self.end


def month_runs(months: list[int]) -> list[tuple[int, ...]]:
    """Group a month set into contiguous runs."""
    ordered = sorted(set(months))
    if not ordered:
        raise ValueError("no months given; there are no seasons to enumerate")
    if 12 in ordered and 1 in ordered:
        raise ValueError(
            f"months {ordered} span the new year. A season crossing 31 December "
            "belongs to two calendar years and the configuration does not say "
            "which one names it; split it into two explicit seasons instead."
        )

    runs: list[list[int]] = [[ordered[0]]]
    for month in ordered[1:]:
        if month == runs[-1][-1] + 1:
            runs[-1].append(month)
        else:
            runs.append([month])
    return [tuple(run) for run in runs]


def season_runs(seasons: SeasonsConfig, interval_minutes: int) -> list[SeasonRun]:
    """Every contiguous season stretch in the configured years, in order."""
    if interval_minutes <= 0:
        raise ValueError(f"interval_minutes must be positive; got {interval_minutes}")
    if _MINUTES_PER_DAY % interval_minutes:
        raise ValueError(
            f"a {interval_minutes}-minute cadence does not divide the day, so "
            "slot times would drift from one day to the next and no fixed grid "
            "of valid times exists"
        )

    runs: list[SeasonRun] = []
    for year in seasons.years:
        for months in month_runs(seasons.months):
            last_day = calendar.monthrange(year, months[-1])[1]
            start = datetime(year, months[0], 1, tzinfo=UTC)
            end = datetime(year, months[-1], last_day, tzinfo=UTC) + timedelta(
                minutes=_MINUTES_PER_DAY - interval_minutes
            )
            runs.append(SeasonRun(year=year, months=months, start=start, end=end))
    return sorted(runs, key=lambda run: run.start)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def enumerate_valid_times(
    seasons: SeasonsConfig,
    *,
    interval_minutes: int,
    sequence_length: int,
    lead_frames: int,
    require_window_within_season: bool = True,
) -> list[datetime]:
    """Every ``t0`` the record can support, oldest first."""
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be at least 1; got {sequence_length}")
    if lead_frames < 1:
        raise ValueError(f"lead_frames must be at least 1; got {lead_frames}")

    step = timedelta(minutes=interval_minutes)
    lookback = (sequence_length - 1) * step
    horizon = lead_frames * step

    if not require_window_within_season:
        logger.warning(
            "require_window_within_season is false: windows within %s of a "
            "season start and %s of its end reference frames outside the "
            "record and will be rejected after a full ingestion pass",
            lookback,
            horizon,
        )

    runs = season_runs(seasons, interval_minutes)
    times: list[datetime] = []
    for run in runs:
        if require_window_within_season:
            first, last = run.start + lookback, run.end - horizon
        else:
            first, last = run.start, run.end

        if last < first:
            logger.warning(
                "%s is shorter than one window (%s of lookback plus %s of "
                "target) and contributes nothing",
                run.label,
                lookback,
                horizon,
            )
            continue

        count = 0
        moment = first
        while moment <= last:
            times.append(moment)
            moment += step
            count += 1
        logger.debug("%s contributes %d windows", run.label, count)

    if not times:
        raise ValueError(
            "no valid times could be enumerated. Every configured season is "
            "shorter than one window's full span; check seasons, "
            "sequence_length and lead_frames against each other."
        )

    logger.info(
        "enumerated %d candidate windows across %d season runs",
        len(times),
        len(runs),
    )
    return times


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSplit:
    """Enumerated windows, partitioned by the reaching-edge rule."""

    train: tuple[datetime, ...]
    validation: tuple[datetime, ...]
    discarded: tuple[datetime, ...]
    lookback: timedelta
    horizon: timedelta

    def __len__(self) -> int:
        return len(self.train) + len(self.validation)

    def summary(self) -> dict[str, object]:
        def envelope(times: tuple[datetime, ...]) -> dict[str, str] | None:
            if not times:
                return None
            return {
                "first_input": (times[0] - self.lookback).isoformat(),
                "first_t0": times[0].isoformat(),
                "last_t0": times[-1].isoformat(),
                "last_target": (times[-1] + self.horizon).isoformat(),
            }

        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "discarded": len(self.discarded),
            "train_envelope": envelope(self.train),
            "validation_envelope": envelope(self.validation),
        }


def split_valid_times(
    times: list[datetime],
    split: SplitConfig,
    *,
    interval_minutes: int,
    sequence_length: int,
    lead_frames: int,
) -> DatasetSplit:
    """Partition enumerated times at their reaching edges."""
    step = timedelta(minutes=interval_minutes)
    lookback = (sequence_length - 1) * step
    horizon = lead_frames * step

    train_until = _as_utc(split.train_until)
    validation_from = _as_utc(split.validation_from)

    train: list[datetime] = []
    validation: list[datetime] = []
    discarded: list[datetime] = []

    for moment in sorted(_as_utc(t) for t in times):
        if moment + horizon <= train_until:
            train.append(moment)
        elif moment - lookback >= validation_from:
            validation.append(moment)
        else:
            discarded.append(moment)

    result = DatasetSplit(
        train=tuple(train),
        validation=tuple(validation),
        discarded=tuple(discarded),
        lookback=lookback,
        horizon=horizon,
    )

    _verify_disjoint(result)

    logger.info(
        "split: %d train, %d validation, %d discarded at the boundary",
        len(train),
        len(validation),
        len(discarded),
    )
    if not train:
        raise ValueError(
            f"no training windows survive train_until={train_until.isoformat()}; "
            "every enumerated window forecasts past the cutoff"
        )
    if not validation:
        raise ValueError(
            f"no validation windows survive "
            f"validation_from={validation_from.isoformat()}; the held-out "
            "season yields nothing to score against"
        )
    return result


def _verify_disjoint(split: DatasetSplit) -> None:
    """Assert the two sides share no instant of observation."""
    if not split.train or not split.validation:
        return

    train_end = split.train[-1] + split.horizon
    validation_start = split.validation[0] - split.lookback

    if train_end > validation_start:
        raise ValueError(
            f"training data reaches {train_end.isoformat()} but validation "
            f"reads from {validation_start.isoformat()}: the two sides overlap "
            f"by {train_end - validation_start}. Every window is more than the "
            "instant it is named after, and the split must be tested at the "
            "reaching edges, not at t0."
        )


# ---------------------------------------------------------------------------
# Configuration entry point
# ---------------------------------------------------------------------------


def build_index(training: TrainingConfig, model: ModelConfig) -> DatasetSplit:
    """Enumerate and split, taking every dimension from validated configuration."""
    interval = model.output.lead_interval_minutes
    times = enumerate_valid_times(
        training.data.seasons,
        interval_minutes=interval,
        sequence_length=model.input.sequence_length,
        lead_frames=model.output.lead_frames,
        require_window_within_season=(training.data.split.require_window_within_season),
    )
    return split_valid_times(
        times,
        training.data.split,
        interval_minutes=interval,
        sequence_length=model.input.sequence_length,
        lead_frames=model.output.lead_frames,
    )


__all__ = [
    "DatasetSplit",
    "SeasonRun",
    "build_index",
    "enumerate_valid_times",
    "month_runs",
    "season_runs",
    "split_valid_times",
]
