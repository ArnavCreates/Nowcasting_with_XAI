"""Window enumeration and the train/validation split.

Pure datetime arithmetic, so these run without any data at all.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import pytest

from indra.config import SeasonsConfig, SplitConfig
from indra.datasets.index import (
    enumerate_valid_times,
    month_runs,
    season_runs,
    split_valid_times,
)

INTERVAL = 30
SEQUENCE = 13
LEADS = 12


def jjas(*years: int) -> SeasonsConfig:
    return SeasonsConfig(years=list(years), months=[6, 7, 8, 9])


class TestMonthRuns:
    def test_contiguous_months_are_one_run(self):
        assert month_runs([6, 7, 8, 9]) == [(6, 7, 8, 9)]

    def test_gap_splits_into_two_runs(self):
        assert month_runs([3, 4, 9, 10]) == [(3, 4), (9, 10)]

    def test_unsorted_input_is_ordered(self):
        assert month_runs([9, 6, 8, 7]) == [(6, 7, 8, 9)]

    def test_new_year_span_is_refused(self):
        # December-January belongs to two calendar years and nothing says
        # which one names the season.
        with pytest.raises(ValueError, match="new year"):
            month_runs([12, 1, 2])

    def test_empty_is_refused(self):
        with pytest.raises(ValueError):
            month_runs([])


class TestSeasonRuns:
    def test_one_run_per_year(self):
        runs = season_runs(jjas(2017, 2018), INTERVAL)
        assert len(runs) == 2
        assert [r.year for r in runs] == [2017, 2018]

    def test_run_starts_at_midnight_on_the_first(self):
        run = season_runs(jjas(2020), INTERVAL)[0]
        assert run.start == datetime(2020, 6, 1, 0, 0, tzinfo=UTC)

    def test_run_ends_on_the_last_slot_not_midnight(self):
        # 23:30, not 00:00 on 1 October -- midnight would admit a slot from
        # outside the season.
        run = season_runs(jjas(2020), INTERVAL)[0]
        assert run.end == datetime(2020, 9, 30, 23, 30, tzinfo=UTC)

    def test_cadence_must_divide_the_day(self):
        with pytest.raises(ValueError, match="does not divide the day"):
            season_runs(jjas(2020), 7)


class TestEnumeration:
    def test_margins_exclude_edge_windows(self):
        times = enumerate_valid_times(
            jjas(2020),
            interval_minutes=INTERVAL,
            sequence_length=SEQUENCE,
            lead_frames=LEADS,
            require_window_within_season=True,
        )
        lookback = timedelta(minutes=(SEQUENCE - 1) * INTERVAL)
        horizon = timedelta(minutes=LEADS * INTERVAL)

        assert times[0] == datetime(2020, 6, 1, tzinfo=UTC) + lookback
        assert times[-1] == (datetime(2020, 9, 30, 23, 30, tzinfo=UTC) - horizon)

    def test_every_window_fits_inside_its_season(self):
        times = enumerate_valid_times(
            jjas(2020),
            interval_minutes=INTERVAL,
            sequence_length=SEQUENCE,
            lead_frames=LEADS,
        )
        lookback = timedelta(minutes=(SEQUENCE - 1) * INTERVAL)
        horizon = timedelta(minutes=LEADS * INTERVAL)
        start = datetime(2020, 6, 1, tzinfo=UTC)
        end = datetime(2020, 9, 30, 23, 30, tzinfo=UTC)

        for t in times:
            assert t - lookback >= start
            assert t + horizon <= end

    def test_times_are_strictly_increasing(self):
        times = enumerate_valid_times(
            jjas(2017, 2018),
            interval_minutes=INTERVAL,
            sequence_length=SEQUENCE,
            lead_frames=LEADS,
        )
        assert all(b > a for a, b in itertools.pairwise(times))

    def test_relaxing_the_margin_yields_more_windows(self):
        kwargs = {
            "interval_minutes": INTERVAL,
            "sequence_length": SEQUENCE,
            "lead_frames": LEADS,
        }
        strict = enumerate_valid_times(
            jjas(2020), require_window_within_season=True, **kwargs
        )
        loose = enumerate_valid_times(
            jjas(2020), require_window_within_season=False, **kwargs
        )
        assert len(loose) > len(strict)


class TestSplit:
    @staticmethod
    def split_config() -> SplitConfig:
        return SplitConfig(
            strategy="chronological",
            train_until=datetime(2019, 9, 30, 23, 59, 59, tzinfo=UTC),
            validation_from=datetime(2020, 6, 1, tzinfo=UTC),
            require_window_within_season=True,
        )

    def build(self):
        times = enumerate_valid_times(
            jjas(2017, 2018, 2019, 2020),
            interval_minutes=INTERVAL,
            sequence_length=SEQUENCE,
            lead_frames=LEADS,
        )
        return split_valid_times(
            times,
            self.split_config(),
            interval_minutes=INTERVAL,
            sequence_length=SEQUENCE,
            lead_frames=LEADS,
        )

    def test_both_sides_are_populated(self):
        split = self.build()
        assert split.train and split.validation

    def test_training_years_are_2017_to_2019(self):
        split = self.build()
        assert {t.year for t in split.train} == {2017, 2018, 2019}

    def test_validation_is_2020_only(self):
        split = self.build()
        assert {t.year for t in split.validation} == {2020}

    def test_no_training_target_crosses_the_cutoff(self):
        # The reaching-edge rule: a training window forecasts six hours past
        # its own t0, and that must still land before the boundary.
        split = self.build()
        cutoff = self.split_config().train_until
        assert all(t + split.horizon <= cutoff for t in split.train)

    def test_no_validation_input_precedes_the_boundary(self):
        split = self.build()
        start = self.split_config().validation_from
        assert all(t - split.lookback >= start for t in split.validation)

    def test_the_two_sides_never_overlap_in_time(self):
        split = self.build()
        train_end = split.train[-1] + split.horizon
        validation_start = split.validation[0] - split.lookback
        assert train_end <= validation_start

    def test_split_is_a_partition(self):
        split = self.build()
        assert not set(split.train) & set(split.validation)


class TestSplitConfigValidation:
    def test_validation_must_follow_training(self):
        with pytest.raises(ValueError, match="must follow"):
            SplitConfig(
                strategy="chronological",
                train_until=datetime(2020, 9, 1, tzinfo=UTC),
                validation_from=datetime(2019, 6, 1, tzinfo=UTC),
                require_window_within_season=True,
            )

    def test_random_split_is_unspellable(self):
        # A random split scores the model on 30-minute neighbours of its own
        # training windows. The Literal makes it a validation error.
        with pytest.raises(ValueError):
            SplitConfig(
                strategy="random",
                train_until=datetime(2019, 9, 30, tzinfo=UTC),
                validation_from=datetime(2020, 6, 1, tzinfo=UTC),
                require_window_within_season=True,
            )
