"""Verification metrics.

Small explicit tensors with known answers, so the arithmetic is checked rather
than assumed. No data and no model.
"""

from __future__ import annotations

import math

import pytest
import torch

from indra.evaluation.metrics import (
    MetricAccumulator,
    contingency_table,
    crps_ensemble,
    higher_is_better,
    masked_absolute_error,
)

THRESHOLD = 15.0


def members(*values: list[float]) -> torch.Tensor:
    """(N, B, T, C, H, W) from one row of cells per member."""
    return torch.tensor(
        [[[[[row]]] for row in [v]] for v in values], dtype=torch.float32
    )


def observation(row: list[float]) -> torch.Tensor:
    """(B, T, C, H, W) from one row of cells."""
    return torch.tensor([[[[row]]]], dtype=torch.float32)


class TestContingencyTable:
    def test_counts_hits_misses_and_false_alarms(self):
        # observed  : wet  wet  dry  dry
        # forecast  : wet  dry  wet  dry
        # so 1 hit, 1 miss, 1 false alarm, 1 correct negative.
        table = contingency_table(
            members([20.0, 2.0, 20.0, 2.0]),
            observation([20.0, 20.0, 2.0, 2.0]),
            THRESHOLD,
        )
        assert table.hits.tolist() == [1]
        assert table.misses.tolist() == [1]
        assert table.false_alarms.tolist() == [1]
        assert table.correct_negatives.tolist() == [1]

    def test_csi_pod_far_from_known_counts(self):
        table = contingency_table(
            members([20.0, 2.0, 20.0, 2.0]),
            observation([20.0, 20.0, 2.0, 2.0]),
            THRESHOLD,
        )
        assert table.csi == pytest.approx(1 / 3)  # 1 / (1 + 1 + 1)
        assert table.pod == pytest.approx(1 / 2)  # 1 / (1 + 1)
        assert table.far == pytest.approx(1 / 2)  # 1 / (1 + 1)

    def test_threshold_is_inclusive(self):
        # Exactly 15.0 counts as heavy rather than falling between classes.
        table = contingency_table(
            members([THRESHOLD]), observation([THRESHOLD]), THRESHOLD
        )
        assert table.hits.tolist() == [1]

    def test_mask_excludes_cells_entirely(self):
        mask = torch.tensor([[[[[True, False]]]]])
        table = contingency_table(
            members([20.0, 20.0]), observation([20.0, 2.0]), THRESHOLD, mask
        )
        # The masked false alarm is not counted anywhere.
        assert table.hits.tolist() == [1]
        assert table.false_alarms.tolist() == [0]

    def test_empty_table_is_undefined_not_perfect(self):
        # No event observed and none forecast. Scoring this 1.0 would make
        # csi_heavy a measure of dry-day behaviour.
        table = contingency_table(
            members([1.0, 2.0]), observation([1.0, 2.0]), THRESHOLD
        )
        assert math.isnan(table.csi)

    def test_scores_average_over_members(self):
        # One perfect member, one that misses everything: mean CSI is 0.5.
        table = contingency_table(
            members([20.0, 20.0], [2.0, 2.0]),
            observation([20.0, 20.0]),
            THRESHOLD,
        )
        assert table.csi == pytest.approx(0.5)

    def test_tables_pool_by_addition(self):
        first = contingency_table(members([20.0]), observation([20.0]), THRESHOLD)
        second = contingency_table(members([2.0]), observation([20.0]), THRESHOLD)
        pooled = first + second
        assert pooled.hits.tolist() == [1]
        assert pooled.misses.tolist() == [1]
        assert pooled.csi == pytest.approx(0.5)

    def test_pooling_across_thresholds_is_refused(self):
        a = contingency_table(members([20.0]), observation([20.0]), 15.0)
        b = contingency_table(members([20.0]), observation([20.0]), 7.5)
        with pytest.raises(ValueError, match="different thresholds"):
            _ = a + b

    def test_events_observed_is_reported(self):
        table = contingency_table(
            members([20.0, 2.0]), observation([20.0, 20.0]), THRESHOLD
        )
        assert table.events_observed == 2


class TestPooling:
    def test_pooled_csi_differs_from_the_mean_of_batch_csis(self):
        """The reason MetricAccumulator pools counts rather than averaging.

        One batch with many events and one with almost none: averaging the
        two ratios weights them equally, pooling weights them by event count.
        """
        heavy_fc = members([20.0] * 10)
        heavy_ob = observation([20.0] * 10)  # 10 hits
        light_fc = members([20.0])
        light_ob = observation([2.0])  # 1 false alarm

        pooled = contingency_table(heavy_fc, heavy_ob, THRESHOLD) + contingency_table(
            light_fc, light_ob, THRESHOLD
        )
        assert pooled.csi == pytest.approx(10 / 11)

        per_batch = [
            contingency_table(heavy_fc, heavy_ob, THRESHOLD).csi,
            contingency_table(light_fc, light_ob, THRESHOLD).csi,
        ]
        naive = sum(per_batch) / len(per_batch)
        assert naive == pytest.approx(0.5)
        assert abs(pooled.csi - naive) > 0.4


class TestScalarScores:
    def test_mae_uses_the_ensemble_mean(self):
        # Members at 10 and 20 average to 15 against an observation of 12.
        total, count = masked_absolute_error(
            members([10.0], [20.0]), observation([12.0])
        )
        assert count == 1
        assert total == pytest.approx(3.0)

    def test_crps_of_a_collapsed_ensemble_is_absolute_error(self):
        # With no spread the pairwise term vanishes and CRPS reduces to |x-y|.
        total, count = crps_ensemble(members([10.0], [10.0]), observation([12.0]))
        assert count == 1
        assert total == pytest.approx(2.0)

    def test_crps_rewards_spread_that_brackets_the_observation(self):
        tight, _ = crps_ensemble(members([10.0], [10.0]), observation([12.0]))
        spread, _ = crps_ensemble(members([10.0], [14.0]), observation([12.0]))
        assert spread < tight

    def test_crps_is_never_negative_here(self):
        total, _ = crps_ensemble(members([1.0], [5.0], [30.0]), observation([12.0]))
        assert total >= 0.0


class TestAccumulator:
    def test_reports_band_suffixed_names(self):
        accumulator = MetricAccumulator(threshold=THRESHOLD, threshold_name="heavy")
        accumulator.update(members([20.0]), observation([20.0]))
        results = accumulator.compute()
        assert "csi_heavy" in results
        assert results["csi_heavy"] == pytest.approx(1.0)

    def test_windows_are_counted(self):
        accumulator = MetricAccumulator(threshold=THRESHOLD)
        accumulator.update(members([20.0]), observation([20.0]))
        accumulator.update(members([20.0]), observation([20.0]))
        assert accumulator.compute()["windows"] == 2

    def test_reset_clears_state(self):
        accumulator = MetricAccumulator(threshold=THRESHOLD)
        accumulator.update(members([20.0]), observation([20.0]))
        accumulator.reset()
        assert accumulator.compute()["windows"] == 0

    def test_unsupported_metric_is_refused(self):
        with pytest.raises(ValueError, match="unsupported metrics"):
            MetricAccumulator(threshold=THRESHOLD, metrics=["bogus"])

    def test_threshold_must_be_positive(self):
        with pytest.raises(ValueError, match="threshold must be positive"):
            MetricAccumulator(threshold=-1.0)


class TestDirection:
    @pytest.mark.parametrize("metric", ["csi", "csi_heavy", "pod"])
    def test_skill_scores_ascend(self, metric):
        assert higher_is_better(metric) is True

    @pytest.mark.parametrize("metric", ["far", "crps", "mae_mm_h"])
    def test_error_scores_descend(self, metric):
        assert higher_is_better(metric) is False

    def test_unknown_metric_raises(self):
        # A monitor whose direction is unknown would otherwise silently ship
        # the worst checkpoint of the run.
        with pytest.raises(KeyError):
            higher_is_better("bogus")


class TestShapeValidation:
    def test_member_axis_is_required(self):
        with pytest.raises(ValueError, match=r"\(N, B, T, C, H, W\)"):
            contingency_table(observation([20.0]), observation([20.0]), THRESHOLD)

    def test_mismatched_target_is_refused(self):
        with pytest.raises(ValueError, match="do not match"):
            contingency_table(members([20.0, 20.0]), observation([20.0]), THRESHOLD)
