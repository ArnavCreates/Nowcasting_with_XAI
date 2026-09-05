"""The promotion gate.

Pure comparison logic, so the rule governing what reaches production is
checked without a model or a validation run.
"""

from __future__ import annotations

import pytest

from indra.evaluation.gate import GateCriteria, evaluate_gate

CRITERIA = GateCriteria(
    monitor="csi_heavy",
    min_delta=0.005,
    guarded=("far_heavy", "crps"),
    max_regression=0.02,
    min_events=50,
)


def scored(csi: float, *, events: int = 500, far: float = 0.30, crps: float = 1.0):
    return {
        "csi_heavy": csi,
        "far_heavy": far,
        "crps": crps,
        "events_observed": float(events),
    }


class TestImprovement:
    def test_clear_improvement_promotes(self):
        decision = evaluate_gate(scored(0.42), scored(0.40), CRITERIA)
        assert decision.promote
        assert decision.delta == pytest.approx(0.02)

    def test_regression_does_not_promote(self):
        decision = evaluate_gate(scored(0.38), scored(0.40), CRITERIA)
        assert not decision.promote

    def test_tie_does_not_promote(self):
        # Swapping a deployed model for one no better costs a deployment and
        # gains nothing.
        decision = evaluate_gate(scored(0.40), scored(0.40), CRITERIA)
        assert not decision.promote

    def test_improvement_below_min_delta_does_not_promote(self):
        decision = evaluate_gate(scored(0.4020), scored(0.4000), CRITERIA)
        assert not decision.promote
        assert "short of" in " ".join(decision.reasons)


class TestMetricDirection:
    def test_lower_is_better_for_far(self):
        # far descends: 0.20 beats 0.30. A gate that assumed every metric
        # ascends would promote the worse model here.
        criteria = GateCriteria(monitor="far_heavy", min_delta=0.01, guarded=())
        decision = evaluate_gate(
            scored(0.40, far=0.20), scored(0.40, far=0.30), criteria
        )
        assert decision.promote
        assert decision.delta == pytest.approx(0.10)

    def test_rising_far_does_not_promote(self):
        criteria = GateCriteria(monitor="far_heavy", min_delta=0.01, guarded=())
        decision = evaluate_gate(
            scored(0.40, far=0.40), scored(0.40, far=0.30), criteria
        )
        assert not decision.promote


class TestGuardedRegression:
    def test_guarded_regression_blocks_a_better_monitor(self):
        # csi improves, but false alarms rise sharply: not a trade worth
        # deploying unattended.
        decision = evaluate_gate(
            scored(0.45, far=0.40), scored(0.40, far=0.30), CRITERIA
        )
        assert not decision.promote
        assert any("regressed" in r for r in decision.reasons)

    def test_small_guarded_movement_is_tolerated(self):
        decision = evaluate_gate(
            scored(0.45, far=0.31), scored(0.40, far=0.30), CRITERIA
        )
        assert decision.promote

    def test_missing_guarded_metric_is_skipped(self):
        candidate = scored(0.45)
        del candidate["crps"]
        assert evaluate_gate(candidate, scored(0.40), CRITERIA).promote


class TestEventFloor:
    def test_too_few_events_does_not_promote(self):
        # A CSI computed from four events is a number, not evidence.
        decision = evaluate_gate(scored(0.90, events=4), scored(0.40), CRITERIA)
        assert not decision.promote
        assert any("observed events" in r for r in decision.reasons)

    def test_event_floor_applies_to_the_first_model_too(self):
        decision = evaluate_gate(scored(0.50, events=4), None, CRITERIA)
        assert not decision.promote


class TestUndefined:
    def test_nan_monitor_never_promotes(self):
        decision = evaluate_gate(scored(float("nan")), scored(0.40), CRITERIA)
        assert not decision.promote
        assert "undefined" in " ".join(decision.reasons)

    def test_missing_monitor_never_promotes(self):
        candidate = scored(0.45)
        del candidate["csi_heavy"]
        assert not evaluate_gate(candidate, scored(0.40), CRITERIA).promote

    def test_unknown_incumbent_judges_the_candidate_alone(self):
        decision = evaluate_gate(scored(0.45), {"csi_heavy": float("nan")}, CRITERIA)
        assert decision.promote
        assert any("unknown" in r for r in decision.reasons)


class TestFirstModel:
    def test_first_model_promotes_with_enough_events(self):
        # Refusing would mean never deploying a first model at all.
        decision = evaluate_gate(scored(0.30), None, CRITERIA)
        assert decision.promote
        assert decision.incumbent is None


class TestDecisionPayload:
    def test_every_decision_carries_a_reason(self):
        for candidate, incumbent in [
            (scored(0.42), scored(0.40)),
            (scored(0.38), scored(0.40)),
            (scored(0.90, events=4), scored(0.40)),
            (scored(float("nan")), scored(0.40)),
        ]:
            assert evaluate_gate(candidate, incumbent, CRITERIA).reasons

    def test_describe_is_serialisable(self):
        payload = evaluate_gate(scored(0.42), scored(0.40), CRITERIA).describe()
        assert payload["promote"] is True
        assert payload["monitor"] == "csi_heavy"
        assert isinstance(payload["reasons"], list)
