"""Unit tests for core data model invariants (T016)."""

from __future__ import annotations

import pytest

from quotient_learning.dataset import Trajectory, TrajectoryDataset
from quotient_learning.errors import InsufficientSamplesError, WindowMismatchError
from quotient_learning.types import (
    CanonicalQuotient,
    ContinuationSample,
    EquivalenceTestResult,
    ObservationHistory,
    RCritEstimate,
)


class TestObservationHistory:
    def test_empty_history_is_valid(self) -> None:
        h = ObservationHistory(())
        assert h.length == 0

    def test_history_is_hashable(self) -> None:
        h1 = ObservationHistory((0, 1, 0))
        h2 = ObservationHistory((0, 1, 0))
        assert hash(h1) == hash(h2)
        assert {h1, h2} == {h1}

    def test_history_is_immutable(self) -> None:
        h = ObservationHistory((0, 1))
        with pytest.raises(Exception):
            h.observations = (2,)  # type: ignore[misc]

    def test_history_rejects_negative_observation(self) -> None:
        with pytest.raises(ValueError):
            ObservationHistory((0, -1))

    def test_extend_appends_token(self) -> None:
        h = ObservationHistory((0, 1))
        e = h.extend(2)
        assert e == ObservationHistory((0, 1, 2))
        assert h == ObservationHistory((0, 1))  # original unchanged


class TestContinuationSample:
    def test_continuation_window_length_enforced(self) -> None:
        with pytest.raises(ValueError):
            ContinuationSample(
                history=ObservationHistory(()),
                continuations=((0, 1), (0,)),  # mismatched lengths
                window_length=2,
            )

    def test_n_samples_matches_continuation_count(self) -> None:
        cs = ContinuationSample(
            history=ObservationHistory((0,)),
            continuations=((0, 1), (1, 0), (1, 1)),
            window_length=2,
        )
        assert cs.n_samples == 3


class TestTrajectory:
    def test_observation_action_length_relationship(self) -> None:
        # observations must be exactly one longer than actions
        with pytest.raises(ValueError):
            Trajectory(observations=(0, 1), actions=(0, 0), rewards=(0.0, 0.0), seed=0)

    def test_actions_rewards_length_match(self) -> None:
        with pytest.raises(ValueError):
            Trajectory(observations=(0, 1, 0), actions=(0, 0), rewards=(0.0,), seed=0)

    def test_valid_trajectory_constructs(self) -> None:
        t = Trajectory(observations=(0, 1, 0), actions=(0, 1), rewards=(0.0, 1.0), seed=42)
        assert t.length == 2


class TestTrajectoryDataset:
    def test_immutability(self) -> None:
        t = Trajectory(observations=(0,), actions=(), rewards=(), seed=0)
        ds = TrajectoryDataset(trajectories=(t,), environment_id="x", policy_id="rand")
        with pytest.raises(Exception):
            ds.environment_id = "y"  # type: ignore[misc]

    def test_total_steps(self) -> None:
        t1 = Trajectory(observations=(0, 1), actions=(0,), rewards=(0.0,), seed=0)
        t2 = Trajectory(observations=(0, 1, 1), actions=(0, 1), rewards=(0.0, 0.0), seed=1)
        ds = TrajectoryDataset.from_iterable([t1, t2], environment_id="x", policy_id="rand")
        assert ds.total_steps() == 3
        assert ds.n_trajectories == 2


class TestCanonicalQuotient:
    def test_class_of_returns_none_for_unknown(self) -> None:
        q = CanonicalQuotient(n_classes=1, partition={ObservationHistory(()): 0})
        assert q.class_of(ObservationHistory((0,))) is None
        assert q.class_of(ObservationHistory(())) == 0


class TestRCritEstimate:
    def test_relative_error_none_when_no_truth(self) -> None:
        q = CanonicalQuotient(n_classes=1, partition={ObservationHistory(()): 0})
        e = RCritEstimate(quotient_a=q, quotient_b=q, r_crit=1.0, true_r_crit=None)
        assert e.relative_error is None

    def test_relative_error_computed(self) -> None:
        q = CanonicalQuotient(n_classes=1, partition={ObservationHistory(()): 0})
        e = RCritEstimate(quotient_a=q, quotient_b=q, r_crit=1.1, true_r_crit=1.0)
        assert e.relative_error == pytest.approx(0.1)


class TestExceptions:
    def test_insufficient_samples_message(self) -> None:
        e = InsufficientSamplesError(n_samples=10, min_required=30)
        assert "30" in str(e) and "10" in str(e)

    def test_window_mismatch_message(self) -> None:
        e = WindowMismatchError(window_a=2, window_b=3)
        assert "2" in str(e) and "3" in str(e)


def test_equivalence_result_construction() -> None:
    r = EquivalenceTestResult(
        history_a=ObservationHistory((0,)),
        history_b=ObservationHistory((1,)),
        test_statistic=0.5,
        p_value=0.04,
        same_class=False,
        alpha=0.05,
    )
    assert r.same_class is False
