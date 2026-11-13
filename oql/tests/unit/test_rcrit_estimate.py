"""Unit tests for RCritEstimate (T065)."""

from __future__ import annotations

import math

import pytest

from quotient_learning.dataset import Trajectory, TrajectoryDataset
from quotient_learning.rcrit import (
    classify_phase_transition,
    compute_rcrit,
    compute_rcrit_exact,
    compute_rcrit_surrogate,
)
from quotient_learning.types import CanonicalQuotient, ObservationHistory


def _quotient(n: int) -> CanonicalQuotient:
    return CanonicalQuotient(
        n_classes=n,
        partition={ObservationHistory((i,)): i for i in range(n)},
    )


def _single_obs_dataset(obs_seq: tuple[int, ...]) -> TrajectoryDataset:
    """Return a dataset with a single trajectory whose observations are obs_seq."""
    n = len(obs_seq)
    traj = Trajectory(
        observations=obs_seq,
        actions=tuple(0 for _ in range(n - 1)),
        rewards=tuple(0.0 for _ in range(n - 1)),
        seed=0,
    )
    return TrajectoryDataset.from_iterable(
        [traj], environment_id="test", policy_id="test"
    )


@pytest.mark.unit
class TestRCritEstimate:
    def test_rcrit_zero_for_singleton_quotients(self) -> None:
        est = compute_rcrit(_quotient(1), _quotient(1))
        assert est.r_crit == 0.0

    def test_rcrit_log2_of_max_quotient(self) -> None:
        est = compute_rcrit(_quotient(2), _quotient(4))
        assert est.r_crit == pytest.approx(math.log2(4))

    def test_relative_error_computed(self) -> None:
        est = compute_rcrit(_quotient(2), _quotient(2),
                            true_r_crit=math.log2(2))
        assert est.relative_error == pytest.approx(0.0)

    def test_relative_error_none_when_no_truth(self) -> None:
        est = compute_rcrit(_quotient(2), _quotient(2))
        assert est.relative_error is None

    def test_phase_classifier_feasible(self) -> None:
        est = compute_rcrit(_quotient(2), _quotient(2),
                            true_r_crit=math.log2(2))
        phase = classify_phase_transition(est, channel_capacity=2.0)
        assert phase.feasible is True
        assert phase.margin == pytest.approx(1.0)

    def test_phase_classifier_infeasible(self) -> None:
        est = compute_rcrit(_quotient(8), _quotient(8),
                            true_r_crit=math.log2(8))
        phase = classify_phase_transition(est, channel_capacity=1.0)
        assert phase.feasible is False
        assert phase.margin < 0


@pytest.mark.unit
class TestRCritExact:
    """Unit tests for compute_rcrit_exact (rate-distortion H(A|B) estimator)."""

    def _make_perfectly_correlated_quotients(self, n: int):
        """Two quotients with identical partitions over the same histories.

        When Q_A == Q_B, knowing B's class perfectly predicts A's class, so
        H(A|B) = 0 and R_crit should be 0.
        """
        partition = {ObservationHistory((i,)): i for i in range(n)}
        q = CanonicalQuotient(n_classes=n, partition=partition)
        return q, q

    def _make_independent_quotients(self, n: int):
        """Two quotients with *disjoint* history sets.

        Since no observation history is in both partitions, the joint
        distribution cannot be estimated from shared data; the fallback
        R_crit = log2(|Q_A|) is returned with using_independent_fallback=True.
        """
        # Q_A owns observations 0..n-1; Q_B owns observations n..2n-1.
        partition_a = {ObservationHistory((i,)): i for i in range(n)}
        partition_b = {ObservationHistory((n + i,)): i for i in range(n)}
        q_a = CanonicalQuotient(n_classes=n, partition=partition_a)
        q_b = CanonicalQuotient(n_classes=n, partition=partition_b)
        return q_a, q_b

    # ------------------------------------------------------------------
    # Test 1: perfectly correlated quotients → R_crit = 0
    # ------------------------------------------------------------------

    def test_perfectly_correlated_rcrit_zero(self) -> None:
        """When Q_A == Q_B, H(A|B) = 0 — no bits needed."""
        n = 4
        q_a, q_b = self._make_perfectly_correlated_quotients(n)
        # Dataset whose observations cycle through all classes many times.
        obs = tuple(i % n for i in range(4 * n + 1))
        dataset = _single_obs_dataset(obs)
        est = compute_rcrit_exact(q_a, q_b, dataset)
        assert est.r_crit == pytest.approx(0.0, abs=1e-9)
        assert est.using_independent_fallback is False

    # ------------------------------------------------------------------
    # Test 2: independent quotients (disjoint histories) → fallback
    # ------------------------------------------------------------------

    def test_independent_fallback_flag_and_value(self) -> None:
        """When histories are disjoint, fallback is log2(|Q_A|) with flag."""
        n = 4
        q_a, q_b = self._make_independent_quotients(n)
        # Observations only in Q_A's range — Q_B has nothing to match.
        obs = tuple(i % n for i in range(4 * n + 1))
        dataset = _single_obs_dataset(obs)
        est = compute_rcrit_exact(q_a, q_b, dataset)
        assert est.using_independent_fallback is True
        assert est.r_crit == pytest.approx(math.log2(n))

    # ------------------------------------------------------------------
    # Test 3: dispatch — compute_rcrit with dataset uses exact path
    # ------------------------------------------------------------------

    def test_dispatch_uses_exact_when_dataset_provided(self) -> None:
        n = 4
        q_a, q_b = self._make_perfectly_correlated_quotients(n)
        obs = tuple(i % n for i in range(4 * n + 1))
        dataset = _single_obs_dataset(obs)
        est_dispatch = compute_rcrit(q_a, q_b, dataset=dataset)
        est_exact = compute_rcrit_exact(q_a, q_b, dataset)
        assert est_dispatch.r_crit == pytest.approx(est_exact.r_crit)
        assert est_dispatch.using_independent_fallback == est_exact.using_independent_fallback

    def test_dispatch_uses_surrogate_when_no_dataset(self) -> None:
        n = 4
        q_a, q_b = self._make_perfectly_correlated_quotients(n)
        est = compute_rcrit(q_a, q_b)
        est_sur = compute_rcrit_surrogate(q_a, q_b)
        assert est.r_crit == pytest.approx(est_sur.r_crit)

    # ------------------------------------------------------------------
    # Test 4: mutual_information field is populated on exact path
    # ------------------------------------------------------------------

    def test_mutual_information_populated(self) -> None:
        n = 4
        q_a, q_b = self._make_perfectly_correlated_quotients(n)
        obs = tuple(i % n for i in range(4 * n + 1))
        dataset = _single_obs_dataset(obs)
        est = compute_rcrit_exact(q_a, q_b, dataset)
        assert est.mutual_information is not None
        assert est.mutual_information >= 0.0
