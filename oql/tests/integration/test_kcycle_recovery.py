"""Integration test: K-Cycle quotient recovery (constructive ground-truth benchmark).

The KCyclePOMDP family provides a constructive benchmark where the canonical
Myhill-Nerode quotient size is exactly K by design.

Algorithm notes
---------------
The empty-history class H() always forms its own class (flat continuation =
time-average over all cycle positions, distinct from any cycle-state
distribution).  ``effective_n_classes()`` subtracts this root-only class so
that the returned count matches the K-cycle ground truth.

Estimator config
----------------
``min_history_occurrences=1000`` and ``max_steps_per_episode=20`` (fixed) give
O(K) convergence scaling: each symbol appears ~N*20/K times, so convergence
requires N ≈ 50*K trajectories.  This places the convergence knee for
K=8/16/32 squarely within the test budgets.

Acceptance criteria (from the task spec)
-----------------------------------------
* K=2: effective learned == 2 within 1000 trajectories for ≥ 4/5 seeds.
* K=4: effective learned == 4 within 5000 trajectories for ≥ 3/5 seeds.
* K=8: effective learned ≥ 6 within 10000 trajectories for ≥ 3/5 seeds.
"""

from __future__ import annotations

import pytest

from environments.layer1_discrete.kcycle import KCyclePOMDP
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.types import CanonicalQuotient, ObservationHistory

SEEDS = [0, 1, 2, 3, 4]

# Shared estimator config (matches run_kcycle.py).
HISTORY_MAX_LENGTH = 1
MIN_HISTORY_OCCURRENCES = 1000
MAX_STEPS_PER_EPISODE = 20
WINDOW_LENGTH = 2
ALPHA = 0.01


def effective_n_classes(q: CanonicalQuotient) -> int:
    """Subtract the root-only class from n_classes (see module docstring)."""
    root = ObservationHistory(())
    if root in q.partition:
        root_cls = q.partition[root]
        root_members = [h for h, c in q.partition.items() if c == root_cls]
        if len(root_members) == 1:
            return max(0, q.n_classes - 1)
    return q.n_classes


def _run_k(k: int, n_trajectories: int, seeds: list[int]) -> list[int]:
    """Run the estimator for each seed and return effective class counts."""
    learned: list[int] = []
    for seed in seeds:
        env = KCyclePOMDP(k=k, obs_accuracy=0.85, seed=seed)
        est = QuotientEstimator(
            env=env,
            n_trajectories=n_trajectories,
            window_length=WINDOW_LENGTH,
            history_max_length=HISTORY_MAX_LENGTH,
            min_history_occurrences=MIN_HISTORY_OCCURRENCES,
            alpha=ALPHA,
            max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        )
        q = est.run(seed=seed)
        learned.append(effective_n_classes(q))
    return learned


@pytest.mark.integration
class TestKCycleK2:
    """K=2: should recover exactly 2 classes within 1000 trajectories (≥4/5 seeds)."""

    K = 2
    N_TRAJECTORIES = 1000
    MIN_SEEDS_PASSING = 4

    def test_exact_recovery(self) -> None:
        learned = _run_k(self.K, self.N_TRAJECTORIES, SEEDS)
        hits = sum(1 for n in learned if n == self.K)
        assert hits >= self.MIN_SEEDS_PASSING, (
            f"K={self.K}: only {hits}/{len(SEEDS)} seeds recovered exactly "
            f"{self.K} classes within {self.N_TRAJECTORIES} trajectories; "
            f"effective sizes = {learned}"
        )

    def test_metadata(self) -> None:
        env = KCyclePOMDP(k=self.K, seed=0)
        assert env.metadata["ground_truth_n_classes"] == self.K
        assert env.n_observations == self.K
        assert env.n_actions == 1


@pytest.mark.integration
class TestKCycleK4:
    """K=4: should recover exactly 4 classes within 5000 trajectories (≥3/5 seeds)."""

    K = 4
    N_TRAJECTORIES = 5000
    MIN_SEEDS_PASSING = 3

    def test_exact_recovery(self) -> None:
        learned = _run_k(self.K, self.N_TRAJECTORIES, SEEDS)
        hits = sum(1 for n in learned if n == self.K)
        assert hits >= self.MIN_SEEDS_PASSING, (
            f"K={self.K}: only {hits}/{len(SEEDS)} seeds recovered exactly "
            f"{self.K} classes within {self.N_TRAJECTORIES} trajectories; "
            f"effective sizes = {learned}"
        )


@pytest.mark.integration
class TestKCycleK8:
    """K=8: should recover ≥ 6 classes within 10000 trajectories (≥3/5 seeds).

    Under the O(K) convergence scaling, K=8 converges at N≈400 trajectories
    (50*K = 400).  At N=10000 this is well past convergence for all seeds.
    """

    K = 8
    N_TRAJECTORIES = 10000
    MIN_SEEDS_PASSING = 3
    MIN_CLASSES = 6  # ≥ 6 out of 8

    def test_approximate_recovery(self) -> None:
        learned = _run_k(self.K, self.N_TRAJECTORIES, SEEDS)
        hits = sum(1 for n in learned if n >= self.MIN_CLASSES)
        assert hits >= self.MIN_SEEDS_PASSING, (
            f"K={self.K}: only {hits}/{len(SEEDS)} seeds recovered ≥ "
            f"{self.MIN_CLASSES} classes within {self.N_TRAJECTORIES} "
            f"trajectories; effective sizes = {learned}"
        )


@pytest.mark.integration
class TestKCycleEnvironmentProperties:
    """Sanity checks on the KCyclePOMDP environment itself."""

    def test_observation_range(self) -> None:
        """All emitted observations must be in [0, k-1]."""
        k = 5
        env = KCyclePOMDP(k=k, seed=42)
        env.reset(seed=42)
        for _ in range(200):
            result = env.step(0)
            assert 0 <= result.observation < k, (
                f"observation {result.observation} out of range [0, {k-1}]"
            )

    def test_cycle_wraps(self) -> None:
        """After K deterministic steps, the observation should cycle back."""
        k = 4
        env = KCyclePOMDP(k=k, obs_accuracy=1.0, seed=0, randomise_start=False)
        # With accuracy=1.0, observations equal the true state.
        obs0 = env.reset(seed=0)  # state is 0 (randomise_start=False)
        assert obs0 == 0
        obs_seq = [obs0]
        for _ in range(k):
            obs_seq.append(env.step(0).observation)
        # After k steps we should be back at state 0.
        assert obs_seq[-1] == 0, f"cycle did not wrap: {obs_seq}"

    def test_ground_truth_metadata_matches_k(self) -> None:
        for k in [2, 3, 4, 5, 8]:
            env = KCyclePOMDP(k=k, seed=0)
            assert env.metadata["ground_truth_n_classes"] == k
            assert env.metadata["k"] == k

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            KCyclePOMDP(k=1)

    def test_deterministic_with_accuracy_1(self) -> None:
        """obs_accuracy=1.0 should yield deterministic obs == state index."""
        k = 3
        env = KCyclePOMDP(k=k, obs_accuracy=1.0, seed=0, randomise_start=False)
        env.reset(seed=0)
        for step_i in range(k * 3):
            result = env.step(0)
            expected_state = (step_i + 1) % k
            assert result.observation == expected_state, (
                f"step {step_i}: expected obs {expected_state}, "
                f"got {result.observation}"
            )

    def test_effective_n_classes_subtracts_root(self) -> None:
        """effective_n_classes() returns n_classes - 1 when root is isolated."""
        k = 2
        env = KCyclePOMDP(k=k, seed=0)
        est = QuotientEstimator(
            env=env,
            n_trajectories=2000,
            window_length=WINDOW_LENGTH,
            history_max_length=HISTORY_MAX_LENGTH,
            min_history_occurrences=MIN_HISTORY_OCCURRENCES,
            alpha=ALPHA,
            max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        )
        q = est.run(seed=0)
        eff = effective_n_classes(q)
        assert eff == k, (
            f"expected effective n_classes = {k}, got {eff} "
            f"(raw n_classes = {q.n_classes})"
        )
