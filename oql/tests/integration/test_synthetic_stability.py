"""Integration test for synthetic POMDP stability (T036)."""

from __future__ import annotations

import numpy as np
import pytest

from environments.layer1_discrete.synthetic import SyntheticPOMDP
from quotient_learning.convergence import run_convergence_ladder
from quotient_learning.estimator import QuotientEstimator


@pytest.mark.integration
@pytest.mark.slow
class TestSyntheticStability:
    def test_quotient_size_finite_and_bounded(self) -> None:
        env = SyntheticPOMDP(n_states=50, n_obs=5, n_actions=3, seed=0)
        checkpoints, q = run_convergence_ladder(
            env=env, budgets=[500, 1000, 2000],
            estimator_kwargs=dict(
                window_length=2, history_max_length=1,
                min_history_occurrences=30, alpha=0.01,
                max_steps_per_episode=30,
            ),
            seed=0,
        )
        assert q.n_classes >= 1
        assert q.n_classes <= 2 * env.n_observations
        # Stability: the final two checkpoints agree (the partition has settled).
        sizes = [c.n_classes for c in checkpoints]
        assert sizes[-1] == sizes[-2], (
            f"final two checkpoints disagree (no stability): {sizes}"
        )

    def test_synthetic_200_nontrivial_quotient(self) -> None:
        """SyntheticPOMDP(S=200, O=10) with default estimator must recover
        ≥ 5 classes consistently across seeds (std ≤ 2).

        Regression test for the degeneracy bug fixed in:
        - SyntheticPOMDP: changed to block-diagonal structured construction.
        - QuotientEstimator: changed defaults to history_max_length=1, alpha=0.01.
        """
        env = SyntheticPOMDP(n_states=200, n_obs=10, seed=0)
        n_classes_per_seed = []
        for seed in range(5):
            est = QuotientEstimator(
                env, n_trajectories=5000, max_steps_per_episode=30,
            )
            q = est.run(seed=seed)
            n_classes_per_seed.append(q.n_classes)
        assert all(n >= 5 for n in n_classes_per_seed), (
            f"n_classes < 5 on some seed: {n_classes_per_seed}"
        )
        assert np.std(n_classes_per_seed) <= 2.0, (
            f"n_classes unstable across seeds (std > 2): {n_classes_per_seed}"
        )
