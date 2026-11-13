"""Equivalence-test ablation: G-test vs MMD vs EmpiricalBernstein on Tiger (T043)."""

from __future__ import annotations

import pytest

from environments.layer1_discrete.tiger import TigerPOMDP
from quotient_learning.equivalence import (
    EmpiricalBernsteinTest,
    GTestEquivalenceTest,
    MMDEquivalenceTest,
)
from quotient_learning.estimator import QuotientEstimator


@pytest.mark.integration
@pytest.mark.slow
class TestEquivalenceAblation:
    """Each equivalence test should produce a finite, reasonable quotient on Tiger."""

    @pytest.mark.parametrize("test_name", ["gtest", "mmd", "empirical_bernstein"])
    def test_each_test_produces_finite_quotient(self, test_name: str) -> None:
        env = TigerPOMDP(seed=0)
        if test_name == "gtest":
            test = GTestEquivalenceTest()
        elif test_name == "mmd":
            test = MMDEquivalenceTest(n_observations=2, n_permutations=50, seed=0)
        else:
            test = EmpiricalBernsteinTest(n_observations=2, seed=0)
        est = QuotientEstimator(
            env=env, n_trajectories=600, window_length=2,
            history_max_length=2, min_history_occurrences=40,
            equivalence_test=test, alpha=0.01,
            max_steps_per_episode=8,
        )
        q = est.run(seed=0)
        assert 1 <= q.n_classes <= 6, (
            f"{test_name} produced unreasonable quotient: {q.n_classes}"
        )
