"""Integration test: RockSample(7,8) quotient recovery (T021).

Smoke test that the estimator runs to completion on a structured larger POMDP
and produces a finite, well-formed quotient. We don't have a tight
ground-truth quotient size for RockSample, so we assert structural sanity
properties instead.
"""

from __future__ import annotations

import pytest

from environments.layer1_discrete.rocksample import RockSamplePOMDP
from quotient_learning.estimator import QuotientEstimator


@pytest.mark.integration
@pytest.mark.slow
class TestRockSampleRecovery:
    def test_completes_and_returns_finite_quotient(self) -> None:
        env = RockSamplePOMDP(grid_size=7, n_rocks=8, seed=0)
        est = QuotientEstimator(
            env=env,
            n_trajectories=500,        # small budget so the test is CI-friendly
            window_length=2,
            history_max_length=1,
            min_history_occurrences=30,
            alpha=0.01,
            max_steps_per_episode=20,
        )
        q = est.run(seed=0)
        assert q.n_classes >= 1
        assert q.n_classes <= 50, f"unexpectedly large quotient: {q.n_classes} classes"
        # Class IDs are dense [0, n_classes).
        assert all(0 <= c < q.n_classes for c in q.partition.values())
        # Transition function values must reference valid class IDs.
        for (cid, _obs), cid_next in q.transition.items():
            assert 0 <= cid < q.n_classes
            assert 0 <= cid_next < q.n_classes
