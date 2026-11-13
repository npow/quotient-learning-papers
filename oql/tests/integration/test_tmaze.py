"""Integration test for T-Maze memory horizon (T037)."""

from __future__ import annotations

import pytest

from environments.layer2_memory.tmaze import TMazePOMDP
from quotient_learning.estimator import QuotientEstimator


@pytest.mark.integration
@pytest.mark.slow
class TestTMaze:
    def test_runs_to_completion(self) -> None:
        env = TMazePOMDP(corridor_length=4, seed=0)
        est = QuotientEstimator(
            env=env, n_trajectories=1000, window_length=1,
            history_max_length=1, min_history_occurrences=30,
            alpha=0.01, max_steps_per_episode=10,
        )
        q = est.run(seed=0)
        assert q.n_classes >= 1
        assert q.n_classes <= 10
