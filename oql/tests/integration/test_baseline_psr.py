"""PSR baseline parity test (T040)."""

from __future__ import annotations

import pytest

from environments.layer1_discrete.tiger import TigerPOMDP
from experiments.baselines.psr_spectral import psr_quotient
from quotient_learning.collection import collect_trajectories, uniform_random_policy


@pytest.mark.integration
class TestBaselinePSR:
    def test_runs_on_tiger(self) -> None:
        env = TigerPOMDP(seed=0)
        dataset = collect_trajectories(
            env, uniform_random_policy(env.n_actions), n_episodes=500,
            max_steps=8, seed=0,
        )
        q = psr_quotient(dataset, n_obs=env.n_observations, rank=2,
                        history_length=1, future_length=2, seed=0)
        assert q.n_classes >= 1
        assert q.n_classes <= 10
