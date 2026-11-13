"""POMDP-EM baseline parity test (T042)."""

from __future__ import annotations

import pytest

from environments.layer1_discrete.tiger import TigerPOMDP
from experiments.baselines.pomdp_em import em_quotient
from quotient_learning.collection import collect_trajectories, uniform_random_policy


@pytest.mark.integration
class TestBaselineEM:
    def test_runs_on_tiger(self) -> None:
        env = TigerPOMDP(seed=0)
        dataset = collect_trajectories(
            env, uniform_random_policy(env.n_actions), n_episodes=300,
            max_steps=8, seed=0,
        )
        q = em_quotient(dataset, n_states=4, n_obs=env.n_observations,
                       n_iterations=5, seed=0)
        assert 1 <= q.n_classes <= 4
