"""The policy's RNG must be independent of the environment's noise."""

import numpy as np
import pytest

from environments.layer1_discrete.tiger import TigerPOMDP
from quotient_learning.collection import collect_trajectories


@pytest.mark.unit
def test_stochastic_policy_independent_of_env_noise() -> None:
    cum = np.cumsum([0.8, 0.1, 0.1])

    def policy(_obs, rng):
        return int(np.searchsorted(cum, rng.random(), side="right"))

    ds = collect_trajectories(
        TigerPOMDP(seed=0), policy, n_episodes=20000, max_steps=15, seed=3
    )
    n = left = 0
    for t in ds.trajectories:
        if len(t.observations) >= 3 and t.observations[1] == 0:
            n += 1
            left += t.observations[2] == 0
    # Exact value: 0.8*(0.85^2+0.15^2) + 0.2*0.5 = 0.696 (coupled seeds gave ~0.78).
    assert abs(left / n - 0.696) < 0.02
