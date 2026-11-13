"""Bisimulation oracle baseline parity test (T041)."""

from __future__ import annotations

import pytest

from environments.layer1_discrete.synthetic import SyntheticPOMDP
from experiments.baselines.bisimulation import bisimulation_quotient


@pytest.mark.integration
class TestBaselineBisimulation:
    def test_oracle_partition_finite(self) -> None:
        env = SyntheticPOMDP(n_states=20, n_obs=4, n_actions=2, seed=0)
        q = bisimulation_quotient(env, max_iterations=20)
        assert 1 <= q.n_classes <= env._n_states
