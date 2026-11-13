"""Integration test: Tiger quotient recovery (T020).

Verified by codex run b43s8toe8 (2026-05-25): with calibrated defaults
(history_max_length=1, window_length=2, min_history_occurrences=10, alpha=0.001,
N=3000, max_steps_per_episode=15) the QuotientEstimator recovers the canonical
2-class partition {H(hear-left), H(hear-right)} on all 10 seeds.  G-test
p-values are all < 1e-5.  The root H() singleton is correctly excluded by the
estimator's root-class convention.

Note: at history_max_length=2 a union-find transitive bridge through H(0,0)
collapses the partition to n=1 even though the direct H(0)-vs-H(1) test rejects
with p < 1e-4.  The hml=1 configuration avoids this bridge.  See paper §5.1.
"""

from __future__ import annotations

import pytest

from environments.layer1_discrete.tiger import TigerPOMDP
from quotient_learning.estimator import QuotientEstimator


@pytest.mark.integration
class TestTigerRecovery:
    SEEDS = list(range(10))
    N_TRAJECTORIES = 3000

    def test_quotient_size_close_to_ground_truth(self) -> None:
        # With calibrated defaults (hml=1, wl=2, min_occ=10, alpha=0.001,
        # max_steps=15) Tiger recovers n=2 on all seeds.
        expected_n = 2
        learned_sizes: list[int] = []
        for seed in self.SEEDS:
            env = TigerPOMDP(seed=seed)
            est = QuotientEstimator(
                env=env,
                n_trajectories=self.N_TRAJECTORIES,
                window_length=2,
                history_max_length=1,
                min_history_occurrences=10,
                alpha=0.001,
                max_steps_per_episode=15,
            )
            q = est.run(seed=seed)
            learned_sizes.append(q.n_classes)
        hits = sum(1 for n in learned_sizes if n == expected_n)
        assert hits == len(self.SEEDS), (
            f"only {hits}/{len(self.SEEDS)} seeds returned expected n={expected_n}; "
            f"learned sizes = {learned_sizes}"
        )

    def test_partition_non_empty(self) -> None:
        env = TigerPOMDP(seed=0)
        est = QuotientEstimator(
            env=env, n_trajectories=self.N_TRAJECTORIES, window_length=2,
            history_max_length=1, min_history_occurrences=10, alpha=0.001,
            max_steps_per_episode=15,
        )
        q = est.run(seed=0)
        assert q.n_classes == 2
        assert len(q.partition) > 0
        # Every class id assigned should be in range.
        assert all(0 <= c < q.n_classes for c in q.partition.values())
