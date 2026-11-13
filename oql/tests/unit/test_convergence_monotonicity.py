"""Convergence monotonicity test (T022).

`ConvergenceTracker.is_stable` should be True once the learned quotient size
has not changed across the last `window` checkpoints, and the size should
remain bounded as trajectory count grows.
"""

from __future__ import annotations

import pytest

from environments.layer1_discrete.tiger import TigerPOMDP
from quotient_learning.convergence import run_convergence_ladder


@pytest.mark.unit
class TestConvergenceMonotonicity:
    def test_tiger_quotient_size_bounded(self) -> None:
        env = TigerPOMDP(seed=11)
        checkpoints, q = run_convergence_ladder(
            env=env,
            budgets=[500, 1000, 2000, 4000],
            estimator_kwargs=dict(
                window_length=2, history_max_length=2,
                min_history_occurrences=50, alpha=0.01,
                max_steps_per_episode=8,
            ),
            seed=11,
        )
        sizes = [c.n_classes for c in checkpoints]
        # Should never explode — quotient size is bounded by a small constant.
        assert max(sizes) <= 8, f"learned quotient sizes {sizes} exceed sanity bound"
        # Last two checkpoints should agree (stability).
        assert sizes[-1] == sizes[-2], f"final two checkpoints disagree: {sizes}"

    def test_stability_window(self) -> None:
        env = TigerPOMDP(seed=12)
        checkpoints, _ = run_convergence_ladder(
            env=env,
            budgets=[500, 1000, 1500, 2000],
            estimator_kwargs=dict(
                window_length=2, history_max_length=2,
                min_history_occurrences=50, alpha=0.01,
                max_steps_per_episode=8,
            ),
            seed=12,
        )
        # We constructed 4 checkpoints; a stability window of 3 should be checkable.
        sizes = [c.n_classes for c in checkpoints]
        # The last three sizes should be identical for Tiger under random policy.
        assert len(set(sizes[-3:])) == 1, f"final 3 sizes not stable: {sizes}"
