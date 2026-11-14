"""Integration test: continuous K-cycle recovers n=K on ≥4/5 seeds (T028).

Marked @pytest.mark.integration and @pytest.mark.slow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experiments" / "layer1_ground_truth"))

from continuous_quotient import ContinuousEstimatorConfig, ContinuousQuotientEstimator
from envs.kcycle_continuous import KCycleContinuousEnv


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("K", [2, 3, 4])
def test_kcycle_recovers_k_classes(K: int):
    """ContinuousQuotientEstimator should recover n=K on ≥4/5 seeds."""
    config = ContinuousEstimatorConfig(
        n_trajectories=3000,
        max_steps_per_episode=50,
        window_length=2,
        history_max_length=1,
        min_history_occurrences=20,
        alpha=0.01,
        n_permutations=199,
        history_n_bins=K,   # aligned resolution B=K (paper Table 1); the package default B=8 over-splits (Table 2)
    )
    estimator = ContinuousQuotientEstimator(config)
    correct = 0
    for seed in range(5):
        env = KCycleContinuousEnv(K=K, obs_noise=0.1, episode_length=50)
        quotient = estimator.fit_from_env(env, seed=seed)
        if quotient.n_classes == K:
            correct += 1
    assert correct >= 4, f"K={K}: only {correct}/5 seeds recovered n=K"
