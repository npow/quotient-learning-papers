"""Integration test: continuous Tiger recovers n=2 on ≥4/5 seeds (T017).

Marked as @pytest.mark.integration — requires only numpy/scipy, no MuJoCo.
Fast enough to run in CI (~60s for 5 seeds at N=1000).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow running from project root without install.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experiments" / "layer1_ground_truth"))

from continuous_quotient import ContinuousEstimatorConfig, ContinuousQuotientEstimator
from envs.tiger_continuous import TigerContinuousEnv


@pytest.mark.integration
def test_tiger_continuous_recovers_n2():
    """ContinuousQuotientEstimator should recover n=2 on ≥4/5 seeds."""
    config = ContinuousEstimatorConfig(
        n_trajectories=1000,
        max_steps_per_episode=20,
        window_length=2,
        history_max_length=1,
        min_history_occurrences=15,
        alpha=0.05,         # raw per-pair level; Bonferroni via larger n_permutations for paper
        n_permutations=99,
        history_n_bins=8,
        max_samples_per_test=300,
    )
    estimator = ContinuousQuotientEstimator(config)
    correct = 0
    for seed in range(5):
        env = TigerContinuousEnv(sigma=0.5, episode_length=20)
        quotient = estimator.fit_from_env(env, seed=seed)
        if quotient.n_classes == 2:
            correct += 1
    assert correct >= 4, (
        f"Only {correct}/5 seeds recovered n=2. "
        "Increase N or check continuation window extraction."
    )


@pytest.mark.integration
def test_tiger_continuous_quotient_nontrivial():
    """Quotient should have n_classes ≥ 1 and partition non-empty."""
    config = ContinuousEstimatorConfig(
        n_trajectories=500,
        max_steps_per_episode=20,
        window_length=2,
        history_max_length=1,
        min_history_occurrences=10,
        alpha=0.05,
        n_permutations=49,
    )
    env = TigerContinuousEnv(sigma=0.5, episode_length=20)
    estimator = ContinuousQuotientEstimator(config)
    quotient = estimator.fit_from_env(env, seed=0)
    assert quotient.n_classes >= 1
    assert len(quotient.partition) >= 1
