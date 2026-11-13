"""Unit tests for auto_select_bins() (T-adaptive-bins).

Verifies that the silhouette-based bin selection correctly identifies
B=2 for a Tiger-like dataset whose observations form two well-separated
Gaussian clusters centred at ±1 (representing 'growl-left' / 'growl-right').
"""

from __future__ import annotations

import numpy as np
import pytest

from continuous_quotient.estimator import auto_select_bins
from continuous_quotient.types import ContinuousTrajectory, ContinuousTrajectoryDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiger_like_dataset(
    n_episodes: int = 200,
    steps_per_episode: int = 20,
    seed: int = 0,
) -> ContinuousTrajectoryDataset:
    """Synthetic Tiger-like dataset.

    Tiger has two latent states (tiger-left / tiger-right).  Observations
    are scalars drawn from N(-1, 0.2²) in state LEFT and N(+1, 0.2²) in
    state RIGHT, matching the two Gaussian clusters around ±1.

    The silhouette maximisation should unambiguously prefer k=2 because the
    clusters are ~10σ apart.
    """
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n_episodes):
        # Each episode uses a fixed latent state (LEFT or RIGHT).
        latent = rng.choice([-1.0, 1.0])
        obs_list = []
        for _ in range(steps_per_episode + 1):
            o = rng.normal(loc=latent, scale=0.2)
            obs_list.append(np.array([o], dtype=np.float32))
        acts = [rng.integers(0, 2) for _ in range(steps_per_episode)]
        rews = [0.0] * steps_per_episode
        trajs.append(ContinuousTrajectory(
            observations=obs_list, actions=acts, rewards=rews
        ))
    return ContinuousTrajectoryDataset(trajectories=trajs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAutoSelectBins:
    """auto_select_bins should return B=2 for a Tiger-like two-cluster dataset."""

    def test_tiger_like_selects_two_bins(self):
        """Two well-separated Gaussian clusters → silhouette peak at k=2."""
        dataset = _make_tiger_like_dataset(n_episodes=200, steps_per_episode=20, seed=0)
        b = auto_select_bins(dataset, b_min=2, b_max=16, n_sample=2000, seed=0)
        assert b == 2, (
            f"Expected auto_select_bins to return 2 for Tiger-like data, got {b}"
        )

    def test_returns_int(self):
        """Return type must be int."""
        dataset = _make_tiger_like_dataset(n_episodes=50, seed=1)
        b = auto_select_bins(dataset, b_min=2, b_max=8, seed=1)
        assert isinstance(b, int), f"Expected int, got {type(b)}"

    def test_respects_b_min(self):
        """Result must be >= b_min regardless of data."""
        dataset = _make_tiger_like_dataset(n_episodes=50, seed=2)
        for b_min in [2, 3, 4]:
            b = auto_select_bins(dataset, b_min=b_min, b_max=8, seed=0)
            assert b >= b_min, f"b={b} < b_min={b_min}"

    def test_small_dataset_does_not_crash(self):
        """Edge case: very few observations should not raise."""
        rng = np.random.default_rng(42)
        tiny = ContinuousTrajectoryDataset(trajectories=[
            ContinuousTrajectory(
                observations=[np.array([rng.normal()], dtype=np.float32) for _ in range(5)],
                actions=[0, 0, 0, 0],
                rewards=[0.0, 0.0, 0.0, 0.0],
            )
        ])
        # Should not raise even with very few points.
        b = auto_select_bins(tiny, b_min=2, b_max=8, seed=0)
        assert isinstance(b, int)

    def test_subsample_reproduces_same_result(self):
        """With the same seed, subsampling is deterministic."""
        dataset = _make_tiger_like_dataset(n_episodes=300, seed=5)
        b1 = auto_select_bins(dataset, n_sample=500, seed=7)
        b2 = auto_select_bins(dataset, n_sample=500, seed=7)
        assert b1 == b2, "Same seed should give same result"
