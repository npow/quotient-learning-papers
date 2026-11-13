"""Continuous-observation Gaussian-mixture random-walk POMDP.

State space:  N discrete states {0, 1, ..., N-1}
Transition:   uniform over {max(0,s-1), s, min(N-1,s+1)}  (random walk, clamped)
Observation:  obs = state + N(0, sigma²)  ∈  R

Ground-truth quotient: n* = N
  - Boundary states (0 and N-1) have 2 reachable neighbours; interior states
    have 3.  All N continuation distributions are therefore distinct, giving
    exactly N quotient classes.
  - When sigma << 1 (default 0.15), the Gaussian blobs centred at {0,…,N-1}
    are well-separated (spacing 1.0), so the observations reliably identify
    the current state.

This environment is the "continuous-observation" analogue of K-cycle: ground
truth is analytically known (n* = N) but observations are fully continuous.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GaussianMixturePOMDP(gym.Env):
    """N-state random-walk POMDP with continuous Gaussian observations.

    Ground truth: n* = N (each state is a distinct quotient class).
    Boundary states (0 and N-1) have different transition distributions
    from interior states → all N states are continuation-distinguishable.

    Observation: obs = state + N(0, sigma²) ∈ R
    Transition: uniform over {max(0,s-1), s, min(N-1,s+1)}
    """

    metadata = {"render_modes": []}

    def __init__(self, N: int = 5, sigma: float = 0.15, episode_length: int = 50):
        super().__init__()
        assert N >= 2, f"N must be >= 2, got {N}"
        assert sigma > 0, f"sigma must be > 0, got {sigma}"
        self.N = N
        self.sigma = sigma
        self.episode_length = episode_length
        self.observation_space = spaces.Box(
            low=np.array([-2.0], dtype=np.float32),
            high=np.array([float(N) + 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(1)  # no-op action
        self._state: int = 0
        self._step: int = 0
        self._rng: np.random.Generator = np.random.default_rng(0)

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = int(self._rng.integers(0, self.N))
        self._step = 0
        return self._observe(), {}

    def step(self, action):
        delta = self._rng.choice([-1, 0, 1])
        self._state = int(np.clip(self._state + delta, 0, self.N - 1))
        self._step += 1
        done = self._step >= self.episode_length
        return self._observe(), 0.0, done, False, {}

    def _observe(self) -> np.ndarray:
        return np.array(
            [self._state + self._rng.normal(0, self.sigma)], dtype=np.float32
        )
