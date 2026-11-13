"""Continuous-observation K-cycle POMDP.

State: discrete phase index s ∈ {0, 1, ..., K-1}
Transition: s → (s+1) mod K  (deterministic, action-independent)
Observation: o = sin(2π·s/K) + N(0, obs_noise²)   ∈ ℝ

Ground-truth quotient: n=K classes (each phase is a distinct class).

Actions: 0 (no-op), ignored.  Episode resets after episode_length steps.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class KCycleContinuousEnv(gym.Env):
    """Continuous-observation K-cycle environment.

    Observation space: Box([-2], [2], float32)
    Action space: Discrete(1) — one dummy action
    """

    metadata = {"render_modes": []}

    def __init__(self, K: int = 4, obs_noise: float = 0.1, episode_length: int = 50):
        super().__init__()
        assert K >= 2, f"K must be ≥ 2, got {K}"
        self.K = K
        self.obs_noise = obs_noise
        self.episode_length = episode_length
        self.observation_space = spaces.Box(
            low=np.array([-2.0], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(1)
        self._state: int = 0
        self._step: int = 0
        self._rng: np.random.Generator = np.random.default_rng(0)

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = int(self._rng.integers(0, self.K))
        self._step = 0
        return self._observe(), {}

    def step(self, action):
        self._state = (self._state + 1) % self.K
        self._step += 1
        done = self._step >= self.episode_length
        return self._observe(), 0.0, done, False, {}

    def _observe(self) -> np.ndarray:
        # Use uniform spacing in [0, 1) so all K states have distinct means.
        # sin(2πs/K) maps states {0,K/2} both to 0 for even K — not distinguishable.
        signal = float(self._state) / self.K
        return np.array([signal + self._rng.normal(0, self.obs_noise)], dtype=np.float32)
