"""Gaussian-observation Tiger POMDP.

State: s ∈ {0=Left, 1=Right}
Observations: o = state_signal + N(0, σ²)
    state_signal = -1.0 if Left, +1.0 if Right
Actions: 0=listen, 1=open-left, 2=open-right  (discrete)

Ground-truth quotient: n=2 (two distinguishable history classes)
under observation-only history with window_length ≥ 2.

This environment is a drop-in Gymnasium env for ContinuousQuotientEstimator.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TigerContinuousEnv(gym.Env):
    """Tiger POMDP with Gaussian observation noise.

    Observation space: Box([-5], [5], dtype=float32) — scalar signal
    Action space: Discrete(3)  — listen, open-left, open-right

    Rewards:
      listen    : -1
      open-left : +10 if tiger-right, -100 if tiger-left
      open-right: +10 if tiger-left, -100 if tiger-right
    """

    metadata = {"render_modes": []}

    def __init__(self, sigma: float = 0.5, episode_length: int = 20):
        super().__init__()
        self.sigma = sigma
        self.episode_length = episode_length
        self.observation_space = spaces.Box(
            low=np.array([-5.0], dtype=np.float32),
            high=np.array([5.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)
        self._state: int = 0
        self._step: int = 0
        self._rng: np.random.Generator = np.random.default_rng(0)

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = int(self._rng.integers(0, 2))
        self._step = 0
        return self._observe(), {}

    def step(self, action: int):
        action = int(action)
        reward = self._reward(action)
        # Reset state after opening (reset tiger position).
        if action in (1, 2):
            self._state = int(self._rng.integers(0, 2))
        self._step += 1
        done = self._step >= self.episode_length
        return self._observe(), reward, done, False, {}

    def _observe(self) -> np.ndarray:
        signal = -1.0 if self._state == 0 else 1.0
        return np.array([signal + self._rng.normal(0, self.sigma)], dtype=np.float32)

    def _reward(self, action: int) -> float:
        if action == 0:
            return -1.0
        if action == 1:  # open-left
            return 10.0 if self._state == 1 else -100.0
        # action == 2: open-right
        return 10.0 if self._state == 0 else -100.0
