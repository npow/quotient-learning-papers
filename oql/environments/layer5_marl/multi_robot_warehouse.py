"""Multi-Robot Warehouse cooperative POMDP (T067).

Simplified Semaphore-AI style warehouse: 2 robots navigate a corridor to
collect goods. Each robot observes its own position (discretized) and a
binary 'good in pickup zone' signal. The other robot's position is hidden.

Single-agent view exposed for quotient learning; the second agent acts under
a uniform-random policy.
"""

from __future__ import annotations

import numpy as np

from environments.base import POMDPEnvironment, StepResult


class MultiRobotWarehouseSingleAgent(POMDPEnvironment):
    """One robot's view of a 2-robot warehouse."""

    def __init__(self, corridor_length: int = 5, max_steps: int = 30, seed: int = 0):
        if corridor_length < 3:
            raise ValueError("corridor_length must be >= 3")
        self.corridor_length = corridor_length
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._x = 0
        self._other_x = 0
        self._good_pos = 0
        self._t = 0

    @property
    def n_observations(self) -> int:
        # (own_position bucket [0..corridor_length-1]) × (good_present binary)
        return 2 * self.corridor_length

    @property
    def n_actions(self) -> int:
        return 3  # left, stay, right

    @property
    def environment_id(self) -> str:
        return f"MRW(L={self.corridor_length})"

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._x = 0
        self._other_x = self.corridor_length - 1
        self._good_pos = int(self._rng.integers(0, self.corridor_length))
        self._t = 0
        return self._observe()

    def _observe(self) -> int:
        good_present = int(self._x == self._good_pos)
        return self._x * 2 + good_present

    def step(self, action: int) -> StepResult:
        # Self moves.
        if action == 0:
            self._x = max(0, self._x - 1)
        elif action == 2:
            self._x = min(self.corridor_length - 1, self._x + 1)
        # Other agent does a uniform-random move.
        other_action = int(self._rng.integers(0, 3))
        if other_action == 0:
            self._other_x = max(0, self._other_x - 1)
        elif other_action == 2:
            self._other_x = min(self.corridor_length - 1, self._other_x + 1)
        reward = -0.1
        if self._x == self._good_pos:
            reward += 1.0
            self._good_pos = int(self._rng.integers(0, self.corridor_length))
        if self._other_x == self._x:
            reward -= 1.0  # collision
        self._t += 1
        truncated = self._t >= self.max_steps
        return StepResult(observation=self._observe(), reward=reward,
                          terminated=False, truncated=truncated)
