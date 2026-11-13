"""Small partially observable gridworld.

The agent navigates an `n x n` grid where the only observation is whether the
agent is on a wall-adjacent cell (binary). The hidden state is the agent's
(x, y) position. Used as a sanity-check environment between Tiger (trivial)
and RockSample (complex).
"""

from __future__ import annotations

import numpy as np

from environments.base import POMDPEnvironment, StepResult

ACT_NORTH = 0
ACT_SOUTH = 1
ACT_EAST = 2
ACT_WEST = 3


class GridWorldPOMDP(POMDPEnvironment):
    """`n x n` gridworld with binary wall-adjacency observation."""

    _k1_full_support = True

    def __init__(self, size: int = 4, max_steps: int = 30, seed: int | None = None):
        if size < 2:
            raise ValueError("grid size must be >= 2")
        self.size = size
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._x = 0
        self._y = 0
        self._t = 0

    @property
    def n_observations(self) -> int:
        return 2  # {not-on-wall, on-wall}

    @property
    def n_actions(self) -> int:
        return 4

    @property
    def environment_id(self) -> str:
        return f"GridWorld({self.size}x{self.size})"

    @property
    def metadata(self) -> dict[str, object]:
        return {"size": self.size, "max_steps": self.max_steps}

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._x = int(self._rng.integers(0, self.size))
        self._y = int(self._rng.integers(0, self.size))
        self._t = 0
        return self._observe()

    def _observe(self) -> int:
        on_wall = (
            self._x == 0 or self._y == 0
            or self._x == self.size - 1 or self._y == self.size - 1
        )
        return 1 if on_wall else 0

    def step(self, action: int) -> StepResult:
        dx, dy = {ACT_NORTH: (0, 1), ACT_SOUTH: (0, -1),
                  ACT_EAST: (1, 0), ACT_WEST: (-1, 0)}.get(action, (0, 0))
        nx = max(0, min(self.size - 1, self._x + dx))
        ny = max(0, min(self.size - 1, self._y + dy))
        self._x, self._y = nx, ny
        self._t += 1
        truncated = self._t >= self.max_steps
        return StepResult(observation=self._observe(), reward=-0.1,
                          terminated=False, truncated=truncated)
