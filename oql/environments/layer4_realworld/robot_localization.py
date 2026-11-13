"""Robot-localization POMDP (T050).

A robot navigates an `n x n` grid with `k` colored landmarks. The agent's
hidden state is its (x, y) position; the observation is the bearing-class
(N/S/E/W/none) to the nearest landmark. Actions are N/S/E/W moves with 10%
slip.
"""

from __future__ import annotations

import numpy as np

from environments.base import POMDPEnvironment, StepResult


class RobotLocalizationPOMDP(POMDPEnvironment):
    """`n x n` grid with k landmarks; observation is nearest-landmark bearing."""

    _k1_full_support = True

    def __init__(
        self,
        size: int = 6,
        n_landmarks: int = 4,
        slip_prob: float = 0.1,
        max_steps: int = 50,
        seed: int = 0,
    ):
        if size < 3 or n_landmarks < 1:
            raise ValueError("invalid grid configuration")
        self.size = size
        self.n_landmarks = n_landmarks
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.seed = seed
        construction = np.random.default_rng(seed)
        cells = [(x, y) for x in range(size) for y in range(size)]
        positions = construction.choice(len(cells), size=n_landmarks, replace=False)
        self.landmarks: list[tuple[int, int]] = [cells[i] for i in positions]
        self._rng = np.random.default_rng(seed + 1)
        self._x = 0
        self._y = 0
        self._t = 0

    @property
    def n_observations(self) -> int:
        return 5  # {N, S, E, W, none}

    @property
    def n_actions(self) -> int:
        return 4

    @property
    def environment_id(self) -> str:
        return f"RobotLocalization({self.size}x{self.size},k={self.n_landmarks})"

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._x = int(self._rng.integers(0, self.size))
        self._y = int(self._rng.integers(0, self.size))
        self._t = 0
        return self._observe()

    def _observe(self) -> int:
        if not self.landmarks:
            return 4
        dists = [abs(lx - self._x) + abs(ly - self._y) for lx, ly in self.landmarks]
        nearest = self.landmarks[int(np.argmin(dists))]
        dx = nearest[0] - self._x
        dy = nearest[1] - self._y
        if abs(dx) + abs(dy) == 0:
            return 4
        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 3  # E or W
        return 0 if dy > 0 else 1  # N or S

    def step(self, action: int) -> StepResult:
        if self._rng.random() < self.slip_prob:
            action = int(self._rng.integers(0, 4))
        dx, dy = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0)}.get(action, (0, 0))
        self._x = max(0, min(self.size - 1, self._x + dx))
        self._y = max(0, min(self.size - 1, self._y + dy))
        self._t += 1
        truncated = self._t >= self.max_steps
        return StepResult(observation=self._observe(), reward=-0.1,
                          terminated=False, truncated=truncated)
