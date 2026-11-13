"""RockSample(n, k) POMDP (Smith & Simmons 2004).

Parameterized: an `n x n` grid with `k` rocks placed at fixed positions.
Each rock is independently 'good' or 'bad' (hidden binary state). Actions:
- 0..3: move N/S/E/W
- 4: sample current cell (no-op unless on a rock; deterministic +/-)
- 5..(5+k-1): query rock `i` — observe noisy good/bad reading whose accuracy
  decreases with Euclidean distance.

Observation alphabet is {0: no-reading, 1: rock-good, 2: rock-bad}. After the
agent exits east of column n-1 the episode terminates with reward +10.

This is the standard RockSample variant adapted to the model-free trajectory
collection workflow. Concrete instances used by US1 / US2 tests: (7,8),
(15,15), (20,20).
"""

from __future__ import annotations

import numpy as np

from environments.base import POMDPEnvironment, StepResult

OBS_NONE = 0
OBS_GOOD = 1
OBS_BAD = 2


class RockSamplePOMDP(POMDPEnvironment):
    """RockSample(n, k)."""

    def __init__(
        self,
        grid_size: int,
        n_rocks: int,
        max_steps: int | None = None,
        sensor_decay: float = 20.0,
        seed: int | None = None,
    ):
        self.grid_size = grid_size
        self.n_rocks = n_rocks
        self.sensor_decay = sensor_decay
        self.max_steps = max_steps or (4 * grid_size * grid_size)
        self._rng = np.random.default_rng(seed)
        # Fixed rock positions, deterministic per seed (placement is part of the env).
        placement_rng = np.random.default_rng(1234567)
        # Sample n_rocks distinct positions from the grid (excluding exit column).
        positions: set[tuple[int, int]] = set()
        while len(positions) < n_rocks:
            x = int(placement_rng.integers(0, max(1, grid_size - 1)))
            y = int(placement_rng.integers(0, grid_size))
            positions.add((x, y))
        self.rock_positions: list[tuple[int, int]] = sorted(positions)
        self._rock_qualities: np.ndarray = np.zeros(n_rocks, dtype=bool)
        self._x = 0
        self._y = 0
        self._t = 0

    @property
    def n_observations(self) -> int:
        return 3  # {none, good, bad}

    @property
    def n_actions(self) -> int:
        return 5 + self.n_rocks

    @property
    def environment_id(self) -> str:
        return f"RockSample({self.grid_size},{self.n_rocks})"

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "grid_size": self.grid_size,
            "n_rocks": self.n_rocks,
            "max_steps": self.max_steps,
        }

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Agent starts west-center.
        self._x = 0
        self._y = self.grid_size // 2
        self._t = 0
        self._rock_qualities = self._rng.random(self.n_rocks) < 0.5
        return OBS_NONE

    def step(self, action: int) -> StepResult:
        self._t += 1
        truncated = self._t >= self.max_steps
        reward = -0.01
        terminated = False
        observation = OBS_NONE
        if action == 0:  # N
            self._y = min(self.grid_size - 1, self._y + 1)
        elif action == 1:  # S
            self._y = max(0, self._y - 1)
        elif action == 2:  # E
            self._x += 1
            if self._x >= self.grid_size:
                terminated = True
                reward = 10.0
                self._x = self.grid_size - 1
        elif action == 3:  # W
            self._x = max(0, self._x - 1)
        elif action == 4:  # sample
            for ri, (rx, ry) in enumerate(self.rock_positions):
                if rx == self._x and ry == self._y:
                    reward = 10.0 if self._rock_qualities[ri] else -10.0
                    self._rock_qualities[ri] = False  # consumed
                    break
        else:
            rock_idx = action - 5
            if 0 <= rock_idx < self.n_rocks:
                rx, ry = self.rock_positions[rock_idx]
                dist = float(np.hypot(rx - self._x, ry - self._y))
                accuracy = 0.5 * (1.0 + np.exp(-dist / self.sensor_decay))
                truth_good = bool(self._rock_qualities[rock_idx])
                if self._rng.random() < accuracy:
                    observation = OBS_GOOD if truth_good else OBS_BAD
                else:
                    observation = OBS_BAD if truth_good else OBS_GOOD
        return StepResult(observation=observation, reward=reward,
                          terminated=terminated, truncated=truncated)
