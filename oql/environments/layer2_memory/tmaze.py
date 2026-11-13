"""T-Maze and Passive T-Maze POMDPs (Bakker 2001) — T045.

The agent traverses an east-bound corridor of length `corridor_length`. At
position 0 it observes a clue (LEFT or RIGHT). At the T-junction (position
corridor_length - 1) it must turn into the matching arm to receive +1 reward;
the wrong turn gives -1. Inside the corridor the observation is a single
"in-corridor" symbol — no information leakage.

Memory horizon = corridor_length. The minimal quotient grows linearly with
corridor length (each position-in-corridor with a remembered clue is its own
class).
"""

from __future__ import annotations

import numpy as np

from environments.base import POMDPEnvironment, StepResult

OBS_CORRIDOR = 0
OBS_CLUE_LEFT = 1
OBS_CLUE_RIGHT = 2
OBS_JUNCTION = 3

ACT_FORWARD = 0
ACT_LEFT = 1
ACT_RIGHT = 2


class TMazePOMDP(POMDPEnvironment):
    """Active T-Maze: agent sees the clue at the start and acts on it later."""

    _k1_full_support = True

    def __init__(self, corridor_length: int = 5, seed: int | None = None):
        if corridor_length < 2:
            raise ValueError("corridor_length must be >= 2")
        self.corridor_length = corridor_length
        self._rng = np.random.default_rng(seed)
        self._pos = 0
        self._clue_left = True

    @property
    def n_observations(self) -> int:
        return 4

    @property
    def n_actions(self) -> int:
        return 3

    @property
    def environment_id(self) -> str:
        return f"TMaze(L={self.corridor_length})"

    @property
    def metadata(self) -> dict[str, object]:
        # Memory-horizon hypothesis: quotient grows ~linearly with corridor length.
        return {
            "corridor_length": self.corridor_length,
            "expected_quotient_lower_bound": 2,
        }

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._clue_left = bool(self._rng.integers(0, 2))
        self._pos = 0
        return OBS_CLUE_LEFT if self._clue_left else OBS_CLUE_RIGHT

    def step(self, action: int) -> StepResult:
        reward = -0.01
        if self._pos < self.corridor_length - 1:
            if action == ACT_FORWARD:
                self._pos += 1
            obs = (OBS_JUNCTION if self._pos == self.corridor_length - 1
                   else OBS_CORRIDOR)
            return StepResult(observation=obs, reward=reward,
                              terminated=False, truncated=False)
        # At junction: turning terminates the episode.
        if action == ACT_LEFT:
            reward = 1.0 if self._clue_left else -1.0
            return StepResult(observation=OBS_JUNCTION, reward=reward,
                              terminated=True, truncated=False)
        if action == ACT_RIGHT:
            reward = 1.0 if not self._clue_left else -1.0
            return StepResult(observation=OBS_JUNCTION, reward=reward,
                              terminated=True, truncated=False)
        return StepResult(observation=OBS_JUNCTION, reward=reward,
                          terminated=False, truncated=False)


class PassiveTMazePOMDP(TMazePOMDP):
    """Passive variant: agent's actions don't advance the corridor — time does.

    The corridor automatically advances by one step per turn; the only action
    that matters is the left/right turn at the junction. Tests pure memory
    capacity.
    """

    def step(self, action: int) -> StepResult:
        reward = -0.01
        if self._pos < self.corridor_length - 1:
            self._pos += 1
            obs = (OBS_JUNCTION if self._pos == self.corridor_length - 1
                   else OBS_CORRIDOR)
            return StepResult(observation=obs, reward=reward,
                              terminated=False, truncated=False)
        if action == ACT_LEFT:
            reward = 1.0 if self._clue_left else -1.0
            return StepResult(observation=OBS_JUNCTION, reward=reward,
                              terminated=True, truncated=False)
        if action == ACT_RIGHT:
            reward = 1.0 if not self._clue_left else -1.0
            return StepResult(observation=OBS_JUNCTION, reward=reward,
                              terminated=True, truncated=False)
        return StepResult(observation=OBS_JUNCTION, reward=reward,
                          terminated=False, truncated=False)
