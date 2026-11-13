"""Tiger POMDP (Cassandra/Kaelbling/Littman 1998).

Two hidden states (tiger-left / tiger-right), three actions (listen, open-left,
open-right), two observations (hear-left / hear-right). The standard noisy
listen has 85% accuracy. Opening a door terminates the episode and starts a
fresh one with a freshly sampled hidden state.

Used by US1 as the canonical correctness benchmark — the Myhill-Nerode
quotient of a capacity-2 bounded agent is 2 classes (lean-left, lean-right).
"""

from __future__ import annotations

import numpy as np

from environments.base import POMDPEnvironment, StepResult

ACT_LISTEN = 0
ACT_OPEN_LEFT = 1
ACT_OPEN_RIGHT = 2

OBS_HEAR_LEFT = 0
OBS_HEAR_RIGHT = 1

STATE_TIGER_LEFT = 0
STATE_TIGER_RIGHT = 1


class TigerPOMDP(POMDPEnvironment):
    """Standard Tiger POMDP."""

    _k1_full_support = True

    def __init__(
        self,
        listen_accuracy: float = 0.85,
        listen_reward: float = -1.0,
        correct_open_reward: float = 10.0,
        wrong_open_reward: float = -100.0,
        seed: int | None = None,
    ):
        if not 0.5 <= listen_accuracy <= 1.0:
            raise ValueError("listen_accuracy must be in [0.5, 1.0]")
        self.listen_accuracy = listen_accuracy
        self.listen_reward = listen_reward
        self.correct_open_reward = correct_open_reward
        self.wrong_open_reward = wrong_open_reward
        self._rng = np.random.default_rng(seed)
        self._state: int = STATE_TIGER_LEFT

    @property
    def n_observations(self) -> int:
        return 2

    @property
    def n_actions(self) -> int:
        return 3

    @property
    def environment_id(self) -> str:
        return f"Tiger(acc={self.listen_accuracy})"

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "ground_truth_n_classes": 2,
            "listen_accuracy": self.listen_accuracy,
        }

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = int(self._rng.integers(0, 2))
        # Initial observation: uninformative listen-style draw from uniform prior.
        return int(self._rng.integers(0, 2))

    def step(self, action: int) -> StepResult:
        if action == ACT_LISTEN:
            correct = self._state == STATE_TIGER_LEFT
            if self._rng.random() < self.listen_accuracy:
                obs = OBS_HEAR_LEFT if correct else OBS_HEAR_RIGHT
            else:
                obs = OBS_HEAR_RIGHT if correct else OBS_HEAR_LEFT
            return StepResult(observation=obs, reward=self.listen_reward,
                              terminated=False, truncated=False)
        if action in (ACT_OPEN_LEFT, ACT_OPEN_RIGHT):
            opened_left = action == ACT_OPEN_LEFT
            tiger_left = self._state == STATE_TIGER_LEFT
            correct_open = opened_left != tiger_left  # opened away from tiger
            reward = self.correct_open_reward if correct_open else self.wrong_open_reward
            # Episode terminates; emit a noisy observation as the final draw.
            obs = int(self._rng.integers(0, 2))
            return StepResult(observation=obs, reward=reward,
                              terminated=True, truncated=False)
        raise ValueError(f"unknown action {action}; must be in {{0,1,2}}")
