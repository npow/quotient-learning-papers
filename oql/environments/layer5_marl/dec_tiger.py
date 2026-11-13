"""Dec-Tiger Dec-POMDP (Nair et al. 2003) — T066.

Two agents in a Tiger-like POMDP. The hidden state (tiger-left/right) is the
same for both. Each agent independently chooses: listen, open-left, open-right.
Each listen returns a noisy observation; the joint reward depends on both
agents' actions.

For the quotient learning experiment, the environment exposes one
single-agent view at a time via `set_agent(i)` — the learner runs once per
agent to produce per-agent quotients, then `compute_rcrit` is applied to the
pair.
"""

from __future__ import annotations

import numpy as np

from environments.base import POMDPEnvironment, StepResult

ACT_LISTEN = 0
ACT_OPEN_LEFT = 1
ACT_OPEN_RIGHT = 2


class DecTigerSingleAgentView(POMDPEnvironment):
    """A single-agent slice of Dec-Tiger.

    The 'other agent' acts uniformly at random; the controlled agent's
    observations and rewards are exposed. After both agents open a door, the
    episode resets with a fresh hidden state. Each agent gets a noisy
    observation per listen.
    """

    def __init__(
        self,
        listen_accuracy: float = 0.85,
        listen_reward: float = -2.0,    # both listen
        coordinated_open_reward: float = 20.0,  # both open correct
        uncoordinated_open_reward: float = -101.0,  # one opens, other listens
        miscoordinated_open_reward: float = -50.0,  # both open wrong
        seed: int | None = None,
    ):
        if not 0.5 <= listen_accuracy <= 1.0:
            raise ValueError("listen_accuracy must be in [0.5, 1.0]")
        self.listen_accuracy = listen_accuracy
        self.listen_reward = listen_reward
        self.coordinated_open_reward = coordinated_open_reward
        self.uncoordinated_open_reward = uncoordinated_open_reward
        self.miscoordinated_open_reward = miscoordinated_open_reward
        self._rng = np.random.default_rng(seed)
        self._state: int = 0
        # The other agent's policy is uniform random over its 3 actions.

    @property
    def n_observations(self) -> int:
        return 2

    @property
    def n_actions(self) -> int:
        return 3

    @property
    def environment_id(self) -> str:
        return f"DecTigerSingleAgent(acc={self.listen_accuracy})"

    @property
    def metadata(self) -> dict[str, object]:
        return {"two_agent_dec_pomdp": True}

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = int(self._rng.integers(0, 2))
        return int(self._rng.integers(0, 2))

    def _listen_obs(self) -> int:
        correct = self._state == 0
        if self._rng.random() < self.listen_accuracy:
            return 0 if correct else 1
        return 1 if correct else 0

    def step(self, action: int) -> StepResult:
        other = int(self._rng.integers(0, 3))
        # Joint reward + per-agent observation logic.
        if action == ACT_LISTEN and other == ACT_LISTEN:
            return StepResult(observation=self._listen_obs(),
                              reward=self.listen_reward,
                              terminated=False, truncated=False)
        if action == ACT_LISTEN:
            # The other opened a door; agent observes uniformly.
            return StepResult(observation=int(self._rng.integers(0, 2)),
                              reward=self.uncoordinated_open_reward,
                              terminated=True, truncated=False)
        # action is open-left / open-right
        opened_left_a = action == ACT_OPEN_LEFT
        if other == ACT_LISTEN:
            return StepResult(observation=int(self._rng.integers(0, 2)),
                              reward=self.uncoordinated_open_reward,
                              terminated=True, truncated=False)
        opened_left_b = other == ACT_OPEN_LEFT
        tiger_left = self._state == 0
        # Both opened. Reward depends on whether both opened safely.
        a_safe = (opened_left_a != tiger_left)
        b_safe = (opened_left_b != tiger_left)
        if a_safe and b_safe:
            reward = self.coordinated_open_reward
        else:
            reward = self.miscoordinated_open_reward
        return StepResult(observation=int(self._rng.integers(0, 2)),
                          reward=reward, terminated=True, truncated=False)
