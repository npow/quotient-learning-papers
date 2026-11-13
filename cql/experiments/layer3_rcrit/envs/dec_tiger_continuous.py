"""Dec-Tiger POMDP with independent Gaussian observations for each agent.

State: s ∈ {0=Left, 1=Right}
Agent A sees: obs_A = signal + N(0, sigma_A²)
Agent B sees: obs_B = signal + N(0, sigma_B²)  (same signal, independent noise)
signal = -1.0 if Left, +1.0 if Right

Both agents independently observe the same underlying tiger state through their
own noisy channel.  The Dec-POMDP structure means they act without communicating,
but both observations carry the same mutual information about the state.

Analytical R_crit: H(Q_A | Q_B) where Q_A = Q_B = 2 classes.
When both agents share the same 2-class quotient structure and the same
underlying state, the conditional entropy H(Q̂_A | Q̂_B) ≈ 0 bits — the agents
can perfectly coordinate because their quotient classes are aligned.

For the R_crit *estimation* experiment the key result is:
  estimated R_crit = H(Q̂_A | Q̂_B) should be near 0 when both agents
  have the same quotient structure (n_A = n_B = 2).
"""

from __future__ import annotations

import numpy as np


class DecTigerContinuousEnv:
    """Two-agent Dec-Tiger with Gaussian observations.

    State: s ∈ {0=Left, 1=Right}
    Agent A sees: obs_A = signal + N(0, sigma_A²)
    Agent B sees: obs_B = signal + N(0, sigma_B²)  (same signal, independent noise)
    signal = -1.0 if Left, +1.0 if Right

    Analytical R_crit: H(Q_A | Q_B) where Q_A = Q_B = 2 classes
    → R_crit = I(Z_A; Z_B) - H(Z_A) + H(Z_B) but since both agents have
      identical 2-class quotients and share the underlying state,
      R_crit ≈ 0 bits (agents with equal capacity and same environment
      can perfectly coordinate).

    But for the R_crit *estimation* experiment, the key result is:
    The estimated R_crit = H(Q̂_A | Q̂_B) should be near 0 when both
    agents have the same quotient structure (n_A = n_B = 2).

    Parameters
    ----------
    sigma_a : float
        Standard deviation of agent A's observation noise.
    sigma_b : float
        Standard deviation of agent B's observation noise.
    episode_length : int
        Maximum steps per episode before truncation.
    """

    def __init__(
        self,
        sigma_a: float = 0.5,
        sigma_b: float = 0.5,
        episode_length: int = 20,
    ) -> None:
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        self.episode_length = episode_length

        self._state: int = 0
        self._step: int = 0
        self._rng: np.random.Generator = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Reset the environment and return initial observations.

        Parameters
        ----------
        seed : int or None
            If provided, re-seeds the RNG before sampling the initial state.

        Returns
        -------
        obs_A : np.ndarray, shape (1,)
        obs_B : np.ndarray, shape (1,)
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = int(self._rng.integers(0, 2))
        self._step = 0
        return self._observe()

    def step(
        self,
        action_A: int,
        action_B: int,
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """Advance the environment by one step.

        Actions for each agent:
          0 = listen  (reward -1 each)
          1 = open-left
          2 = open-right

        The joint reward uses the *average* of agent rewards; the state
        resets after any opening action.

        Parameters
        ----------
        action_A : int
            Action chosen by agent A.
        action_B : int
            Action chosen by agent B.

        Returns
        -------
        obs_A : np.ndarray, shape (1,)
        obs_B : np.ndarray, shape (1,)
        reward : float
            Joint team reward (mean of individual rewards).
        done : bool
            True when the episode has ended.
        """
        action_A = int(action_A)
        action_B = int(action_B)

        reward_A = self._agent_reward(action_A)
        reward_B = self._agent_reward(action_B)
        reward = 0.5 * (reward_A + reward_B)

        # Reset tiger position after any opening action.
        if action_A in (1, 2) or action_B in (1, 2):
            self._state = int(self._rng.integers(0, 2))

        self._step += 1
        done = self._step >= self.episode_length
        obs_A, obs_B = self._observe()
        return obs_A, obs_B, reward, done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _signal(self) -> float:
        """Return the noiseless signal for the current state."""
        return -1.0 if self._state == 0 else 1.0

    def _observe(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample independent Gaussian observations for both agents."""
        signal = self._signal()
        obs_A = np.array(
            [signal + self._rng.normal(0.0, self.sigma_a)],
            dtype=np.float32,
        )
        obs_B = np.array(
            [signal + self._rng.normal(0.0, self.sigma_b)],
            dtype=np.float32,
        )
        return obs_A, obs_B

    def _agent_reward(self, action: int) -> float:
        """Individual reward for one agent's action against the current state."""
        if action == 0:  # listen
            return -1.0
        if action == 1:  # open-left
            return 10.0 if self._state == 1 else -100.0
        # action == 2: open-right
        return 10.0 if self._state == 0 else -100.0
