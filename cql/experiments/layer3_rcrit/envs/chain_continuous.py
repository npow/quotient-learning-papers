"""N-state random-walk chain Dec-POMDP with independent Gaussian observations.

State space: {0, 1, ..., N-1}
Transitions: random walk — from state s, move to s-1, s, or s+1 with equal
             probability (1/3 each), clamped to [0, N-1].
Agent A observes: obs_A = state/N + N(0, sigma_A²)   (1-d float)
Agent B observes: obs_B = state/N + N(0, sigma_B²)   (1-d float)
Actions: 0 = move left, 1 = move right (used only to define the action space;
         the transition is a pure random walk and ignores actions, making this
         a pure observation-only coordination task).

Why this environment is interesting for R_crit validation:
  - 4 states (N=4) vs 2 states in Dec-Tiger → richer quotient structure.
  - Each state is a distinct quotient class (σ_A=0.1 is small enough to
    distinguish adjacent states whose signals are 0.25 apart).
  - Analytical R_crit = H(Q_A | Q_B) is computable via numerical integration
    over a 4×4 joint distribution.
  - With σ_A=0.1, σ_B=0.5 the asymmetry is clear: R_crit > 0.
"""

from __future__ import annotations

import numpy as np


class ChainContinuousDecEnv:
    """N-state random-walk chain with independent Gaussian observations per agent.

    Parameters
    ----------
    N : int
        Number of states in the chain (default 4).
    sigma_a : float
        Observation noise std for agent A.
    sigma_b : float
        Observation noise std for agent B.
    episode_length : int
        Maximum steps per episode before truncation.
    """

    def __init__(
        self,
        N: int = 4,
        sigma_a: float = 0.1,
        sigma_b: float = 0.5,
        episode_length: int = 20,
    ) -> None:
        self.N = N
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        self.episode_length = episode_length
        self.n_actions = 2

        self._state: int = 0
        self._step: int = 0
        self._rng: np.random.Generator = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Reset environment; return (obs_A, obs_B) of shape (1,) each."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = int(self._rng.integers(0, self.N))
        self._step = 0
        return self._observe()

    def step(
        self,
        action_a: int,
        action_b: int,
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """Advance by one step.

        The state transitions as a random walk (actions are ignored for the
        state dynamics — agents can only communicate implicitly via their
        shared coordination target, not by affecting the world).

        Returns
        -------
        obs_A, obs_B : np.ndarray, shape (1,)
        reward : float
            +1.0 if the state lands on the middle state (N//2), else 0.0.
        done : bool
        """
        # Random walk: move -1, 0, or +1, clamped to [0, N-1].
        delta = self._rng.choice([-1, 0, 1])
        self._state = int(np.clip(self._state + delta, 0, self.N - 1))

        reward = 1.0 if self._state == self.N // 2 else 0.0

        self._step += 1
        done = self._step >= self.episode_length

        obs_a, obs_b = self._observe()
        return obs_a, obs_b, reward, done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observe(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample independent Gaussian observations for both agents."""
        signal = self._state / self.N
        obs_a = np.array(
            [signal + self._rng.normal(0.0, self.sigma_a)],
            dtype=np.float32,
        )
        obs_b = np.array(
            [signal + self._rng.normal(0.0, self.sigma_b)],
            dtype=np.float32,
        )
        return obs_a, obs_b
