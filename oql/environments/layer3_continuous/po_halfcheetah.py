"""PO-MuJoCo HalfCheetah wrapper (T047).

Wraps `gymnasium`'s HalfCheetah with the qvel components (joint velocities)
masked from the observation. The remaining qpos observation is then
discretized via k-means binning so the quotient learner sees a discrete
observation alphabet.

Gracefully degrades when MuJoCo / gymnasium are not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from environments.base import POMDPEnvironment, StepResult

# Cap on discrete observation alphabet size — prevents 3^8 = 6561 codes that
# are nearly all unvisited. Hash-modding into this range keeps the alphabet
# learnable by the QuotientEstimator.
_OBS_ALPHABET_CAP = 64

class POMuJoCoUnavailable(RuntimeError):
    """Raised when the wrapper is instantiated without gymnasium / mujoco."""


try:
    import gymnasium as _gym
    _HAVE_GYM = True
except Exception:
    _gym = None  # type: ignore[assignment]
    _HAVE_GYM = False


def _require_gym() -> None:
    if not _HAVE_GYM:
        raise POMuJoCoUnavailable(
            "gymnasium with mujoco support is required. "
            "pip install -e .[continuous]"
        )


class POHalfCheetah(POMDPEnvironment):
    """HalfCheetah with velocity dimensions masked.

    Observation discretization: each qpos dimension is bucketed into
    `bins_per_dim` quantile bins (fit from a short warm-up rollout). The
    resulting integer code is encoded as a single observation token, then
    hash-modded to ``_OBS_ALPHABET_CAP`` so the alphabet stays learnable.
    """

    #: Number of discrete actions we expose to the quotient learner.
    #: We define 5 coarse directions for the first joint.
    _N_DISCRETE_ACTIONS = 5

    def __init__(
        self,
        bins_per_dim: int = 4,
        max_steps: int = 200,
        warmup_episodes: int = 5,
        seed: int = 0,
    ):
        _require_gym()
        import numpy as np
        self._env = _gym.make("HalfCheetah-v4")
        self.bins_per_dim = bins_per_dim
        self.max_steps = max_steps
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._bin_edges = None
        # Determine qpos dimensionality from the environment observation space
        # *before* fitting bins (uses action_space, not a reset call).
        shape = self._env.observation_space.shape
        assert shape is not None, "gymnasium observation space must have a shape"
        # HalfCheetah obs = qpos[1:] (8 dims) + qvel (9 dims) = 17.
        # We use obs[:half] as the "position" part to mask velocities.
        self._qpos_dim: int = shape[0] // 2
        # Now fit quantile bins from warm-up rollouts.
        self._fit_bins(warmup_episodes=warmup_episodes)

    def _fit_bins(self, warmup_episodes: int) -> None:
        import numpy as np
        observations = []
        for _ in range(warmup_episodes):
            obs, _ = self._env.reset()
            for _ in range(self.max_steps):
                a = self._env.action_space.sample()
                obs, _, term, trunc, _ = self._env.step(a)
                observations.append(obs[: self._qpos_dim])
                if term or trunc:
                    break
        if not observations:
            self._bin_edges = None
            return
        x = np.array(observations)
        qs = np.linspace(0.0, 1.0, self.bins_per_dim + 1)
        self._bin_edges = np.quantile(x, qs, axis=0).T  # (qpos_dim, bins+1)

    @property
    def n_observations(self) -> int:
        """Capped discrete alphabet size."""
        return int(min(self.bins_per_dim ** self._qpos_dim, _OBS_ALPHABET_CAP))

    @property
    def n_actions(self) -> int:
        return self._N_DISCRETE_ACTIONS

    @property
    def environment_id(self) -> str:
        return f"POHalfCheetah(bins={self.bins_per_dim})"

    def _encode(self, obs: "np.ndarray[Any, np.dtype[Any]]") -> int:
        if self._bin_edges is None:
            return 0
        qpos = obs[: self._qpos_dim]
        idx = 0
        for d, x in enumerate(qpos):
            edges = self._bin_edges[d]
            b = int(np.clip(np.searchsorted(edges, x) - 1, 0, self.bins_per_dim - 1))
            idx = idx * self.bins_per_dim + b
        return idx % self.n_observations

    def reset(self, seed: int | None = None) -> int:
        obs, _ = self._env.reset(seed=seed if seed is not None else None)
        self._t = 0
        return self._encode(obs)

    def step(self, action: int) -> StepResult:
        # Map discrete action index to a continuous action vector.
        a = self._env.action_space.sample()
        a[:] = 0.0
        # Map integer 0..4 to a coarse direction on the first joint.
        a[0] = -1.0 + 2.0 * ((action % self._N_DISCRETE_ACTIONS) / (self._N_DISCRETE_ACTIONS - 1))
        obs, reward, term, trunc, _ = self._env.step(a)
        self._t += 1
        truncated = trunc or self._t >= self.max_steps
        return StepResult(observation=self._encode(obs), reward=float(reward),
                          terminated=bool(term), truncated=bool(truncated))

    def close(self) -> None:
        self._env.close()
