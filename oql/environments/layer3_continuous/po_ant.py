"""PO-MuJoCo Ant wrapper (T048).

Same pattern as PO-HalfCheetah: wrap gymnasium's Ant-v4, mask joint-angle
velocities, discretize the remaining qpos via quantile bins.
"""

from __future__ import annotations

from environments.layer3_continuous.po_halfcheetah import (
    _HAVE_GYM,
    _OBS_ALPHABET_CAP,
    POHalfCheetah,
    POMuJoCoUnavailable,
    _gym,
)


class POAnt(POHalfCheetah):
    """Ant-v4 with joint velocities masked.

    Inherits the bin-fitting / discretization logic from POHalfCheetah; only
    the underlying gymnasium env id differs.
    """

    def __init__(
        self,
        bins_per_dim: int = 3,
        max_steps: int = 200,
        warmup_episodes: int = 3,
        seed: int = 0,
    ):
        if not _HAVE_GYM:
            raise POMuJoCoUnavailable(
                "gymnasium with mujoco support is required. "
                "pip install -e .[continuous]"
            )
        import numpy as np
        self._env = _gym.make("Ant-v4")
        self.bins_per_dim = bins_per_dim
        self.max_steps = max_steps
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._bin_edges = None
        # Ant obs = qpos (13 dims) + qvel (14 dims) = 27. obs//2 = 13 qpos dims.
        obs_dim = self._env.observation_space.shape[0]
        self._qpos_dim: int = obs_dim // 2
        self._fit_bins(warmup_episodes=warmup_episodes)

    @property
    def environment_id(self) -> str:
        return f"POAnt(bins={self.bins_per_dim})"
