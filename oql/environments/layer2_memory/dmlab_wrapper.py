"""DM Lab memory-task wrapper (T046).

Wraps DeepMind Lab memory-and-planning levels via the `deepmind_lab` Python
binding. The binding is non-trivial to install on workbench-class hosts (it
requires the bazel-built `dm_lab.so`), so this wrapper *gracefully degrades*:
if the binding is unavailable at import time, `DMLabWrapper(level)` raises
`DMLabUnavailable` at instantiation rather than at import time.

This pattern keeps the rest of the package importable even when DM Lab is not
present — required so US1 / Layer-1 work continues to function in
DM-Lab-less environments.
"""

from __future__ import annotations

from environments.base import POMDPEnvironment, StepResult


class DMLabUnavailable(RuntimeError):
    """Raised when DMLabWrapper is instantiated without `deepmind_lab` installed."""


try:
    import deepmind_lab as _dm_lab
    _HAVE_DMLAB = True
except Exception:
    _dm_lab = None
    _HAVE_DMLAB = False


class DMLabWrapper(POMDPEnvironment):
    """Wrap a DM Lab level as a POMDPEnvironment with discretized observations.

    Quantizes the (84, 84, 3) RGB observation to a small integer code via
    color-histogram bucketing. This is intentionally coarse — fine-grained
    visual encoding is out of scope for the quotient-learning experiments.
    """

    def __init__(
        self,
        level: str = "nav_maze_static_01",
        n_obs_buckets: int = 32,
        max_steps: int = 200,
    ):
        if not _HAVE_DMLAB:
            raise DMLabUnavailable(
                "deepmind_lab Python binding not available; install via "
                "https://github.com/google-deepmind/lab or run without this layer."
            )
        self.level = level
        self.n_obs_buckets = n_obs_buckets
        self.max_steps = max_steps
        self._env = _dm_lab.Lab(level, ["RGB_INTERLEAVED"])
        self._t = 0

    @property
    def n_observations(self) -> int:
        return self.n_obs_buckets

    @property
    def n_actions(self) -> int:
        return 6  # standard DM Lab discrete action set: forward/back/turn/strafe.

    @property
    def environment_id(self) -> str:
        return f"DMLab({self.level})"

    def _encode(self, obs_dict: dict) -> int:
        rgb = obs_dict["RGB_INTERLEAVED"]
        mean = float(rgb.mean())
        return int(mean / 256.0 * self.n_obs_buckets) % self.n_obs_buckets

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._env.reset(seed=seed)
        else:
            self._env.reset()
        self._t = 0
        return self._encode(self._env.observations())

    def step(self, action: int) -> StepResult:
        import numpy as np
        action_vec = np.zeros(7, dtype=np.intc)
        # Map our small discrete actions onto DM Lab's 7-dim continuous-style spec.
        if action == 0:
            action_vec[3] = 1  # forward
        elif action == 1:
            action_vec[3] = -1  # back
        elif action == 2:
            action_vec[0] = -20  # turn left
        elif action == 3:
            action_vec[0] = 20  # turn right
        elif action == 4:
            action_vec[4] = 1  # strafe left
        elif action == 5:
            action_vec[4] = -1  # strafe right
        reward = float(self._env.step(action_vec, num_steps=4))
        self._t += 1
        truncated = self._t >= self.max_steps
        terminated = not self._env.is_running()
        if terminated or truncated:
            obs = 0
        else:
            obs = self._encode(self._env.observations())
        return StepResult(observation=obs, reward=reward,
                          terminated=terminated, truncated=truncated)
