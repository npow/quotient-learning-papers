"""Position-flip chain: the estimand-mismatch benchmark (review item B4).

A binary observation process whose one-step transition kernel depends on the
parity of the episode position:

    even t:  P(next = 1 | cur = 0) = 0.9,  P(next = 1 | cur = 1) = 0.1
    odd  t:  P(next = 1 | cur = 0) = 0.1,  P(next = 1 | cur = 1) = 0.9

The initial observation is uniform, and one checks by induction that every
observation is marginally uniform at every position.  Consequently:

- Episode-position estimand at r = 0, k = 1: the two length-one contexts
  (0,) and (1,) have continuation laws Bernoulli(0.9) and Bernoulli(0.1)
  respectively -- distinct classes at TV = 0.8.
- Position-pooled law (what an overlapping-window scan estimates): each
  context symbol occurs equally often at even and odd predecessor
  positions, so the pooled continuation law after either symbol is exactly
  Bernoulli(0.5).  The pooled laws of the two contexts are identical.

An estimator that scans all window positions therefore converges to a
*merge* of the two contexts -- not through miscalibration, but because
pooling positions changes the estimand.  The certified path at fixed r = 0
separates them with TV gap 0.8.
"""

from __future__ import annotations

import numpy as np

from environments.base import POMDPEnvironment, StepResult

P_HI = 0.9
P_LO = 0.1


class PositionFlipChain(POMDPEnvironment):
    """Binary chain with parity-dependent transitions (see module docstring)."""

    _k1_full_support = True

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self._cur: int = 0
        self._t: int = 0

    @property
    def n_observations(self) -> int:
        return 2

    @property
    def n_actions(self) -> int:
        return 1

    @property
    def environment_id(self) -> str:
        return "PositionFlipChain(p_hi=0.9,p_lo=0.1)"

    @property
    def metadata(self) -> dict[str, object]:
        return {
            # Episode-position estimand at r=0, k=1 (the certified target):
            "ground_truth_n_classes": 2,
            "epsilon_min": self.analytic_tv_gap_r0(),
            "pooled_tv_gap": 0.0,
            "recommended_window_length": 1,
            "recommended_history_max_length": 1,
        }

    @staticmethod
    def analytic_tv_gap_r0() -> float:
        """TV between the r=0 continuation laws of contexts (0,) and (1,)."""
        return P_HI - P_LO

    @staticmethod
    def transition_prob_next_one(cur: int, t: int) -> float:
        """P(next observation = 1 | current = cur, position = t)."""
        if t % 2 == 0:
            return P_HI if cur == 0 else P_LO
        return P_LO if cur == 0 else P_HI

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._cur = int(self._rng.integers(0, 2))
        self._t = 0
        return self._cur

    def step(self, action: int) -> StepResult:
        p1 = self.transition_prob_next_one(self._cur, self._t)
        nxt = int(self._rng.random() < p1)
        self._cur = nxt
        self._t += 1
        return StepResult(observation=nxt, reward=0.0, terminated=False, truncated=False)
