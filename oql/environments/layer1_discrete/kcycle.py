"""K-Cycle POMDP family (constructive ground-truth benchmark).

The hidden state is a position on a cycle of length K.  Each state s emits
observation ``o = s`` with probability ``obs_accuracy`` and a uniformly random
observation otherwise.  The single action advances the hidden state forward by
one on the cycle (deterministic).

Canonical Myhill-Nerode quotient is *exactly* K classes by construction:
state s's k-step continuation distribution differs from state s' iff s != s'.
The continuation from position s is a periodic orbit of period K, and each
offset within the cycle produces a distinct marginal distribution over the
noisy observations.

The K-chain-with-reset variant is identical in quotient structure but keeps
the transition explicit: state 0 → 1 → … → (K-1) → 0.

For large K (K ≥ 8) the environment randomises its starting state on each
episode reset so that every state position is sampled and every history prefix
has sufficient coverage.

metadata keys
-------------
``ground_truth_n_classes``  : K (exact)
``obs_accuracy``            : α
``k``                       : K
"""

from __future__ import annotations

import numpy as np

from environments.base import POMDPEnvironment, StepResult

# Single deterministic action.
ACT_FORWARD = 0


class KCyclePOMDP(POMDPEnvironment):
    """Cycle of K hidden states; observations are noisy state index.

    Canonical Myhill-Nerode quotient is K classes by construction:
    state s's k-step continuation distribution differs from state s'
    iff s != s'.

    Parameters
    ----------
    k:
        Number of hidden states (cycle length).  Must be >= 2.
    obs_accuracy:
        Probability that the emitted observation equals the true state index.
        Must be in (1/k, 1].  Default 0.85.
    seed:
        Optional RNG seed for the environment's internal randomness.
    randomise_start:
        If True (default), ``reset()`` samples the start state uniformly from
        {0, …, k-1} rather than always starting at 0.  This ensures all cycle
        positions are visited across episodes, which is required for the
        estimator to build histories for every class.
    """

    _k1_full_support = True

    def __init__(
        self,
        k: int,
        obs_accuracy: float = 0.85,
        seed: int | None = None,
        randomise_start: bool = True,
    ):
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        if not (1.0 / k) < obs_accuracy <= 1.0:
            raise ValueError(
                f"obs_accuracy must be in (1/k, 1.0], got {obs_accuracy} for k={k}"
            )
        self._k = k
        self._obs_accuracy = obs_accuracy
        self._randomise_start = randomise_start
        self._rng = np.random.default_rng(seed)
        self._state: int = 0

    # ------------------------------------------------------------------
    # POMDPEnvironment interface
    # ------------------------------------------------------------------

    @property
    def n_observations(self) -> int:
        """Observation alphabet: {0, 1, …, k-1}."""
        return self._k

    @property
    def n_actions(self) -> int:
        """Single deterministic forward action."""
        return 1

    @property
    def environment_id(self) -> str:
        return f"KCycle(k={self._k},acc={self._obs_accuracy})"

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "ground_truth_n_classes": self._k,
            "obs_accuracy": self._obs_accuracy,
            "k": self._k,
        }

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._randomise_start:
            self._state = int(self._rng.integers(0, self._k))
        else:
            self._state = 0
        return self._emit()

    def step(self, action: int) -> StepResult:  # noqa: ARG002
        # Deterministic forward transition.
        self._state = (self._state + 1) % self._k
        obs = self._emit()
        return StepResult(
            observation=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self) -> int:
        """Emit noisy observation: true state with prob α, uniform otherwise."""
        if self._rng.random() < self._obs_accuracy:
            return self._state
        # Uniform over {0, …, k-1} \ {state} to be strictly noisy, but we
        # keep it uniform over the full alphabet for simplicity (as in Tiger).
        return int(self._rng.integers(0, self._k))
