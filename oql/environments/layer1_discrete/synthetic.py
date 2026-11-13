"""Synthetic POMDP generator (T044).

Produces a reproducible discrete POMDP with `n_states` hidden states,
`n_observations` observation symbols, and `n_actions` actions. Transition and
observation matrices are sampled from Dirichlet priors keyed by `seed`.

Used by US2 for the 1000-state stress test that validates the O(n²) sample
complexity bound (FR-4).

Construction modes
------------------
``structured=False`` (legacy):
    Fully random transitions and observations.  With many states and few
    observations the Myhill-Nerode quotient is trivial (n_classes ≈ 1) because
    every single-observation history leads to an indistinguishable mixture over
    hidden states — the G-test has no power to separate them.

``structured=True`` (default for n_states > 20):
    Block-diagonal construction.  States are grouped into ``n_obs`` blocks; the
    transition matrix favours within-block moves (``within_block_prob``), and
    the observation matrix makes each block emit its characteristic symbol with
    high probability (``obs_fidelity``).  This creates a proper quotient of size
    ≥ n_obs that the estimator reliably recovers.  Documented as the
    "evaluation-friendly" instance in the paper (Section 4.1).
"""

from __future__ import annotations

import numpy as np

from environments.base import POMDPEnvironment, StepResult


class SyntheticPOMDP(POMDPEnvironment):
    """Random tabular POMDP with reproducible dynamics.

    Parameters
    ----------
    structured:
        When *True* (default) the POMDP uses a block-diagonal transition matrix
        and block-specific observation distributions so the Myhill-Nerode
        quotient has ``n_obs`` well-separated classes.  Set to *False* to use
        the original fully-random construction (which collapses to 1 class on
        high-dimensional instances).
    within_block_prob:
        For structured mode: probability mass kept within the state block on
        each transition step.  Higher values make blocks more isolated.
    obs_fidelity:
        For structured mode: concentration boost for the block-dominant
        observation.  Higher values make observations more diagnostic.
    """

    _k1_full_support = True

    def __init__(
        self,
        n_states: int = 1000,
        n_obs: int = 100,
        n_actions: int = 5,
        sparsity: float = 0.05,
        observation_sharpness: float = 5.0,
        max_steps: int = 50,
        seed: int = 0,
        structured: bool = True,
        within_block_prob: float = 0.80,
        obs_fidelity: float = 3.0,
    ):
        if n_states < 2 or n_obs < 2 or n_actions < 1:
            raise ValueError("invalid POMDP sizes")
        self._n_states = n_states
        self._n_obs = n_obs
        self._n_actions = n_actions
        self.max_steps = max_steps
        self.seed = seed

        # Use a separate RNG for env construction (deterministic given seed).
        construction = np.random.default_rng(seed)

        if structured:
            self._transition, self._observation = self._build_structured(
                construction, n_states, n_obs, n_actions,
                within_block_prob, obs_fidelity,
            )
        else:
            self._transition, self._observation = self._build_random(
                construction, n_states, n_obs, n_actions, sparsity,
                observation_sharpness,
            )

        self._rng = np.random.default_rng(seed + 1)
        self._state: int = 0
        self._t: int = 0

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_random(
        construction: np.random.Generator,
        n_states: int,
        n_obs: int,
        n_actions: int,
        sparsity: float,
        observation_sharpness: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fully random (legacy) construction — collapses to 1 class on large instances."""
        transition = np.zeros((n_actions, n_states, n_states), dtype=np.float32)
        k = max(2, int(np.ceil(sparsity * n_states)))
        for a in range(n_actions):
            for s in range(n_states):
                support = construction.choice(n_states, size=k, replace=False)
                probs = construction.dirichlet(np.ones(k))
                transition[a, s, support] = probs
        observation = construction.dirichlet(
            np.ones(n_obs) / observation_sharpness, size=n_states,
        ).astype(np.float32)
        return transition, observation

    @staticmethod
    def _build_structured(
        construction: np.random.Generator,
        n_states: int,
        n_obs: int,
        n_actions: int,
        within_block_prob: float,
        obs_fidelity: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Block-diagonal construction — produces a learnable n_obs-class quotient.

        States are divided into ``n_obs`` equal blocks.  Within each block,
        transitions concentrate probability mass on other block members
        (``within_block_prob``), and the observation for state *s* in block *b*
        is biased towards symbol *b* (concentration boost ``obs_fidelity``).

        Result: after observing symbol *o*, you are almost certainly in block *o*,
        and subsequent observations are heavily correlated with *o*.  This gives
        continuation distributions that differ strongly across observation
        histories — the Myhill-Nerode quotient has exactly ``n_obs`` classes.
        """
        states_per_block = n_states // n_obs
        # Pad remainder into the last block
        block_sizes = [states_per_block] * n_obs
        for extra in range(n_states - states_per_block * n_obs):
            block_sizes[extra] += 1
        block_starts = [0]
        for sz in block_sizes[:-1]:
            block_starts.append(block_starts[-1] + sz)

        transition = np.zeros((n_actions, n_states, n_states), dtype=np.float32)
        for a in range(n_actions):
            for b, (bs, bsz) in enumerate(zip(block_starts, block_sizes)):
                within_states = np.arange(bs, bs + bsz)
                outside_states = np.array(
                    [s for s in range(n_states) if not (bs <= s < bs + bsz)],
                    dtype=np.intp,
                )
                for s in within_states:
                    within_dir = construction.dirichlet(np.ones(bsz))
                    transition[a, s, within_states] = within_block_prob * within_dir
                    if len(outside_states) > 0:
                        cross_dir = construction.dirichlet(np.ones(len(outside_states)))
                        transition[a, s, outside_states] = (1 - within_block_prob) * cross_dir

        # Observation matrix: block-specific dominant symbol
        observation = np.zeros((n_states, n_obs), dtype=np.float32)
        for b, (bs, bsz) in enumerate(zip(block_starts, block_sizes)):
            for s in range(bs, bs + bsz):
                alpha = np.ones(n_obs) * 0.5          # uniform background
                alpha[b] += obs_fidelity               # boost block symbol
                observation[s] = construction.dirichlet(alpha)

        return transition, observation

    @property
    def n_observations(self) -> int:
        return self._n_obs

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def environment_id(self) -> str:
        return f"SyntheticPOMDP(S={self._n_states},O={self._n_obs},A={self._n_actions},seed={self.seed})"

    @property
    def metadata(self) -> dict[str, object]:
        return {"n_states": self._n_states, "n_obs": self._n_obs, "n_actions": self._n_actions}

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = int(self._rng.integers(0, self._n_states))
        self._t = 0
        return self._draw_observation()

    def _draw_observation(self) -> int:
        return int(self._rng.choice(self._n_obs, p=self._observation[self._state]))

    def step(self, action: int) -> StepResult:
        if not 0 <= action < self._n_actions:
            raise ValueError(f"action {action} out of range")
        self._t += 1
        probs = self._transition[action, self._state]
        # Guard against numerical issues from float32 dirichlet rows.
        probs = probs / probs.sum()
        self._state = int(self._rng.choice(self._n_states, p=probs))
        obs = self._draw_observation()
        truncated = self._t >= self.max_steps
        return StepResult(observation=obs, reward=-0.01, terminated=False, truncated=truncated)
