"""Misspecification-probe chain: an engineered sub-gap TV pair (REV-5).

A first-order Markov observation process over the alphabet {0, 1, 2}.  The
first symbol X_0 is uniform on {0, 1, 2}; thereafter the next symbol is
drawn from the row of the current symbol:

    row(0) = [0.45,       0.45,       0.10]
    row(1) = [0.45 + tau, 0.45 - tau, 0.10]
    row(2) = [0.05,       0.05,       0.90]

Because the process is first-order Markov, the continuation law of ANY
context is exactly the row of its final symbol, at every episode position.
At endpoint r=0 with horizon k=1 the three length-one contexts (0,), (1,),
(2,) therefore have closed-form pairwise total-variation distances:

    TV(P_(0,), P_(1,)) = tau      -- the engineered SUB-GAP pair
    TV(P_(0,), P_(2,)) = 0.80
    TV(P_(1,), P_(2,)) = 0.80     (exact for any tau in (0, 0.40])

The engineered sub-gap pair
---------------------------
With the default tau = 0.10, the pair ((0,), (1,)) violates the TV gap
model (Eq. 2 of the paper) for any eps_min > 0.10: its laws are neither
equal nor eps_min-separated.  In particular, under the harness default
eps_min = 0.5 the pair's true TV (0.10) lies strictly below the SAME
threshold eps_min / 2 = 0.25, so the certified test's empirical TV
concentrates near 0.10 and it returns a confident SAME verdict — the
silent, uncertified merge probed by experiments/main_table/run_misspec.py.
The true quotient at r=0, k=1 has THREE classes; the honest gap for this
environment is epsilon_min(1) = tau.

The POMDP wrapper has hidden state equal to the current symbol with the
identity emission; a single action advances the chain.  Recommended
estimator configuration: window_length=1, history_max_length=1,
independent_context_end=0.
"""

from __future__ import annotations

import itertools

import numpy as np

from environments.base import POMDPEnvironment, StepResult

ACT_FORWARD = 0

_LAW_TOL = 1e-9
_PROB_TOL = 1e-12

#: Separation of each context law from row(2); exact for tau in (0, 0.40].
SEPARATED_TV = 0.80


class MisspecChain(POMDPEnvironment):
    """Three-context chain with one sub-gap pair at exact TV = tau.

    Parameters
    ----------
    tau:
        Total-variation distance between the laws of contexts (0,) and (1,).
        Must lie in (0, 0.40] so all rows remain valid distributions and the
        separated pairs keep TV exactly 0.80.  The misspecification probe
        targets tau = 0.10, well below eps_min / 2 = 0.25 at eps_min = 0.5.
    seed:
        Optional RNG seed.
    """

    def __init__(self, tau: float = 0.10, seed: int | None = None):
        if not 0.0 < tau <= 0.40:
            raise ValueError(f"tau must be in (0, 0.40], got {tau}")
        self.tau = float(tau)
        self._rng = np.random.default_rng(seed)
        self._cur: int = 0
        self._k1_full_support = True  # every row is strictly positive

    # ------------------------------------------------------------------
    # The designed table
    # ------------------------------------------------------------------

    def transition_row(self, cur: int) -> np.ndarray:
        """P(next symbol | current symbol) — the exact model."""
        if cur == 0:
            return np.array([0.45, 0.45, 0.10])
        if cur == 1:
            return np.array([0.45 + self.tau, 0.45 - self.tau, 0.10])
        if cur == 2:
            return np.array([0.05, 0.05, 0.90])
        raise ValueError(f"symbol out of range: {cur}")

    def initial_distribution(self) -> np.ndarray:
        """Law of the first observation X_0 (uniform by design)."""
        return np.full(self.n_observations, 1.0 / self.n_observations)

    # ------------------------------------------------------------------
    # POMDPEnvironment interface
    # ------------------------------------------------------------------

    @property
    def n_observations(self) -> int:
        return 3

    @property
    def n_actions(self) -> int:
        return 1

    @property
    def environment_id(self) -> str:
        return f"MisspecChain(tau={self.tau})"

    @property
    def metadata(self) -> dict[str, object]:
        return {
            # True quotient at the r=0, k=1 estimand: all three length-one
            # context laws are pairwise distinct (the sub-gap pair differs
            # by tau > 0), so exact recovery must report 3 classes.
            "ground_truth_n_classes": len(set(self.target_partition(1).values())),
            "tau": self.tau,
            "subgap_pair": ((0,), (1,)),
            "subgap_tv": self.tau,
            "separated_tv": SEPARATED_TV,
            # The honest gap parameter: eps_min must not exceed this.
            "epsilon_min": self.epsilon_min(max_length=1),
            "recommended_window_length": 1,
            "recommended_history_max_length": 1,
            "recommended_context_end": 0,
        }

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._cur = int(self._rng.choice(self.n_observations,
                                         p=self.initial_distribution()))
        return self._cur

    def step(self, action: int) -> StepResult:  # noqa: ARG002
        row = self.transition_row(self._cur)
        nxt = int(self._rng.choice(self.n_observations, p=row))
        self._cur = nxt
        return StepResult(observation=nxt, reward=0.0, terminated=False, truncated=False)

    # ------------------------------------------------------------------
    # Exact model quantities (no simulation)
    # ------------------------------------------------------------------

    def word_probability(self, word: tuple[int, ...]) -> float:
        """Exact probability of observing `word` as the episode prefix."""
        if not word:
            raise ValueError("word must be non-empty")
        p = float(self.initial_distribution()[word[0]])
        for prev, nxt in zip(word, word[1:]):
            p *= float(self.transition_row(prev)[nxt])
        return p

    def exact_continuation_law(self, word: tuple[int, ...]) -> list[float]:
        """P(next symbol | observed suffix = word), exactly.

        First-order Markov: the law depends only on the final symbol of the
        word, so it equals ``transition_row(word[-1])`` at every episode
        position (no posterior computation is required).
        """
        if not word:
            raise ValueError("word must be non-empty")
        for o in word:
            if not 0 <= int(o) < self.n_observations:
                raise ValueError(f"symbol out of range: {o}")
        return [float(x) for x in self.transition_row(int(word[-1]))]

    def true_pairwise_tvs(self) -> dict[frozenset[tuple[int, ...]], float]:
        """Exact pairwise TVs between the three length-one context laws."""
        words = [(0,), (1,), (2,)]
        laws = {w: self.exact_continuation_law(w) for w in words}
        return {
            frozenset((a, b)): self._tv(laws[a], laws[b])
            for a, b in itertools.combinations(words, 2)
        }

    def target_partition(self, max_length: int):
        """Exact suffix partition (words of length 1..max_length, k=1 window)."""
        from quotient_learning.types import ObservationHistory

        words = [
            w
            for length in range(1, max_length + 1)
            for w in itertools.product(range(self.n_observations), repeat=length)
            if self.word_probability(w) > _PROB_TOL
        ]
        laws = {w: self.exact_continuation_law(w) for w in words}
        reps: list[tuple[int, ...]] = []
        partition: dict[ObservationHistory, int] = {}
        for w in sorted(words, key=lambda x: (len(x), x)):
            for cid, rep in enumerate(reps):
                if self._tv(laws[w], laws[rep]) < _LAW_TOL:
                    partition[ObservationHistory(w)] = cid
                    break
            else:
                reps.append(w)
                partition[ObservationHistory(w)] = len(reps) - 1
        return partition

    def epsilon_min(self, max_length: int) -> float:
        """Minimum TV separation between distinct ground-truth classes.

        Equals ``tau`` (the sub-gap pair is the closest distinct pair); any
        certified configuration for this environment must use
        eps_min <= this value.  The misspecification probe deliberately
        violates that requirement.
        """
        partition = self.target_partition(max_length)
        by_class: dict[int, tuple[int, ...]] = {}
        for h, cid in partition.items():
            by_class.setdefault(cid, h.observations)
        laws = {cid: self.exact_continuation_law(w) for cid, w in by_class.items()}
        return min(
            self._tv(laws[a], laws[b])
            for a, b in itertools.combinations(laws, 2)
        )

    @staticmethod
    def _tv(p, q) -> float:
        return 0.5 * sum(abs(a - b) for a, b in zip(p, q))
