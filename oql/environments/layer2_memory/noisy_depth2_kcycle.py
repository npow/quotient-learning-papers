"""Noisy depth-2 benchmark: second-order Markov observation process (E1).

The observation process over the alphabet {0, 1, ..., m} is an explicit
second-order Markov chain: the law of the next symbol depends on the last
TWO symbols, but only through the designed table

    row(prev, cur) = L_odd                      if cur != 0
                   = (1-eta) * delta_succ(prev) + eta * U   if cur == 0, prev != 0
                   = U (uniform)                if cur == 0, prev == 0

where L_odd = [1-nu, nu/m, ..., nu/m] (return to the aliasing symbol 0 with
probability 1-nu) and succ(x) cycles 1 -> 2 -> ... -> m -> 1.

Ground-truth suffix structure (exact, for any eta in [0, 1)):
- every context ending in a non-zero symbol x has law L_odd - one class,
  EXACTLY, at every noise level (the table rows are identical, so no
  spurious near-classes appear);
- the depth-2 contexts (x, 0) have distinct laws separated pairwise by
  TV = (1-eta)(1-1/(m+1)) if you compare against uniform and 1-eta against
  each other - recovering them REQUIRES depth-2 suffixes because the
  depth-1 context (0,) only sees the stationary mixture;
- eps_min(eta) is exactly computable and strictly decreasing in eta.

The POMDP wrapper has hidden state (prev, cur) with the identity emission
of cur; a single action advances the chain. Recommended estimator
configuration: window_length=1, history_max_length=2.
"""

from __future__ import annotations

import itertools

import numpy as np

from environments.base import POMDPEnvironment, StepResult

ACT_FORWARD = 0

_LAW_TOL = 1e-9
_PROB_TOL = 1e-12


class NoisyDepth2KCycle(POMDPEnvironment):
    """Second-order Markov symbol process whose quotient needs depth-2 contexts.

    Parameters
    ----------
    m:
        Number of informative symbols; alphabet is {0, 1, ..., m}. Must be >= 2.
    eta:
        Noise level in [0, 1): the (x, 0) rows are
        (1-eta) * point-mass(succ(x)) + eta * uniform. eps_min is strictly
        decreasing in eta.
    nu:
        Escape probability of the L_odd row (mass off the aliasing symbol 0).
    seed:
        Optional RNG seed.
    """

    def __init__(self, m: int, eta: float = 0.1, nu: float = 0.2, seed: int | None = None):
        if m < 2:
            raise ValueError(f"m must be >= 2, got {m}")
        if not 0.0 <= eta < 1.0:
            raise ValueError(f"eta must be in [0, 1), got {eta}")
        if not 0.0 < nu < 1.0:
            raise ValueError(f"nu must be in (0, 1), got {nu}")
        self.m = m
        self.eta = eta
        self.nu = nu
        self._rng = np.random.default_rng(seed)
        self._prev: int = 1
        self._cur: int = 0
        self._k1_full_support = True  # every symbol has positive stationary mass

    # ------------------------------------------------------------------
    # The designed table
    # ------------------------------------------------------------------

    def transition_row(self, prev: int, cur: int) -> np.ndarray:
        """P(next symbol | last two symbols = (prev, cur)) — the exact model."""
        n = self.n_observations
        uniform = np.full(n, 1.0 / n)
        if cur != 0:
            row = np.full(n, self.nu / self.m)
            row[0] = 1.0 - self.nu
            return row
        if prev == 0:
            return uniform
        succ = prev % self.m + 1
        row = self.eta * uniform
        row[succ] += 1.0 - self.eta
        return row

    # ------------------------------------------------------------------
    # POMDPEnvironment interface
    # ------------------------------------------------------------------

    @property
    def n_observations(self) -> int:
        return self.m + 1

    @property
    def n_actions(self) -> int:
        return 1

    @property
    def environment_id(self) -> str:
        return f"NoisyDepth2KCycle(m={self.m},eta={self.eta},nu={self.nu})"

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "ground_truth_n_classes": len(set(self.target_partition(2).values())),
            "eta": self.eta,
            "nu": self.nu,
            "m": self.m,
            "epsilon_min": self.epsilon_min(max_length=2),
            "recommended_window_length": 1,
            "recommended_history_max_length": 2,
        }

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Start from the stationary pair distribution so trajectories are
        # stationary from step one.
        pairs, probs = self._stationary_items()
        idx = self._rng.choice(len(pairs), p=probs)
        self._prev, self._cur = pairs[idx]
        return self._cur

    def step(self, action: int) -> StepResult:  # noqa: ARG002
        row = self.transition_row(self._prev, self._cur)
        nxt = int(self._rng.choice(self.n_observations, p=row))
        self._prev, self._cur = self._cur, nxt
        return StepResult(observation=nxt, reward=0.0, terminated=False, truncated=False)

    # ------------------------------------------------------------------
    # Exact model quantities (no simulation)
    # ------------------------------------------------------------------

    def _stationary_items(self):
        """Stationary distribution over (prev, cur) pairs, exactly."""
        n = self.n_observations
        pairs = [(p, c) for p in range(n) for c in range(n)]
        index = {pc: i for i, pc in enumerate(pairs)}
        P = np.zeros((len(pairs), len(pairs)))
        for (p, c), i in index.items():
            row = self.transition_row(p, c)
            for nxt in range(n):
                P[i, index[(c, nxt)]] = row[nxt]
        vals, vecs = np.linalg.eig(P.T)
        k = int(np.argmin(np.abs(vals - 1.0)))
        pi = np.real(vecs[:, k])
        pi = np.abs(pi) / np.abs(pi).sum()
        return pairs, pi

    def word_probability(self, word: tuple[int, ...]) -> float:
        """Exact stationary probability of observing `word` as a suffix."""
        pairs, pi = self._stationary_items()
        if len(word) == 1:
            return float(sum(p for (a, b), p in zip(pairs, pi) if b == word[0]))
        total = 0.0
        for (a, b), p in zip(pairs, pi):
            w = p
            prev, cur = a, b
            ok = cur == word[0]
            if not ok:
                continue
            for nxt_sym in word[1:]:
                row = self.transition_row(prev, cur)
                w *= row[nxt_sym]
                prev, cur = cur, nxt_sym
            total += w
        return float(total)

    def exact_continuation_law(self, word: tuple[int, ...]) -> list[float]:
        """P(next symbol | observed suffix = word), exactly."""
        n = self.n_observations
        pairs, pi = self._stationary_items()
        # Posterior over the pair state (prev, cur) after observing `word`.
        post = np.zeros(n * n)
        idx = {pc: i for i, pc in enumerate(pairs)}
        for (a, b), p in zip(pairs, pi):
            if b != word[0]:
                continue
            w = p
            prev, cur = a, b
            for nxt_sym in word[1:]:
                w *= self.transition_row(prev, cur)[nxt_sym]
                prev, cur = cur, nxt_sym
            post[idx[(prev, cur)]] += w
        total = post.sum()
        if total <= _PROB_TOL:
            raise ValueError(f"word {word} has zero stationary probability")
        post /= total
        law = np.zeros(n)
        for (p, c), i in idx.items():
            if post[i] > 0:
                law += post[i] * self.transition_row(p, c)
        return [float(x) for x in law]

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
        """Minimum TV separation between distinct ground-truth classes."""
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
