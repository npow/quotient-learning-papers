"""Randomly parameterized discrete HMM with post-hoc (not designed) quotient.

Every published exact-recovery experiment runs on an environment DESIGNED so
the episode-position quotient and its TV gap are known in closed form before
any data is drawn.  This environment deliberately leaves that regime: the
initial distribution, transition matrix, and emission matrix are Dirichlet(1)
draws from a dedicated *instance seed*, so whether any two length-one
contexts at r=0, k=1 share (or nearly share) a continuation law is a
numerical accident of the draw, not a construction.  Ground truth exists
only POST HOC, by brute-force computation from the drawn matrices.

Instance-generation protocol (fixed before looking at any estimator output;
see experiments/main_table/run_posthoc.py for the experiment that uses it):

1. ``instance_seed`` drives a dedicated ``numpy`` Generator used ONLY for the
   Dirichlet draws of (pi0, A, B).  Trajectory randomness uses a separate
   ``seed`` argument, exactly like every other environment in the repo.
2. Scan rule: within a fixed generation family (n_states, n_obs,
   Dirichlet concentration), starting at instance seed 1000 and incrementing
   by 1, take the FIRST instance whose exact minimum pairwise TV between the
   continuation laws of the eligible length-one contexts at r=0, k=1 is
   >= 0.15 (``min_pairwise_tv``).  The rule guards against an inconveniently
   tiny accidental gap; it is evaluated on brute-force ground truth ONLY and
   was fixed before any estimator was run on any candidate instance.  A
   consequence of the rule is that the accepted instance has no exact ties,
   i.e. an all-singleton quotient.
3. Scan history (all decisions made on ground-truth TV alone): the first
   family tried, (n_states=4, n_obs=5, concentration=1.0), produced NO
   instance with min TV >= 0.15 across seeds 1000-1019 (min pairwise TV over
   5 mixture laws of 4 transition rows is generically tiny: observed range
   0.004-0.019).  The family was therefore re-parameterized to sharper,
   smaller draws, (n_states=3, n_obs=3, concentration=0.3), and the scan
   restarted at 1000.  Seed 1000 itself passes: min pairwise TV = 0.3872,
   max = 0.6417, minimum context mass 0.2426.  That instance is the class
   default and the one used by run_posthoc.py.

The class also exposes brute-force helpers computing, exactly from the
matrices: per-context masses, per-context continuation laws over O^k, the
pairwise TV matrix, and the induced true partition.  Episodes never
terminate on their own (they are truncated by ``max_steps``), so the
estimand's "conditioned on reaching r+k" event is certain whenever
``max_steps >= r + k`` and the helpers need no termination correction.

The ``pi0`` / ``A`` / ``B`` keyword arguments allow explicit matrices for
hand-computable unit tests; they bypass the Dirichlet draw.
"""

from __future__ import annotations

import itertools

import numpy as np

from environments.base import POMDPEnvironment, StepResult

ACT_FORWARD = 0

_LAW_TOL = 1e-9
_PROB_TOL = 1e-12


class PosthocRandomHMM(POMDPEnvironment):
    """Discrete HMM whose parameters are random Dirichlet draws.

    Parameters
    ----------
    n_states:
        Number of latent states (>= 2).
    n_obs:
        Observation alphabet size (>= 2).
    instance_seed:
        Seed of the dedicated Generator that draws (pi0, A, B).  Separate
        from trajectory seeds by construction.
    concentration:
        Dirichlet concentration of every drawn row.  Part of the generation
        family; see the scan history in the module docstring.
    seed:
        Trajectory RNG seed (same convention as the other environments).
    pi0, A, B:
        Optional explicit matrices (unit-test hook).  When given they must
        be row-stochastic with shapes (n_states,), (n_states, n_states),
        (n_states, n_obs); the Dirichlet draw is skipped.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_obs: int = 3,
        instance_seed: int = 1000,
        concentration: float = 0.3,
        seed: int | None = None,
        *,
        pi0: np.ndarray | None = None,
        A: np.ndarray | None = None,
        B: np.ndarray | None = None,
    ):
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}")
        if n_obs < 2:
            raise ValueError(f"n_obs must be >= 2, got {n_obs}")
        if concentration <= 0.0:
            raise ValueError(f"concentration must be > 0, got {concentration}")
        self.n_states = n_states
        self.n_obs = n_obs
        self.instance_seed = instance_seed
        self.concentration = concentration
        if pi0 is not None or A is not None or B is not None:
            if pi0 is None or A is None or B is None:
                raise ValueError("pi0, A, and B must all be given together")
            self.pi0 = np.asarray(pi0, dtype=float)
            self.A = np.asarray(A, dtype=float)
            self.B = np.asarray(B, dtype=float)
        else:
            inst_rng = np.random.default_rng(instance_seed)
            self.pi0 = inst_rng.dirichlet(np.full(n_states, concentration))
            self.A = inst_rng.dirichlet(np.full(n_states, concentration), size=n_states)
            self.B = inst_rng.dirichlet(np.full(n_obs, concentration), size=n_states)
        self._validate_matrices()
        # Full support of the k=1 observation alphabet holds iff some state
        # with positive reachable mass emits each symbol; Dirichlet draws
        # are strictly positive a.s., so check positivity directly.
        self._k1_full_support = bool((self.B > 0).all() and (self.pi0 > 0).any())
        # Precomputed cumulative rows for fast exact sampling.
        self._pi0_cum = np.cumsum(self.pi0)
        self._A_cum = np.cumsum(self.A, axis=1)
        self._B_cum = np.cumsum(self.B, axis=1)
        self._rng = np.random.default_rng(seed)
        self._state: int = 0

    def _validate_matrices(self) -> None:
        if self.pi0.shape != (self.n_states,):
            raise ValueError(f"pi0 must have shape ({self.n_states},)")
        if self.A.shape != (self.n_states, self.n_states):
            raise ValueError(f"A must have shape ({self.n_states}, {self.n_states})")
        if self.B.shape != (self.n_states, self.n_obs):
            raise ValueError(f"B must have shape ({self.n_states}, {self.n_obs})")
        for name, arr, axis_sum in (
            ("pi0", self.pi0, self.pi0.sum()),
            ("A", self.A, self.A.sum(axis=1)),
            ("B", self.B, self.B.sum(axis=1)),
        ):
            if (arr < -_PROB_TOL).any():
                raise ValueError(f"{name} has negative entries")
            if not np.allclose(axis_sum, 1.0, atol=1e-9):
                raise ValueError(f"{name} rows must sum to 1")

    # ------------------------------------------------------------------
    # POMDPEnvironment interface
    # ------------------------------------------------------------------

    @property
    def n_observations(self) -> int:
        return self.n_obs

    @property
    def n_actions(self) -> int:
        return 1

    @property
    def environment_id(self) -> str:
        return (
            f"PosthocRandomHMM(n_states={self.n_states},n_obs={self.n_obs},"
            f"instance_seed={self.instance_seed},conc={self.concentration})"
        )

    @property
    def metadata(self) -> dict[str, object]:
        # No designed ground truth: the quotient is a post-hoc numerical
        # fact of the drawn matrices (see true_partition / exact_tv_matrix).
        return {
            "instance_seed": self.instance_seed,
            "n_states": self.n_states,
            "n_obs": self.n_obs,
            "concentration": self.concentration,
        }

    def _draw(self, cum: np.ndarray) -> int:
        return int(np.searchsorted(cum, self._rng.random(), side="right"))

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._draw(self._pi0_cum)
        return self._draw(self._B_cum[self._state])

    def step(self, action: int) -> StepResult:  # noqa: ARG002
        self._state = self._draw(self._A_cum[self._state])
        obs = self._draw(self._B_cum[self._state])
        return StepResult(observation=obs, reward=0.0, terminated=False, truncated=False)

    # ------------------------------------------------------------------
    # Brute-force post-hoc ground truth (exact, from the matrices)
    # ------------------------------------------------------------------

    def _filter(self, context: tuple[int, ...], r: int) -> tuple[float, np.ndarray]:
        """Forward filter to the context endpoint.

        Returns ``(mass, posterior)`` where ``mass`` is the exact probability
        that the observations at indices ``r - len(context) + 1 .. r`` equal
        ``context`` (earlier observations marginalized) and ``posterior`` is
        the latent-state law at index ``r`` given that event.  Because the
        chain never terminates, this mass equals the estimand's
        mu_h = P(H = h, Z reaches r + k) for any k <= max_steps - r.
        """
        ell = len(context)
        if ell < 1:
            raise ValueError("context must be non-empty")
        if ell > r + 1:
            raise ValueError(f"context of length {ell} cannot end at index {r}")
        constrained_from = r - ell + 1
        alpha = self.pi0.copy()
        if constrained_from == 0:
            alpha = alpha * self.B[:, context[0]]
        for t in range(1, r + 1):
            alpha = alpha @ self.A
            if t >= constrained_from:
                alpha = alpha * self.B[:, context[t - constrained_from]]
        mass = float(alpha.sum())
        if mass <= _PROB_TOL:
            return 0.0, np.zeros(self.n_states)
        return mass, alpha / mass

    def exact_context_mass(self, context: tuple[int, ...], r: int) -> float:
        """Exact visitation mass mu_h of ``context`` at endpoint ``r``."""
        mass, _ = self._filter(context, r)
        return mass

    def exact_continuation_law(
        self, context: tuple[int, ...], r: int, k: int
    ) -> dict[tuple[int, ...], float]:
        """Exact law of (O_{r+1}, ..., O_{r+k}) given ``context`` at endpoint ``r``.

        Brute force over the latent chain: forward filter to the endpoint,
        then enumerate all |O|^k continuation words, propagating the joint
        (state, word) weight through A and B.  Termination conditioning is
        trivial (the chain never terminates before r + k).
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        mass, post = self._filter(context, r)
        if mass <= _PROB_TOL:
            raise ValueError(
                f"context {context} at r={r} has zero probability"
            )
        law: dict[tuple[int, ...], float] = {}
        # weight[s] = P(word so far, state = s | context); extend step by step.
        stack: list[tuple[tuple[int, ...], np.ndarray]] = [((), post)]
        while stack:
            word, weight = stack.pop()
            if len(word) == k:
                law[word] = float(weight.sum())
                continue
            propagated = weight @ self.A
            for o in range(self.n_obs):
                w = propagated * self.B[:, o]
                if w.sum() > 0.0:
                    stack.append((word + (o,), w))
        total = sum(law.values())
        assert abs(total - 1.0) < 1e-9, f"law sums to {total}, not 1"
        return {
            word: law.get(word, 0.0)
            for word in itertools.product(range(self.n_obs), repeat=k)
        }

    def eligible_contexts(
        self, r: int, max_length: int = 1, min_mass: float = _PROB_TOL
    ) -> list[tuple[int, ...]]:
        """All contexts of length 1..min(max_length, r+1) ending at index ``r``
        with visitation mass above ``min_mass`` (root excluded, as in the paper)."""
        out: list[tuple[int, ...]] = []
        for ell in range(1, min(max_length, r + 1) + 1):
            for word in itertools.product(range(self.n_obs), repeat=ell):
                if self.exact_context_mass(word, r) > min_mass:
                    out.append(word)
        return out

    def exact_tv_matrix(
        self, r: int, k: int, max_length: int = 1
    ) -> tuple[list[tuple[int, ...]], np.ndarray, dict, dict]:
        """Exact pairwise TV matrix between all eligible-context continuation laws.

        Returns ``(contexts, tv, laws, masses)`` where ``tv[i, j]`` is the total
        variation distance between the laws of ``contexts[i]`` and ``contexts[j]``.
        """
        contexts = self.eligible_contexts(r, max_length)
        laws = {c: self.exact_continuation_law(c, r, k) for c in contexts}
        masses = {c: self.exact_context_mass(c, r) for c in contexts}
        n = len(contexts)
        tv = np.zeros((n, n))
        words = list(itertools.product(range(self.n_obs), repeat=k))
        for i in range(n):
            for j in range(i + 1, n):
                li, lj = laws[contexts[i]], laws[contexts[j]]
                d = 0.5 * sum(abs(li[w] - lj[w]) for w in words)
                tv[i, j] = tv[j, i] = d
        return contexts, tv, laws, masses

    def min_pairwise_tv(self, r: int, k: int, max_length: int = 1) -> float:
        """Post-hoc gap: minimum pairwise TV over eligible contexts (0 iff a tie)."""
        contexts, tv, _, _ = self.exact_tv_matrix(r, k, max_length)
        if len(contexts) < 2:
            raise ValueError("need at least two eligible contexts")
        iu = np.triu_indices(len(contexts), 1)
        return float(tv[iu].min())

    def true_partition(
        self, r: int, k: int, max_length: int = 1, tol: float = _LAW_TOL
    ) -> dict[tuple[int, ...], int]:
        """Post-hoc true quotient: contexts grouped by exact law equality (< tol)."""
        contexts, tv, _, _ = self.exact_tv_matrix(r, k, max_length)
        partition: dict[tuple[int, ...], int] = {}
        reps: list[int] = []
        for i, c in enumerate(contexts):
            for cid, j in enumerate(reps):
                if tv[i, j] < tol:
                    partition[c] = cid
                    break
            else:
                reps.append(i)
                partition[c] = len(reps) - 1
        return partition
