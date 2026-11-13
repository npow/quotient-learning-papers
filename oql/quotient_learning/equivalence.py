"""Equivalence-test interfaces and implementations.

- `EquivalenceTest`: abstract base.
- `GTestEquivalenceTest`: χ²/G-test on the joint histogram of continuation
  windows. Its chi-square calibration is asymptotic.
- `TVThresholdEquivalenceTest`: finite-sample histogram test used by the
  certified estimator.
- `MMDEquivalenceTest`: kernel maximum mean discrepancy with an RBF kernel over
  one-hot embeddings; permutation test for the p-value.
- `EmpiricalBernsteinTest`: empirical-Bernstein concentration on MMD² (US2).
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
from scipy import stats

from quotient_learning.errors import InsufficientSamplesError, WindowMismatchError
from quotient_learning.types import (
    ContinuationAlphabet,
    ContinuationSample,
    EquivalenceTestResult,
    PairOutcome,
    Verdict,
)

MIN_SAMPLES_FOR_CLT: int = 30


class NotCertifiedWarning(UserWarning):
    """Emitted when a test variant outside the certified analysis is used."""


class EquivalenceTest(ABC):
    """Abstract statistical equivalence test on continuation distributions."""

    @abstractmethod
    def test(
        self,
        samples_a: ContinuationSample,
        samples_b: ContinuationSample,
        alpha: float = 0.05,
    ) -> EquivalenceTestResult:
        ...

    @staticmethod
    def _check_preconditions(
        samples_a: ContinuationSample,
        samples_b: ContinuationSample,
        alpha: float,
        min_samples: int = MIN_SAMPLES_FOR_CLT,
    ) -> None:
        if not (0.0 < alpha < 0.5):
            raise ValueError(f"alpha must be in (0, 0.5); got {alpha}")
        if samples_a.window_length != samples_b.window_length:
            raise WindowMismatchError(samples_a.window_length, samples_b.window_length)
        if samples_a.n_samples < min_samples:
            raise InsufficientSamplesError(samples_a.n_samples, min_samples)
        if samples_b.n_samples < min_samples:
            raise InsufficientSamplesError(samples_b.n_samples, min_samples)


class GTestEquivalenceTest(EquivalenceTest):
    """Two-sample G-test (log-likelihood ratio) on continuation histograms.

    Fixed-alphabet path (``alphabet`` given): computes the raw unsmoothed
    statistic of the paper's Eq. (gstat) over the FIXED environment-supplied
    alphabet — no pseudocounts, zero-count cells contribute 0 (0*ln0 := 0) —
    with degrees of freedom from the alphabet contract
    (``ContinuationAlphabet.df``).  The resulting p-value uses the usual
    asymptotic chi-square approximation; it is not the finite-sample certified
    test.

    Legacy path (``smoothed=True`` or no alphabet): the historical variant
    that derives categories from the observed data, adds ``pseudocount`` to
    every cell of both samples, and sets df from the observed category
    count. It is NOT covered by the paper's guarantees and emits
    ``NotCertifiedWarning``.
    """

    def __init__(
        self,
        alphabet: ContinuationAlphabet | None = None,
        smoothed: bool = False,
        pseudocount: float = 0.5,
    ):
        if alphabet is not None and smoothed:
            raise ValueError("smoothed=True is incompatible with a fixed alphabet")
        self.alphabet = alphabet
        self.smoothed = smoothed or alphabet is None
        self.pseudocount = pseudocount
        if self.smoothed:
            warnings.warn(
                "GTestEquivalenceTest without a fixed ContinuationAlphabet uses the "
                "legacy smoothed observed-category statistic, which the paper's "
                "finite-sample guarantees do not cover. Pass alphabet=... for the "
                "fixed-alphabet asymptotic variant, or use "
                "TVThresholdEquivalenceTest for certification.",
                NotCertifiedWarning,
                stacklevel=2,
            )

    def test(
        self,
        samples_a: ContinuationSample,
        samples_b: ContinuationSample,
        alpha: float = 0.05,
    ) -> PairOutcome | EquivalenceTestResult:
        self._check_preconditions(samples_a, samples_b, alpha)
        if self.smoothed:
            return self._test_legacy_smoothed(samples_a, samples_b, alpha)
        return self._test_fixed_alphabet(samples_a, samples_b, alpha)

    def _test_fixed_alphabet(
        self,
        samples_a: ContinuationSample,
        samples_b: ContinuationSample,
        alpha: float,
    ) -> PairOutcome:
        assert self.alphabet is not None
        cat_list = sorted(self.alphabet.categories)
        ca = Counter(samples_a.continuations)
        cb = Counter(samples_b.continuations)
        unknown = (set(ca) | set(cb)) - self.alphabet.categories
        if unknown:
            raise ValueError(
                f"continuations outside the declared alphabet: {sorted(unknown)[:5]}"
            )
        obs_a = np.array([ca[c] for c in cat_list], dtype=float)
        obs_b = np.array([cb[c] for c in cat_list], dtype=float)
        n_a, n_b = obs_a.sum(), obs_b.sum()
        totals = obs_a + obs_b
        exp_a = totals * (n_a / (n_a + n_b))
        exp_b = totals * (n_b / (n_a + n_b))
        # G = 2 * sum O ln(O/E) over raw counts, with 0*ln0 := 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            term_a = np.where(obs_a > 0, obs_a * np.log(obs_a / exp_a), 0.0)
            term_b = np.where(obs_b > 0, obs_b * np.log(obs_b / exp_b), 0.0)
        g = float(2.0 * (term_a.sum() + term_b.sum()))
        df = self.alphabet.df
        p_value = float(stats.chi2.sf(g, df=df))
        verdict = Verdict.SAME if p_value >= alpha else Verdict.DIFFERENT
        return PairOutcome(
            history_a=samples_a.history, history_b=samples_b.history,
            verdict=verdict, alpha=alpha, m_required=0,
            test_statistic=g, p_value=p_value,
        )

    def _test_legacy_smoothed(
        self,
        samples_a: ContinuationSample,
        samples_b: ContinuationSample,
        alpha: float,
    ) -> EquivalenceTestResult:
        categories = set(samples_a.continuations) | set(samples_b.continuations)
        if len(categories) < 2:
            # Degenerate: only one unique outcome — same distribution by construction.
            return EquivalenceTestResult(
                history_a=samples_a.history, history_b=samples_b.history,
                test_statistic=0.0, p_value=1.0, same_class=True, alpha=alpha,
            )
        cat_list = sorted(categories)
        ca = Counter(samples_a.continuations)
        cb = Counter(samples_b.continuations)
        obs_a = np.array([ca[c] + self.pseudocount for c in cat_list], dtype=float)
        obs_b = np.array([cb[c] + self.pseudocount for c in cat_list], dtype=float)
        n_a, n_b = obs_a.sum(), obs_b.sum()
        n_total = n_a + n_b
        totals = obs_a + obs_b
        exp_a = totals * (n_a / n_total)
        exp_b = totals * (n_b / n_total)
        with np.errstate(divide="ignore", invalid="ignore"):
            term_a = np.where(obs_a > 0, obs_a * np.log(obs_a / exp_a), 0.0)
            term_b = np.where(obs_b > 0, obs_b * np.log(obs_b / exp_b), 0.0)
        g = float(2.0 * (term_a.sum() + term_b.sum()))
        df = len(cat_list) - 1
        p_value = float(stats.chi2.sf(g, df=df))
        return EquivalenceTestResult(
            history_a=samples_a.history, history_b=samples_b.history,
            test_statistic=g, p_value=p_value,
            same_class=(p_value >= alpha), alpha=alpha,
        )


class TVThresholdEquivalenceTest(EquivalenceTest):
    """Finite-sample two-sample test for an ``epsilon_min``-gap model.

    The test declares two laws equal when their empirical total-variation
    distance is less than ``epsilon_min / 2``.  For a fixed alphabet of size
    ``d`` and at least ``m`` iid samples from each law, both error types are at
    most ``2**(d+1) * exp(-m * epsilon_min**2 / 8)`` whenever the true laws are
    either identical or separated by at least ``epsilon_min``.  This follows
    directly from the multinomial L1 concentration inequality and the
    triangle inequality, without an asymptotic calibration.

    The estimator's power gate is responsible for checking that this bound is
    no larger than the requested pairwise error budget.
    """

    def __init__(self, alphabet: ContinuationAlphabet, epsilon_min: float):
        if not (0.0 < epsilon_min <= 1.0):
            raise ValueError("epsilon_min must be in (0, 1]")
        self.alphabet = alphabet
        self.epsilon_min = float(epsilon_min)

    def test(
        self,
        samples_a: ContinuationSample,
        samples_b: ContinuationSample,
        alpha: float = 0.05,
    ) -> PairOutcome:
        self._check_preconditions(samples_a, samples_b, alpha)
        categories = sorted(self.alphabet.categories)
        ca, cb = Counter(samples_a.continuations), Counter(samples_b.continuations)
        unknown = (set(ca) | set(cb)) - self.alphabet.categories
        if unknown:
            raise ValueError(
                f"continuations outside the declared alphabet: {sorted(unknown)[:5]}"
            )
        na, nb = samples_a.n_samples, samples_b.n_samples
        pa = np.array([ca[c] / na for c in categories], dtype=float)
        pb = np.array([cb[c] / nb for c in categories], dtype=float)
        tv = float(0.5 * np.abs(pa - pb).sum())
        m = min(na, nb)
        log_bound = (self.alphabet.size + 1) * np.log(2.0) \
            - m * self.epsilon_min**2 / 8.0
        error_bound = float(min(1.0, np.exp(log_bound)))
        verdict = Verdict.SAME if tv < self.epsilon_min / 2.0 else Verdict.DIFFERENT
        return PairOutcome(
            history_a=samples_a.history,
            history_b=samples_b.history,
            verdict=verdict,
            alpha=alpha,
            m_required=0,
            test_statistic=tv,
            p_value=error_bound,
        )


class MMDEquivalenceTest(EquivalenceTest):
    """Two-sample kernel MMD² test with an RBF kernel over one-hot embeddings.

    Computes the biased MMD² estimator on integer-encoded continuation windows
    embedded as one-hot vectors (one bit per (position, observation) pair). The
    p-value comes from a permutation test (default 200 permutations).
    """

    def __init__(
        self,
        n_observations: int,
        bandwidth: float | None = None,
        n_permutations: int = 200,
        seed: int = 0,
    ):
        self.n_observations = n_observations
        self.bandwidth = bandwidth
        self.n_permutations = n_permutations
        self._rng = np.random.default_rng(seed)

    def _embed(self, samples: ContinuationSample) -> np.ndarray:
        """Embed each continuation (o_1, ..., o_k) as a flattened one-hot vector
        of length window_length * n_observations.
        """
        k = samples.window_length
        m = samples.n_samples
        d = k * self.n_observations
        x = np.zeros((m, d), dtype=np.float32)
        for i, cont in enumerate(samples.continuations):
            for j, o in enumerate(cont):
                if 0 <= o < self.n_observations:
                    x[i, j * self.n_observations + o] = 1.0
        return x

    def _rbf_gram(self, x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
        # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 x_i · y_j
        x_norm = (x * x).sum(axis=1, keepdims=True)
        y_norm = (y * y).sum(axis=1, keepdims=True)
        sq = x_norm + y_norm.T - 2.0 * (x @ y.T)
        np.maximum(sq, 0.0, out=sq)
        return np.exp(-sq / (2.0 * bandwidth * bandwidth))

    def _mmd2(self, kxx: np.ndarray, kyy: np.ndarray, kxy: np.ndarray) -> float:
        return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())

    def test(
        self,
        samples_a: ContinuationSample,
        samples_b: ContinuationSample,
        alpha: float = 0.05,
    ) -> EquivalenceTestResult:
        self._check_preconditions(samples_a, samples_b, alpha)
        x = self._embed(samples_a)
        y = self._embed(samples_b)
        # Median heuristic for bandwidth if not specified.
        if self.bandwidth is None:
            stacked = np.vstack([x, y])
            n = stacked.shape[0]
            idx = self._rng.choice(n, size=min(n, 200), replace=False)
            sub = stacked[idx]
            d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
            med = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
            bandwidth = float(np.sqrt(0.5 * med)) or 1.0
        else:
            bandwidth = self.bandwidth
        kxx = self._rbf_gram(x, x, bandwidth)
        kyy = self._rbf_gram(y, y, bandwidth)
        kxy = self._rbf_gram(x, y, bandwidth)
        observed = self._mmd2(kxx, kyy, kxy)
        # Permutation test on the pooled gram matrix.
        combined = np.vstack([x, y])
        m = x.shape[0]
        ge = 0
        for _ in range(self.n_permutations):
            perm = self._rng.permutation(combined.shape[0])
            x_perm = combined[perm[:m]]
            y_perm = combined[perm[m:]]
            kxx_p = self._rbf_gram(x_perm, x_perm, bandwidth)
            kyy_p = self._rbf_gram(y_perm, y_perm, bandwidth)
            kxy_p = self._rbf_gram(x_perm, y_perm, bandwidth)
            stat_p = self._mmd2(kxx_p, kyy_p, kxy_p)
            if stat_p >= observed:
                ge += 1
        p_value = float((ge + 1) / (self.n_permutations + 1))
        return EquivalenceTestResult(
            history_a=samples_a.history, history_b=samples_b.history,
            test_statistic=float(observed), p_value=p_value,
            same_class=(p_value >= alpha), alpha=alpha,
        )


class EmpiricalBernsteinTest(EquivalenceTest):
    """Empirical-Bernstein concentration on MMD² (US2, T051).

    For two samples of size m, declares same-class iff
        observed MMD² ≤ 2 * sqrt(2 V log(2/alpha) / m) + 7 log(2/alpha) / (3 (m-1))
    where V is the empirical variance of the MMD² unbiased estimator. This is a
    confidence bound — when MMD² falls inside the bound we cannot reject the
    null at level alpha.
    """

    def __init__(self, n_observations: int, bandwidth: float | None = None, seed: int = 0):
        self.n_observations = n_observations
        self.bandwidth = bandwidth
        self._mmd = MMDEquivalenceTest(
            n_observations=n_observations, bandwidth=bandwidth,
            n_permutations=0, seed=seed,
        )

    def test(
        self,
        samples_a: ContinuationSample,
        samples_b: ContinuationSample,
        alpha: float = 0.05,
    ) -> EquivalenceTestResult:
        self._check_preconditions(samples_a, samples_b, alpha)
        x = self._mmd._embed(samples_a)
        y = self._mmd._embed(samples_b)
        if self.bandwidth is None:
            stacked = np.vstack([x, y])
            d2 = np.sum(
                (stacked[:, None, :] - stacked[None, :, :]) ** 2, axis=-1
            )
            med = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
            bandwidth = float(np.sqrt(0.5 * med)) or 1.0
        else:
            bandwidth = self.bandwidth
        kxx = self._mmd._rbf_gram(x, x, bandwidth)
        kyy = self._mmd._rbf_gram(y, y, bandwidth)
        kxy = self._mmd._rbf_gram(x, y, bandwidth)
        mmd2 = float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
        m = min(x.shape[0], y.shape[0])
        # Empirical variance of per-row MMD-like contributions (rough).
        row_diffs = kxx.mean(axis=1) - kxy.mean(axis=1)
        v = float(np.var(row_diffs, ddof=1)) if row_diffs.size > 1 else 0.0
        log_term = float(np.log(2.0 / alpha))
        bound = 2.0 * np.sqrt(2.0 * v * log_term / m) + 7.0 * log_term / (3.0 * max(m - 1, 1))
        same_class = mmd2 <= bound
        return EquivalenceTestResult(
            history_a=samples_a.history, history_b=samples_b.history,
            test_statistic=mmd2,
            p_value=float(np.exp(-mmd2 * m / max(2.0 * v + 1e-12, 1e-12))),
            same_class=same_class, alpha=alpha,
        )
