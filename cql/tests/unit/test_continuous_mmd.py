"""Unit tests for ContinuousMMDTest (T008).

Verifies:
  (a) Type I error control: equal-distribution pairs are not rejected at rate > α
  (b) Type II decay: well-separated distributions are reliably rejected
  (c) Median bandwidth heuristic works when bandwidth=None
  (d) Laplace kernel alternative works
"""

from __future__ import annotations

import numpy as np
import pytest

from continuous_quotient.equivalence import ContinuousMMDTest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_samples(loc: float, scale: float, n: int, seed: int, d: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=loc, scale=scale, size=(n, d)).astype(np.float64)


# ---------------------------------------------------------------------------
# Type I error control
# ---------------------------------------------------------------------------

class TestTypeI:
    """Type I error should be ≤ α + noise (permutation calibration is exact)."""

    def test_same_distribution_not_rejected(self):
        """H0: P = Q = N(0, 1). At α=0.05 and 100 replications, expect ~5% rejection."""
        alpha = 0.05
        n_rep = 100
        test = ContinuousMMDTest(kernel="rbf", n_permutations=99, alpha=alpha, seed=42)
        rejections = 0
        rng = np.random.default_rng(7)
        for _ in range(n_rep):
            a = rng.normal(0, 1, (40, 4))
            b = rng.normal(0, 1, (40, 4))
            res = test.test(a, b, alpha=alpha)
            if res.reject:
                rejections += 1
        # Allow up to 3x α due to randomness (loose bound for speed).
        assert rejections / n_rep <= 3 * alpha + 0.05, (
            f"Type I rate {rejections/n_rep:.3f} exceeds 3α={3*alpha}"
        )

    def test_p_value_in_unit_interval(self):
        a = _make_samples(0.0, 1.0, 50, 1)
        b = _make_samples(0.0, 1.0, 50, 2)
        res = ContinuousMMDTest(n_permutations=49).test(a, b)
        assert 0.0 <= res.p_value <= 1.0


# ---------------------------------------------------------------------------
# Type II decay (power)
# ---------------------------------------------------------------------------

class TestTypeII:
    """Well-separated distributions should be reliably detected."""

    def test_large_separation_rejected(self):
        """N(0,1) vs N(5,1): MMD should reject on every seed."""
        test = ContinuousMMDTest(kernel="rbf", n_permutations=99, alpha=0.05)
        rng = np.random.default_rng(99)
        for seed in range(20):
            a = rng.normal(0, 1, (40, 4))
            b = rng.normal(5, 1, (40, 4))
            res = test.test(a, b)
            assert res.reject, f"Failed to reject on seed {seed}: p={res.p_value:.4f}"

    def test_moderate_separation_power(self):
        """N(0,1) vs N(1,1) with n=100: ≥80% power across 50 replications."""
        test = ContinuousMMDTest(kernel="rbf", n_permutations=99, alpha=0.05)
        rng = np.random.default_rng(0)
        power = sum(
            test.test(rng.normal(0, 1, (100, 2)), rng.normal(1, 1, (100, 2))).reject
            for _ in range(50)
        ) / 50
        assert power >= 0.80, f"Power {power:.2f} < 0.80"


# ---------------------------------------------------------------------------
# Bandwidth heuristic
# ---------------------------------------------------------------------------

class TestBandwidth:
    def test_auto_bandwidth_works(self):
        """bandwidth=None should not raise and should return positive bandwidth."""
        a = _make_samples(0.0, 1.0, 30, 3)
        b = _make_samples(2.0, 1.0, 30, 4)
        res = ContinuousMMDTest(bandwidth=None, n_permutations=49).test(a, b)
        assert res.bandwidth > 0

    def test_manual_bandwidth_consistent(self):
        """Two separate instances with the same seed produce identical p-values."""
        a = _make_samples(0.0, 1.0, 30, 5)
        b = _make_samples(0.0, 1.0, 30, 6)
        t1 = ContinuousMMDTest(bandwidth=1.0, n_permutations=99, seed=0)
        t2 = ContinuousMMDTest(bandwidth=1.0, n_permutations=99, seed=0)
        p1 = t1.test(a, b).p_value
        p2 = t2.test(a, b).p_value
        assert abs(p1 - p2) < 1e-9  # same seed, same data → same p-value


# ---------------------------------------------------------------------------
# Alternative kernel
# ---------------------------------------------------------------------------

class TestLaplaceKernel:
    def test_laplace_rejects_separated(self):
        test = ContinuousMMDTest(kernel="laplace", n_permutations=99, alpha=0.05)
        rng = np.random.default_rng(11)
        a = rng.normal(0, 1, (50, 3))
        b = rng.normal(4, 1, (50, 3))
        res = test.test(a, b)
        assert res.reject

    def test_laplace_does_not_reject_same(self):
        test = ContinuousMMDTest(kernel="laplace", n_permutations=99, alpha=0.05, seed=13)
        rng = np.random.default_rng(13)
        rejections = sum(
            test.test(rng.normal(0, 1, (50, 3)), rng.normal(0, 1, (50, 3))).reject
            for _ in range(30)
        )
        assert rejections / 30 <= 0.20  # loose bound


# ---------------------------------------------------------------------------
# Bounds module
# ---------------------------------------------------------------------------

class TestBounds:
    def test_trajectory_budget_monotone_in_n(self):
        from continuous_quotient.bounds import trajectory_budget
        budgets = [trajectory_budget(n, k=2, tau=2.0, delta=0.05, T=50, eps_mmd=0.3)
                   for n in [2, 4, 8, 16]]
        for i in range(len(budgets) - 1):
            assert budgets[i] < budgets[i + 1], "budget should increase with n_classes"

    def test_trajectory_budget_monotone_decreasing_in_eps(self):
        from continuous_quotient.bounds import trajectory_budget
        budgets = [trajectory_budget(4, k=2, tau=2.0, delta=0.05, T=50, eps_mmd=e)
                   for e in [0.5, 0.3, 0.1, 0.05]]
        for i in range(len(budgets) - 1):
            assert budgets[i] < budgets[i + 1], "budget should increase as eps_mmd shrinks"
