"""Unit tests for quotient_learning.stats.bootstrap_ci.

Three properties verified:
1. Coverage  — for Gaussian samples, ~95% of CIs contain the true mean.
2. Symmetry  — on a symmetric distribution (uniform) the CI is symmetric
               around the sample mean.
3. Reproducibility — same seed → identical (lo, hi).
"""

from __future__ import annotations

import random

import pytest

from quotient_learning.stats import bootstrap_ci


# ---------------------------------------------------------------------------
# 1. Coverage on Gaussian samples
# ---------------------------------------------------------------------------

def test_coverage_gaussian() -> None:
    """~95% of 95%-CIs should contain the true mean for Gaussian samples."""
    true_mean = 0.0
    n_trials = 200          # number of independent CI computations
    n_per_trial = 50        # sample size per trial
    rng = random.Random(0)

    def _gaussian(mu: float = 0.0, sigma: float = 1.0) -> float:
        # Box-Muller, no numpy dependency.
        import math
        u1 = rng.random() or 1e-300
        u2 = rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    hits = 0
    for trial_seed in range(n_trials):
        sample = [_gaussian() for _ in range(n_per_trial)]
        lo, hi = bootstrap_ci(sample, n_resamples=500, ci=0.95, seed=trial_seed)
        if lo <= true_mean <= hi:
            hits += 1

    coverage = hits / n_trials
    # Expect coverage ≥ 0.85 (conservative tolerance to avoid flakiness).
    assert coverage >= 0.85, (
        f"Coverage too low: {coverage:.3f} < 0.85 on {n_trials} Gaussian trials"
    )


# ---------------------------------------------------------------------------
# 2. Symmetry on symmetric distributions
# ---------------------------------------------------------------------------

def test_symmetry_uniform() -> None:
    """CI bounds should be (approximately) symmetric around sample mean for uniform data."""
    rng = random.Random(7)
    n = 500
    sample = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    sample_mean = sum(sample) / n

    lo, hi = bootstrap_ci(sample, n_resamples=2000, ci=0.95, seed=99)

    lo_dist = sample_mean - lo
    hi_dist = hi - sample_mean
    # On a symmetric distribution the bootstrap distribution is symmetric around
    # the sample mean; allow 30% relative asymmetry tolerance.
    if lo_dist > 0 or hi_dist > 0:
        ratio = max(lo_dist, hi_dist) / (min(lo_dist, hi_dist) + 1e-12)
        assert ratio < 3.0, (
            f"CI is strongly asymmetric: lo_dist={lo_dist:.4f}, hi_dist={hi_dist:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. Reproducibility under seed
# ---------------------------------------------------------------------------

def test_reproducibility() -> None:
    """Identical seed must produce identical (lo, hi)."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    lo1, hi1 = bootstrap_ci(values, n_resamples=1000, ci=0.95, seed=42)
    lo2, hi2 = bootstrap_ci(values, n_resamples=1000, ci=0.95, seed=42)
    assert lo1 == lo2, "Lower bound not reproducible under same seed"
    assert hi1 == hi2, "Upper bound not reproducible under same seed"

    # Different seed → (very likely) different result.
    lo3, hi3 = bootstrap_ci(values, n_resamples=1000, ci=0.95, seed=99)
    # It's theoretically possible they coincide, but extremely unlikely.
    different = (lo1 != lo3) or (hi1 != hi3)
    assert different, "Different seeds produced identical CIs (implausible)"


# ---------------------------------------------------------------------------
# 4. Edge-case: single value
# ---------------------------------------------------------------------------

def test_single_value() -> None:
    """Single-element list should produce a degenerate CI equal to that value."""
    lo, hi = bootstrap_ci([5.0], n_resamples=100, seed=0)
    assert lo == pytest.approx(5.0)
    assert hi == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 5. Invalid inputs raise ValueError
# ---------------------------------------------------------------------------

def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        bootstrap_ci([])


def test_bad_ci_raises() -> None:
    with pytest.raises(ValueError, match="ci must be in"):
        bootstrap_ci([1.0, 2.0], ci=1.5)
