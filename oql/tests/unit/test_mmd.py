"""Contract test for MMDEquivalenceTest (T018).

Smaller trial count than the G-test contract because the permutation-based
p-value is computationally heavier. Still strong evidence the test discriminates
same- vs different-distribution categorical samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from quotient_learning.equivalence import MMDEquivalenceTest
from quotient_learning.types import ContinuationSample, ObservationHistory


def _draw(probs: np.ndarray, n: int, rng: np.random.Generator) -> tuple[tuple[int, ...], ...]:
    indices = rng.choice(len(probs), size=n, p=probs)
    outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return tuple(outcomes[i] for i in indices)


@pytest.mark.unit
class TestMMDContract:
    N_TRIALS = 20
    N_SAMPLES = 200
    ALPHA = 0.05

    def test_type_i_error_on_same_distribution(self) -> None:
        rng = np.random.default_rng(20260601)
        mmd = MMDEquivalenceTest(n_observations=2, n_permutations=100, seed=0)
        p = np.array([0.4, 0.2, 0.2, 0.2])
        rejections = 0
        for _ in range(self.N_TRIALS):
            a = ContinuationSample(
                history=ObservationHistory(()),
                continuations=_draw(p, self.N_SAMPLES, rng), window_length=2,
            )
            b = ContinuationSample(
                history=ObservationHistory(()),
                continuations=_draw(p, self.N_SAMPLES, rng), window_length=2,
            )
            res = mmd.test(a, b, alpha=self.ALPHA)
            if not res.same_class:
                rejections += 1
        # Type I error tolerance: 5x nominal alpha for N=20 trials due to small sample variance.
        assert rejections / self.N_TRIALS <= 0.25

    def test_power_on_different_distribution(self) -> None:
        rng = np.random.default_rng(20260602)
        mmd = MMDEquivalenceTest(n_observations=2, n_permutations=100, seed=0)
        p1 = np.array([0.85, 0.05, 0.05, 0.05])
        p2 = np.array([0.05, 0.05, 0.05, 0.85])
        false_accepts = 0
        for _ in range(self.N_TRIALS):
            a = ContinuationSample(
                history=ObservationHistory((0,)),
                continuations=_draw(p1, self.N_SAMPLES, rng), window_length=2,
            )
            b = ContinuationSample(
                history=ObservationHistory((1,)),
                continuations=_draw(p2, self.N_SAMPLES, rng), window_length=2,
            )
            res = mmd.test(a, b, alpha=self.ALPHA)
            if res.same_class:
                false_accepts += 1
        assert false_accepts / self.N_TRIALS <= 0.1
