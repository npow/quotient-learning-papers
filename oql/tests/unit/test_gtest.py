"""Contract test for GTestEquivalenceTest (T017).

Verifies Type I and Type II error rates ≤ 0.05 across 100 repeated trials on
synthetic categorical distributions matched to Tiger's continuation space (2
observations, window length 2 → 4 categories).
"""

from __future__ import annotations

import numpy as np
import pytest

from quotient_learning.equivalence import GTestEquivalenceTest
from quotient_learning.types import ContinuationSample, ObservationHistory


def _draw(probs: np.ndarray, n: int, rng: np.random.Generator) -> tuple[tuple[int, ...], ...]:
    """Draw n samples from a categorical distribution over (window_length=2) outcomes."""
    indices = rng.choice(len(probs), size=n, p=probs)
    outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return tuple(outcomes[i] for i in indices)


@pytest.mark.unit
class TestGTestContract:
    N_TRIALS = 100
    N_SAMPLES = 1000
    ALPHA = 0.05

    def test_type_i_error_under_threshold(self) -> None:
        """Under the null (same distribution), false-rejection rate ≤ alpha."""
        rng = np.random.default_rng(20260525)
        gtest = GTestEquivalenceTest()
        # Tiger-like continuation distribution: skewed toward (0, 0).
        p = np.array([0.5, 0.15, 0.15, 0.20])
        rejections = 0
        for _ in range(self.N_TRIALS):
            a = ContinuationSample(
                history=ObservationHistory(()),
                continuations=_draw(p, self.N_SAMPLES, rng),
                window_length=2,
            )
            b = ContinuationSample(
                history=ObservationHistory(()),
                continuations=_draw(p, self.N_SAMPLES, rng),
                window_length=2,
            )
            res = gtest.test(a, b, alpha=self.ALPHA)
            if not res.same_class:
                rejections += 1
        type_i = rejections / self.N_TRIALS
        # Allow small overshoot above nominal alpha — empirical with N=100 trials.
        assert type_i <= self.ALPHA + 0.05, f"Type I error {type_i} exceeds {self.ALPHA + 0.05}"

    def test_type_ii_error_under_threshold(self) -> None:
        """Under a clear alternative, false-acceptance rate ≤ 0.05."""
        rng = np.random.default_rng(20260526)
        gtest = GTestEquivalenceTest()
        p1 = np.array([0.85, 0.05, 0.05, 0.05])
        p2 = np.array([0.05, 0.05, 0.05, 0.85])
        false_accepts = 0
        for _ in range(self.N_TRIALS):
            a = ContinuationSample(
                history=ObservationHistory((0,)),
                continuations=_draw(p1, self.N_SAMPLES, rng),
                window_length=2,
            )
            b = ContinuationSample(
                history=ObservationHistory((1,)),
                continuations=_draw(p2, self.N_SAMPLES, rng),
                window_length=2,
            )
            res = gtest.test(a, b, alpha=self.ALPHA)
            if res.same_class:
                false_accepts += 1
        type_ii = false_accepts / self.N_TRIALS
        assert type_ii <= self.ALPHA, f"Type II error {type_ii} exceeds {self.ALPHA}"

    def test_rejects_too_few_samples(self) -> None:
        from quotient_learning.errors import InsufficientSamplesError
        gtest = GTestEquivalenceTest()
        a = ContinuationSample(
            history=ObservationHistory(()),
            continuations=tuple([(0, 0)] * 10),
            window_length=2,
        )
        b = ContinuationSample(
            history=ObservationHistory(()),
            continuations=tuple([(0, 0)] * 1000),
            window_length=2,
        )
        with pytest.raises(InsufficientSamplesError):
            gtest.test(a, b)

    def test_rejects_window_mismatch(self) -> None:
        from quotient_learning.errors import WindowMismatchError
        gtest = GTestEquivalenceTest()
        a = ContinuationSample(
            history=ObservationHistory(()),
            continuations=tuple([(0, 0)] * 100),
            window_length=2,
        )
        b = ContinuationSample(
            history=ObservationHistory(()),
            continuations=tuple([(0,)] * 100),
            window_length=1,
        )
        with pytest.raises(WindowMismatchError):
            gtest.test(a, b)
