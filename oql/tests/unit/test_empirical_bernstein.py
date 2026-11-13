"""Empirical-Bernstein equivalence test unit test (T039)."""

from __future__ import annotations

import numpy as np
import pytest

from quotient_learning.equivalence import EmpiricalBernsteinTest
from quotient_learning.types import ContinuationSample, ObservationHistory


def _draw(probs: np.ndarray, n: int, rng: np.random.Generator) -> tuple[tuple[int, ...], ...]:
    indices = rng.choice(len(probs), size=n, p=probs)
    outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return tuple(outcomes[i] for i in indices)


@pytest.mark.unit
class TestEmpiricalBernstein:
    def test_same_distribution_accepted(self) -> None:
        rng = np.random.default_rng(0)
        eb = EmpiricalBernsteinTest(n_observations=2, seed=0)
        p = np.array([0.4, 0.2, 0.2, 0.2])
        accepts = 0
        for _ in range(10):
            a = ContinuationSample(history=ObservationHistory(()),
                                   continuations=_draw(p, 200, rng), window_length=2)
            b = ContinuationSample(history=ObservationHistory(()),
                                   continuations=_draw(p, 200, rng), window_length=2)
            r = eb.test(a, b, alpha=0.05)
            if r.same_class:
                accepts += 1
        # Bernstein bound is conservative — should accept most of the time.
        assert accepts >= 7

    def test_returns_finite_statistic(self) -> None:
        rng = np.random.default_rng(0)
        eb = EmpiricalBernsteinTest(n_observations=2, seed=0)
        p = np.array([0.4, 0.2, 0.2, 0.2])
        a = ContinuationSample(history=ObservationHistory(()),
                               continuations=_draw(p, 100, rng), window_length=2)
        b = ContinuationSample(history=ObservationHistory(()),
                               continuations=_draw(p, 100, rng), window_length=2)
        r = eb.test(a, b, alpha=0.05)
        assert np.isfinite(r.test_statistic)
        assert 0.0 <= r.p_value <= 1.0
