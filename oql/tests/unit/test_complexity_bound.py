"""Unit test for sample-complexity bound (T038)."""

from __future__ import annotations

import math

import pytest

from quotient_learning.complexity import (
    samples_per_pair,
    theoretical_complexity_curve,
    trajectories_required,
)


@pytest.mark.unit
class TestComplexityBound:
    def test_polynomial_in_n_classes(self) -> None:
        small = trajectories_required(0.1, 0.05, n_classes=2, window_length=2)
        medium = trajectories_required(0.1, 0.05, n_classes=20, window_length=2)
        big = trajectories_required(0.1, 0.05, n_classes=200, window_length=2)
        # Polynomial growth — should be much less than exponential
        assert small < medium < big
        # Should not grow super-cubically (O(n^3) formula: n=200 yields ~8.7e8).
        assert big < 1e10

    def test_log_in_one_over_delta(self) -> None:
        a = trajectories_required(0.1, 0.05, 10, 2)
        b = trajectories_required(0.1, 0.005, 10, 2)
        # Tighter delta only multiplies by log ratio, not 10x.
        assert b < 3 * a

    def test_inverse_square_in_epsilon(self) -> None:
        a = samples_per_pair(0.1, 0.05, 10)
        b = samples_per_pair(0.05, 0.05, 10)
        # Halving epsilon should ~quadruple required samples.
        assert b >= 3 * a

    def test_curve_monotonic(self) -> None:
        curve = theoretical_complexity_curve(
            [2, 5, 10, 50], epsilon=0.1, delta=0.05, window_length=2,
        )
        ns = [c[1] for c in curve]
        assert all(ns[i] <= ns[i + 1] for i in range(len(ns) - 1))

    def test_validates_inputs(self) -> None:
        with pytest.raises(ValueError):
            samples_per_pair(0.0, 0.05, 10)
        with pytest.raises(ValueError):
            samples_per_pair(0.1, 0.0, 10)
        with pytest.raises(ValueError):
            samples_per_pair(0.1, 0.05, 0)
