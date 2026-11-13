"""Phase-transition classifier accuracy test (T064).

For a sweep of channel capacities across a synthetic distribution of
R_crit values, classify each as feasible / infeasible and compare to ground
truth (R_crit ≤ channel_capacity).
"""

from __future__ import annotations

import math

import pytest

from quotient_learning.rcrit import classify_phase_transition
from quotient_learning.types import CanonicalQuotient, ObservationHistory, RCritEstimate


def _quotient_of_size(n: int) -> CanonicalQuotient:
    return CanonicalQuotient(
        n_classes=n,
        partition={ObservationHistory((i,)): i for i in range(n)},
    )


@pytest.mark.integration
class TestPhaseTransitionClassifier:
    def test_classifier_accuracy_above_90_percent(self) -> None:
        """Across a sweep of (quotient size, channel capacity) pairs the
        classifier should be ≥ 90% accurate."""
        cases: list[tuple[int, float]] = [
            (2, 0.5), (2, 1.0), (2, 1.5),
            (4, 1.0), (4, 2.0), (4, 3.0),
            (8, 2.0), (8, 3.0), (8, 4.0),
            (16, 3.0), (16, 4.0), (16, 5.0),
        ]
        correct = 0
        for n_classes, cap in cases:
            q = _quotient_of_size(n_classes)
            true_r = math.log2(n_classes)
            est = RCritEstimate(quotient_a=q, quotient_b=q, r_crit=true_r,
                                true_r_crit=true_r)
            phase = classify_phase_transition(est, channel_capacity=cap)
            truth = (cap >= true_r)
            if phase.feasible == truth:
                correct += 1
        accuracy = correct / len(cases)
        assert accuracy >= 0.9, f"accuracy {accuracy:.1%} below 90%"
