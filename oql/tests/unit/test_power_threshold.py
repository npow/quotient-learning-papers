"""Hand-computed checks for the m*(eps, alpha, beta) power threshold."""

import math

import pytest

from quotient_learning.stats import required_samples
from quotient_learning.types import ContinuationAlphabet


def _mstar(eps: float, alpha: float, beta: float, size: int) -> int:
    return math.ceil(
        (8.0 / eps**2)
        * ((size + 1) * math.log(2.0) + math.log(1.0 / min(alpha, beta)))
    )


class TestRequiredSamples:
    def test_hand_computed_eps_half(self):
        # eps=0.5, alpha=beta=0.05, |C|=4:
        # 8/0.25 * (5 ln 2 + ln 20) = 206.77... -> 207.
        assert required_samples(0.5, 0.05, 0.05, 4) == 207
        assert required_samples(0.5, 0.05, 0.05, 4) == _mstar(0.5, 0.05, 0.05, 4)

    def test_hand_computed_asymmetric_levels(self):
        # min(alpha, beta) drives the log term: min(0.1, 0.01) = 0.01
        # 8/0.04 * (3 ln 2 + ln 100) = 1336.92... -> 1337.
        assert required_samples(0.2, 0.1, 0.01, 2) == 1337

    def test_accepts_alphabet_object(self):
        alphabet = ContinuationAlphabet(
            k=1, categories=frozenset({(0,), (1,), (2,)}), full_support=True
        )
        assert required_samples(0.5, 0.05, 0.05, alphabet) == _mstar(0.5, 0.05, 0.05, 3)

    def test_monotone_in_separation(self):
        # Smaller separation -> more samples required
        assert required_samples(0.1, 0.05, 0.05, 4) > required_samples(0.5, 0.05, 0.05, 4)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            required_samples(0.0, 0.05, 0.05, 4)
        with pytest.raises(ValueError):
            required_samples(0.5, 0.0, 0.05, 4)
        with pytest.raises(ValueError):
            required_samples(0.5, 0.05, 1.0, 4)
        with pytest.raises(ValueError):
            required_samples(0.5, 0.05, 0.05, 1)
