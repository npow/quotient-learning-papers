"""Hand checks for the non-asymptotic histogram-TV decision rule."""

from quotient_learning.equivalence import TVThresholdEquivalenceTest
from quotient_learning.stats import required_samples
from quotient_learning.types import (
    ContinuationAlphabet,
    ContinuationSample,
    ObservationHistory,
    Verdict,
)


def _sample(label: int, values: tuple[tuple[int, ...], ...]) -> ContinuationSample:
    return ContinuationSample(ObservationHistory((label,)), values, 1)


def test_tv_threshold_separates_and_merges() -> None:
    alphabet = ContinuationAlphabet(1, frozenset({(0,), (1,)}), True)
    test = TVThresholdEquivalenceTest(alphabet, epsilon_min=0.5)
    same_a = _sample(0, ((0,),) * 30 + ((1,),) * 30)
    same_b = _sample(1, ((0,),) * 30 + ((1,),) * 30)
    different = _sample(2, ((1,),) * 60)
    assert test.test(same_a, same_b).verdict is Verdict.SAME
    assert test.test(same_a, different).verdict is Verdict.DIFFERENT


def test_required_samples_makes_bound_meet_budget() -> None:
    alphabet = ContinuationAlphabet(1, frozenset({(0,), (1,)}), True)
    m = required_samples(0.5, 0.01, 0.02, alphabet)
    bound = 2 ** (alphabet.size + 1) * __import__("math").exp(-m * 0.5**2 / 8)
    assert bound <= 0.01
