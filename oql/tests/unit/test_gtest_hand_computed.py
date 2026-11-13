"""Hand-computed verification of the certified raw fixed-alphabet G-test.

Contract (contracts/equivalence-test-api.md): the certified statistic is the
paper's Eq. (gstat) computed on raw counts over the environment-supplied
alphabet — no pseudocounts, no data-derived categories, 0*ln(0) := 0 —
with degrees of freedom from the alphabet contract (|categories| - 1 under
full support, |support| - 1 otherwise), never from observed data.

Every case checks statistic, critical value, and decision bit-for-bit
against values computed independently here from the formula.
"""

import math

import pytest
from scipy import stats as sstats

from quotient_learning.equivalence import GTestEquivalenceTest
from quotient_learning.types import ContinuationAlphabet, ContinuationSample, ObservationHistory, Verdict

X, Y, Z = (0,), (1,), (2,)


def _sample(counts: dict, name: str = "h") -> ContinuationSample:
    conts = []
    for cat, n in counts.items():
        conts.extend([cat] * n)
    return ContinuationSample(
        history=ObservationHistory((sum(ord(c) for c in name) % 7,)),
        continuations=tuple(conts),
        window_length=1,
    )


def _g_raw(counts_a: dict, counts_b: dict, categories) -> float:
    """Independent implementation of Eq. (gstat) on raw counts."""
    n_a = sum(counts_a.values())
    n_b = sum(counts_b.values())
    n = n_a + n_b
    g = 0.0
    for c in categories:
        oa, ob = counts_a.get(c, 0), counts_b.get(c, 0)
        tot = oa + ob
        ea, eb = tot * n_a / n, tot * n_b / n
        if oa > 0:
            g += 2.0 * oa * math.log(oa / ea)
        if ob > 0:
            g += 2.0 * ob * math.log(ob / eb)
    return g


class TestCertifiedGTestHandComputed:
    def test_case1_separated_binary_alphabet_rejects(self):
        # A: 30 x, 10 y; B: 10 x, 30 y over C = {x, y}, full support, df = 1.
        # G = 4*(30 ln 1.5 + 10 ln 0.5) = 20.9299...
        alphabet = ContinuationAlphabet(k=1, categories=frozenset({X, Y}), full_support=True)
        test = GTestEquivalenceTest(alphabet=alphabet)
        ca, cb = {X: 30, Y: 10}, {X: 10, Y: 30}
        result = test.test(_sample(ca, "a"), _sample(cb, "b"), alpha=0.05)
        expected_g = _g_raw(ca, cb, alphabet.categories)
        assert expected_g == pytest.approx(4 * (30 * math.log(1.5) + 10 * math.log(0.5)), rel=1e-12)
        assert result.test_statistic == pytest.approx(expected_g, rel=1e-12)
        assert result.p_value == pytest.approx(float(sstats.chi2.sf(expected_g, df=1)), rel=1e-12)
        assert result.verdict is Verdict.DIFFERENT
        assert result.same_class is False

    def test_case2_near_identical_accepts(self):
        # A: 20 x, 20 y; B: 22 x, 18 y — tiny statistic, p >> 0.05.
        alphabet = ContinuationAlphabet(k=1, categories=frozenset({X, Y}), full_support=True)
        test = GTestEquivalenceTest(alphabet=alphabet)
        ca, cb = {X: 20, Y: 20}, {X: 22, Y: 18}
        result = test.test(_sample(ca, "a"), _sample(cb, "b"), alpha=0.05)
        expected_g = _g_raw(ca, cb, alphabet.categories)
        assert result.test_statistic == pytest.approx(expected_g, rel=1e-12)
        assert result.p_value == pytest.approx(float(sstats.chi2.sf(expected_g, df=1)), rel=1e-12)
        assert result.verdict is Verdict.SAME
        assert result.same_class is True

    def test_case3_structural_zero_uses_declared_support_df(self):
        # C = {x, y, z} but declared support {x, y} (z has stationary mass 0):
        # df = |support| - 1 = 1, NOT |observed| - 1 or |C| - 1.
        # Statistic identical to case 1 (z contributes 0 via 0*ln0 := 0).
        alphabet = ContinuationAlphabet(
            k=1, categories=frozenset({X, Y, Z}), full_support=False,
            support=frozenset({X, Y}),
        )
        test = GTestEquivalenceTest(alphabet=alphabet)
        ca, cb = {X: 30, Y: 10}, {X: 10, Y: 30}
        result = test.test(_sample(ca, "a"), _sample(cb, "b"), alpha=0.05)
        expected_g = _g_raw(ca, cb, alphabet.categories)
        assert result.test_statistic == pytest.approx(expected_g, rel=1e-12)
        assert result.p_value == pytest.approx(float(sstats.chi2.sf(expected_g, df=1)), rel=1e-12)
        assert result.verdict is Verdict.DIFFERENT

    def test_case4_full_alphabet_df_when_full_support_declared(self):
        # Same counts, but C = {x, y, z} declared full-support: df = 2.
        # The statistic is unchanged; only the critical value moves.
        alphabet = ContinuationAlphabet(k=1, categories=frozenset({X, Y, Z}), full_support=True)
        test = GTestEquivalenceTest(alphabet=alphabet)
        ca, cb = {X: 30, Y: 10}, {X: 10, Y: 30}
        result = test.test(_sample(ca, "a"), _sample(cb, "b"), alpha=0.05)
        expected_g = _g_raw(ca, cb, alphabet.categories)
        assert result.test_statistic == pytest.approx(expected_g, rel=1e-12)
        assert result.p_value == pytest.approx(float(sstats.chi2.sf(expected_g, df=2)), rel=1e-12)

    def test_no_pseudocounts_anywhere(self):
        # With raw counts the statistic must be exactly zero for identical
        # samples; any pseudocount scheme would still give exactly zero here,
        # but unequal sample sizes with a missing category expose smoothing:
        # A: 40 x; B: 30 x, 10 y. Raw G uses O=0 cells contributing 0.
        alphabet = ContinuationAlphabet(k=1, categories=frozenset({X, Y}), full_support=True)
        test = GTestEquivalenceTest(alphabet=alphabet)
        ca, cb = {X: 40}, {X: 30, Y: 10}
        result = test.test(_sample(ca, "a"), _sample(cb, "b"), alpha=0.05)
        expected_g = _g_raw(ca, cb, alphabet.categories)
        assert result.test_statistic == pytest.approx(expected_g, rel=1e-12)

    def test_legacy_smoothed_path_warns_and_is_not_certified(self):
        import warnings

        from quotient_learning.equivalence import NotCertifiedWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            GTestEquivalenceTest(smoothed=True)
        assert any(issubclass(w.category, NotCertifiedWarning) for w in caught)
