"""Misspecification probe (REV-5): analytic laws and margin computation.

Covers (a) the engineered TV structure of MisspecChain — the sub-gap pair
sits at exact TV tau in (0, eps_min/2) while every other pair is separated
by >= 0.5 — and (b) the runner's per-pair margin computation on hand-built
histogram pairs, cross-checked against TVThresholdEquivalenceTest.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pytest

from environments.layer2_memory.misspec_chain import SEPARATED_TV, MisspecChain
from experiments.main_table.run_misspec import (
    EPS_MIN,
    TAU,
    empirical_tv,
    margin_fields,
    probe_budget,
)
from quotient_learning.equivalence import TVThresholdEquivalenceTest
from quotient_learning.types import ContinuationSample, ObservationHistory


def _tv(p, q) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


class TestMisspecChainLaws:
    def test_subgap_pair_tv_is_tau_and_strictly_sub_gap(self):
        env = MisspecChain(tau=TAU)
        l0 = env.exact_continuation_law((0,))
        l1 = env.exact_continuation_law((1,))
        tv = _tv(l0, l1)
        assert tv == pytest.approx(TAU, abs=1e-12)
        # The misspecification regime: 0 < TV < eps_min/2 = 0.25.
        assert 0.0 < tv < 0.25

    def test_separated_pairs_tv_at_least_half(self):
        env = MisspecChain(tau=TAU)
        l0 = env.exact_continuation_law((0,))
        l1 = env.exact_continuation_law((1,))
        l2 = env.exact_continuation_law((2,))
        assert _tv(l0, l2) == pytest.approx(SEPARATED_TV, abs=1e-12)
        assert _tv(l1, l2) == pytest.approx(SEPARATED_TV, abs=1e-12)
        assert _tv(l0, l2) >= 0.5
        assert _tv(l1, l2) >= 0.5

    def test_separated_tv_is_tau_independent(self):
        # The 0.80 separation must not drift with tau in the valid range.
        for tau in (0.05, 0.10, 0.20, 0.40):
            env = MisspecChain(tau=tau)
            l1 = env.exact_continuation_law((1,))
            l2 = env.exact_continuation_law((2,))
            assert _tv(l1, l2) == pytest.approx(SEPARATED_TV, abs=1e-12)

    def test_laws_are_valid_distributions(self):
        env = MisspecChain(tau=TAU)
        for x in (0, 1, 2):
            law = env.exact_continuation_law((x,))
            assert len(law) == env.n_observations
            assert all(p > 0 for p in law)
            assert sum(law) == pytest.approx(1.0, abs=1e-12)

    def test_law_depends_only_on_last_symbol(self):
        # First-order Markov: any word's law equals the row of its last symbol.
        env = MisspecChain(tau=TAU)
        for word in [(2, 0), (1, 1), (0, 1, 2), (2, 2, 1)]:
            assert env.exact_continuation_law(word) == pytest.approx(
                env.exact_continuation_law((word[-1],)), abs=1e-15
            )

    def test_target_partition_has_three_classes_at_length_one(self):
        env = MisspecChain(tau=TAU)
        partition = env.target_partition(max_length=1)
        assert len(partition) == 3
        assert len(set(partition.values())) == 3  # all laws pairwise distinct
        assert env.metadata["ground_truth_n_classes"] == 3

    def test_epsilon_min_equals_tau(self):
        # The honest gap parameter is the sub-gap TV itself.
        for tau in (0.05, 0.10, 0.25):
            env = MisspecChain(tau=tau)
            assert env.epsilon_min(max_length=1) == pytest.approx(tau, abs=1e-12)

    def test_true_pairwise_tvs_helper(self):
        env = MisspecChain(tau=TAU)
        tvs = env.true_pairwise_tvs()
        assert tvs[frozenset(((0,), (1,)))] == pytest.approx(TAU, abs=1e-12)
        assert tvs[frozenset(((0,), (2,)))] == pytest.approx(SEPARATED_TV, abs=1e-12)
        assert tvs[frozenset(((1,), (2,)))] == pytest.approx(SEPARATED_TV, abs=1e-12)

    def test_invalid_tau_rejected(self):
        for bad in (0.0, -0.1, 0.41, 1.0):
            with pytest.raises(ValueError):
                MisspecChain(tau=bad)

    def test_rollout_frequencies_match_exact_laws(self):
        env = MisspecChain(tau=TAU, seed=7)
        obs = [env.reset(seed=7)]
        for _ in range(30_000):
            obs.append(env.step(0).observation)
        counts = {x: np.zeros(env.n_observations) for x in range(env.n_observations)}
        for cur, nxt in itertools.pairwise(obs):
            counts[cur][nxt] += 1
        for x in range(env.n_observations):
            emp = counts[x] / counts[x].sum()
            assert _tv(emp, env.exact_continuation_law((x,))) < 0.02

    def test_k1_alphabet_full_support(self):
        env = MisspecChain(tau=TAU)
        alphabet = env.continuation_alphabet(k=1)
        assert alphabet.size == 3
        assert alphabet.full_support


class TestMarginComputation:
    CATS = frozenset({(0,), (1,), (2,)})

    def test_empirical_tv_hand_computed(self):
        # p = (0.5, 0.3, 0.2), q = (0.3, 0.3, 0.4) -> TV = 0.5*(0.2+0+0.2) = 0.2
        counts_a = {(0,): 50, (1,): 30, (2,): 20}
        counts_b = {(0,): 30, (1,): 30, (2,): 40}
        assert empirical_tv(counts_a, counts_b, self.CATS) == pytest.approx(
            0.2, abs=1e-12
        )

    def test_empirical_tv_missing_category_counts_zero(self):
        counts_a = {(0,): 60, (1,): 40}           # (2,) absent -> 0
        counts_b = {(0,): 10, (1,): 40, (2,): 50}
        # p = (0.6, 0.4, 0.0), q = (0.1, 0.4, 0.5) -> TV = 0.5*(0.5+0+0.5) = 0.5
        assert empirical_tv(counts_a, counts_b, self.CATS) == pytest.approx(
            0.5, abs=1e-12
        )

    def test_empirical_tv_rejects_empty_histogram(self):
        with pytest.raises(ValueError):
            empirical_tv({}, {(0,): 10}, self.CATS)

    def test_margin_fields_near_threshold_flagged(self):
        # tv = 0.2 vs threshold 0.25: margin 0.05 < eps/8 = 0.0625 -> flagged
        fields = margin_fields(0.2, eps_min=0.5)
        assert fields["threshold"] == pytest.approx(0.25)
        assert fields["margin"] == pytest.approx(0.05, abs=1e-12)
        assert fields["near_threshold"] is True

    def test_margin_fields_far_from_threshold_not_flagged(self):
        # A separated pair: tv = 0.8 -> margin 0.55.
        fields = margin_fields(0.8, eps_min=0.5)
        assert fields["margin"] == pytest.approx(0.55, abs=1e-12)
        assert fields["near_threshold"] is False
        # The probe's sub-gap pair: tv ~ 0.10 -> margin 0.15 — the wrong
        # merge is confident, NOT a borderline call.
        fields = margin_fields(TAU, eps_min=EPS_MIN)
        assert fields["margin"] == pytest.approx(0.15, abs=1e-12)
        assert fields["near_threshold"] is False

    def test_margin_exactly_at_rule_boundary_not_flagged(self):
        # Strict inequality: margin == eps/8 must not flag.
        eps = 0.5
        tv = eps / 2.0 - eps / 8.0  # margin exactly eps/8
        assert margin_fields(tv, eps)["near_threshold"] is False

    def test_empirical_tv_matches_tv_test_statistic(self):
        # The reporting statistic must equal the statistic that drives the
        # certified verdict (TVThresholdEquivalenceTest.test_statistic).
        env = MisspecChain(tau=TAU)
        alphabet = env.continuation_alphabet(k=1)
        cont_a = ((0,),) * 25 + ((1,),) * 15 + ((2,),) * 10
        cont_b = ((0,),) * 10 + ((1,),) * 25 + ((2,),) * 15
        sample_a = ContinuationSample(
            history=ObservationHistory((0,)), continuations=cont_a, window_length=1
        )
        sample_b = ContinuationSample(
            history=ObservationHistory((1,)), continuations=cont_b, window_length=1
        )
        out = TVThresholdEquivalenceTest(alphabet, EPS_MIN).test(
            sample_a, sample_b, alpha=0.01
        )
        expected = empirical_tv(
            Counter(cont_a), Counter(cont_b), alphabet.categories
        )
        assert out.test_statistic == pytest.approx(expected, abs=1e-12)


class TestProbeBudget:
    def test_budget_clears_gate_in_expectation(self):
        m_star, n_trajectories = probe_budget()
        assert m_star >= 30  # the estimator's CLT floor is dominated by m*
        # Expected per-context inference count (uniform initial symbol,
        # 50/50 split) must clear the gate with head-room.
        assert n_trajectories / 6.0 >= m_star
