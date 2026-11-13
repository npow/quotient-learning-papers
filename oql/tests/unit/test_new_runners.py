"""Analytic derivations behind the new runners (REV-2, S4, S10).

These tests pin the closed-form quotients, gaps, and gate values the three
new runners (run_multiendpoint, run_nested, run_sweep_depth2) rely on,
asserted against the environment helpers -- no data collection.
"""

from __future__ import annotations

import itertools

import pytest

from environments.layer1_discrete.position_flip import (
    P_HI,
    P_LO,
    PositionFlipChain,
)
from environments.layer2_memory.noisy_depth2_kcycle import NoisyDepth2KCycle
from experiments.main_table import run_multiendpoint, run_nested, run_sweep_depth2
from quotient_learning.stats import required_samples
from quotient_learning.types import ObservationHistory

L_ODD = [0.8, 0.1, 0.1]  # the (m=2, nu=0.2) L_odd row shared by cur != 0


def _tv(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


@pytest.fixture(scope="module")
def depth2_env() -> NoisyDepth2KCycle:
    return NoisyDepth2KCycle(m=2, eta=0.1, nu=0.2, seed=0)


# ---------------------------------------------------------------------------
# REV-2: parity-chain quotient at r = 1 (multi-endpoint runner)
# ---------------------------------------------------------------------------

class TestParityChainMultiEndpoint:
    def test_r1_laws_are_parity_flipped(self):
        # r = 0 uses the even kernel: (0,)->Ber(0.9), (1,)->Ber(0.1).
        assert PositionFlipChain.transition_prob_next_one(0, 0) == P_HI
        assert PositionFlipChain.transition_prob_next_one(1, 0) == P_LO
        # r = 1 uses the odd kernel: laws swap, (0,)->Ber(0.1), (1,)->Ber(0.9).
        assert PositionFlipChain.transition_prob_next_one(0, 1) == P_LO
        assert PositionFlipChain.transition_prob_next_one(1, 1) == P_HI

    def test_r1_quotient_two_singletons_gap_08(self):
        p0 = PositionFlipChain.transition_prob_next_one(0, 1)
        p1 = PositionFlipChain.transition_prob_next_one(1, 1)
        assert p0 != p1  # two singleton classes at r = 1
        assert abs(p0 - p1) == pytest.approx(0.8)
        assert abs(p0 - p1) == pytest.approx(PositionFlipChain.analytic_tv_gap_r0())

    def test_runner_config_respects_gap_and_delta_split(self):
        assert run_multiendpoint.ENDPOINTS == (0, 1)
        assert run_multiendpoint.DELTA_PER_ENDPOINT == pytest.approx(0.025)
        assert run_multiendpoint.EPS_MIN <= 0.8
        assert run_multiendpoint.EXPECTED_CLASSES == 2
        run_multiendpoint.check_analytic_quotient()  # must not raise

    def test_per_endpoint_gate_value(self):
        # K = 2 contexts, alpha_0 = (delta/T)/(2K^2) = 0.025/8, d = 2.
        expected = required_samples(0.75, 0.025 / 8.0, 0.05, 2)
        assert expected == 112
        assert run_multiendpoint.M_STAR == expected


# ---------------------------------------------------------------------------
# S4: nested depth-2 quotient at r = 1 (nested runner)
# ---------------------------------------------------------------------------

class TestNestedDepth2:
    def test_twelve_eligible_contexts_five_classes(self, depth2_env):
        part = depth2_env.target_partition(2)
        assert len(part) == 12  # 3 length-1 + 9 positive-mass length-2
        assert len(set(part.values())) == 5
        assert run_nested.EXPECTED_CLASSES == 5
        assert run_nested.N_CONTEXTS == 12

    def test_nested_merge_across_lengths(self, depth2_env):
        # Length-1 and length-2 contexts genuinely share the L_odd law:
        # every (a, b) with b != 0, plus (1,) and (2,).
        members = [(1,), (2,)] + [
            (a, b) for a in range(3) for b in (1, 2)
        ]
        for w in members:
            law = depth2_env.exact_continuation_law(w)
            assert law == pytest.approx(L_ODD, abs=1e-12), w
        part = depth2_env.target_partition(2)
        cids = {part[ObservationHistory(w)] for w in members}
        assert len(cids) == 1  # one class spanning both context lengths

    def test_singleton_laws(self, depth2_env):
        # (0,) mixes rows over the stationary posterior of the hidden
        # predecessor: exactly [1/21, 10/21, 10/21].
        law0 = depth2_env.exact_continuation_law((0,))
        assert law0 == pytest.approx([1 / 21, 10 / 21, 10 / 21], abs=1e-12)
        # Two observed symbols pin the pair state: law(a, b) is the table row.
        assert depth2_env.exact_continuation_law((0, 0)) == pytest.approx(
            [1 / 3, 1 / 3, 1 / 3], abs=1e-12
        )
        eta = depth2_env.eta
        assert depth2_env.exact_continuation_law((1, 0)) == pytest.approx(
            [eta / 3, eta / 3, 1 - 2 * eta / 3], abs=1e-12
        )
        assert depth2_env.exact_continuation_law((2, 0)) == pytest.approx(
            [eta / 3, 1 - 2 * eta / 3, eta / 3], abs=1e-12
        )

    def test_min_between_class_tv_is_two_sevenths(self, depth2_env):
        part = depth2_env.target_partition(2)
        words = [h.observations for h in part]
        laws = {w: depth2_env.exact_continuation_law(w) for w in words}
        gaps = {
            (a, b): _tv(laws[a], laws[b])
            for a, b in itertools.combinations(words, 2)
            if part[ObservationHistory(a)] != part[ObservationHistory(b)]
        }
        min_pair = min(gaps, key=gaps.get)
        assert gaps[min_pair] == pytest.approx(2 / 7, abs=1e-12)
        assert set(min_pair) == {(0,), (0, 0)}
        # env helper agrees, and the runner's eps respects the gap model.
        assert depth2_env.epsilon_min(2) == pytest.approx(2 / 7, abs=1e-12)
        assert run_nested.EPS_MIN <= 2 / 7
        run_nested.check_analytic_quotient(depth2_env)  # must not raise

    def test_nested_gate_value(self):
        # K = 12, alpha_0 = delta/(2K^2) = 0.05/288, d = 3, eps = 0.28.
        expected = required_samples(run_nested.EPS_MIN, 0.05 / 288.0, 0.05, 3)
        assert expected == 1167
        assert run_nested.M_STAR == expected


# ---------------------------------------------------------------------------
# S10: merge-bearing sweep target at r = 0 (sweep runner)
# ---------------------------------------------------------------------------

class TestSweepDepth2Target:
    def test_r0_quotient_two_classes_with_genuine_same_pair(self, depth2_env):
        law1 = depth2_env.exact_continuation_law((1,))
        law2 = depth2_env.exact_continuation_law((2,))
        law0 = depth2_env.exact_continuation_law((0,))
        assert law1 == pytest.approx(law2, abs=1e-12)  # genuine SAME pair
        gap = _tv(law0, law1)
        assert gap == pytest.approx(0.752381, abs=1e-6)
        # The sweep reuses the certified eps for this env; gap model holds.
        from experiments.main_table.run_certified import eps_min_for

        assert eps_min_for("noisy_depth2") <= gap
        # Truth table used by the sweep's incorrect-merge counter.
        assert run_sweep_depth2.TRUTH[ObservationHistory((1,))] == \
            run_sweep_depth2.TRUTH[ObservationHistory((2,))]
        assert run_sweep_depth2.TRUTH[ObservationHistory((0,))] != \
            run_sweep_depth2.TRUTH[ObservationHistory((1,))]
        assert run_sweep_depth2.EXPECTED_CLASSES == 2

    def test_sweep_gate_value(self):
        # K = 3, alpha_0 = delta/(2K^2) = 0.05/18, d = 3, eps = 0.5.
        expected = required_samples(0.5, 0.05 / 18.0, 0.05, 3)
        assert expected == 278
        assert run_sweep_depth2.m_star() == expected

    def test_incorrect_merge_counter(self):
        h0, h1, h2 = (ObservationHistory((i,)) for i in range(3))
        # Correct recovery: no incorrect merges.
        assert run_sweep_depth2.incorrect_merges({h0: 0, h1: 1, h2: 1}) == 0
        # Full over-split: still no incorrect merges.
        assert run_sweep_depth2.incorrect_merges({h0: 0, h1: 1, h2: 2}) == 0
        # Merging (0,) into the L_odd class is the counted failure.
        assert run_sweep_depth2.incorrect_merges({h0: 0, h1: 0, h2: 0}) == 2
