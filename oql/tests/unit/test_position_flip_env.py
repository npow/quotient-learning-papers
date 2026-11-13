"""Position-flip chain: estimand-mismatch benchmark invariants (B4)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from environments.layer1_discrete.position_flip import (
    P_HI,
    P_LO,
    PositionFlipChain,
)
from quotient_learning.config import seed_everything
from quotient_learning.equivalence import (
    GTestEquivalenceTest,
    NotCertifiedWarning,
    TVThresholdEquivalenceTest,
)
from quotient_learning.estimator import QuotientEstimator


class TestModel:
    def test_parity_dependent_kernel(self) -> None:
        env = PositionFlipChain(seed=0)
        assert env.transition_prob_next_one(0, 0) == P_HI
        assert env.transition_prob_next_one(1, 0) == P_LO
        assert env.transition_prob_next_one(0, 1) == P_LO
        assert env.transition_prob_next_one(1, 1) == P_HI

    def test_analytic_r0_gap(self) -> None:
        assert PositionFlipChain.analytic_tv_gap_r0() == pytest.approx(0.8)

    def test_marginals_uniform_every_position(self) -> None:
        # p_{t+1} = P(1|0,t)(1-p_t) + P(1|1,t)p_t stays 1/2 from p_0 = 1/2.
        p = 0.5
        for t in range(10):
            hi, lo = (P_HI, P_LO) if t % 2 == 0 else (P_LO, P_HI)
            p = hi * (1 - p) + lo * p
            assert p == pytest.approx(0.5)

    def test_pooled_law_uniform_empirically(self) -> None:
        env = PositionFlipChain(seed=1)
        rng_counts = {0: [0, 0], 1: [0, 0]}
        for _ in range(2000):
            cur = env.reset()
            for _ in range(20):
                nxt = env.step(0).observation
                rng_counts[cur][nxt] += 1
                cur = nxt
        for cur in (0, 1):
            total = sum(rng_counts[cur])
            frac1 = rng_counts[cur][1] / total
            assert frac1 == pytest.approx(0.5, abs=0.02)


class TestHeadToHead:
    def test_certified_separates_exploratory_merges(self) -> None:
        seed_everything(0)
        env = PositionFlipChain(seed=0)
        alphabet = env.continuation_alphabet(1)
        certified = QuotientEstimator(
            env=env,
            equivalence_test=TVThresholdEquivalenceTest(alphabet, 0.75),
            delta=0.05, epsilon_min=0.75, power_beta=0.05,
            certified_independent=True, selection_fraction=0.5,
            independent_context_end=0, include_root=False,
            n_trajectories=4000, window_length=1, history_max_length=1,
            min_history_occurrences=10, max_steps_per_episode=40,
        )
        q_cert = certified.run(seed=0)
        assert q_cert.n_classes == 2
        assert q_cert.undetermined_rate == 0.0

        seed_everything(0)
        env2 = PositionFlipChain(seed=0)
        exploratory = QuotientEstimator(
            env=env2,
            equivalence_test=GTestEquivalenceTest(),
            n_trajectories=400, window_length=1, history_max_length=1,
            min_history_occurrences=50, alpha=0.05,
            max_steps_per_episode=40, include_root=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotCertifiedWarning)
            q_expl = exploratory.run(seed=0)
        # The pooled estimand is genuinely one class: the scan converges to
        # a merge of the pair that the r=0 estimand separates.
        assert q_expl.n_classes == 1


def test_rng_determinism() -> None:
    a = PositionFlipChain(seed=7)
    b = PositionFlipChain(seed=7)
    seq_a = [a.reset()] + [a.step(0).observation for _ in range(30)]
    seq_b = [b.reset()] + [b.step(0).observation for _ in range(30)]
    assert np.array_equal(seq_a, seq_b)
