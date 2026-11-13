"""Regression tests for the certified-mode retention contract (review A1/A2).

The certified path must never drop a selection-half context on
inference-half evidence: a context whose inference count is below the gate
abstains on all of its pairs and stays in the reported partition, and the
Bonferroni denominator is the selection-half candidate count.  The
per-environment eps_min must respect the recorded analytic TV gap.
"""

from __future__ import annotations

import pytest

from environments.layer1_discrete.kcycle import KCyclePOMDP
from experiments.main_table.run_certified import (
    ANALYTIC_TV_GAPS,
    EPS_MIN,
    EPS_OVERRIDES,
    eps_min_for,
)
from quotient_learning.config import seed_everything
from quotient_learning.equivalence import TVThresholdEquivalenceTest
from quotient_learning.estimator import QuotientEstimator


class TestEpsMinForGapAssertion:
    def test_published_envs_pass(self) -> None:
        assert eps_min_for("tiger") == 0.45
        assert eps_min_for("kcycle_8") == 0.5
        assert eps_min_for("noisy_depth2") == 0.5

    def test_every_override_below_its_gap(self) -> None:
        for env, eps in EPS_OVERRIDES.items():
            assert env in ANALYTIC_TV_GAPS
            assert eps <= ANALYTIC_TV_GAPS[env]

    def test_violation_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(EPS_OVERRIDES, "tiger", 0.5)  # > gap 0.49
        with pytest.raises(ValueError, match="exceeds the analytic TV gap"):
            eps_min_for("tiger")

    def test_default_used_when_no_override(self) -> None:
        assert eps_min_for("hallway") == EPS_MIN  # no recorded gap: default


class TestLowCountContextsRetainedAndAbstaining:
    def test_underpowered_budget_abstains_never_drops(self) -> None:
        # N=400 puts every context's inference count near 25: below both the
        # 30-sample floor and m* (=451 at delta=0.05, K=8, d=8), and below
        # the pre-revision drop filter max(30, q//2).  The old code returned
        # an empty quotient here by silently dropping every context; the
        # contract requires all selected contexts retained, all pairs
        # abstaining, and no merges.
        seed_everything(0)
        env = KCyclePOMDP(8, seed=0)
        alphabet = env.continuation_alphabet(1)
        est = QuotientEstimator(
            env=env,
            equivalence_test=TVThresholdEquivalenceTest(alphabet, 0.5),
            delta=0.05, epsilon_min=0.5, power_beta=0.05,
            certified_independent=True, selection_fraction=0.5,
            independent_context_end=0, include_root=False,
            n_trajectories=400, window_length=1, history_max_length=1,
            min_history_occurrences=10, max_steps_per_episode=20,
        )
        q = est.run(seed=0)
        assert q.n_classes == 8, "selected contexts must be retained as singletons"
        assert q.undetermined_rate == 1.0, "all pairs must abstain below the gate"
        assert len(q.boundary_flags) > 0
        assert q.transition == {}, "certified mode must not emit the uncertified table"
