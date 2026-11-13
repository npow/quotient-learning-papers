"""Unit tests for action-aware history types and sampling functions.

Key property verified: action-aware histories are a refinement of observation-only
histories, meaning they can distinguish histories that obs-only conflates.

Specifically: two action-aware histories AH1, AH2 that project to the same
ObservationHistory must be tracked SEPARATELY by enumerate_observed_action_aware_histories.
This is the property that enables finer quotient recovery on Tiger.
"""

from __future__ import annotations

import pytest

from quotient_learning.dataset import Trajectory, TrajectoryDataset
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.sampling import (
    enumerate_observed_action_aware_histories,
    enumerate_observed_histories,
    extract_action_aware_continuations,
    extract_continuation_samples,
)
from quotient_learning.types import ActionAwareHistory, ObservationHistory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_dataset() -> TrajectoryDataset:
    """Small dataset with 2 observations and 2 actions for unit testing."""
    trajs = []
    t1 = Trajectory(
        observations=(0, 1, 0, 1, 0),
        actions=(0, 1, 0, 1),
        rewards=(0.0, 0.0, 0.0, 0.0),
        seed=1,
    )
    t2 = Trajectory(
        observations=(1, 0, 1, 0, 1),
        actions=(1, 0, 1, 0),
        rewards=(0.0, 0.0, 0.0, 0.0),
        seed=2,
    )
    return TrajectoryDataset.from_iterable([t1, t2], "simple", "uniform_random")


@pytest.fixture
def mixed_action_dataset() -> TrajectoryDataset:
    """Dataset where the SAME observation appears with DIFFERENT actions.

    Trajectory 1: obs=(0,0,0,0,0), actions=(0,0,0,0)  -- obs=0 with action=0
    Trajectory 2: obs=(0,1,0,1,0), actions=(1,0,1,0)  -- obs=0 with action=0 AND action=1

    In t2, step 0: (obs=0, act=1); step 2: (obs=0, act=1); step 1: (obs=1, act=0).
    Combined, obs=0 appears with action=0 (from t1 and t2-step1-after) and action=1 (t2-step0).
    """
    t1 = Trajectory(
        observations=(0, 0, 0, 0, 0),
        actions=(0, 0, 0, 0),
        rewards=(0.0,) * 4,
        seed=1,
    )
    # obs=0 at step 0 with action=1, then at step 2 with action=1
    # obs=1 at step 1 with action=0
    t2 = Trajectory(
        observations=(0, 1, 0, 1, 0),
        actions=(1, 0, 1, 0),
        rewards=(0.0,) * 4,
        seed=2,
    )
    return TrajectoryDataset.from_iterable([t1, t2], "mixed_action", "uniform_random")


@pytest.fixture
def tiger_like_dataset() -> TrajectoryDataset:
    """Dataset that mimics Tiger: obs={0,1}, actions={0=listen, 1=open}.

    After action=0 (listen), obs stays consistent with state (informative).
    After action=1 (open), obs is random (episode reset, uninformative).
    """
    trajs = []
    seed = 0
    # Hear-left sequences (listen only)
    for _ in range(50):
        trajs.append(Trajectory(
            observations=(0, 0, 0, 0, 0),
            actions=(0, 0, 0, 0),
            rewards=(-1.0,) * 4,
            seed=seed,
        ))
        seed += 1
    # Hear-right sequences (listen only)
    for _ in range(50):
        trajs.append(Trajectory(
            observations=(1, 1, 1, 1, 1),
            actions=(0, 0, 0, 0),
            rewards=(-1.0,) * 4,
            seed=seed,
        ))
        seed += 1
    # After open (action=1), reset: obs is random
    for i in range(50):
        obs_after = i % 2
        trajs.append(Trajectory(
            observations=(0, obs_after, obs_after),
            actions=(1, 0),
            rewards=(10.0, -1.0),
            seed=seed,
        ))
        seed += 1
        trajs.append(Trajectory(
            observations=(1, obs_after, obs_after),
            actions=(1, 0),
            rewards=(-100.0, -1.0),
            seed=seed,
        ))
        seed += 1
    return TrajectoryDataset.from_iterable(trajs, "tiger_like", "uniform_random")


# ---------------------------------------------------------------------------
# ActionAwareHistory type tests
# ---------------------------------------------------------------------------

class TestActionAwareHistory:
    def test_empty_history_is_valid(self) -> None:
        h = ActionAwareHistory(())
        assert h.length == 0

    def test_history_is_hashable(self) -> None:
        h1 = ActionAwareHistory(((0, 1), (1, 0)))
        h2 = ActionAwareHistory(((0, 1), (1, 0)))
        assert hash(h1) == hash(h2)
        assert {h1, h2} == {h1}

    def test_history_is_immutable(self) -> None:
        h = ActionAwareHistory(((0, 1),))
        with pytest.raises(Exception):
            h.steps = ((1, 0),)  # type: ignore[misc]

    def test_length_correct(self) -> None:
        h = ActionAwareHistory(((0, 1), (1, 0), (0, 2)))
        assert h.length == 3

    def test_extend_appends_step(self) -> None:
        h = ActionAwareHistory(((0, 1),))
        h2 = h.extend(1, 0)
        assert h2 == ActionAwareHistory(((0, 1), (1, 0)))
        assert h == ActionAwareHistory(((0, 1),))  # original unchanged

    def test_to_observation_history_projects_correctly(self) -> None:
        h = ActionAwareHistory(((0, 1), (1, 0), (0, 2)))
        obs_h = h.to_observation_history()
        assert obs_h == ObservationHistory((0, 1, 0))

    def test_empty_to_observation_history(self) -> None:
        h = ActionAwareHistory(())
        obs_h = h.to_observation_history()
        assert obs_h == ObservationHistory(())

    def test_different_actions_same_obs_are_different_histories(self) -> None:
        h_listen = ActionAwareHistory(((0, 0),))   # obs=0, action=0 (listen)
        h_open = ActionAwareHistory(((0, 1),))      # obs=0, action=1 (open-left)
        assert h_listen != h_open
        # But they project to the same ObservationHistory
        assert h_listen.to_observation_history() == h_open.to_observation_history()

    def test_rejects_invalid_observation(self) -> None:
        with pytest.raises(ValueError):
            ActionAwareHistory(((-1, 0),))

    def test_rejects_invalid_action(self) -> None:
        with pytest.raises(ValueError):
            ActionAwareHistory(((0, -1),))

    def test_none_action_is_valid(self) -> None:
        h = ActionAwareHistory(((0, None),))
        assert h.length == 1


# ---------------------------------------------------------------------------
# enumerate_observed_action_aware_histories tests
# ---------------------------------------------------------------------------

class TestEnumerateActionAwareHistories:
    def test_includes_empty_history(self, simple_dataset: TrajectoryDataset) -> None:
        hs = enumerate_observed_action_aware_histories(
            simple_dataset, max_length=1, min_occurrences=1
        )
        assert ActionAwareHistory(()) in hs

    def test_length_zero_always_present(self, simple_dataset: TrajectoryDataset) -> None:
        hs = enumerate_observed_action_aware_histories(
            simple_dataset, max_length=2, min_occurrences=1
        )
        assert ActionAwareHistory(()) in hs

    def test_min_occurrences_filters(self, simple_dataset: TrajectoryDataset) -> None:
        hs = enumerate_observed_action_aware_histories(
            simple_dataset, max_length=1, min_occurrences=1000
        )
        assert hs == []

    def test_same_obs_different_interleaved_action_are_distinct_keys(self) -> None:
        """Depth-2 contexts with the same observations but different interleaved
        actions are distinct keys. (Under the revised convention a context ends
        at its final observation; the action that refines a length-2 context is
        the one BETWEEN its two observations.)"""
        t_a = Trajectory(observations=(0, 0, 0), actions=(0, 0), rewards=(0.0, 0.0), seed=1)
        t_b = Trajectory(observations=(0, 0, 0), actions=(1, 1), rewards=(0.0, 0.0), seed=2)
        ds = TrajectoryDataset.from_iterable([t_a, t_b], "aa_keys", "uniform_random")
        hs = enumerate_observed_action_aware_histories(ds, max_length=2, min_occurrences=1)
        h_via_act0 = ActionAwareHistory(((0, 0), (0, None)))
        h_via_act1 = ActionAwareHistory(((0, 1), (0, None)))
        assert h_via_act0 in hs, f"AH(((0,0),(0,None))) should be in enumeration; got {hs}"
        assert h_via_act1 in hs, f"AH(((0,1),(0,None))) should be in enumeration; got {hs}"
        # Both project to the same obs-only history
        assert h_via_act0.to_observation_history() == h_via_act1.to_observation_history()

    def test_aa_enumeration_at_least_as_many_as_obs_only(
        self, simple_dataset: TrajectoryDataset
    ) -> None:
        """Action-aware enumeration has >= entries than obs-only enumeration
        (ignoring the empty history difference), because each obs-only history
        can be split by action into one or more action-aware histories."""
        aa_hs = enumerate_observed_action_aware_histories(
            simple_dataset, max_length=2, min_occurrences=1
        )
        obs_hs = enumerate_observed_histories(
            simple_dataset, max_length=2, min_occurrences=1
        )
        assert len(aa_hs) >= len(obs_hs), (
            f"Action-aware enumeration ({len(aa_hs)}) should have >= entries "
            f"than obs-only ({len(obs_hs)})"
        )


# ---------------------------------------------------------------------------
# extract_action_aware_continuations tests
# ---------------------------------------------------------------------------

class TestExtractActionAwareContinuations:
    def test_empty_history_yields_all_windows(self, simple_dataset: TrajectoryDataset) -> None:
        h = ActionAwareHistory(())
        cs = extract_action_aware_continuations(simple_dataset, h, window_length=2)
        assert cs.n_samples > 0
        assert cs.window_length == 2
        for cont in cs.continuations:
            assert len(cont) == 2

    def test_window_length_zero_raises(self, simple_dataset: TrajectoryDataset) -> None:
        h = ActionAwareHistory(())
        with pytest.raises(ValueError):
            extract_action_aware_continuations(simple_dataset, h, window_length=0)

    def test_action_aware_history_filters_correctly(
        self, simple_dataset: TrajectoryDataset
    ) -> None:
        h = ActionAwareHistory(((0, 0),))
        cs = extract_action_aware_continuations(simple_dataset, h, window_length=1)
        for cont in cs.continuations:
            assert len(cont) == 1

    def test_nonexistent_history_gives_zero_samples(
        self, simple_dataset: TrajectoryDataset
    ) -> None:
        h = ActionAwareHistory(((0, 99),))
        cs = extract_action_aware_continuations(simple_dataset, h, window_length=1)
        assert cs.n_samples == 0

    def test_continuation_carries_immediate_action(
        self, mixed_action_dataset: TrajectoryDataset
    ) -> None:
        """Under the revised convention the immediate action a_t is the first
        symbol of the continuation, so a context followed by different actions
        in the data yields continuation categories distinguished by that
        action — the refinement lives in the continuation law, not in a
        trailing context action."""
        h = ActionAwareHistory(((0, None),))
        cs = extract_action_aware_continuations(mixed_action_dataset, h, window_length=1)
        assert cs.n_samples > 0
        first_actions = {cont[0][0] for cont in cs.continuations}
        # obs=0 is followed by action 0 (t1, t2 step 1) and action 1 (t2 steps 0, 2)
        assert first_actions == {0, 1}, (
            f"continuations after context (0,) should carry both actions; got {first_actions}"
        )

    def test_contexts_with_different_interleaved_actions_differ(self) -> None:
        """Depth-2 contexts whose interleaved action differs pool different
        continuation samples."""
        t_a = Trajectory(observations=(0, 0, 5), actions=(0, 2), rewards=(0.0, 0.0), seed=1)
        t_b = Trajectory(observations=(0, 0, 6), actions=(1, 3), rewards=(0.0, 0.0), seed=2)
        ds = TrajectoryDataset.from_iterable([t_a, t_b], "aa_diff", "uniform_random")
        cs_a = extract_action_aware_continuations(
            ds, ActionAwareHistory(((0, 0), (0, None))), window_length=1
        )
        cs_b = extract_action_aware_continuations(
            ds, ActionAwareHistory(((0, 1), (0, None))), window_length=1
        )
        assert cs_a.continuations == (((2, 5),),)
        assert cs_b.continuations == (((3, 6),),)


# ---------------------------------------------------------------------------
# Finer-or-equal partition property
# ---------------------------------------------------------------------------

class TestActionAwareRefinement:
    """Verify that action-aware histories are a structural refinement of obs-only
    histories: each obs-only history H(o1,...,ok) can be split into one or more
    action-aware histories AH((o1,a1),...,(ok,ak)).

    This is tested at the enumeration level (structural property) rather than at
    the clustering level (which depends on data and statistical thresholds).
    """

    def test_aa_histories_project_to_obs_histories(
        self, simple_dataset: TrajectoryDataset
    ) -> None:
        """Every action-aware history's obs-projection should be an observed obs-history."""
        aa_hs = enumerate_observed_action_aware_histories(
            simple_dataset, max_length=2, min_occurrences=1
        )
        obs_hs = set(enumerate_observed_histories(
            simple_dataset, max_length=2, min_occurrences=1
        ))
        for h_aa in aa_hs:
            proj = h_aa.to_observation_history()
            assert proj in obs_hs, (
                f"Action-aware history {h_aa} projects to {proj} "
                f"which is not in obs-only enumeration"
            )

    def test_multiple_aa_histories_can_project_to_same_obs(
        self, mixed_action_dataset: TrajectoryDataset
    ) -> None:
        """Multiple action-aware histories with the same observation sequence but
        different actions should project to the SAME obs-only history."""
        h_act0 = ActionAwareHistory(((0, 0),))
        h_act1 = ActionAwareHistory(((0, 1),))
        assert h_act0.to_observation_history() == h_act1.to_observation_history(), (
            "Same obs, different action → same obs-only projection"
        )

    def test_aa_n_histories_gte_obs_n_histories(
        self, tiger_like_dataset: TrajectoryDataset
    ) -> None:
        """Action-aware enumeration produces >= distinct prefixes than obs-only."""
        aa_hs = enumerate_observed_action_aware_histories(
            tiger_like_dataset, max_length=2, min_occurrences=5
        )
        obs_hs = enumerate_observed_histories(
            tiger_like_dataset, max_length=2, min_occurrences=5
        )
        assert len(aa_hs) >= len(obs_hs), (
            f"Action-aware ({len(aa_hs)}) should enumerate >= histories "
            f"than obs-only ({len(obs_hs)})"
        )

    def test_estimator_action_aware_flag_runs(
        self, tiger_like_dataset: TrajectoryDataset
    ) -> None:
        """QuotientEstimator with action_aware=True should run without error."""
        from unittest.mock import MagicMock
        env = MagicMock()
        env.n_actions = 2
        env.n_observations = 2
        env.environment_id = "mock"

        est = QuotientEstimator(
            env=env,
            n_trajectories=10,
            window_length=1,
            history_max_length=1,
            min_history_occurrences=1,
            alpha=0.05,
            action_aware=True,
        )
        q = est.estimate(tiger_like_dataset)
        assert q.n_classes >= 0  # valid quotient returned

    def test_estimator_action_aware_finer_or_equal_tiger_like(
        self, tiger_like_dataset: TrajectoryDataset
    ) -> None:
        """On tiger-like data with sufficient occurrences, action-aware should
        find >= as many classes as obs-only when both have enough data.

        This is a best-effort test with lenient min_history_occurrences so that
        both modes have enough data to make meaningful comparisons.
        """
        from unittest.mock import MagicMock
        env = MagicMock()
        env.n_actions = 2
        env.n_observations = 2
        env.environment_id = "mock"

        common = dict(
            window_length=1,
            history_max_length=1,
            min_history_occurrences=20,
            alpha=0.05,
        )

        obs_est = QuotientEstimator(env=env, n_trajectories=10, action_aware=False, **common)
        aa_est = QuotientEstimator(env=env, n_trajectories=10, action_aware=True, **common)

        q_obs = obs_est.estimate(tiger_like_dataset)
        q_aa = aa_est.estimate(tiger_like_dataset)

        # The tiger-like data is constructed so that action-aware histories
        # (obs, listen) and (obs, open) have DIFFERENT continuation distributions,
        # while obs-only histories (obs,) conflate them.
        # Action-aware should therefore recover more (or equal) distinct classes.
        assert q_aa.n_classes >= q_obs.n_classes, (
            f"On tiger-like data: action-aware n_classes={q_aa.n_classes} should be >= "
            f"obs-only n_classes={q_obs.n_classes}"
        )
