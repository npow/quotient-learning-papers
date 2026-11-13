"""Integration tests for the .dpomdp parser and DecPOMDPxSingleAgentView (T068).

Tests:
- parse_dpomdp on dec_tiger.dpomdp returns a model with correct dimensions and
  valid probability distributions.
- DecPOMDPxSingleAgentView is a working POMDPEnvironment (reset + steps).
- QuotientEstimator on the single-agent view produces a finite quotient.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from environments.layer5_marl.dec_pomdpx import (
    DecPOMDPxModel,
    DecPOMDPxSingleAgentView,
    parse_dpomdp,
)
from quotient_learning.estimator import QuotientEstimator

# Path to bundled benchmark files
_BENCH = Path(__file__).parents[2] / "environments" / "layer5_marl" / "benchmarks"
DEC_TIGER_PATH = _BENCH / "dec_tiger.dpomdp"
COORD2X2_PATH = _BENCH / "coord2x2.dpomdp"


@pytest.mark.integration
class TestParseDPomdp:
    """Tests for parse_dpomdp on the Dec-Tiger benchmark."""

    def test_returns_dec_pomdp_model(self) -> None:
        model = parse_dpomdp(DEC_TIGER_PATH)
        assert isinstance(model, DecPOMDPxModel)

    def test_state_count(self) -> None:
        model = parse_dpomdp(DEC_TIGER_PATH)
        assert model.n_states == 2, f"expected 2 states, got {model.n_states}"

    def test_action_count_per_agent(self) -> None:
        model = parse_dpomdp(DEC_TIGER_PATH)
        assert model.n_agents == 2
        assert len(model.actions_per_agent[0]) == 3, (
            f"agent 0 should have 3 actions, got {len(model.actions_per_agent[0])}"
        )
        assert len(model.actions_per_agent[1]) == 3, (
            f"agent 1 should have 3 actions, got {len(model.actions_per_agent[1])}"
        )

    def test_obs_count_per_agent(self) -> None:
        model = parse_dpomdp(DEC_TIGER_PATH)
        assert len(model.obs_per_agent[0]) == 2
        assert len(model.obs_per_agent[1]) == 2

    def test_transition_rows_sum_to_one(self) -> None:
        model = parse_dpomdp(DEC_TIGER_PATH)
        # T[ja, s, :] should sum to 1 for all ja, s
        row_sums = model.T.sum(axis=2)  # shape (n_ja, S)
        np.testing.assert_allclose(
            row_sums, np.ones_like(row_sums), atol=1e-6,
            err_msg="T rows do not sum to 1",
        )

    def test_observation_rows_sum_to_one(self) -> None:
        model = parse_dpomdp(DEC_TIGER_PATH)
        # O[ja, s', :] should sum to 1 for all ja, s'
        row_sums = model.O.sum(axis=2)  # shape (n_ja, S)
        np.testing.assert_allclose(
            row_sums, np.ones_like(row_sums), atol=1e-6,
            err_msg="O rows do not sum to 1",
        )

    def test_rewards_are_finite(self) -> None:
        model = parse_dpomdp(DEC_TIGER_PATH)
        assert np.all(np.isfinite(model.R)), "R contains non-finite values"

    def test_start_sums_to_one(self) -> None:
        model = parse_dpomdp(DEC_TIGER_PATH)
        np.testing.assert_allclose(model.start.sum(), 1.0, atol=1e-6)

    def test_gamma(self) -> None:
        model = parse_dpomdp(DEC_TIGER_PATH)
        assert 0.0 < model.gamma < 1.0, f"unexpected gamma {model.gamma}"

    def test_listen_listen_obs_accuracy(self) -> None:
        """In listen-listen, hearing the correct tiger side should be ~72.25%."""
        model = parse_dpomdp(DEC_TIGER_PATH)
        # Joint action 0 = listen listen
        # State 0 = tiger-left; joint obs 0 = tiger-left tiger-left
        ja_ll = model.joint_action_index((0, 0))
        jo_ll = model.joint_obs_index((0, 0))
        jo_rr = model.joint_obs_index((1, 1))
        # Both hear left when tiger is left: 0.85 * 0.85 = 0.7225
        np.testing.assert_allclose(
            model.O[ja_ll, 0, jo_ll], 0.7225, atol=1e-6,
        )
        # Both hear right when tiger is right: 0.7225
        np.testing.assert_allclose(
            model.O[ja_ll, 1, jo_rr], 0.7225, atol=1e-6,
        )

    def test_coord2x2_parses(self) -> None:
        model = parse_dpomdp(COORD2X2_PATH)
        assert model.n_states == 2
        assert model.n_agents == 2
        assert len(model.actions_per_agent[0]) == 2
        assert len(model.obs_per_agent[0]) == 2


@pytest.mark.integration
class TestDecPOMDPxSingleAgentView:
    """Tests for the single-agent environment wrapper."""

    @pytest.fixture
    def dec_tiger_model(self) -> DecPOMDPxModel:
        return parse_dpomdp(DEC_TIGER_PATH)

    def test_is_pomdp_environment(self, dec_tiger_model: DecPOMDPxModel) -> None:
        from environments.base import POMDPEnvironment
        env = DecPOMDPxSingleAgentView(model=dec_tiger_model, agent_idx=0, seed=42)
        assert isinstance(env, POMDPEnvironment)

    def test_n_observations_and_actions(self, dec_tiger_model: DecPOMDPxModel) -> None:
        env = DecPOMDPxSingleAgentView(model=dec_tiger_model, agent_idx=0)
        assert env.n_observations == 2
        assert env.n_actions == 3

    def test_reset_returns_valid_obs(self, dec_tiger_model: DecPOMDPxModel) -> None:
        env = DecPOMDPxSingleAgentView(model=dec_tiger_model, agent_idx=0, seed=0)
        obs = env.reset()
        assert 0 <= obs < env.n_observations

    def test_step_returns_valid_result(self, dec_tiger_model: DecPOMDPxModel) -> None:
        from environments.base import StepResult
        env = DecPOMDPxSingleAgentView(model=dec_tiger_model, agent_idx=0, seed=1)
        env.reset(seed=1)
        result = env.step(0)  # action=listen
        assert isinstance(result, StepResult)
        assert 0 <= result.observation < env.n_observations
        assert np.isfinite(result.reward)

    def test_ten_steps_complete_without_error(self, dec_tiger_model: DecPOMDPxModel) -> None:
        env = DecPOMDPxSingleAgentView(model=dec_tiger_model, agent_idx=0, seed=2)
        env.reset(seed=2)
        for step in range(10):
            result = env.step(step % env.n_actions)
            assert 0 <= result.observation < env.n_observations
            assert np.isfinite(result.reward)
            if result.terminated or result.truncated:
                env.reset()

    def test_episode_truncates_at_max_steps(self, dec_tiger_model: DecPOMDPxModel) -> None:
        env = DecPOMDPxSingleAgentView(
            model=dec_tiger_model, agent_idx=0, max_steps=5, seed=3
        )
        env.reset()
        truncated_seen = False
        for _ in range(10):
            result = env.step(0)  # always listen
            if result.truncated:
                truncated_seen = True
                break
        assert truncated_seen, "Expected truncation within 10 steps (max_steps=5)"

    def test_agent_1_view(self, dec_tiger_model: DecPOMDPxModel) -> None:
        env = DecPOMDPxSingleAgentView(model=dec_tiger_model, agent_idx=1, seed=7)
        obs = env.reset()
        assert 0 <= obs < env.n_observations
        result = env.step(0)
        assert 0 <= result.observation < env.n_observations


@pytest.mark.integration
@pytest.mark.slow
class TestDecPOMDPxQuotientEstimation:
    """Quotient estimator on the Dec-Tiger single-agent view."""

    def test_quotient_estimator_returns_finite_quotient(self) -> None:
        model = parse_dpomdp(DEC_TIGER_PATH)
        env = DecPOMDPxSingleAgentView(model=model, agent_idx=0, seed=0)
        est = QuotientEstimator(
            env=env,
            n_trajectories=500,
            window_length=2,
            history_max_length=1,
            min_history_occurrences=30,
            alpha=0.01,
            max_steps_per_episode=8,
        )
        q = est.run(seed=0)
        assert q.n_classes >= 1, f"Expected at least 1 quotient class, got {q.n_classes}"
        assert q.n_classes <= 6, (
            f"Expected at most 6 quotient classes, got {q.n_classes} "
            f"(Dec-Tiger has 2 distinguishable histories)"
        )
