"""Integration tests for PO-MuJoCo wrappers (T047, T048).

These tests are skipped when mujoco / gymnasium are not installed.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")
gymnasium = pytest.importorskip("gymnasium")


from environments.layer3_continuous.po_halfcheetah import POHalfCheetah  # noqa: E402
from environments.layer3_continuous.po_ant import POAnt  # noqa: E402
from quotient_learning.estimator import QuotientEstimator  # noqa: E402


@pytest.mark.integration
class TestPOHalfCheetah:
    """End-to-end tests for POHalfCheetah."""

    def test_reset_and_step_30(self) -> None:
        """Reset then step 30 times; all observations must be finite and in alphabet."""
        env = POHalfCheetah(bins_per_dim=3, warmup_episodes=2, max_steps=50, seed=0)
        obs = env.reset(seed=0)
        assert isinstance(obs, int)
        assert 0 <= obs < env.n_observations
        for _ in range(30):
            result = env.step(0)
            assert isinstance(result.observation, int)
            assert 0 <= result.observation < env.n_observations, (
                f"observation {result.observation} out of [0, {env.n_observations})"
            )
        env.close()

    def test_quotient_estimator_runs(self) -> None:
        """QuotientEstimator on POHalfCheetah should complete and return n_classes >= 1."""
        env = POHalfCheetah(bins_per_dim=3, warmup_episodes=2, max_steps=50, seed=0)
        est = QuotientEstimator(
            env,
            n_trajectories=100,
            window_length=1,
            history_max_length=1,
            min_history_occurrences=10,
            alpha=0.01,
            max_steps_per_episode=20,
        )
        q = est.run(seed=0)
        assert q.n_classes >= 1
        env.close()

    def test_n_observations_capped(self) -> None:
        """n_observations must be capped and not blow up to bins_per_dim^qpos_dim."""
        env = POHalfCheetah(bins_per_dim=4, warmup_episodes=1, max_steps=20, seed=0)
        # bins_per_dim=4, qpos_dim=8 -> 4^8 = 65536 uncapped; must be <=64.
        assert env.n_observations <= 64
        env.close()


@pytest.mark.integration
class TestPOAnt:
    """End-to-end tests for POAnt."""

    def test_reset_and_step_30(self) -> None:
        """Reset then step 30 times; all observations must be finite and in alphabet."""
        env = POAnt(bins_per_dim=3, warmup_episodes=2, max_steps=50, seed=0)
        obs = env.reset(seed=0)
        assert isinstance(obs, int)
        assert 0 <= obs < env.n_observations
        for _ in range(30):
            result = env.step(0)
            assert isinstance(result.observation, int)
            assert 0 <= result.observation < env.n_observations, (
                f"observation {result.observation} out of [0, {env.n_observations})"
            )
        env.close()

    def test_quotient_estimator_runs(self) -> None:
        """QuotientEstimator on POAnt should complete and return n_classes >= 1."""
        env = POAnt(bins_per_dim=3, warmup_episodes=2, max_steps=50, seed=0)
        est = QuotientEstimator(
            env,
            n_trajectories=100,
            window_length=1,
            history_max_length=1,
            min_history_occurrences=10,
            alpha=0.01,
            max_steps_per_episode=20,
        )
        q = est.run(seed=0)
        assert q.n_classes >= 1
        env.close()

    def test_n_observations_capped(self) -> None:
        """n_observations must be capped and not blow up to bins_per_dim^qpos_dim."""
        env = POAnt(bins_per_dim=3, warmup_episodes=1, max_steps=20, seed=0)
        # bins_per_dim=3, qpos_dim=13 -> 3^13 = 1594323 uncapped; must be <=64.
        assert env.n_observations <= 64
        env.close()
