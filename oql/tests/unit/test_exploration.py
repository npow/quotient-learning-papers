"""Unit tests for count-based exploration policy (T-new).

Verifies that:
1. The policy visits all actions on a simple 2-state deterministic environment
   within K steps.
2. The per-episode obs-window reset works correctly (counts persist across
   episodes, window clears between episodes).
3. The factory produces independent policies (separate counters).
"""

from __future__ import annotations

import numpy as np
import pytest

from quotient_learning.collection import count_based_exploration_policy


# ---------------------------------------------------------------------------
# Minimal stub environment (no quotient_learning dependency)
# ---------------------------------------------------------------------------

class _TwoStateEnv:
    """Deterministic 2-obs, 3-action environment.

    State alternates with each action taken.  Observation = state.
    Used to check action coverage without needing a full POMDPEnvironment.
    """

    n_actions = 3
    n_observations = 2

    def __init__(self) -> None:
        self._state = 0

    def reset(self) -> int:
        self._state = 0
        return self._state

    def step(self, action: int) -> int:
        self._state = (self._state + action + 1) % self.n_observations
        return self._state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rollout(policy, env: _TwoStateEnv, n_steps: int, rng: np.random.Generator) -> list[int]:
    """Run the policy for n_steps, resetting the obs window at episode start."""
    obs = env.reset()
    if hasattr(policy, "_obs_window"):
        policy._obs_window.clear()
    actions_taken = []
    for _ in range(n_steps):
        a = policy(obs, rng)
        actions_taken.append(a)
        obs = env.step(a)
    return actions_taken


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCountBasedPolicy:

    def test_all_actions_visited_within_k_steps(self) -> None:
        """The policy must try every action at least once within 3 * n_actions steps."""
        n_actions = 3
        K = 3 * n_actions  # generous budget
        policy = count_based_exploration_policy(n_actions=n_actions)
        env = _TwoStateEnv()
        rng = np.random.default_rng(42)
        actions = _rollout(policy, env, K, rng)
        assert set(actions) == set(range(n_actions)), (
            f"Expected all {n_actions} actions visited within {K} steps, got: {set(actions)}"
        )

    def test_all_actions_visited_within_2n_steps(self) -> None:
        """With n_actions actions and a bounded horizon, all actions are visited
        within 2 * n_actions steps."""
        n_actions = 4
        policy = count_based_exploration_policy(n_actions=n_actions, history_max_length=1)
        env = _TwoStateEnv()
        rng = np.random.default_rng(0)
        # Use 2 * n_actions steps to account for observation variation.
        actions = _rollout(policy, env, n_steps=2 * n_actions, rng=rng)
        assert len(set(actions)) == n_actions, (
            f"Expected {n_actions} distinct actions in first {2 * n_actions} steps, got {set(actions)}"
        )

    def test_counts_persist_across_episodes(self) -> None:
        """Counts accumulated in episode 1 influence episode 2."""
        n_actions = 2
        policy = count_based_exploration_policy(n_actions=n_actions, history_max_length=1)
        env = _TwoStateEnv()
        rng = np.random.default_rng(7)

        # Episode 1: take only action 0 (hack: override the policy counter after).
        # Instead, just run one episode and inspect the counter.
        actions_ep1 = _rollout(policy, env, n_steps=2, rng=rng)

        # Snapshot counts after episode 1.
        counts_ep1 = {k: list(v) for k, v in policy._counts.items()}

        # Episode 2: start fresh window but counts should still be there.
        if hasattr(policy, "_obs_window"):
            policy._obs_window.clear()
        obs = env.reset()
        rng2 = np.random.default_rng(99)
        _ = policy(obs, rng2)

        # Counts for keys seen in ep1 must be >= the ep1 snapshot values.
        for key, ep1_vals in counts_ep1.items():
            current_vals = list(policy._counts[key])
            assert all(c >= e for c, e in zip(current_vals, ep1_vals)), (
                f"Counts for key {key} regressed between episodes: {ep1_vals} -> {current_vals}"
            )

    def test_obs_window_resets_between_episodes(self) -> None:
        """After clearing _obs_window (as collect_trajectories does), the
        history key used in the next episode starts fresh."""
        n_actions = 2
        policy = count_based_exploration_policy(n_actions=n_actions, history_max_length=2)
        assert hasattr(policy, "_obs_window"), "policy must expose _obs_window for reset"
        env = _TwoStateEnv()
        rng = np.random.default_rng(13)

        # Run episode 1.
        _rollout(policy, env, n_steps=3, rng=rng)
        window_after_ep1 = list(policy._obs_window)
        assert len(window_after_ep1) > 0

        # Reset window (as collect_trajectories would).
        policy._obs_window.clear()
        assert len(policy._obs_window) == 0, "obs_window should be empty after clear()"

    def test_independent_factory_calls_give_independent_counters(self) -> None:
        """Two calls to the factory produce policies with separate counters."""
        n_actions = 3
        p1 = count_based_exploration_policy(n_actions=n_actions)
        p2 = count_based_exploration_policy(n_actions=n_actions)
        env = _TwoStateEnv()
        rng = np.random.default_rng(55)

        # Advance p1.
        _rollout(p1, env, n_steps=5, rng=rng)

        # p2 should still have all-zero counts.
        for vals in p2._counts.values():
            assert all(v == 0 for v in vals), "p2 should have independent (zero) counts"

    def test_explores_all_actions_within_budget_two_state_env(self) -> None:
        """Integration-style: 2-state, 3-action env — all actions within 9 steps."""
        n_actions = 3
        policy = count_based_exploration_policy(n_actions=n_actions, history_max_length=2)
        env = _TwoStateEnv()
        rng = np.random.default_rng(2026)
        actions = _rollout(policy, env, n_steps=9, rng=rng)
        assert set(actions) == {0, 1, 2}, (
            f"Expected all 3 actions in 9 steps, got {set(actions)}"
        )
