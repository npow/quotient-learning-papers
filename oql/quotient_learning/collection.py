"""Trajectory collection: exploration policies + collect_trajectories driver."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable

import numpy as np

from environments.base import POMDPEnvironment
from quotient_learning.dataset import Trajectory, TrajectoryDataset

Policy = Callable[[int, np.random.Generator], int]
"""A policy is `(observation, rng) -> action`."""


def uniform_random_policy(n_actions: int) -> Policy:
    """Return a uniform-random policy over the action space."""

    def _policy(_obs: int, rng: np.random.Generator) -> int:
        return int(rng.integers(0, n_actions))

    return _policy


def count_based_exploration_policy(
    n_actions: int,
    history_max_length: int = 1,
    bonus_weight: float = 1.0,
) -> Policy:
    """Return a count-based (UCB-style) exploration policy.

    Maintains per-history-key action counts across episodes.  The history key
    is the tuple of the last ``history_max_length`` observations.  At each
    step the policy picks the action with the smallest count under the current
    history key; ties are broken uniformly at random.

    The returned callable exposes two attributes for external inspection /
    reset by ``collect_trajectories``:

    ``_counts``
        ``defaultdict`` mapping history key (tuple of ints) to a list of
        per-action visit counts.

    ``_obs_window``
        ``deque`` with ``maxlen=history_max_length`` holding the most recent
        observations.  Callers that want to reset the within-episode window
        between episodes should call ``_obs_window.clear()``.
    """
    counts: dict[tuple, list[int]] = defaultdict(lambda: [0] * n_actions)
    obs_window: deque[int] = deque(maxlen=history_max_length)

    def _policy(obs: int, rng: np.random.Generator) -> int:
        obs_window.append(obs)
        key = tuple(obs_window)
        action_counts = counts[key]
        # Pick the action(s) with minimum count; break ties randomly.
        min_count = min(action_counts)
        candidates = [a for a, c in enumerate(action_counts) if c == min_count]
        action = int(rng.choice(candidates))
        action_counts[action] += 1
        return action

    _policy._counts = counts  # type: ignore[attr-defined]
    _policy._obs_window = obs_window  # type: ignore[attr-defined]
    return _policy


def collect_trajectories(
    env: POMDPEnvironment,
    policy: Policy,
    n_episodes: int,
    max_steps: int = 100,
    seed: int = 0,
    policy_id: str = "uniform_random",
) -> TrajectoryDataset:
    """Roll out `n_episodes` independent episodes and return an immutable
    TrajectoryDataset.

    The environment is reset with a per-episode seed derived from (seed, ep_idx);
    the policy draws from a separate generator seeded by (that seed, 1), so
    action randomness is independent of environment noise.
    """
    trajectories: list[Trajectory] = []
    for ep in range(n_episodes):
        ep_seed = int(seed) * 100003 + ep
        # The policy's generator must be independent of the environment's
        # (which env.reset(seed=ep_seed) seeds with ep_seed itself); sharing
        # the seed couples action draws to latent/observation noise and
        # silently changes the closed-loop law for stochastic policies.
        rng = np.random.default_rng([ep_seed, 1])
        obs = env.reset(seed=ep_seed)
        observations: list[int] = [int(obs)]
        actions: list[int] = []
        rewards: list[float] = []
        for _ in range(max_steps):
            a = policy(observations[-1], rng)
            step = env.step(a)
            observations.append(int(step.observation))
            actions.append(int(a))
            rewards.append(float(step.reward))
            if step.terminated or step.truncated:
                break
        trajectories.append(
            Trajectory(
                observations=tuple(observations),
                actions=tuple(actions),
                rewards=tuple(rewards),
                seed=ep_seed,
            )
        )
    return TrajectoryDataset(
        trajectories=tuple(trajectories),
        environment_id=env.environment_id,
        policy_id=policy_id,
        extra={"n_episodes": n_episodes, "max_steps": max_steps, "seed": seed},
    )
