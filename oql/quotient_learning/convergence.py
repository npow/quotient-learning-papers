"""Convergence tracking and stopping criteria for the quotient learner."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from environments.base import POMDPEnvironment
from quotient_learning.collection import collect_trajectories, uniform_random_policy
from quotient_learning.dataset import TrajectoryDataset
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.types import CanonicalQuotient


@dataclass
class ConvergenceCheckpoint:
    n_trajectories: int
    n_classes: int
    n_partition_entries: int


@dataclass
class ConvergenceTracker:
    """Run the QuotientEstimator at a logarithmic ladder of trajectory budgets
    and record how the learned quotient size evolves.

    `is_stable(window=3, max_classes_delta=0)` returns True when the last
    `window` consecutive checkpoints have agreed on the number of classes — the
    partition-stability convergence criterion used in US2 (T052).
    """

    estimator: QuotientEstimator
    checkpoints: list[ConvergenceCheckpoint] = field(default_factory=list)

    def record(self, dataset: TrajectoryDataset) -> ConvergenceCheckpoint:
        q = self.estimator.estimate(dataset)
        cp = ConvergenceCheckpoint(
            n_trajectories=dataset.n_trajectories,
            n_classes=q.n_classes,
            n_partition_entries=len(q.partition),
        )
        self.checkpoints.append(cp)
        return cp

    def is_stable(self, window: int = 3) -> bool:
        if len(self.checkpoints) < window:
            return False
        recent = self.checkpoints[-window:]
        return all(c.n_classes == recent[0].n_classes for c in recent)

    @property
    def latest_n_classes(self) -> int | None:
        return self.checkpoints[-1].n_classes if self.checkpoints else None


def run_convergence_ladder(
    env: POMDPEnvironment,
    budgets: Iterable[int],
    estimator_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
) -> tuple[list[ConvergenceCheckpoint], CanonicalQuotient]:
    """Run a convergence ladder: estimate the quotient at each trajectory budget.

    Returns the list of checkpoints and the final (largest-budget) quotient.
    Reuses a single growing dataset across budgets — guarantees monotonicity in
    expectation since later estimates strictly include earlier data.
    """
    kwargs = dict(estimator_kwargs or {})
    budgets = sorted(set(int(b) for b in budgets))
    if not budgets:
        raise ValueError("budgets must be non-empty")
    largest = budgets[-1]
    policy = uniform_random_policy(env.n_actions)
    # Collect once at the largest budget; slice for smaller checkpoints.
    full = collect_trajectories(
        env=env, policy=policy, n_episodes=largest,
        max_steps=kwargs.get("max_steps_per_episode", 50),
        seed=seed, policy_id="uniform_random",
    )
    estimator = QuotientEstimator(env=env, **kwargs)
    checkpoints: list[ConvergenceCheckpoint] = []
    last_q: CanonicalQuotient | None = None
    for b in budgets:
        subset = TrajectoryDataset(
            trajectories=full.trajectories[:b],
            environment_id=full.environment_id,
            policy_id=full.policy_id,
            extra={**full.extra, "n_episodes": b},
        )
        last_q = estimator.estimate(subset)
        checkpoints.append(
            ConvergenceCheckpoint(
                n_trajectories=b,
                n_classes=last_q.n_classes,
                n_partition_entries=len(last_q.partition),
            )
        )
    assert last_q is not None
    return checkpoints, last_q
