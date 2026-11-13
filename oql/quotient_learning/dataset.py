"""Trajectory and TrajectoryDataset containers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Trajectory:
    """A single sampled interaction episode."""

    observations: tuple[int, ...]
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    seed: int

    def __post_init__(self) -> None:
        if len(self.actions) != len(self.rewards):
            raise ValueError("actions and rewards must have equal length")
        if len(self.observations) != len(self.actions) + 1:
            raise ValueError(
                "observations must have one more element than actions "
                f"(got {len(self.observations)} obs, {len(self.actions)} act)"
            )

    @property
    def length(self) -> int:
        return len(self.actions)


@dataclass(frozen=True)
class TrajectoryDataset:
    """An immutable collection of trajectories with provenance metadata."""

    trajectories: tuple[Trajectory, ...]
    environment_id: str
    policy_id: str
    extra: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_iterable(
        cls,
        trajectories: Iterable[Trajectory],
        environment_id: str,
        policy_id: str,
        extra: dict[str, object] | None = None,
    ) -> TrajectoryDataset:
        return cls(
            trajectories=tuple(trajectories),
            environment_id=environment_id,
            policy_id=policy_id,
            extra=dict(extra or {}),
        )

    @property
    def n_trajectories(self) -> int:
        return len(self.trajectories)

    def total_steps(self) -> int:
        return sum(t.length for t in self.trajectories)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self.trajectories)

    def __len__(self) -> int:
        return self.n_trajectories

    def split(self, fraction: float = 0.5) -> tuple[TrajectoryDataset, TrajectoryDataset]:
        """Deterministically split whole trajectories into disjoint datasets.

        Splitting at trajectory boundaries preserves the independence supplied
        by independent environment resets.  The first part is intended for
        candidate-context selection and the second for inference.
        """
        if not (0.0 < fraction < 1.0):
            raise ValueError(f"fraction must be in (0, 1), got {fraction}")
        cut = int(len(self.trajectories) * fraction)
        if cut < 1 or cut >= len(self.trajectories):
            raise ValueError("split requires at least two trajectories")

        def _part(trajectories: tuple[Trajectory, ...], role: str) -> TrajectoryDataset:
            extra = dict(self.extra)
            extra.update({"split_role": role, "split_fraction": fraction})
            return TrajectoryDataset(
                trajectories=trajectories,
                environment_id=self.environment_id,
                policy_id=self.policy_id,
                extra=extra,
            )

        return (
            _part(self.trajectories[:cut], "selection"),
            _part(self.trajectories[cut:], "inference"),
        )
