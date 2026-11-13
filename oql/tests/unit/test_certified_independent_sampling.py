"""Regression tests for the finite-sample certified sampling path."""

from quotient_learning.dataset import Trajectory, TrajectoryDataset
from quotient_learning.sampling import (
    extract_action_aware_continuations,
    extract_independent_continuation_samples,
)
from quotient_learning.types import ActionAwareHistory, ObservationHistory


def _dataset() -> TrajectoryDataset:
    trajectories = tuple(
        Trajectory(
            observations=(i % 2, (i + 1) % 2, i % 2, (i + 1) % 2),
            actions=(0, 1, 0), rewards=(0.0, 0.0, 0.0), seed=i,
        )
        for i in range(8)
    )
    return TrajectoryDataset(trajectories, "toy", "uniform_random")


def test_split_is_disjoint_and_covers_trajectories() -> None:
    data = _dataset()
    selection, inference = data.split(0.5)
    assert len(selection) == len(inference) == 4
    assert {t.seed for t in selection}.isdisjoint({t.seed for t in inference})
    assert selection.trajectories + inference.trajectories == data.trajectories


def test_independent_sampler_takes_at_most_one_per_trajectory() -> None:
    data = _dataset()
    sample = extract_independent_continuation_samples(
        data, ObservationHistory((0,)), window_length=1, context_end=0
    )
    assert sample.n_samples == 4
    assert sample.max_per_trajectory == 1


def test_action_aware_stride_is_honoured() -> None:
    data = _dataset()
    root = ActionAwareHistory(())
    dense = extract_action_aware_continuations(data, root, 1, stride=1)
    sparse = extract_action_aware_continuations(data, root, 1, stride=2)
    assert dense.n_samples == 24
    assert sparse.n_samples == 16
