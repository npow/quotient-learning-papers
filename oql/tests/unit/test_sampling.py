"""Unit tests for continuation-sample extraction (T019)."""

from __future__ import annotations

import pytest

from quotient_learning.dataset import Trajectory, TrajectoryDataset
from quotient_learning.sampling import (
    enumerate_observed_histories,
    extract_continuation_samples,
)
from quotient_learning.types import ObservationHistory


@pytest.fixture
def toy_dataset() -> TrajectoryDataset:
    t1 = Trajectory(
        observations=(0, 1, 0, 1, 0),
        actions=(0, 0, 0, 0), rewards=(0.0, 0.0, 0.0, 0.0), seed=1,
    )
    t2 = Trajectory(
        observations=(1, 0, 0, 1, 1),
        actions=(0, 0, 0, 0), rewards=(0.0, 0.0, 0.0, 0.0), seed=2,
    )
    return TrajectoryDataset.from_iterable([t1, t2], "toy", "rand")


class TestExtractContinuationSamples:
    def test_empty_history_yields_all_windows(self, toy_dataset: TrajectoryDataset) -> None:
        cs = extract_continuation_samples(toy_dataset, ObservationHistory(()), window_length=2)
        # 5-obs traj has 4 length-2 windows; two trajectories → 8 samples.
        assert cs.n_samples == 8
        assert cs.window_length == 2

    def test_single_obs_history_extracts_only_after(
        self, toy_dataset: TrajectoryDataset
    ) -> None:
        cs = extract_continuation_samples(toy_dataset, ObservationHistory((0,)), window_length=2)
        for cont in cs.continuations:
            assert len(cont) == 2
        # Verify continuations correspond to obs after a 0 in either trajectory.
        # Traj 1: positions of 0 are [0, 2, 4]. After pos 0 → (1,0); after 2 → (1,0). pos 4 has only 0 obs after — no length-2 window.
        # Traj 2: positions of 0 are [1, 2]. After 1 → (0, 1); after 2 → (1, 1).
        assert cs.n_samples == 4

    def test_window_length_zero_raises(self, toy_dataset: TrajectoryDataset) -> None:
        with pytest.raises(ValueError):
            extract_continuation_samples(toy_dataset, ObservationHistory(()), window_length=0)


class TestEnumerateObservedHistories:
    def test_includes_empty_and_singletons(self, toy_dataset: TrajectoryDataset) -> None:
        hs = enumerate_observed_histories(toy_dataset, max_length=1, min_occurrences=1)
        assert ObservationHistory(()) in hs
        assert ObservationHistory((0,)) in hs
        assert ObservationHistory((1,)) in hs

    def test_min_occurrences_filters_rare(self, toy_dataset: TrajectoryDataset) -> None:
        hs = enumerate_observed_histories(toy_dataset, max_length=2, min_occurrences=100)
        assert hs == []
