"""Unit tests for QuotientEstimator Bonferroni auto-α₀ and root-singleton exclusion.

Covers:
- delta parameter triggers α₀ = delta / (2 * |H_N|²) computation.
- root ObservationHistory(()) is excluded from partition and n_classes when
  it forms a singleton group.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quotient_learning.estimator import QuotientEstimator
from quotient_learning.types import (
    CanonicalQuotient,
    ContinuationSample,
    EquivalenceTestResult,
    ObservationHistory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(n_actions: int = 1, n_observations: int = 2) -> MagicMock:
    env = MagicMock()
    env.n_actions = n_actions
    env.n_observations = n_observations
    env.environment_id = "mock_env"
    return env


def _make_dataset_with_histories(histories_with_counts: dict[ObservationHistory, int]):
    """Build a minimal mock dataset that makes enumerate_observed_histories
    return exactly the requested histories, and extract_continuation_samples
    return a fixed ContinuationSample with the given count."""
    from quotient_learning.dataset import Trajectory, TrajectoryDataset

    # Build trajectories that contain each history the requested number of times.
    # We use length-1 histories only and embed them as individual 3-obs trajectories.
    trajs = []
    seed = 0
    for h, count in histories_with_counts.items():
        obs_seq = h.observations if h.observations else ()
        # For each occurrence, create a trajectory that includes this history context.
        for _ in range(count):
            # Make a 3-observation trajectory; obs[0..len(h)] matches the history.
            obs = tuple(obs_seq) + (0, 0)  # pad to at least 3 obs
            actions = (0,) * (len(obs) - 1)
            rewards = (0.0,) * (len(obs) - 1)
            trajs.append(Trajectory(observations=obs, actions=actions, rewards=rewards, seed=seed))
            seed += 1
    return TrajectoryDataset(
        trajectories=tuple(trajs),
        environment_id="mock_env",
        policy_id="uniform_random",
    )


# ---------------------------------------------------------------------------
# Bonferroni auto-α₀ tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBonferroniAlpha:
    """Verify that delta triggers α₀ = delta / (2 * n_viable²) in pairwise tests."""

    def test_alpha_passed_to_equivalence_test_with_delta(self) -> None:
        """When delta is set, the test() calls use α₀ = delta/(2*n²)."""
        env = _make_env()
        captured_alphas: list[float] = []

        class CapturingTest:
            """Equivalence test that records the alpha it's called with."""
            def test(self, sa, sb, alpha=0.05):
                captured_alphas.append(alpha)
                return EquivalenceTestResult(
                    history_a=sa.history, history_b=sb.history,
                    test_statistic=0.0, p_value=1.0,
                    same_class=True, alpha=alpha,
                )

        delta = 0.1
        # We'll patch enumerate_observed_histories and extract_continuation_samples
        # to return exactly 3 viable histories so n_viable=3 → α₀ = 0.1 / (2*9).
        h0 = ObservationHistory(())
        h1 = ObservationHistory((0,))
        h2 = ObservationHistory((1,))
        histories = [h0, h1, h2]
        n_viable = len(histories)
        expected_alpha = delta / (2.0 * n_viable ** 2)

        def fake_enumerate(dataset, max_length, min_occurrences):
            return histories

        def fake_extract(dataset, history, window_length):
            return ContinuationSample(
                history=history,
                continuations=tuple([(0, 0)] * 50),
                window_length=2,
            )

        est = QuotientEstimator(
            env=env,
            n_trajectories=10,
            window_length=2,
            history_max_length=1,
            min_history_occurrences=1,
            equivalence_test=CapturingTest(),
            delta=delta,
            max_steps_per_episode=5,
        )

        from quotient_learning import estimator as est_module
        with patch.object(est_module, "enumerate_observed_histories", fake_enumerate):
            with patch.object(est_module, "extract_continuation_samples", fake_extract):
                # Build a minimal dataset stub
                dataset = MagicMock()
                dataset.trajectories = []
                est.estimate(dataset)

        assert len(captured_alphas) > 0, "No alpha values were captured — test() was not called"
        for a in captured_alphas:
            assert a == pytest.approx(expected_alpha, rel=1e-9), (
                f"Expected α₀={expected_alpha} (delta/{2 * n_viable**2}), got {a}"
            )

    def test_fixed_alpha_used_when_delta_is_none(self) -> None:
        """When delta=None, the fixed config alpha is used unchanged."""
        env = _make_env()
        captured_alphas: list[float] = []

        class CapturingTest:
            def test(self, sa, sb, alpha=0.05):
                captured_alphas.append(alpha)
                return EquivalenceTestResult(
                    history_a=sa.history, history_b=sb.history,
                    test_statistic=0.0, p_value=1.0,
                    same_class=True, alpha=alpha,
                )

        fixed_alpha = 0.01
        h0 = ObservationHistory(())
        h1 = ObservationHistory((0,))
        histories = [h0, h1]

        def fake_enumerate(dataset, max_length, min_occurrences):
            return histories

        def fake_extract(dataset, history, window_length):
            return ContinuationSample(
                history=history,
                continuations=tuple([(0, 0)] * 50),
                window_length=2,
            )

        est = QuotientEstimator(
            env=env,
            n_trajectories=10,
            window_length=2,
            history_max_length=1,
            min_history_occurrences=1,
            equivalence_test=CapturingTest(),
            alpha=fixed_alpha,
            delta=None,
            max_steps_per_episode=5,
        )

        from quotient_learning import estimator as est_module
        with patch.object(est_module, "enumerate_observed_histories", fake_enumerate):
            with patch.object(est_module, "extract_continuation_samples", fake_extract):
                dataset = MagicMock()
                dataset.trajectories = []
                est.estimate(dataset)

        assert len(captured_alphas) > 0
        for a in captured_alphas:
            assert a == pytest.approx(fixed_alpha, rel=1e-9), (
                f"Expected fixed alpha={fixed_alpha}, got {a}"
            )

    def test_delta_formula_values(self) -> None:
        """Verify the arithmetic: α₀ = delta / (2 * n²) for representative inputs."""
        cases = [
            (0.1, 3, 0.1 / 18.0),
            (0.05, 5, 0.05 / 50.0),
            (1.0, 10, 1.0 / 200.0),
        ]
        for delta, n, expected in cases:
            computed = delta / (2.0 * n ** 2)
            assert computed == pytest.approx(expected, rel=1e-12), (
                f"delta={delta}, n={n}: got {computed}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Root-singleton exclusion tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRootSingletonExclusion:
    """Verify that root ObservationHistory(()) singleton is excluded from output."""

    def _run_with_histories(
        self,
        histories: list[ObservationHistory],
        merge_pairs: list[tuple[int, int]] | None = None,
    ) -> CanonicalQuotient:
        """Run estimate() with mocked enumerate/extract so we control which
        histories are viable and which are merged."""
        env = _make_env()
        if merge_pairs is None:
            merge_pairs = []
        pair_set = {(i, j) for i, j in merge_pairs}

        def fake_enumerate(dataset, max_length, min_occurrences):
            return histories

        def fake_extract(dataset, history, window_length):
            return ContinuationSample(
                history=history,
                continuations=tuple([(0, 0)] * 50),
                window_length=2,
            )

        call_count = [0]

        class ControlledTest:
            def test(self, sa, sb, alpha=0.05):
                i = histories.index(sa.history)
                j = histories.index(sb.history)
                same = (i, j) in pair_set or (j, i) in pair_set
                return EquivalenceTestResult(
                    history_a=sa.history, history_b=sb.history,
                    test_statistic=0.0, p_value=1.0 if same else 0.0,
                    same_class=same, alpha=alpha,
                )

        est = QuotientEstimator(
            env=env,
            n_trajectories=10,
            window_length=2,
            history_max_length=1,
            min_history_occurrences=1,
            equivalence_test=ControlledTest(),
            alpha=0.01,
            max_steps_per_episode=5,
        )

        from quotient_learning import estimator as est_module
        with patch.object(est_module, "enumerate_observed_histories", fake_enumerate):
            with patch.object(est_module, "extract_continuation_samples", fake_extract):
                dataset = MagicMock()
                dataset.trajectories = []
                return est.estimate(dataset)

    def test_root_singleton_excluded_from_partition(self) -> None:
        """Root history in a singleton group should not appear in partition."""
        root = ObservationHistory(())
        h1 = ObservationHistory((0,))
        h2 = ObservationHistory((1,))
        # No merges — root stays singleton, h1 and h2 are separate groups.
        q = self._run_with_histories([root, h1, h2], merge_pairs=[])
        assert root not in q.partition, (
            "Root singleton must be excluded from partition"
        )

    def test_root_singleton_excluded_from_n_classes(self) -> None:
        """n_classes should not count the root singleton."""
        root = ObservationHistory(())
        h1 = ObservationHistory((0,))
        h2 = ObservationHistory((1,))
        q = self._run_with_histories([root, h1, h2], merge_pairs=[])
        # h1 and h2 are distinct non-root classes → n_classes should be 2
        assert q.n_classes == 2, (
            f"Expected n_classes=2 (root excluded), got {q.n_classes}"
        )

    def test_root_merged_with_other_is_kept(self) -> None:
        """When root merges with another history, the merged class IS included."""
        root = ObservationHistory(())
        h1 = ObservationHistory((0,))
        h2 = ObservationHistory((1,))
        # Root merges with h1 → merged group is not a singleton → should be kept.
        q = self._run_with_histories([root, h1, h2], merge_pairs=[(0, 1)])
        # The merged {root, h1} group plus {h2} → 2 classes total.
        assert q.n_classes == 2, (
            f"Expected n_classes=2 when root merges with h1, got {q.n_classes}"
        )
        # Both root and h1 should appear in the partition under the same class.
        assert root in q.partition, "root in merged group should be in partition"
        assert h1 in q.partition
        assert q.partition[root] == q.partition[h1]

    def test_no_root_in_histories_unaffected(self) -> None:
        """When root is not in the viable set, behaviour is unchanged."""
        h1 = ObservationHistory((0,))
        h2 = ObservationHistory((1,))
        q = self._run_with_histories([h1, h2], merge_pairs=[])
        assert q.n_classes == 2
        assert h1 in q.partition
        assert h2 in q.partition
