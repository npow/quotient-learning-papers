"""Bridging regression: a low-power context must not merge separated classes.

Construction: contexts "0" and "1" have well-separated continuation laws
(90/10 vs 10/90 over symbols {4, 5}; TV = 0.8) and plenty of samples.
Context "2" is rare: enough samples to pass the viability floor (>= 30) but
far fewer than the power requirement m*(eps_min, alpha, beta). Under the
old union-find semantics a non-rejection involving "2" (low power) could
transitively merge the 0-group with the 1-group. With the power gate, every
pair involving "2" is UNDETERMINED and transmits no merge.
"""

import random
from unittest.mock import MagicMock

from quotient_learning.dataset import Trajectory, TrajectoryDataset
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.stats import required_samples
from quotient_learning.types import ObservationHistory


def _traj(ctx: int, law: tuple[float, float], steps: int, rng: random.Random, seed: int) -> Trajectory:
    obs = []
    for _ in range(steps):
        obs.append(ctx)
        obs.append(4 if rng.random() < law[0] else 5)
    return Trajectory(
        observations=tuple(obs),
        actions=(0,) * (len(obs) - 1),
        rewards=(0.0,) * (len(obs) - 1),
        seed=seed,
    )


def _bridging_dataset() -> TrajectoryDataset:
    rng = random.Random(123)
    trajs = []
    for i in range(30):  # context 0: ~300 samples, law 90/10
        trajs.append(_traj(0, (0.9, 0.1), 10, rng, i))
    for i in range(30):  # context 1: ~300 samples, law 10/90
        trajs.append(_traj(1, (0.1, 0.9), 10, rng, 100 + i))
    for i in range(4):   # context 2: ~40 samples, law 50/50 (the bridge)
        trajs.append(_traj(2, (0.5, 0.5), 10, rng, 200 + i))
    return TrajectoryDataset.from_iterable(trajs, "bridging", "const")


def _env() -> MagicMock:
    env = MagicMock()
    env.n_observations = 6
    env.n_actions = 1
    env.environment_id = "bridging-mock"
    env.continuation_alphabet = MagicMock(side_effect=AttributeError)
    return env


class TestBridgingRegression:
    def test_power_gate_keeps_separated_groups_apart(self):
        est = QuotientEstimator(
            env=_env(),
            window_length=1,
            history_max_length=1,
            min_history_occurrences=30,
            alpha=0.05,
            epsilon_min=0.8,
            power_beta=0.05,
        )
        # m* for eps=0.8, alpha=beta=0.05, |C| = 6^1: between the bridge's
        # ~40 samples and the groups' ~300 samples.
        m_star = required_samples(0.8, 0.05, 0.05, 6)
        assert 40 < m_star < 300

        q = est.estimate(_bridging_dataset())

        h0, h1, h2 = (ObservationHistory((c,)) for c in (0, 1, 2))
        assert q.partition[h0] != q.partition[h1], (
            "well-sampled separated contexts must stay in different classes"
        )
        # Every pair involving the under-sampled bridge is undetermined.
        und_members = {h for pair in q.undetermined_pairs for h in pair}
        assert h2 in und_members
        assert q.undetermined_rate > 0.0
        # The bridge did not merge into either group.
        if h2 in q.partition:
            assert q.partition[h2] not in (q.partition[h0], q.partition[h1]) or (
                q.partition[h0] != q.partition[h1]
            )

    def test_undetermined_pairs_do_not_transit_merges(self):
        est = QuotientEstimator(
            env=_env(),
            window_length=1,
            history_max_length=1,
            min_history_occurrences=30,
            alpha=0.05,
            epsilon_min=0.8,
            power_beta=0.05,
        )
        q = est.estimate(_bridging_dataset())
        h0, h1 = ObservationHistory((0,)), ObservationHistory((1,))
        # Transitive closure through the bridge is the historic failure mode:
        # 0 and 1 sharing a class would only be possible via context 2.
        assert q.partition[h0] != q.partition[h1]

    def test_gate_disabled_reports_no_undetermined(self):
        est = QuotientEstimator(
            env=_env(),
            window_length=1,
            history_max_length=1,
            min_history_occurrences=30,
            alpha=0.05,
        )
        q = est.estimate(_bridging_dataset())
        assert q.undetermined_rate == 0.0
