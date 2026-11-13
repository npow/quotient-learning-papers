"""Validate that pooled suffix sampling estimates the stationary suffix law.

On a synthetic 2-state Markov chain whose observation IS the state, the
k-step continuation law conditional on the suffix (o,) is exactly the
transition row P[o], and conditioning on a longer suffix (o', o) gives the
same row (Markov property). Pooling every occurrence across positions — the
released sampling semantics — must therefore reproduce these conditionals.
This is the estimand identity the revised paper proves (Definition of the
stationary sliding-window suffix quotient, revision R1).
"""

import random

from quotient_learning.dataset import Trajectory, TrajectoryDataset
from quotient_learning.sampling import extract_continuation_samples
from quotient_learning.types import ObservationHistory

P = {0: (0.7, 0.3), 1: (0.4, 0.6)}


def _chain_dataset(n_traj: int = 200, length: int = 100, seed: int = 7) -> TrajectoryDataset:
    rng = random.Random(seed)
    trajs = []
    for i in range(n_traj):
        obs = [rng.randint(0, 1)]
        for _ in range(length):
            obs.append(0 if rng.random() < P[obs[-1]][0] else 1)
        trajs.append(
            Trajectory(
                observations=tuple(obs),
                actions=(0,) * length,
                rewards=(0.0,) * length,
                seed=i,
            )
        )
    return TrajectoryDataset.from_iterable(trajs, environment_id="chain2", policy_id="const")


def _freq_of_one(sample) -> float:
    return sum(1 for c in sample.continuations if c == (1,)) / sample.n_samples


class TestSuffixEstimand:
    def test_depth1_suffix_matches_transition_row(self):
        ds = _chain_dataset()
        s0 = extract_continuation_samples(ds, ObservationHistory((0,)), window_length=1)
        s1 = extract_continuation_samples(ds, ObservationHistory((1,)), window_length=1)
        assert abs(_freq_of_one(s0) - 0.3) < 0.02
        assert abs(_freq_of_one(s1) - 0.6) < 0.02

    def test_depth2_suffix_agrees_with_depth1(self):
        # Markov: P(next | o', o) == P(next | o); pooled sampling must agree.
        ds = _chain_dataset()
        s_0 = extract_continuation_samples(ds, ObservationHistory((0,)), window_length=1)
        s_10 = extract_continuation_samples(ds, ObservationHistory((1, 0)), window_length=1)
        s_00 = extract_continuation_samples(ds, ObservationHistory((0, 0)), window_length=1)
        assert abs(_freq_of_one(s_10) - _freq_of_one(s_0)) < 0.03
        assert abs(_freq_of_one(s_00) - _freq_of_one(s_0)) < 0.03
