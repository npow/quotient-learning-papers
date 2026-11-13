"""Action-aware refinement demonstration (paper Remark 1, exercised).

On AASeparationPOMDP the observation-only suffix quotient provably merges
the two marker contexts (identical next-observation marginals), while the
action-aware laws have disjoint supports (TV = 1). The estimator must
reproduce both facts.
"""

from environments.layer1_discrete.aa_separation import AASeparationPOMDP
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.types import ObservationHistory

COMMON = dict(
    n_trajectories=1500,
    window_length=1,
    history_max_length=1,
    min_history_occurrences=30,
    max_steps_per_episode=20,
    alpha=0.01,
)


class TestActionAwareRefinement:
    def test_obs_only_merges_markers(self):
        env = AASeparationPOMDP(seed=5)
        q = QuotientEstimator(env=env, action_aware=False, **COMMON).run(seed=5)
        h2, h3 = ObservationHistory((2,)), ObservationHistory((3,))
        assert q.partition[h2] == q.partition[h3], (
            "obs-only must merge the two marker contexts (identical marginals)"
        )
        assert q.n_classes == env.metadata["ground_truth_n_classes_obs_only"]

    def test_action_aware_separates_markers(self):
        env = AASeparationPOMDP(seed=5)
        q = QuotientEstimator(env=env, action_aware=True, **COMMON).run(seed=5)
        h2, h3 = ObservationHistory((2,)), ObservationHistory((3,))
        assert q.partition[h2] != q.partition[h3], (
            "action-aware continuations (immediate action included) must "
            "separate the markers: disjoint (a, o') supports, TV = 1"
        )
        assert q.n_classes == env.metadata["ground_truth_n_classes_action_aware"]
