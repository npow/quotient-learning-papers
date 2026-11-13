"""Tiger L=2 regression: the depth-2 collapse must be corrected or flagged.

The paper documents that at history_max_length=2 the old estimator
confidently collapsed Tiger's two listen classes via low-power depth-2
contexts (union-find bridging). With the power gate the estimator must
either recover the correct depth-1 separation or surface the uncertainty
(undetermined pairs / boundary flags) instead of reporting a confident
wrong partition.
"""

from environments.layer1_discrete.tiger import TigerPOMDP
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.types import ObservationHistory


class TestTigerL2Regression:
    def test_no_confident_wrong_partition_at_l2(self):
        env = TigerPOMDP(seed=7)
        est = QuotientEstimator(
            env=env,
            n_trajectories=2000,
            window_length=1,
            history_max_length=2,
            min_history_occurrences=30,
            alpha=0.05,
            epsilon_min=0.5,
            power_beta=0.05,
        )
        q = est.run(seed=7)

        h_left, h_right = ObservationHistory((0,)), ObservationHistory((1,))
        depth1_separated = (
            h_left in q.partition
            and h_right in q.partition
            and q.partition[h_left] != q.partition[h_right]
        )
        flagged = q.undetermined_rate > 0.0 or len(q.boundary_flags) > 0
        assert depth1_separated or flagged, (
            "estimator reported a confident partition that conflates the two "
            f"listen contexts: n_classes={q.n_classes}, "
            f"undetermined_rate={q.undetermined_rate}, "
            f"boundary_flags={q.boundary_flags}"
        )

    def test_depth1_gate_passes_and_separates(self):
        # At L=1 both listen contexts are abundant; the gate must not block
        # the well-powered depth-1 separation.
        env = TigerPOMDP(seed=11)
        est = QuotientEstimator(
            env=env,
            n_trajectories=3000,
            window_length=1,
            history_max_length=1,
            min_history_occurrences=30,
            alpha=0.05,
            epsilon_min=0.5,
            power_beta=0.05,
        )
        q = est.run(seed=11)
        h_left, h_right = ObservationHistory((0,)), ObservationHistory((1,))
        assert h_left in q.partition and h_right in q.partition
        assert q.partition[h_left] != q.partition[h_right]
