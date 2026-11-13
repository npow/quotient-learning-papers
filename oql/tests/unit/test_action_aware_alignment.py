"""Pin the temporal alignment of action-aware continuations.

The corrected convention (paper Def. 2, revision R3): a context ends at an
observation o_t, and the k-step action-aware continuation is the k pairs

    (a_t, o_{t+1}), (a_{t+1}, o_{t+2}), ..., (a_{t+k-1}, o_{t+k})

— the FIRST symbol of the first pair is the immediate action a_t that
generates o_{t+1}. A context of length l is (o, a) step pairs with the final
step's action None (the context conditions on nothing after its last
observation).
"""

from quotient_learning.dataset import Trajectory, TrajectoryDataset
from quotient_learning.sampling import (
    enumerate_observed_action_aware_histories,
    extract_action_aware_continuations,
)
from quotient_learning.types import ActionAwareHistory


def _dataset() -> TrajectoryDataset:
    # Sequence: o0=5 a0=1 o1=6 a1=2 o2=7 a2=3 o3=8
    traj = Trajectory(
        observations=(5, 6, 7, 8), actions=(1, 2, 3), rewards=(0.0, 0.0, 0.0), seed=0
    )
    return TrajectoryDataset.from_iterable([traj], environment_id="hand", policy_id="hand")


class TestActionAwareAlignment:
    def test_immediate_action_is_first_continuation_symbol(self):
        # Context (6,): matches at o1. Immediate action is a1=2.
        # k=2 continuation: ((a1, o2), (a2, o3)) = ((2, 7), (3, 8)).
        ds = _dataset()
        history = ActionAwareHistory(((6, None),))
        sample = extract_action_aware_continuations(ds, history, window_length=2)
        assert sample.continuations == (((2, 7), (3, 8)),)

    def test_depth2_context_conditions_on_interleaved_action(self):
        # Context ((6, 2), (7, None)): o1=6, a1=2, o2=7. Immediate action a2=3.
        # k=1 continuation: ((3, 8),).
        ds = _dataset()
        history = ActionAwareHistory(((6, 2), (7, None)))
        sample = extract_action_aware_continuations(ds, history, window_length=1)
        assert sample.continuations == (((3, 8),),)

    def test_root_context_pools_every_position(self):
        # Empty context, k=1: one continuation per action position.
        ds = _dataset()
        history = ActionAwareHistory(())
        sample = extract_action_aware_continuations(ds, history, window_length=1)
        assert sorted(sample.continuations) == [((1, 6),), ((2, 7),), ((3, 8),)]

    def test_enumeration_emits_observation_ending_contexts(self):
        # Enumerated contexts must end with a (obs, None) step: they condition
        # on nothing after their final observation.
        ds = _dataset()
        contexts = enumerate_observed_action_aware_histories(ds, max_length=2)
        for ctx in contexts:
            if ctx.length > 0:
                final_obs, final_act = ctx.steps[-1]
                assert final_act is None, f"context {ctx} ends with an action"
        # The depth-2 context seen above must be among them.
        assert ActionAwareHistory(((6, 2), (7, None))) in contexts
