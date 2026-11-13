"""Action-aware refinement demonstration environment (paper Remark 1).

A deterministic 4-phase cycle constructed so that the OBSERVATION-ONLY
suffix quotient provably merges two contexts that the ACTION-AWARE quotient
separates with maximal gap:

    phase M_A emits marker 2;  the next observation is the payload
        f_A(a) = a           (a = action taken at the marker step)
    phase M_B emits marker 3;  the next observation is the payload
        f_B(a) = 1 - a       (the OPPOSITE wiring)

Under uniform-random exploration the observation-only continuation law
after either marker is {0: 1/2, 1: 1/2} — identical, so contexts (2,) and
(3,) are one class of the observation-only suffix quotient (k = 1).  The
action-aware continuation laws are uniform on {(0,0), (1,1)} after (2,)
and uniform on {(0,1), (1,0)} after (3,): disjoint supports, TV = 1.
Recovering the separation therefore REQUIRES the action-aware variant, and
the instance exercises exactly the division argument of Remark 1 (the
policy factor is constant and divides out; the open-loop laws differ).

All four observation symbols are reachable, so the k=1 alphabet is honestly
full-support.  metadata records ground-truth counts for both variants.
"""

from __future__ import annotations

import numpy as np

from environments.base import POMDPEnvironment, StepResult

MARKER_A, MARKER_B = 2, 3
PHASES = 4  # M_A -> payload_A -> M_B -> payload_B -> ...


class AASeparationPOMDP(POMDPEnvironment):
    """Deterministic marker/payload cycle separating aa from obs-only."""

    _k1_full_support = True

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self._phase = 0
        self._pending_payload = 0

    @property
    def n_observations(self) -> int:
        return 4  # payloads {0, 1}, markers {2, 3}

    @property
    def n_actions(self) -> int:
        return 2

    @property
    def environment_id(self) -> str:
        return "AASeparation"

    @property
    def metadata(self) -> dict[str, object]:
        return {
            # obs-only, k=1, L=1: (2,) ~ (3,) (both {0: 1/2, 1: 1/2}) and
            # (0,) ~ (1,) (both {2: 1/2, 3: 1/2}) -> 2 classes.
            "ground_truth_n_classes_obs_only": 2,
            # action-aware: the marker class splits (disjoint (a, o') laws,
            # TV = 1) while the payload class stays merged -> 3 classes.
            "ground_truth_n_classes_action_aware": 3,
        }

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._phase = int(self._rng.integers(0, PHASES))
        self._pending_payload = int(self._rng.integers(0, 2))
        return self._emit_current()

    def _emit_current(self) -> int:
        if self._phase == 0:
            return MARKER_A
        if self._phase == 2:
            return MARKER_B
        return self._pending_payload

    def step(self, action: int) -> StepResult:
        if self._phase == 0:      # leaving marker A: payload f_A(a) = a
            self._pending_payload = int(action)
        elif self._phase == 2:    # leaving marker B: payload f_B(a) = 1 - a
            self._pending_payload = 1 - int(action)
        self._phase = (self._phase + 1) % PHASES
        return StepResult(
            observation=self._emit_current(), reward=0.0,
            terminated=False, truncated=False,
        )
