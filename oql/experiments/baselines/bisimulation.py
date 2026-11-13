"""Bisimulation-distance baseline (Ferns et al. 2004) — model-access oracle.

Requires the full POMDP model (transition + observation matrices). Used as
an oracle upper bound — our model-free learner should approach this quotient
size in the data-rich limit.

Implementation: build the bisimulation partition via the standard fixed-point
algorithm — start with classes by observation distribution, refine until
stable. Operates on a SyntheticPOMDP instance which exposes its internal
matrices.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from environments.layer1_discrete.synthetic import SyntheticPOMDP
from quotient_learning.types import CanonicalQuotient, ObservationHistory


def bisimulation_quotient(
    env: SyntheticPOMDP,
    max_iterations: int = 100,
    tol: float = 1e-6,
) -> CanonicalQuotient:
    """Compute the exact bisimulation partition over states.

    Wraps each state as its own length-1 ObservationHistory for compatibility
    with the CanonicalQuotient type.
    """
    n_states = env._n_states
    obs_matrix = env._observation
    trans_matrix = env._transition  # shape (A, S, S)

    def signature(state: int, partition: dict[int, int]) -> tuple:
        # Combine observation distribution + class-distribution under every action.
        sig = (tuple(np.round(obs_matrix[state], 4).tolist()),)
        for a in range(trans_matrix.shape[0]):
            class_probs: dict[int, float] = defaultdict(float)
            for s_next in range(n_states):
                p = float(trans_matrix[a, state, s_next])
                if p > 0:
                    class_probs[partition[s_next]] += p
            sig += (tuple(sorted(class_probs.items())),)
        return sig

    partition = {s: 0 for s in range(n_states)}
    for _ in range(max_iterations):
        signatures = {s: signature(s, partition) for s in range(n_states)}
        sig_to_class: dict[tuple, int] = {}
        new_partition: dict[int, int] = {}
        for s, sig in signatures.items():
            if sig not in sig_to_class:
                sig_to_class[sig] = len(sig_to_class)
            new_partition[s] = sig_to_class[sig]
        if new_partition == partition:
            break
        partition = new_partition

    n_classes = max(partition.values()) + 1 if partition else 0
    history_partition = {ObservationHistory((s,)): partition[s] for s in range(n_states)}
    return CanonicalQuotient(n_classes=n_classes, partition=history_partition, transition={})
