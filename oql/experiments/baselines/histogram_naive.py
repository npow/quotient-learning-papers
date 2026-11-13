"""Naive fixed-window histogram clustering baseline (T058).

For each observed history, build the empirical next-observation histogram and
cluster histories whose histograms are within L¹-distance `eps` of each
other. No statistical test — pure threshold clustering.

This is the equivalence-test ablation: same data, same window, no G-test /
MMD / Bernstein. Used to show that the statistical machinery in the
quotient-learner actually buys something on top of naive histograms.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from quotient_learning.dataset import TrajectoryDataset
from quotient_learning.sampling import (
    enumerate_observed_histories,
    extract_continuation_samples,
)
from quotient_learning.types import CanonicalQuotient, ObservationHistory


def histogram_naive_quotient(
    dataset: TrajectoryDataset,
    n_obs: int,
    window_length: int = 2,
    history_max_length: int = 2,
    min_occurrences: int = 30,
    epsilon: float = 0.1,
) -> CanonicalQuotient:
    histories = enumerate_observed_histories(
        dataset, max_length=history_max_length, min_occurrences=min_occurrences,
    )
    histograms: dict[ObservationHistory, np.ndarray] = {}
    for h in histories:
        cs = extract_continuation_samples(dataset, h, window_length)
        if cs.n_samples < min_occurrences:
            continue
        counts = Counter(cs.continuations)
        keys = sorted(counts.keys())
        if not keys:
            continue
        v = np.array([counts[k] for k in keys], dtype=float)
        v = v / v.sum()
        # Pad histograms to a fixed cardinality so they can be L1-compared.
        full = np.zeros(n_obs ** window_length, dtype=float)
        for k, val in zip(keys, v):
            idx = sum(o * (n_obs ** (window_length - 1 - i)) for i, o in enumerate(k))
            idx = min(idx, full.size - 1)
            full[idx] = val
        histograms[h] = full
    history_keys = list(histograms.keys())
    partition: dict[ObservationHistory, int] = {}
    class_id = 0
    for i, h in enumerate(history_keys):
        if h in partition:
            continue
        partition[h] = class_id
        for h2 in history_keys[i + 1 :]:
            if h2 in partition:
                continue
            if np.sum(np.abs(histograms[h] - histograms[h2])) < epsilon:
                partition[h2] = class_id
        class_id += 1
    return CanonicalQuotient(n_classes=class_id, partition=partition, transition={})
