"""Predictive State Representations baseline (Boots, Siddiqi, Gordon 2011).

Adapted to emit a quotient: we estimate the PSR Hankel matrix from observed
trajectories, take its rank-k SVD truncation, and cluster histories by their
projected predictive vectors (k-means on the U_k columns).

Sample-complexity is O(n³/ε²) per the spectral learning bound — strictly
worse than the equivalence-test quotient learner for small quotients.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.cluster.vq import kmeans2

from quotient_learning.dataset import TrajectoryDataset
from quotient_learning.types import CanonicalQuotient, ObservationHistory


def _build_hankel(
    dataset: TrajectoryDataset,
    history_length: int,
    future_length: int,
    n_obs: int,
) -> tuple[np.ndarray, list[ObservationHistory], list[tuple[int, ...]]]:
    """Empirical Hankel matrix H[h, f] = freq(h followed by f) / freq(h)."""
    counts_joint: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = defaultdict(int)
    counts_h: dict[tuple[int, ...], int] = defaultdict(int)
    for traj in dataset.trajectories:
        obs = traj.observations
        n = len(obs)
        for start in range(n - history_length - future_length + 1):
            h = obs[start : start + history_length]
            f = obs[start + history_length : start + history_length + future_length]
            counts_joint[(h, f)] += 1
            counts_h[h] += 1
    histories = sorted(counts_h.keys())
    futures = sorted({f for (_, f) in counts_joint})
    H = np.zeros((len(histories), len(futures)), dtype=np.float64)
    h_index = {h: i for i, h in enumerate(histories)}
    f_index = {f: i for i, f in enumerate(futures)}
    for (h, f), c in counts_joint.items():
        H[h_index[h], f_index[f]] = c / max(counts_h[h], 1)
    return H, [ObservationHistory(h) for h in histories], futures


def psr_quotient(
    dataset: TrajectoryDataset,
    n_obs: int,
    rank: int = 8,
    history_length: int = 2,
    future_length: int = 2,
    seed: int = 0,
    spectral_gap_threshold: float | None = None,
) -> CanonicalQuotient:
    """Estimate a quotient by clustering rank-k PSR predictive states.

    If ``spectral_gap_threshold`` is given, the rank is chosen data-adaptively
    as the number of singular values of the Hankel matrix exceeding
    ``spectral_gap_threshold * sigma_max`` (capped at ``rank``); otherwise the
    fixed ``rank`` is used.
    """
    H, histories, _ = _build_hankel(dataset, history_length, future_length, n_obs)
    if H.size == 0 or H.shape[0] == 0:
        return CanonicalQuotient(n_classes=0, partition={}, transition={})
    U, S, _ = np.linalg.svd(H, full_matrices=False)
    k = min(rank, S.size)
    if spectral_gap_threshold is not None and S.size > 0 and S[0] > 0:
        k_gap = int(np.sum(spectral_gap_threshold * S[0] < S))
        k = max(1, min(k, k_gap))
    embeddings = U[:, :k] * S[:k]
    rng = np.random.default_rng(seed)
    n_clusters = min(k, embeddings.shape[0])
    init = embeddings[rng.choice(embeddings.shape[0], size=n_clusters, replace=False)]
    centroids, labels = kmeans2(embeddings, init, minit="matrix", seed=seed)
    partition = {h: int(labels[i]) for i, h in enumerate(histories)}
    n_classes = int(labels.max()) + 1 if labels.size else 0
    return CanonicalQuotient(n_classes=n_classes, partition=partition, transition={})
