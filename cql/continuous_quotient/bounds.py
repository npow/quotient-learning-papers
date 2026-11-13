"""Sample complexity bounds for continuous-space quotient learning.

The key result (research.md §R-3): the per-pair MMD test requires
m = O(1/ε²_MMD) samples independently of ambient dimension d.
The overall trajectory budget replaces |C| in Theorem 5 (finite case) with
a dimension-free RKHS convergence rate.

Trajectory budget (paper Theorem 1):
    N = C · K_hist · (τ + k) · log(K_hist / δ) / (T · ε²_MMD),   K_hist = O(n)

(``trajectory_budget`` below still evaluates the looser n²·k·τ form used by an
earlier draft; it is kept for backward compatibility of the figure script.)

where:
    n        = number of quotient classes (target)
    k        = continuation window length
    τ        = β-mixing time
    δ        = failure probability
    T        = trajectory length
    ε_MMD    = minimum MMD separation between distinct quotient classes
    K_hist   = number of observed distinct histories (Bonferroni correction)
    C        = universal constant (absorbed into O(·))
"""

from __future__ import annotations

import math

import numpy as np

from continuous_quotient.kernels import compute_gram, median_bandwidth
from continuous_quotient.types import CanonicalContinuousQuotient, ContinuousTrajectoryDataset


def trajectory_budget(
    n_classes: int,
    k: int,
    tau: float,
    delta: float,
    T: int,
    eps_mmd: float,
    K_hist: int | None = None,
    C: float = 1.0,
) -> int:
    """Upper bound on N (number of trajectories) for ε-approximate recovery.

    Parameters
    ----------
    n_classes:   target quotient size n
    k:           continuation window length
    tau:         β-mixing time (Assumption 4)
    delta:       failure probability
    T:           trajectory length (steps per episode)
    eps_mmd:     minimum MMD separation between distinct classes
    K_hist:      number of observed distinct histories; defaults to n² if None
    C:           multiplicative constant (default 1.0; tune to empirical data)

    Returns
    -------
    N: int — required number of trajectories (ceiling)
    """
    if eps_mmd <= 0:
        raise ValueError("eps_mmd must be > 0")
    if K_hist is None:
        K_hist = max(1, n_classes * n_classes)
    log_term = math.log(n_classes * K_hist / max(delta, 1e-300))
    numerator = C * n_classes * n_classes * k * tau * log_term
    denominator = T * eps_mmd * eps_mmd
    return math.ceil(numerator / denominator)


def mmd_separation_estimate(
    quotient: CanonicalContinuousQuotient,
    dataset: ContinuousTrajectoryDataset,
    kernel: str = "rbf",
    n_samples_per_class: int = 100,
    seed: int = 0,
) -> float:
    """Empirically estimate ε_min = min_{[h1]≠[h2]} MMD(P_{h1}, P_{h2}).

    Builds a small continuation sample per class (up to n_samples_per_class
    vectors) and computes pairwise biased MMD² across different-class pairs.
    Returns sqrt of the minimum positive MMD² as the estimated ε_MMD.

    This is an *underestimate* of the true ε_min because we use finite samples.
    """
    from continuous_quotient.kernels import biased_mmd2

    rng = np.random.default_rng(seed)

    # Group continuation vectors by class id.
    # We re-extract from the dataset here using the partition keys.
    class_vecs: dict[int, list[np.ndarray]] = {}
    for traj in dataset.trajectories:
        obs = traj.observations
        acts = traj.actions
        T = traj.length
        for start in range(T - 1):
            from continuous_quotient.types import make_history_key
            hkey = make_history_key(obs[:start + 1], acts[:start])
            cid = quotient.partition.get(hkey)
            if cid is None:
                continue
            if cid not in class_vecs:
                class_vecs[cid] = []
            o = np.asarray(obs[start + 1], dtype=np.float64)
            class_vecs[cid].append(o)

    if len(class_vecs) < 2:
        return 0.0

    # Subsample per class.
    class_arrays = {}
    for cid, vecs in class_vecs.items():
        arr = np.stack(vecs, axis=0)
        if len(arr) > n_samples_per_class:
            idx = rng.choice(len(arr), n_samples_per_class, replace=False)
            arr = arr[idx]
        class_arrays[cid] = arr

    class_ids = sorted(class_arrays.keys())
    min_mmd2 = float("inf")

    for i, ci in enumerate(class_ids):
        xa = class_arrays[ci]
        bw = median_bandwidth(xa, xa)
        for cj in class_ids[i + 1:]:
            xb = class_arrays[cj]
            bw_pair = median_bandwidth(xa, xb)
            kxx = compute_gram(xa, xa, kernel, bw_pair)
            kyy = compute_gram(xb, xb, kernel, bw_pair)
            kxy = compute_gram(xa, xb, kernel, bw_pair)
            mmd2 = biased_mmd2(kxx, kyy, kxy)
            if mmd2 > 0:
                min_mmd2 = min(min_mmd2, mmd2)

    return float(math.sqrt(max(min_mmd2, 0.0))) if min_mmd2 < float("inf") else 0.0
