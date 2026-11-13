"""Kernel functions for continuous MMD equivalence tests.

All kernels operate on 2-D float32 arrays of shape (n, d) and return
Gram matrices of shape (n, n) or cross-Gram matrices of shape (n, m).
"""

from __future__ import annotations

import numpy as np


def _sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances ||x_i - y_j||^2, shape (n, m)."""
    x_n = (x * x).sum(axis=1, keepdims=True)
    y_n = (y * y).sum(axis=1, keepdims=True)
    sq = x_n + y_n.T - 2.0 * (x @ y.T)
    np.maximum(sq, 0.0, out=sq)
    return sq


def rbf_gram(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """RBF (Gaussian) kernel: k(x,y) = exp(-||x-y||^2 / (2σ^2))."""
    return np.exp(-_sq_dists(x, y) / (2.0 * bandwidth * bandwidth))


def laplace_gram(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """Laplace kernel: k(x,y) = exp(-||x-y||_1 / σ)."""
    n, m = x.shape[0], y.shape[0]
    gram = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        gram[i] = np.exp(-np.sum(np.abs(x[i] - y), axis=1) / bandwidth)
    return gram


def median_bandwidth(x: np.ndarray, y: np.ndarray, max_samples: int = 500) -> float:
    """Median heuristic: σ = sqrt(0.5 * median(||x_i - x_j||^2)).

    Uses a random subsample of the pooled data for efficiency.
    Returns 1.0 if all pairwise distances are zero (degenerate case).
    """
    rng = np.random.default_rng(0)
    stacked = np.vstack([x, y])
    n = stacked.shape[0]
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        sub = stacked[idx].astype(np.float64)
    else:
        sub = stacked.astype(np.float64)
    sq = _sq_dists(sub, sub)
    positive = sq[sq > 0]
    if positive.size == 0:
        return 1.0
    return float(np.sqrt(0.5 * np.median(positive)))


def compute_gram(
    x: np.ndarray,
    y: np.ndarray,
    kernel: str,
    bandwidth: float,
) -> np.ndarray:
    """Dispatch to the appropriate kernel."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if kernel == "rbf":
        return rbf_gram(x, y, bandwidth)
    if kernel == "laplace":
        return laplace_gram(x, y, bandwidth)
    raise ValueError(f"Unknown kernel: {kernel!r}. Choose 'rbf' or 'laplace'.")


def biased_mmd2(kxx: np.ndarray, kyy: np.ndarray, kxy: np.ndarray) -> float:
    """Biased MMD² = E[k(X,X')] + E[k(Y,Y')] - 2·E[k(X,Y)]."""
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
