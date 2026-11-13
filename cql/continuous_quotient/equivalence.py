"""ContinuousMMDTest: kernel MMD² equivalence test for continuous continuations.

Operates on raw float32 continuation matrices rather than integer-encoded
discrete tokens. Replaces the one-hot embedding step in the upstream
MMDEquivalenceTest with identity (the vectors are already in ℝ^d).

Statistical guarantees (Gretton et al. 2012):
  - Type I error ≤ α for all finite m  (permutation calibration)
  - Type II error → 0 as m → ∞          (characteristic kernel)
"""

from __future__ import annotations

import numpy as np

from continuous_quotient.kernels import biased_mmd2, compute_gram, median_bandwidth
from continuous_quotient.types import ContinuousEquivalenceResult, HistoryKey


class BandwidthEstimationError(Exception):
    pass


class ContinuousMMDTest:
    """Two-sample MMD permutation test for continuous k-step continuation distributions.

    Parameters
    ----------
    kernel:       "rbf" (default) or "laplace"
    bandwidth:    kernel bandwidth σ; None → median heuristic at test time
    n_permutations: number of permutations for the null distribution (default 199)
    alpha:        significance level (used as default when not overridden in .test())
    seed:         RNG seed for permutations
    """

    def __init__(
        self,
        kernel: str = "rbf",
        bandwidth: float | None = None,
        n_permutations: int = 199,
        alpha: float = 0.01,
        seed: int = 0,
    ):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.n_permutations = n_permutations
        self.alpha = alpha
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def test(
        self,
        samples_a: np.ndarray,
        samples_b: np.ndarray,
        h1_key: HistoryKey = (),
        h2_key: HistoryKey = (),
        alpha: float | None = None,
    ) -> ContinuousEquivalenceResult:
        """Test whether two sets of continuation samples share the same distribution.

        Parameters
        ----------
        samples_a: float32 array of shape (m_a, d) — continuation vectors from history h1
        samples_b: float32 array of shape (m_b, d) — continuation vectors from history h2
        h1_key, h2_key: hashable history keys (for bookkeeping)
        alpha: significance level; defaults to self.alpha

        Returns
        -------
        ContinuousEquivalenceResult with reject=True if distributions differ.
        """
        if alpha is None:
            alpha = self.alpha
        a = np.asarray(samples_a, dtype=np.float64)
        b = np.asarray(samples_b, dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        if b.ndim == 1:
            b = b[:, None]

        # Bandwidth selection.
        bw = self.bandwidth
        if bw is None:
            bw = median_bandwidth(a, b)
        if bw <= 0:
            raise BandwidthEstimationError(
                f"Estimated bandwidth {bw} ≤ 0. All pairwise distances are zero."
            )

        # Compute Gram matrices.
        kxx = compute_gram(a, a, self.kernel, bw)
        kyy = compute_gram(b, b, self.kernel, bw)
        kxy = compute_gram(a, b, self.kernel, bw)
        observed = biased_mmd2(kxx, kyy, kxy)

        # Permutation null distribution.
        combined = np.vstack([a, b])
        m_a = a.shape[0]
        ge = 0
        for _ in range(self.n_permutations):
            perm = self._rng.permutation(combined.shape[0])
            xp = combined[perm[:m_a]]
            yp = combined[perm[m_a:]]
            kxx_p = compute_gram(xp, xp, self.kernel, bw)
            kyy_p = compute_gram(yp, yp, self.kernel, bw)
            kxy_p = compute_gram(xp, yp, self.kernel, bw)
            if biased_mmd2(kxx_p, kyy_p, kxy_p) >= observed:
                ge += 1

        p_value = float((ge + 1) / (self.n_permutations + 1))
        return ContinuousEquivalenceResult(
            h1_key=h1_key,
            h2_key=h2_key,
            mmd2=float(observed),
            p_value=p_value,
            reject=(p_value < alpha),
            n_permutations=self.n_permutations,
            bandwidth=bw,
        )
