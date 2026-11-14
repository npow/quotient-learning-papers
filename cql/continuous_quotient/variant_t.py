"""Analysed variant (T) of the quotient estimator (Theorem 1 of the paper).

Differences from :class:`ContinuousQuotientEstimator` (the "implemented"
procedure P), each removing one unanalysed feature:

* **Fixed bins and kernel.**  Quantile edges and the RBF bandwidth (median
  heuristic) are fitted on a *design* half of the trajectories; all tests use
  the other half, so bins and kernel are non-random w.r.t. the test data.
* **i.i.d. samples, no overlapping windows.**  Each test-half trajectory
  contributes at most ONE continuation window per bin (a uniformly chosen
  visit), so samples inside a bin come from distinct i.i.d. trajectories.
* **Independent sides.**  For every pair (b1, b2) the test-half trajectories
  are randomly split in two; side 1 uses b1-windows of half 1, side 2 uses
  b2-windows of half 2, so the two samples are independent.
* **Analytic threshold + union bound.**  A pair is declared different iff
  sqrt(biased MMD^2) > t_{alpha0}(m) with the null threshold of Gretton et al.
  (2012, Cor. 9) and alpha0 = delta / K_hist^2 (no permutations, no BH).

``mode="perm_bh"`` keeps the same data pipeline (fixed bins/kernel, i.i.d.
independent samples) but calibrates by permutation and applies BH; it is the
intermediate configuration used to attribute differences between P and T.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from continuous_quotient.estimator import _UnionFind, _obs_scalar
from continuous_quotient.kernels import biased_mmd2, compute_gram, median_bandwidth
from continuous_quotient.types import ContinuousTrajectory, ContinuousTrajectoryDataset


@dataclass
class VariantTConfig:
    n_bins: int = 8
    window_length: int = 2
    delta: float = 0.05            # overall failure probability (analytic mode)
    mode: str = "analytic"         # "analytic" | "perm_bh"
    alpha: float = 0.05            # BH level (perm_bh mode)
    n_permutations: int = 199      # perm_bh mode
    min_samples: int = 20          # min per-side sample size to test a pair
    design_frac: float = 0.5
    seed: int = 0
    # "per_bin": one uniformly chosen visit per (trajectory, bin).  Samples are i.i.d. across trajectories, but
    #            the sampled law is the "one visit per trajectory" law, which equals the pooled bin law P_b only
    #            if within-trajectory visits are exchangeable (fast mixing).
    # "per_traj": ONE uniformly chosen time per trajectory (window assigned to the bin of that time).  The
    #            sampled law is exactly P_b (pooled law) with no mixing assumption, at the price of ~B times
    #            fewer samples per bin.
    sampling: str = "per_bin"


@dataclass
class VariantTResult:
    n_classes: int
    bin_class: dict[int, int]      # active bin -> class id
    boundaries: np.ndarray
    bandwidth: float
    n_active: int
    pair_info: list[dict] = field(default_factory=list)  # per pair: m, stat, thr/p
    untestable_pairs: int = 0


def traj_windows(traj: ContinuousTrajectory, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (z[t], cont[t]) for t with a full k-step continuation (same layout as the estimator)."""
    obs, acts = traj.observations, traj.actions
    T = len(obs)
    z = np.array([_obs_scalar(o) for o in obs])
    conts = []
    for t in range(T):
        if t + 1 + k > T:
            break
        parts = []
        for ci in range(k):
            parts.append(np.asarray(obs[t + 1 + ci], dtype=np.float64).ravel())
            if ci < k - 1:
                a = acts[t + 1 + ci] if t + 1 + ci < len(acts) else 0
                if isinstance(a, (int, np.integer)):
                    parts.append(np.array([float(a)]))
                else:
                    parts.append(np.asarray(a, dtype=np.float64).ravel())
        conts.append(np.concatenate(parts))
    return z, (np.stack(conts) if conts else np.zeros((0, 0)))


def null_threshold(m: int, alpha0: float, kappa: float = 1.0) -> float:
    """t_alpha = sqrt(2 kappa^2 / m) (1 + sqrt(2 log(1/alpha)))  [Gretton 2012, Cor. 9]."""
    return float(np.sqrt(2.0 * kappa**2 / m) * (1.0 + np.sqrt(2.0 * np.log(1.0 / alpha0))))


def required_m(eps: float, alpha0: float, delta_prime: float, kappa: float = 1.0) -> float:
    """Per-pair sample size of Lemma A.1 (eq. m-pair) guaranteeing detection of MMD gap eps."""
    return 16.0 * kappa**2 / eps**2 * (8.0 + np.log(2.0 / (alpha0 * delta_prime)))


def fit_variant_t(
    dataset: ContinuousTrajectoryDataset | list[ContinuousTrajectory],
    cfg: VariantTConfig,
) -> VariantTResult:
    trajs = dataset.trajectories if isinstance(dataset, ContinuousTrajectoryDataset) else list(dataset)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(trajs))
    n_design = int(len(trajs) * cfg.design_frac)
    design = [trajs[i] for i in order[:n_design]]
    test = [trajs[i] for i in order[n_design:]]
    k = cfg.window_length

    # --- design half: edges and bandwidth ---------------------------------
    zs = np.concatenate([np.array([_obs_scalar(o) for o in t.observations]) for t in design])
    pcts = np.linspace(0, 100, cfg.n_bins + 1)[1:-1]
    boundaries = np.percentile(zs, pcts)
    dconts = [traj_windows(t, k)[1] for t in design]
    dconts = np.concatenate([c for c in dconts if c.shape[0] > 0])
    sub = dconts[rng.choice(len(dconts), size=min(1000, len(dconts)), replace=False)]
    bw = median_bandwidth(sub, sub)

    # --- test half: one window per (trajectory, bin) -----------------------
    per_bin: dict[int, list[tuple[int, np.ndarray]]] = {b: [] for b in range(cfg.n_bins)}
    for ti, t in enumerate(test):
        z, c = traj_windows(t, k)
        if c.shape[0] == 0:
            continue
        bins = np.searchsorted(boundaries, z[: c.shape[0]])
        if cfg.sampling == "per_traj":
            tstar = int(rng.integers(0, c.shape[0]))
            per_bin[int(bins[tstar])].append((ti, c[tstar]))
            continue
        for b in np.unique(bins):
            idx = np.flatnonzero(bins == b)
            per_bin[int(b)].append((ti, c[rng.choice(idx)]))
    active = [b for b in range(cfg.n_bins) if len(per_bin[b]) >= 2 * cfg.min_samples]
    K = len(active)
    alpha0 = cfg.delta / max(K, 1) ** 2

    pairs = [(active[i], active[j]) for i in range(K) for j in range(i + 1, K)]
    info: list[dict] = []
    untestable = 0
    decisions: dict[tuple[int, int], bool | None] = {}
    pvals: list[float] = []
    ptest: list[tuple[int, int]] = []
    for (b1, b2) in pairs:
        half = rng.permutation(len(test)) < len(test) // 2  # True -> half 1
        X = np.array([w for ti, w in per_bin[b1] if half[ti]])
        Y = np.array([w for ti, w in per_bin[b2] if not half[ti]])
        m = min(len(X), len(Y))
        if m < cfg.min_samples:
            untestable += 1
            decisions[(b1, b2)] = None  # not testable -> merged (no evidence of difference)
            continue
        X = X[rng.choice(len(X), m, replace=False)]
        Y = Y[rng.choice(len(Y), m, replace=False)]
        kxx, kyy, kxy = (compute_gram(X, X, "rbf", bw), compute_gram(Y, Y, "rbf", bw),
                         compute_gram(X, Y, "rbf", bw))
        stat = float(np.sqrt(max(biased_mmd2(kxx, kyy, kxy), 0.0)))
        if cfg.mode == "analytic":
            thr = null_threshold(m, alpha0)
            decisions[(b1, b2)] = stat > thr
            info.append(dict(b1=b1, b2=b2, m=m, stat=stat, thr=thr))
        else:
            comb = np.vstack([X, Y])
            kall = compute_gram(comb, comb, "rbf", bw)
            ge = 0
            for _ in range(cfg.n_permutations):
                p = rng.permutation(2 * m)
                i1, i2 = p[:m], p[m:]
                s = kall[np.ix_(i1, i1)].mean() + kall[np.ix_(i2, i2)].mean() - 2 * kall[np.ix_(i1, i2)].mean()
                ge += s >= stat**2
            pv = (ge + 1) / (cfg.n_permutations + 1)
            pvals.append(pv)
            ptest.append((b1, b2))
            info.append(dict(b1=b1, b2=b2, m=m, stat=stat, p=pv))
    if cfg.mode == "perm_bh":
        mp = len(pvals)
        thr_bh = 0.0
        if mp:
            srt = np.argsort(pvals)
            for rank, i in enumerate(srt, start=1):
                if pvals[i] <= rank * cfg.alpha / mp:
                    thr_bh = pvals[i]
        for pr, pv in zip(ptest, pvals):
            decisions[pr] = pv <= thr_bh

    uf = _UnionFind(active)
    for pr, rej in decisions.items():
        if not rej:  # None (untestable) or False -> merge
            uf.union(*pr)
    groups = uf.groups()
    bin_class: dict[int, int] = {}
    for cid, members in enumerate(groups.values()):
        for b in members:
            bin_class[b] = cid
    return VariantTResult(n_classes=len(groups), bin_class=bin_class, boundaries=boundaries,
                          bandwidth=bw, n_active=K, pair_info=info, untestable_pairs=untestable)
