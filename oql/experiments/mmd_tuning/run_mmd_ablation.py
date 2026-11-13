"""Focused MMD ablation on Tiger to match G-test performance.

Tests combinations of:
  - bandwidth heuristics: median, 0.5x median, 2x median, Silverman, cross-val
  - n_permutations: 200, 500, 1000
  - kernels: RBF, Laplacian
  - subsample_max: 200, 500, None (all)

Fixed Tiger config: history_max_length=2, min_history_occurrences=80, N=5000
10 seeds (0-9).

Reports mean ± std n_classes for each variant.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quotient_learning.types import ContinuationSample as _CS
from quotient_learning.equivalence import GTestEquivalenceTest, MMDEquivalenceTest
from quotient_learning.estimator import QuotientEstimator, _UnionFind
from quotient_learning.sampling import enumerate_observed_histories, extract_continuation_samples
from quotient_learning.errors import InsufficientSamplesError
from environments.layer1_discrete.tiger import TigerPOMDP


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
TIGER_CFG = {
    "window_length": 2,
    "history_max_length": 2,
    "min_history_occurrences": 80,
    "alpha": 0.01,
    "n_trajectories": 5000,
    "max_steps_per_episode": 20,
}


def _subsample_cs(cs: _CS, max_n: int, rng: np.random.Generator) -> _CS:
    if cs.n_samples <= max_n:
        return cs
    idx = rng.choice(cs.n_samples, size=max_n, replace=False)
    conts = tuple(cs.continuations[i] for i in sorted(idx))
    return _CS(history=cs.history, continuations=conts, window_length=cs.window_length)


def _estimate_with_test(dataset, eqtest, cfg, subsample_max: int | None = None) -> int:
    """Run quotient estimation on pre-collected dataset. Returns n_classes."""
    rng = np.random.default_rng(42)
    histories = enumerate_observed_histories(
        dataset,
        max_length=cfg["history_max_length"],
        min_occurrences=cfg["min_history_occurrences"],
    )
    if not histories:
        return 0
    samples = {}
    viable = []
    for h in histories:
        try:
            cs = extract_continuation_samples(dataset, h, cfg["window_length"])
        except Exception:
            continue
        if subsample_max is not None:
            cs = _subsample_cs(cs, subsample_max, rng)
        if cs.n_samples >= max(30, cfg["min_history_occurrences"] // 2):
            samples[h] = cs
            viable.append(h)
    if not viable:
        return 0
    uf = _UnionFind(viable)
    for i, ha in enumerate(viable):
        for hb in viable[i + 1:]:
            try:
                res = eqtest.test(samples[ha], samples[hb], alpha=cfg["alpha"])
            except InsufficientSamplesError:
                continue
            if res.same_class:
                uf.union(ha, hb)
    return len(uf.groups())


class MMDVariantTest(MMDEquivalenceTest):
    """Extended MMD test with pluggable bandwidth and kernel."""

    def __init__(
        self,
        n_observations: int,
        bw_mode: str = "median",
        n_permutations: int = 200,
        kernel: str = "rbf",
        seed: int = 0,
    ):
        super().__init__(n_observations=n_observations, bandwidth=None,
                         n_permutations=n_permutations, seed=seed)
        self.bw_mode = bw_mode
        self.kernel = kernel

    def _compute_bandwidth(self, x: np.ndarray, y: np.ndarray) -> float:
        stacked = np.vstack([x, y])
        n = stacked.shape[0]
        # Use all points if <= 500, otherwise subsample
        idx = self._rng.choice(n, size=min(n, 500), replace=False)
        sub = stacked[idx]
        d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
        pos_d2 = d2[d2 > 0]
        med_d2 = float(np.median(pos_d2)) if len(pos_d2) > 0 else 1.0

        if self.bw_mode == "median":
            # Standard median heuristic: sigma = sqrt(0.5 * median(||xi-xj||^2))
            return float(np.sqrt(0.5 * med_d2)) or 1.0
        elif self.bw_mode == "0.5x_median":
            return float(0.5 * np.sqrt(0.5 * med_d2)) or 1.0
        elif self.bw_mode == "2x_median":
            return float(2.0 * np.sqrt(0.5 * med_d2)) or 1.0
        elif self.bw_mode == "silverman":
            # Silverman's rule for 1D: h = 1.06 * sigma * n^(-1/5)
            # For d-dimensional one-hot, use mean pairwise distance as proxy
            # h = 1.06 * std(||xi||) * n^(-1/(d+4))
            d = sub.shape[1]
            norms = np.sqrt((sub * sub).sum(axis=1))
            sigma = float(np.std(norms)) or 1.0
            bw = 1.06 * sigma * (sub.shape[0] ** (-1.0 / (d + 4)))
            return max(bw, 0.01)
        elif self.bw_mode == "cross_val":
            # Approximate cross-validation: maximize held-out MMD² power
            # Test a grid of bandwidths and pick the one maximising test statistic
            candidates = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
            # Scale by median distance
            median_dist = float(np.sqrt(med_d2)) or 1.0
            best_bw = median_dist
            best_stat = -np.inf
            xa, xb = x, y
            for scale in candidates:
                bw = scale * median_dist
                kxx = self._rbf_gram(xa, xa, bw)
                kyy = self._rbf_gram(xb, xb, bw)
                kxy = self._rbf_gram(xa, xb, bw)
                stat = float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
                if stat > best_stat:
                    best_stat = stat
                    best_bw = bw
            return best_bw
        else:
            raise ValueError(f"Unknown bw_mode: {self.bw_mode}")

    def _gram(self, x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
        if self.kernel == "rbf":
            return self._rbf_gram(x, y, bandwidth)
        elif self.kernel == "laplacian":
            # K(x,y) = exp(-||x-y||_1 / bandwidth)
            # For one-hot vectors, L1 = sum of |x_i - y_i|
            # We compute elementwise L1
            diff = np.abs(x[:, None, :] - y[None, :, :]).sum(axis=-1)
            return np.exp(-diff / max(bandwidth, 1e-9))
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def test(self, samples_a, samples_b, alpha: float = 0.05):
        self._check_preconditions(samples_a, samples_b, alpha)
        x = self._embed(samples_a)
        y = self._embed(samples_b)
        bandwidth = self._compute_bandwidth(x, y)

        kxx = self._gram(x, x, bandwidth)
        kyy = self._gram(y, y, bandwidth)
        kxy = self._gram(x, y, bandwidth)
        observed = self._mmd2(kxx, kyy, kxy)

        combined = np.vstack([x, y])
        m = x.shape[0]
        ge = 0
        for _ in range(self.n_permutations):
            perm = self._rng.permutation(combined.shape[0])
            xp = combined[perm[:m]]
            yp = combined[perm[m:]]
            kxx_p = self._gram(xp, xp, bandwidth)
            kyy_p = self._gram(yp, yp, bandwidth)
            kxy_p = self._gram(xp, yp, bandwidth)
            stat_p = self._mmd2(kxx_p, kyy_p, kxy_p)
            if stat_p >= observed:
                ge += 1
        p_value = float((ge + 1) / (self.n_permutations + 1))
        from quotient_learning.types import EquivalenceTestResult
        return EquivalenceTestResult(
            history_a=samples_a.history, history_b=samples_b.history,
            test_statistic=float(observed), p_value=p_value,
            same_class=(p_value >= alpha), alpha=alpha,
        )


def run_ablation():
    print("Collecting Tiger datasets for 10 seeds...")
    datasets = {}
    n_obs = None
    for seed in range(N_SEEDS):
        env = TigerPOMDP(seed=seed)
        if n_obs is None:
            n_obs = env.n_observations
        collector = QuotientEstimator(
            env=env,
            n_trajectories=TIGER_CFG["n_trajectories"],
            window_length=TIGER_CFG["window_length"],
            history_max_length=TIGER_CFG["history_max_length"],
            min_history_occurrences=TIGER_CFG["min_history_occurrences"],
            alpha=TIGER_CFG["alpha"],
            max_steps_per_episode=TIGER_CFG["max_steps_per_episode"],
            equivalence_test=GTestEquivalenceTest(),
        )
        datasets[seed] = collector.collect(seed=seed)
        print(f"  collected seed={seed}")

    # Reference: G-test
    print("\nG-test reference (all seeds):")
    gtest = GTestEquivalenceTest()
    gtest_results = []
    for seed in range(N_SEEDS):
        nc = _estimate_with_test(datasets[seed], gtest, TIGER_CFG, subsample_max=None)
        gtest_results.append(nc)
    arr = np.array(gtest_results)
    print(f"  GTest: mean={arr.mean():.2f} std={arr.std(ddof=1):.2f} per_seed={gtest_results}")

    # Variants to test
    bw_modes = ["median", "0.5x_median", "2x_median", "silverman", "cross_val"]
    n_perms_list = [200, 500, 1000]
    kernels = ["rbf", "laplacian"]
    subsample_maxes = [200, 500, None]

    results = []
    best_score = -1
    best_cfg = None

    total_variants = len(bw_modes) * len(n_perms_list) * len(kernels) * len(subsample_maxes)
    print(f"\nTesting {total_variants} MMD variants across {N_SEEDS} seeds...\n")

    variant_num = 0
    for bw_mode in bw_modes:
        for n_perms in n_perms_list:
            for kernel in kernels:
                for subsample_max in subsample_maxes:
                    variant_num += 1
                    t0 = time.time()
                    nc_list = []
                    for seed in range(N_SEEDS):
                        mmd = MMDVariantTest(
                            n_observations=n_obs,
                            bw_mode=bw_mode,
                            n_permutations=n_perms,
                            kernel=kernel,
                            seed=seed,
                        )
                        nc = _estimate_with_test(
                            datasets[seed], mmd, TIGER_CFG,
                            subsample_max=subsample_max,
                        )
                        nc_list.append(nc)
                    arr = np.array(nc_list)
                    mean_nc = arr.mean()
                    std_nc = arr.std(ddof=1)
                    elapsed = time.time() - t0
                    # Score: how many seeds get n_classes=2
                    n_correct = int((arr == 2).sum())
                    sub_str = str(subsample_max) if subsample_max else "all"
                    label = f"bw={bw_mode} perms={n_perms} kernel={kernel} sub={sub_str}"
                    print(f"[{variant_num:3d}/{total_variants}] {label}")
                    print(f"         mean={mean_nc:.2f} std={std_nc:.2f} correct={n_correct}/10 "
                          f"per_seed={nc_list} ({elapsed:.1f}s)")
                    results.append({
                        "bw_mode": bw_mode,
                        "n_permutations": n_perms,
                        "kernel": kernel,
                        "subsample_max": sub_str,
                        "mean_n_classes": round(mean_nc, 3),
                        "std_n_classes": round(std_nc, 3),
                        "n_correct": n_correct,
                        "per_seed": nc_list,
                    })
                    if n_correct > best_score or (n_correct == best_score and mean_nc > (best_cfg or {}).get("mean_n_classes", 0)):
                        best_score = n_correct
                        best_cfg = results[-1]

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION:")
    if best_cfg:
        print(f"  bw_mode={best_cfg['bw_mode']} n_perms={best_cfg['n_permutations']} "
              f"kernel={best_cfg['kernel']} subsample_max={best_cfg['subsample_max']}")
        print(f"  mean={best_cfg['mean_n_classes']:.2f} std={best_cfg['std_n_classes']:.2f} "
              f"correct={best_cfg['n_correct']}/10")

    # Write summary
    import csv
    out_csv = RESULTS_DIR / "mmd_ablation_tiger.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "bw_mode", "n_permutations", "kernel", "subsample_max",
            "mean_n_classes", "std_n_classes", "n_correct", "per_seed",
        ])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nResults written to {out_csv}")

    return results, best_cfg


if __name__ == "__main__":
    run_ablation()
