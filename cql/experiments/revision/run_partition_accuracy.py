"""Partition-level accuracy (ARI vs latent class) of the implemented estimator (P) and of variant (T)
under B-selection rules that do not use the latent index n.

B rules:  sil  = silhouette auto-selection (Alg. 2, uses observations only)
          f8   = fixed B = 8 (package default)
          oracle = B = n  (reference only; uses n)
Estimators: P (implemented: permutation+BH, median bandwidth per pair, overlapping windows),
            T (analysed: fixed kernel/bins from a design half, i.i.d. one-window-per-trajectory,
               independent sides, analytic threshold, alpha0=delta/K^2, delta=0.05),
            Pfix (T's data pipeline, permutation + BH).
Output: experiments/revision/results/partition_accuracy.csv  (one row per run)
Usage: python run_partition_accuracy.py [--seeds 5] [--workers 8] [--envs tiger kcycle2 ...]
"""
from __future__ import annotations
import argparse, csv, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import env_spec, collect, ari_from_bin_labels, ari_bin_ceiling, ROOT  # noqa: E402

import numpy as np
from continuous_quotient import ContinuousEstimatorConfig, ContinuousQuotientEstimator
from continuous_quotient.estimator import auto_select_bins, _fit_obs_bins, _make_coarse_history_key
from continuous_quotient.variant_t import VariantTConfig, fit_variant_t

OUT = Path(__file__).resolve().parent / "results" / "partition_accuracy.csv"
FIELDS = ["env", "n_true", "N", "seed", "estimator", "b_rule", "B", "alpha", "R", "n_hat", "ari", "ari_ceiling", "coverage",
          "extra", "runtime_s"]


def run_one(args):
    env, seed, b_rule, estimator = args
    sp = env_spec(env)
    t0 = time.perf_counter()
    data = collect(sp.make(), sp.N, sp.T, seed)
    if b_rule == "sil":
        B = auto_select_bins(data.dataset)
    elif b_rule == "f8":
        B = 8
    else:
        B = sp.n_latent
    extra = ""
    if estimator == "P":
        cfg = ContinuousEstimatorConfig(n_trajectories=sp.N, max_steps_per_episode=sp.T, window_length=2,
                                        history_max_length=1, min_history_occurrences=20, alpha=sp.alpha,
                                        n_permutations=sp.R, max_samples_per_test=300, history_n_bins=B)
        q = ContinuousQuotientEstimator(cfg).fit(data.dataset)
        bnd = _fit_obs_bins(data.dataset, B)
        bin_class = {key[0]: cid for key, cid in q.partition.items()}
        n_hat = q.n_classes
    else:
        mode = "analytic" if estimator == "T" else "perm_bh"
        cfg = VariantTConfig(n_bins=B, mode=mode, alpha=sp.alpha, n_permutations=sp.R, seed=seed)
        res = fit_variant_t(data.dataset, cfg)
        bnd, bin_class, n_hat = res.boundaries, res.bin_class, res.n_classes
        ms = [p["m"] for p in res.pair_info]
        extra = f"m_med={int(np.median(ms)) if ms else 0};untestable={res.untestable_pairs};K={res.n_active}"
    ari, cov = ari_from_bin_labels(data, bnd, bin_class)
    ceil = ari_bin_ceiling(data, bnd)
    return dict(env=env, n_true=sp.n_latent, N=sp.N, seed=seed, estimator=estimator, b_rule=b_rule, B=B,
                alpha=sp.alpha if estimator != "T" else "delta=0.05", R=sp.R if estimator != "T" else 0,
                n_hat=n_hat, ari=round(ari, 4), ari_ceiling=round(ceil, 4), coverage=round(cov, 4), extra=extra,
                runtime_s=round(time.perf_counter() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--envs", nargs="+", default=["tiger", "kcycle2", "kcycle3", "kcycle4", "kcycle5", "kcycle8",
                                                  "gauss3", "gauss4", "gauss5", "gauss6"])
    ap.add_argument("--estimators", nargs="+", default=["P", "T", "Pfix"])
    ap.add_argument("--rules", nargs="+", default=["sil", "f8", "oracle"])
    ap.add_argument("--pfix-envs", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    jobs = [(e, s, r, est) for e in a.envs for s in range(a.seeds) for r in a.rules for est in a.estimators]
    # most expensive first
    jobs = [j for j in jobs if not (j[3] == 'Pfix' and j[2] == 'sil')]
    if a.pfix_envs is not None:   # Pfix is ~3x slower than P: run it only on the listed envs
        jobs = [j for j in jobs if j[3] != 'Pfix' or j[0] in a.pfix_envs]
    jobs.sort(key=lambda j: (j[3] == 'T', ))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(a.workers) as ex, open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in ex.map(run_one, jobs):
            rows.append(r); w.writerow(r); f.flush()
            print(r, flush=True)


if __name__ == "__main__":
    main()
