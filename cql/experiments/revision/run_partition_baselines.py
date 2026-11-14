"""Baselines scored by ARI vs the latent state (same data as run_partition_accuracy.py: same envs/seeds).
Methods: W1 (multivariate, eps in {0.1,0.2,0.5,1.0} and silhouette), Ferns bisimulation metric (binned chain),
DBC objective, causal-state learning (GRU + k-means), RKHS epsilon-machine (DBSCAN / k-means-silhouette).
Binned methods use B=8 (fixed) and B=n (oracle reference). Nothing here uses n except the 'oracle' rows.
Output: experiments/revision/results/partition_baselines.csv
"""
from __future__ import annotations
import argparse, csv, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import env_spec, collect  # noqa: E402
import baselines_lib as BL  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "partition_baselines.csv"
FIELDS = ["env", "n_true", "seed", "method", "variant", "B", "n_hat", "ari", "runtime_s"]


def run_one(args):
    env, seed = args
    sp = env_spec(env)
    data = collect(sp.make(), sp.N, sp.T, seed)
    rows = []

    def add(method, variant, B, n_hat, ari, t0):
        rows.append(dict(env=env, n_true=sp.n_latent, seed=seed, method=method, variant=variant, B=B,
                         n_hat=n_hat, ari=round(ari, 4), runtime_s=round(time.perf_counter() - t0, 1)))

    def safe(fn):
        try:
            fn()
        except Exception:
            traceback.print_exc()

    for Bname, B in [("f8", 8), ("oracle", sp.n_latent)]:
        if B < 2:
            continue
        def w1():
            D, bidx = BL.w1_distance_matrix(data, B)
            for eps in [0.1, 0.2, 0.5, 1.0, None]:
                t0 = time.perf_counter()
                import numpy as np
                if eps is None:
                    bl = BL.cluster_distance_matrix(np.where(np.isfinite(D), D, D[np.isfinite(D)].max() * 2))
                    name = "sil"
                else:
                    bl = BL.threshold_merge(D, eps); name = f"eps{eps}"
                lab = BL.bins_to_labels(bidx, bl)
                add("W1-multivariate", f"{name}/{Bname}", B, len(set(bl.tolist())), BL.score_grid(data, lab), t0)
        safe(w1)
        def fe():
            t0 = time.perf_counter()
            lab, k = BL.ferns_labels(data, env, B)
            add("Ferns-bisim-metric", Bname, B, k, BL.score_grid(data, lab), t0)
        safe(fe)
    def dbc():
        t0 = time.perf_counter(); lab, k = BL.dbc_labels(data, env, seed=seed)
        add("DBC", "kmeans-sil", 0, k, BL.score_grid(data, lab), t0)
    safe(dbc)
    def cs():
        t0 = time.perf_counter(); lab, k = BL.causal_labels(data, env, seed=seed)
        add("CausalState-GRU", "kmeans-sil", 0, k, BL.score_grid(data, lab), t0)
    safe(cs)
    def rk():
        for c in ["dbscan", "kmeans"]:
            t0 = time.perf_counter(); ti, tt, lab, k = BL.rkhs_labels(data, seed=seed, cluster=c)
            add("RKHS-eps-machine", c, 0, k, BL.score_flat(data, ti, tt, lab), t0)
    safe(rk)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--envs", nargs="+", default=["tiger", "kcycle2", "kcycle4", "kcycle8", "gauss3", "gauss5"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    jobs = [(e, s) for e in a.envs for s in range(a.seeds)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex, open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for rows in ex.map(run_one, jobs):
            for r in rows:
                w.writerow(r); print(r, flush=True)
            f.flush()
