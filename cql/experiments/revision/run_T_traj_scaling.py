"""Variant (T) with the exact-pooled-law sampler (per_traj; the sampler Theorem 1 covers) as N grows, at B=n (oracle) and
B=2n, for envs where it merges everything at the paper's N.  5 seeds.  Output results/T_traj_scaling.csv"""
import csv, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import env_spec, collect, ari_from_bin_labels
import numpy as np
from continuous_quotient.variant_t import VariantTConfig, fit_variant_t

def job(a):
    env, N, seed = a
    sp = env_spec(env)
    data = collect(sp.make(), N, sp.T, seed)
    out = []
    for rule, B in [("oracle", sp.n_latent), ("2n", 2 * sp.n_latent)]:
        res = fit_variant_t(data.dataset, VariantTConfig(n_bins=B, mode="analytic", seed=seed, sampling="per_traj"))
        ari, _ = ari_from_bin_labels(data, res.boundaries, res.bin_class)
        ms = [p["m"] for p in res.pair_info]
        out.append(dict(env=env, n_true=sp.n_latent, N=N, seed=seed, b_rule=rule, B=B, sampling="per_traj", n_hat=res.n_classes,
                        ari=round(ari, 4), m_med=int(np.median(ms)) if ms else 0))
    return out

if __name__ == "__main__":
    envs = ["kcycle3", "kcycle4", "kcycle8", "gauss3", "gauss4", "gauss6"]
    jobs = [(e, N, s) for N in [10000, 30000, 100000] for e in envs for s in range(5)]
    jobs.sort(key=lambda j: -j[1])
    with ProcessPoolExecutor(6) as ex, open(Path(__file__).parent / "results" / "T_traj_scaling.csv", "w", newline="") as f:
        w = None
        for rows in ex.map(job, jobs):
            for r in rows:
                if w is None: w = csv.DictWriter(f, fieldnames=list(r)); w.writeheader()
                w.writerow(r); f.flush()
            print(rows[0]["env"], rows[0]["N"], [(r["b_rule"], r["n_hat"]) for r in rows], flush=True)
