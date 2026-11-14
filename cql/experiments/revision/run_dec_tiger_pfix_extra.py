"""Extra seeds for Dec-Tiger permutation+BH on variant (T)'s data pipeline (P^fix, per_traj sampler) at N=20000,
B=4: seeds 3..9 (seeds 0..2 are in dec_tiger_variant_t.csv), both sigma pairs.  Output:
results/dec_tiger_pfix_N20000_seeds3_9.csv  (use 'sampling' column; same settings as run_dec_tiger_variant_t.py)"""
from __future__ import annotations
import csv, importlib.util as ilu, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "layer3_rcrit"))
from continuous_quotient.variant_t import VariantTConfig, fit_variant_t
from continuous_quotient.estimator import _obs_scalar
from envs.dec_tiger_continuous import DecTigerContinuousEnv

def _dt():
    spec = ilu.spec_from_file_location("dt", ROOT / "experiments/layer3_rcrit/run_dec_tiger_continuous.py")
    m = ilu.module_from_spec(spec); spec.loader.exec_module(m); return m

def labels(ds, res):
    out = []
    for t in ds.trajectories:
        z = np.array([_obs_scalar(o) for o in t.observations])
        b = np.searchsorted(res.boundaries, z)
        out.append([res.bin_class.get(int(x), -1) for x in b])
    return np.array(out)

def job(args):
    sa, sb, N, seed = args
    dt = _dt()
    true = dt.analytical_rcrit(sa, sb)
    env = DecTigerContinuousEnv(sigma_a=sa, sigma_b=sb, episode_length=20)
    dA, dB = dt.collect_dec_tiger_trajs(env, N, 20, seed)
    cfg = VariantTConfig(n_bins=4, mode="perm_bh", alpha=0.05, n_permutations=99, min_samples=10, seed=seed, sampling="per_traj")
    rA, rB = fit_variant_t(dA, cfg), fit_variant_t(dB, cfg)
    LA, LB = labels(dA, rA), labels(dB, rB)
    ok = (LA >= 0) & (LB >= 0)
    est = dt._estimate_rcrit(LA[ok].tolist(), LB[ok].tolist()) if ok.sum() else float("nan")
    return dict(sigma_a=sa, sigma_b=sb, mode="perm_bh", N=N, sampling="per_traj", seed=seed, n_A=rA.n_classes, n_B=rB.n_classes,
                rcrit_est=round(est, 4), rcrit_true=round(true, 4), abs_err=round(abs(est - true), 4))

if __name__ == "__main__":
    jobs = [(sa, sb, 20000, s) for (sa, sb) in [(0.5, 0.5), (0.1, 1.0)] for s in range(3, 10)]
    rows = []
    with ProcessPoolExecutor(10) as ex, open(HERE / "results" / "dec_tiger_pfix_N20000_seeds3_9.csv", "w", newline="") as f:
        w = None
        for r in ex.map(job, jobs):
            if w is None: w = csv.DictWriter(f, fieldnames=list(r)); w.writeheader()
            w.writerow(r); f.flush(); print(r, flush=True)
