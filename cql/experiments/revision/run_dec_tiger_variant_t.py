"""Dec-Tiger R_crit pipeline under variant (T) (and the intermediate perm+BH on T's data pipeline).
Same environment/policy/N/B as experiments/layer3_rcrit/run_dec_tiger_continuous.py (N=1000, T=20, B=4,
listen-only, sigma pairs (0.5,0.5) and (0.1,1.0), seeds 0..9).  Output: results/dec_tiger_variant_t.csv
"""
from __future__ import annotations
import csv, importlib.util as ilu, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "layer3_rcrit"))
spec = ilu.spec_from_file_location("dt", ROOT / "experiments/layer3_rcrit/run_dec_tiger_continuous.py")
dt = ilu.module_from_spec(spec); spec.loader.exec_module(dt)
from continuous_quotient.variant_t import VariantTConfig, fit_variant_t
from continuous_quotient.estimator import _obs_scalar
from envs.dec_tiger_continuous import DecTigerContinuousEnv

def labels(ds, res):
    out = []
    for t in ds.trajectories:
        z = np.array([_obs_scalar(o) for o in t.observations])
        b = np.searchsorted(res.boundaries, z)
        out.append([res.bin_class.get(int(x), -1) for x in b])
    return np.array(out)

rows = []
for (sa, sb) in [(0.5, 0.5), (0.1, 1.0)]:
    true = dt.analytical_rcrit(sa, sb)
    for N, mode, nseeds in [(1000, "analytic", 10), (1000, "perm_bh", 10), (20000, "analytic", 10), (20000, "perm_bh", 3)]:
        for seed in range(nseeds):   # perm_bh at N=20000 is ~2 min/seed, so only 3 seeds
            env = DecTigerContinuousEnv(sigma_a=sa, sigma_b=sb, episode_length=20)
            dA, dB = dt.collect_dec_tiger_trajs(env, N, 20, seed)
            cfg = VariantTConfig(n_bins=4, mode=mode, alpha=0.05, n_permutations=99, min_samples=10, seed=seed, sampling='per_traj')
            rA, rB = fit_variant_t(dA, cfg), fit_variant_t(dB, cfg)
            LA, LB = labels(dA, rA), labels(dB, rB)
            ok = (LA >= 0) & (LB >= 0)
            est = dt._estimate_rcrit(LA[ok].tolist(), LB[ok].tolist()) if ok.sum() else float("nan")
            rows.append(dict(sigma_a=sa, sigma_b=sb, mode=mode, N=N, sampling='per_traj', seed=seed, n_A=rA.n_classes, n_B=rB.n_classes,
                             rcrit_est=round(est, 4), rcrit_true=round(true, 4),
                             abs_err=round(abs(est - true), 4)))
            print(rows[-1], flush=True)
with open(HERE / "results" / "dec_tiger_variant_t.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
