"""Variant (T) on Tiger (sigma=0.5, random policy, T=20) as N grows: does the analytic threshold eventually
resolve the two classes?  B in {2, 8}, 5 seeds.  Output results/tiger_T_scaling.csv"""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import env_spec, collect, ari_from_bin_labels
import numpy as np
from continuous_quotient.variant_t import VariantTConfig, fit_variant_t, null_threshold
rows = []
sp = env_spec("tiger")
for N in [3000, 10000, 30000]:
    for seed in range(5):
        data = collect(sp.make(), N, 20, seed)
        for B in [2, 8]:
            res = fit_variant_t(data.dataset, VariantTConfig(n_bins=B, mode="analytic", seed=seed))
            ari, _ = ari_from_bin_labels(data, res.boundaries, res.bin_class)
            ms = [p["m"] for p in res.pair_info]
            stats = [p["stat"] for p in res.pair_info]
            rows.append(dict(N=N, B=B, seed=seed, n_hat=res.n_classes, ari=round(ari, 4), m_med=int(np.median(ms)),
                             max_stat=round(max(stats), 4), thr_med=round(float(np.median([p["thr"] for p in res.pair_info])), 4)))
            print(rows[-1], flush=True)
with open(Path(__file__).parent / "results" / "tiger_T_scaling.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
