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
SAMPLING = sys.argv[1] if len(sys.argv) > 1 else "per_bin"
NS = [3000, 10000, 30000] if SAMPLING == "per_bin" else [3000, 10000, 30000, 100000]
for N in NS:
    for seed in range(5):
        data = collect(sp.make(), N, 20, seed)
        for B in [2, 8]:
            res = fit_variant_t(data.dataset, VariantTConfig(n_bins=B, mode="analytic", seed=seed, sampling=SAMPLING))
            ari, _ = ari_from_bin_labels(data, res.boundaries, res.bin_class)
            ms = [p["m"] for p in res.pair_info]
            stats = [p["stat"] for p in res.pair_info]
            rows.append(dict(sampling=SAMPLING, N=N, B=B, seed=seed, n_hat=res.n_classes, ari=round(ari, 4), m_med=int(np.median(ms)),
                             max_stat=round(max(stats), 4), thr_med=round(float(np.median([p["thr"] for p in res.pair_info])), 4)))
            print(rows[-1], flush=True)
with open(Path(__file__).parent / "results" / ("tiger_T_scaling.csv" if SAMPLING == "per_bin" else "tiger_T_scaling_pertraj.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
