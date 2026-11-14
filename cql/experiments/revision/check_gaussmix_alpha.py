"""Resolve the GaussMix alpha discrepancy: the shipped CSV has no alpha column and the script default is 0.05.
Re-run the manual-B configuration (B=N_s, R=99, N=2000 trajectories, seeds 0-2) at alpha=0.05 and alpha=0.01.
Output: results/gaussmix_alpha_check.csv"""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import env_spec, collect, ROOT
from continuous_quotient import ContinuousEstimatorConfig, ContinuousQuotientEstimator
rows = []
for Ns in [3, 4, 5, 6]:
    sp = env_spec(f"gauss{Ns}")
    for seed in range(3):
        data = collect(sp.make(), 2000, 50, seed)
        for alpha in [0.05, 0.01]:
            cfg = ContinuousEstimatorConfig(n_trajectories=2000, max_steps_per_episode=50, window_length=2,
                    history_max_length=1, min_history_occurrences=20, alpha=alpha, n_permutations=99,
                    max_samples_per_test=300, history_n_bins=Ns)
            q = ContinuousQuotientEstimator(cfg).fit(data.dataset)
            rows.append(dict(Ns=Ns, seed=seed, alpha=alpha, R=99, n_hat=q.n_classes))
            print(rows[-1], flush=True)
with open(Path(__file__).parent / "results" / "gaussmix_alpha_check.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
