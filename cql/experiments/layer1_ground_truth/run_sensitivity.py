"""Sensitivity runs used in the paper (main Table 2; supplement D.2-D.3).

Usage:
    python experiments/layer1_ground_truth/run_sensitivity.py kcycle  # K-cycle, B=2K -> results/kcycle_bins_sensitivity.csv
    python experiments/layer1_ground_truth/run_sensitivity.py tiger   # Tiger, B=8, N in {1000,3000,10000} -> results/tiger_N_sensitivity.csv
"""
import csv, sys, time, importlib.util as ilu
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from continuous_quotient import ContinuousEstimatorConfig, ContinuousQuotientEstimator, collect_continuous_trajectories

def load(p, name):
    spec = ilu.spec_from_file_location(name, p); m = ilu.module_from_spec(spec); spec.loader.exec_module(m); return m
KC = load(ROOT/"experiments/layer1_ground_truth/envs/kcycle_continuous.py", "kc").KCycleContinuousEnv
TG = load(ROOT/"experiments/layer1_ground_truth/envs/tiger_continuous.py", "tg").TigerContinuousEnv
OUT = ROOT/"experiments/layer1_ground_truth/results"

which = sys.argv[1]
if which == "kcycle":
    rows = []
    out = OUT/"kcycle_bins_sensitivity.csv"
    for K in [2,3,4,5,8,16]:
        for mult in [2]:
            B = mult*K
            N = 5000 if K == 16 else 3000
            cfg = ContinuousEstimatorConfig(n_trajectories=N, max_steps_per_episode=50, window_length=2,
                    history_max_length=1, min_history_occurrences=20, alpha=0.01, n_permutations=199,
                    max_samples_per_test=300, history_n_bins=B)
            for seed in range(5):
                env = KC(K=K, obs_noise=0.1)
                t0 = time.perf_counter()
                ds = collect_continuous_trajectories(env, N, env.episode_length, seed=seed)
                q = ContinuousQuotientEstimator(cfg).fit(ds)
                rows.append(dict(K=K, B=B, N=N, seed=seed, n_classes=q.n_classes, runtime_s=round(time.perf_counter()-t0,1)))
                print(rows[-1], flush=True)
                with open(out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
elif which == "tiger":
    rows = []
    out = OUT/"tiger_N_sensitivity.csv"
    for N in [1000, 3000, 10000]:
        cfg = ContinuousEstimatorConfig(n_trajectories=N, max_steps_per_episode=20, window_length=2,
                history_max_length=1, min_history_occurrences=20, alpha=0.01, n_permutations=199,
                max_samples_per_test=300, history_n_bins=8)
        for seed in range(5):
            env = TG(sigma=0.5, episode_length=20)
            t0 = time.perf_counter()
            q = ContinuousQuotientEstimator(cfg).fit_from_env(env, seed=seed)
            rows.append(dict(N=N, B=8, seed=seed, n_classes=q.n_classes, runtime_s=round(time.perf_counter()-t0,1)))
            print(rows[-1], flush=True)
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
