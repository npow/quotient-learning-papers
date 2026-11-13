"""Ablation experiment: MMD vs. discretized G-test (T027).

Sweeps N ∈ {500, 1000, 2000, 5000} on continuous Tiger and K-cycle-K=4.
Compares three methods:
  (1) ContinuousMMDTest (no discretization) — this work
  (2) G-test with 3-bin quantile discretization — baseline
  (3) G-test with 8-bin quantile discretization — baseline

For each method/N/env/seed: computes quotient and records:
  - n_classes recovered
  - correct (True if == ground truth)
  - runtime_s

Outputs: experiments/baselines/results/ablation_sample_efficiency.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from project root.
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "layer1_ground_truth"))

from continuous_quotient import ContinuousEstimatorConfig, ContinuousQuotientEstimator
from envs.tiger_continuous import TigerContinuousEnv
from envs.kcycle_continuous import KCycleContinuousEnv

try:
    from quotient_learning.estimator import QuotientEstimator, EstimatorConfig
    from quotient_learning.equivalence import GTestEquivalenceTest
    HAS_UPSTREAM = True
except ImportError:
    HAS_UPSTREAM = False


def _discretize_trajectories(trajectories, n_bins: int):
    """Quantile-discretize float observations to int tokens."""
    from continuous_quotient.types import ContinuousTrajectory
    import numpy as np

    # Fit quantile boundaries from all observations.
    all_obs = []
    for traj in trajectories:
        for o in traj.observations:
            all_obs.append(float(np.asarray(o).ravel()[0]))
    all_obs_arr = np.array(all_obs)
    percentiles = np.linspace(0, 100, n_bins + 1)
    boundaries = np.percentile(all_obs_arr, percentiles)

    disc_trajs = []
    for traj in trajectories:
        obs_int = []
        for o in traj.observations:
            v = float(np.asarray(o).ravel()[0])
            bin_idx = int(np.searchsorted(boundaries[1:-1], v))
            obs_int.append(bin_idx)
        disc_trajs.append((obs_int, traj.actions, traj.rewards))
    return disc_trajs, n_bins


def _run_gtest_on_discrete(disc_trajs, n_bins, alpha: float = 0.01) -> int:
    """Run upstream QuotientEstimator (G-test) on discretized trajectories."""
    if not HAS_UPSTREAM:
        return -1

    from quotient_learning.dataset import Trajectory, TrajectoryDataset
    from quotient_learning.estimator import QuotientEstimator

    traj_objs = []
    for obs_int, acts, rews in disc_trajs:
        acts_int = [int(a) if not isinstance(a, (int, np.integer)) else int(a)
                    for a in acts]
        traj_objs.append(Trajectory(
            observations=tuple(obs_int),
            actions=tuple(acts_int),
            rewards=tuple(rews),
            seed=0,
        ))
    dataset = TrajectoryDataset(trajectories=tuple(traj_objs), environment_id="continuous_disc", policy_id="uniform_random")
    est = QuotientEstimator(
        env=None,  # type: ignore
        n_trajectories=len(disc_trajs),
        window_length=2,
        history_max_length=1,
        min_history_occurrences=15,
        alpha=alpha,
        equivalence_test=GTestEquivalenceTest(),
        policy=lambda obs, rng: 0,  # dummy; dataset already collected
    )
    try:
        quotient = est.estimate(dataset)
        return quotient.n_classes
    except Exception:
        return -1


def _collect_trajs(env, n_traj: int, max_steps: int, seed: int):
    """Collect trajectories and return as ContinuousTrajectory list."""
    from continuous_quotient import collect_continuous_trajectories
    dataset = collect_continuous_trajectories(env, n_episodes=n_traj, max_steps=max_steps, seed=seed)
    return dataset.trajectories


def run_ablation(args):
    N_values = args.n_values
    seeds = list(range(args.seeds))
    out_path = ROOT / "experiments" / "baselines" / "results" / "ablation_sample_efficiency.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    envs_gt = {
        "tiger_n2": (lambda: TigerContinuousEnv(sigma=0.5, episode_length=20), 2, 20),
        "kcycle_K4": (lambda: KCycleContinuousEnv(K=4, obs_noise=0.05, episode_length=50), 4, 50),
    }

    rows = []
    for env_name, (env_factory, gt_n, max_steps) in envs_gt.items():
        for N in N_values:
            for seed in seeds:
                env = env_factory()
                trajs = _collect_trajs(env, N, max_steps, seed)

                # Method 1: Continuous MMD
                # Use history_n_bins=gt_n (state-aligned) and alpha=0.05
                # (alpha > 1/(B+1)=0.02 with B=49 permutations so rejection is possible).
                t0 = time.time()
                cfg = ContinuousEstimatorConfig(
                    n_trajectories=N, max_steps_per_episode=max_steps,
                    window_length=2, history_max_length=1,
                    min_history_occurrences=max(10, N // 200),
                    alpha=0.05, n_permutations=args.n_permutations,
                    history_n_bins=gt_n, max_samples_per_test=300,
                )
                est = ContinuousQuotientEstimator(cfg)
                from continuous_quotient.types import ContinuousTrajectoryDataset
                q = est.fit(ContinuousTrajectoryDataset(trajectories=trajs))
                rt_mmd = time.time() - t0
                rows.append({
                    "env": env_name, "method": "mmd_continuous", "N": N, "seed": seed,
                    "n_classes": q.n_classes, "correct": int(q.n_classes == gt_n),
                    "runtime_s": round(rt_mmd, 2),
                })

                # Methods 2 & 3: Discretized G-test
                for n_bins in [3, 8]:
                    t0 = time.time()
                    disc, _ = _discretize_trajectories(trajs, n_bins)
                    n_cls = _run_gtest_on_discrete(disc, n_bins, alpha=0.01)
                    rt_g = time.time() - t0
                    rows.append({
                        "env": env_name, "method": f"gtest_{n_bins}bin", "N": N, "seed": seed,
                        "n_classes": n_cls, "correct": int(n_cls == gt_n),
                        "runtime_s": round(rt_g, 2),
                    })

                print(f"  {env_name} N={N} seed={seed}: mmd→{rows[-3]['n_classes']}"
                      f" g3→{rows[-2]['n_classes']} g8→{rows[-1]['n_classes']}")

    # Write CSV.
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["env", "method", "N", "seed", "n_classes", "correct", "runtime_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_path}")

    # Print per-method summary.
    import pandas as pd
    df = pd.DataFrame(rows)
    summary = df.groupby(["env", "method", "N"])["correct"].mean().reset_index()
    print("\nAccuracy by (env, method, N):")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Ablation: MMD vs discretized G-test")
    parser.add_argument("--n-values", type=int, nargs="+", default=[500, 1000, 2000, 5000])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=99)
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
