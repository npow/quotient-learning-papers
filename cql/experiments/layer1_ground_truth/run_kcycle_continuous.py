"""Experiment: K-cycle continuous POMDP ground-truth quotient recovery.

For each K in --K-values and each seed in 0..(--seeds - 1), collect
trajectories from KCycleContinuousEnv(K) and run ContinuousQuotientEstimator.
The ground-truth quotient has exactly K classes (one per phase).

Results are written to experiments/layer1_ground_truth/results/kcycle_continuous.csv.
A per-K summary is printed and a PASS/FAIL assertion checks that every K
achieves >= 80% correct rate (n_classes == K).

Usage
-----
    python experiments/layer1_ground_truth/run_kcycle_continuous.py
    python experiments/layer1_ground_truth/run_kcycle_continuous.py \
        --K-values 2 3 4 --seeds 10 --n-trajectories 8000
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Ensure project root is importable when script is run directly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from continuous_quotient import (
    ContinuousEstimatorConfig,
    ContinuousQuotientEstimator,
    collect_continuous_trajectories,
)

# Import env directly to avoid requiring __init__.py up the experiments/ tree.
_ENV_PATH = Path(__file__).resolve().parent / "envs" / "kcycle_continuous.py"
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("kcycle_continuous", _ENV_PATH)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
KCycleContinuousEnv = _mod.KCycleContinuousEnv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ground-truth quotient recovery experiment on K-cycle continuous POMDP."
    )
    parser.add_argument(
        "--K-values",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 8],
        metavar="K",
        help="List of K values to evaluate (default: 2 3 4 5 8).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        metavar="INT",
        help="Number of random seeds per K (default: 5).",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=5000,
        metavar="INT",
        help="Number of episodes to collect per run (default: 5000).",
    )
    parser.add_argument(
        "--obs-noise",
        type=float,
        default=0.1,
        metavar="FLOAT",
        help="Observation noise std for KCycleContinuousEnv (default: 0.1).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        metavar="FLOAT",
        help="Base significance level for MMD tests (default: 0.01).",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=2,
        metavar="INT",
        help="Continuation window length k (default: 2).",
    )
    parser.add_argument(
        "--history-max-length",
        type=int,
        default=1,
        metavar="INT",
        help="Maximum history prefix length L (default: 1).",
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=20,
        metavar="INT",
        help="Minimum history occurrences to include in test (default: 20).",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=199,
        metavar="INT",
        help="Number of permutations for MMD permutation test (default: 199).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_single(
    K: int,
    seed: int,
    n_trajectories: int,
    obs_noise: float,
    config: ContinuousEstimatorConfig,
) -> dict:
    """Run one (K, seed) trial and return a result dict."""
    env = KCycleContinuousEnv(K=K, obs_noise=obs_noise)

    t0 = time.perf_counter()

    dataset = collect_continuous_trajectories(
        env=env,
        n_episodes=n_trajectories,
        max_steps=env.episode_length,
        seed=seed,
    )

    estimator = ContinuousQuotientEstimator(config=config)
    quotient = estimator.fit(dataset, env_id=f"KCycle-K{K}", verbose=False)

    runtime_s = time.perf_counter() - t0

    return {
        "K": K,
        "seed": seed,
        "n_classes": quotient.n_classes,
        "correct": int(quotient.n_classes == K),
        "runtime_s": round(runtime_s, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # history_n_bins is set per-K below (= K) so that bin boundaries align with
    # the K equally-spaced state signals.  A global config is not used here.
    _base_config_kwargs = dict(
        n_trajectories=args.n_trajectories,
        max_steps_per_episode=50,
        window_length=args.window_length,
        history_max_length=args.history_max_length,
        min_history_occurrences=args.min_occurrences,
        alpha=args.alpha,
        n_permutations=args.n_permutations,
        max_samples_per_test=300,
    )

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "kcycle_continuous.csv"

    fieldnames = ["K", "seed", "n_classes", "correct", "runtime_s"]
    rows: list[dict] = []

    K_values = args.K_values
    seeds = list(range(args.seeds))
    total_runs = len(K_values) * len(seeds)

    print(f"Running {total_runs} trials "
          f"({len(K_values)} K-values x {len(seeds)} seeds) ...")
    print(f"  n_trajectories={args.n_trajectories}, obs_noise={args.obs_noise}, "
          f"alpha={args.alpha}, window_length={args.window_length}, "
          f"history_max_length={args.history_max_length}, "
          f"min_occurrences={args.min_occurrences}, "
          f"n_permutations={args.n_permutations}")
    print()

    with tqdm(total=total_runs, unit="trial") as pbar:
        for K in K_values:
            # Use K bins so each bin aligns with one of the K equally-spaced states.
            k_config = ContinuousEstimatorConfig(**_base_config_kwargs, history_n_bins=K)
            for seed in seeds:
                pbar.set_description(f"K={K} seed={seed}")
                row = run_single(
                    K=K,
                    seed=seed,
                    n_trajectories=args.n_trajectories,
                    obs_noise=args.obs_noise,
                    config=k_config,
                )
                rows.append(row)
                pbar.update(1)

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {output_path}\n")

    # Per-K summary
    import statistics

    all_pass = True
    print(f"{'K':>4}  {'mean_classes':>12}  {'std_classes':>11}  {'frac_correct':>12}")
    print("-" * 46)
    for K in K_values:
        k_rows = [r for r in rows if r["K"] == K]
        n_classes_list = [r["n_classes"] for r in k_rows]
        correct_list = [r["correct"] for r in k_rows]

        mean_nc = statistics.mean(n_classes_list)
        std_nc = statistics.stdev(n_classes_list) if len(n_classes_list) > 1 else 0.0
        frac_correct = sum(correct_list) / len(correct_list)

        print(f"{K:>4}  {mean_nc:>12.2f}  {std_nc:>11.2f}  {frac_correct:>11.0%}")

        if frac_correct < 0.80:
            all_pass = False

    print()

    # PASS / FAIL assertion
    failed_Ks = []
    for K in K_values:
        k_rows = [r for r in rows if r["K"] == K]
        frac_correct = sum(r["correct"] for r in k_rows) / len(k_rows)
        if frac_correct < 0.80:
            failed_Ks.append((K, frac_correct))

    if not failed_Ks:
        print("PASS: all K values achieved >= 80% correct quotient recovery.")
    else:
        for K, frac in failed_Ks:
            print(f"FAIL: K={K} achieved only {frac:.0%} correct (threshold: 80%).")
        sys.exit(1)


if __name__ == "__main__":
    main()
