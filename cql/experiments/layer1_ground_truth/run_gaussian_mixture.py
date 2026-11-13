"""Experiment: Gaussian-mixture POMDP continuous-observation ground-truth recovery.

For each N in --N-values and each seed in 0..(--seeds - 1), collect
trajectories from GaussianMixturePOMDP(N) and run ContinuousQuotientEstimator
twice:
  - manual_B: history_n_bins = N   (state-aligned, should give n_classes = N)
  - auto_B:   history_n_bins = 0   (auto-select via silhouette, records chosen B)

Results are written to experiments/layer1_ground_truth/results/gaussian_mixture.csv
with columns: N, seed, auto_B, n_classes_manual_B, n_classes_auto_B,
              correct_manual, correct_auto, runtime_s

A per-N summary is printed and a PASS/FAIL assertion checks that every N
achieves >= 80% correct rate with the manual (state-aligned) bin count.

Usage
-----
    python experiments/layer1_ground_truth/run_gaussian_mixture.py
    python experiments/layer1_ground_truth/run_gaussian_mixture.py \\
        --N-values 3 4 5 6 --seeds 10 --n-trajectories 3000
"""

from __future__ import annotations

import argparse
import csv
import importlib.util as _ilu
import os
import statistics
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
    auto_select_bins,
    collect_continuous_trajectories,
)

# Load env without requiring __init__.py imports up the experiments/ tree.
_ENV_PATH = Path(__file__).resolve().parent / "envs" / "gaussian_mixture_pomdp.py"
_spec = _ilu.spec_from_file_location("gaussian_mixture_pomdp", _ENV_PATH)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GaussianMixturePOMDP = _mod.GaussianMixturePOMDP


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ground-truth quotient recovery on Gaussian-mixture POMDP "
            "(continuous observations, n* = N analytically)."
        )
    )
    parser.add_argument(
        "--N-values",
        nargs="+",
        type=int,
        default=[3, 4, 5, 6],
        metavar="N",
        help="List of N values (state-space sizes) to evaluate (default: 3 4 5 6).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        metavar="INT",
        help="Number of random seeds per N (default: 5).",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=2000,
        metavar="INT",
        help="Number of episodes to collect per run (default: 2000).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.15,
        metavar="FLOAT",
        help="Observation noise std for GaussianMixturePOMDP (default: 0.15).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        metavar="FLOAT",
        help="Base significance level for MMD tests (default: 0.05).",
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
        default=99,
        metavar="INT",
        help="Number of permutations for MMD permutation test (default: 99).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=300,
        metavar="INT",
        help="Max continuation samples per MMD test pair (default: 300).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_single(
    N: int,
    seed: int,
    n_trajectories: int,
    sigma: float,
    base_config_kwargs: dict,
) -> dict:
    """Run one (N, seed) trial with both manual-B and auto-B estimators.

    Both estimators share the same collected dataset to avoid redundant rollouts.
    auto_B is resolved once via auto_select_bins(), then the auto estimator reuses
    that resolved value (history_n_bins = auto_b) so the inner fit() does NOT repeat
    the silhouette scan — halving runtime relative to using the sentinel 0.

    Returns a result dict with all CSV columns.
    """
    env = GaussianMixturePOMDP(N=N, sigma=sigma)

    t0 = time.perf_counter()

    dataset = collect_continuous_trajectories(
        env=env,
        n_episodes=n_trajectories,
        max_steps=env.episode_length,
        seed=seed,
    )

    # --- Manual B = N (state-aligned bins) ---
    cfg_manual = ContinuousEstimatorConfig(
        **base_config_kwargs,
        history_n_bins=N,
    )
    est_manual = ContinuousQuotientEstimator(config=cfg_manual)
    q_manual = est_manual.fit(dataset, env_id=f"GaussianMixture-N{N}", verbose=False)
    n_classes_manual = q_manual.n_classes

    # --- Auto B (silhouette-based bin selection) ---
    # Resolve auto_b once, then pass directly to avoid re-running silhouette inside fit().
    auto_b = auto_select_bins(dataset)
    cfg_auto = ContinuousEstimatorConfig(
        **base_config_kwargs,
        history_n_bins=auto_b,
    )
    est_auto = ContinuousQuotientEstimator(config=cfg_auto)
    q_auto = est_auto.fit(dataset, env_id=f"GaussianMixture-N{N}-autoB", verbose=False)
    n_classes_auto = q_auto.n_classes

    runtime_s = time.perf_counter() - t0

    return {
        "N": N,
        "seed": seed,
        "auto_B": auto_b,
        "n_classes_manual_B": n_classes_manual,
        "n_classes_auto_B": n_classes_auto,
        "correct_manual": int(n_classes_manual == N),
        "correct_auto": int(n_classes_auto == N),
        "runtime_s": round(runtime_s, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    base_config_kwargs = dict(
        n_trajectories=args.n_trajectories,
        max_steps_per_episode=50,
        window_length=args.window_length,
        history_max_length=args.history_max_length,
        min_history_occurrences=args.min_occurrences,
        alpha=args.alpha,
        n_permutations=args.n_permutations,
        max_samples_per_test=args.max_samples,
    )

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "gaussian_mixture.csv"

    fieldnames = [
        "N", "seed", "auto_B",
        "n_classes_manual_B", "n_classes_auto_B",
        "correct_manual", "correct_auto",
        "runtime_s",
    ]

    rows: list[dict] = []

    N_values = args.N_values
    seeds = list(range(args.seeds))
    total_runs = len(N_values) * len(seeds)

    print(f"Running {total_runs} trials "
          f"({len(N_values)} N-values x {len(seeds)} seeds) ...")
    print(f"  n_trajectories={args.n_trajectories}, sigma={args.sigma}, "
          f"alpha={args.alpha}, window_length={args.window_length}, "
          f"history_max_length={args.history_max_length}, "
          f"min_occurrences={args.min_occurrences}, "
          f"n_permutations={args.n_permutations}, "
          f"max_samples={args.max_samples}")
    print()

    with tqdm(total=total_runs, unit="trial") as pbar:
        for N in N_values:
            for seed in seeds:
                pbar.set_description(f"N={N} seed={seed}")
                row = run_single(
                    N=N,
                    seed=seed,
                    n_trajectories=args.n_trajectories,
                    sigma=args.sigma,
                    base_config_kwargs=base_config_kwargs,
                )
                rows.append(row)
                pbar.update(1)

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {output_path}\n")

    # Per-N summary
    print(f"{'N':>4}  {'mean_manual':>11}  {'std_manual':>10}  "
          f"{'frac_corr_manual':>16}  {'mean_auto_B':>11}  {'frac_corr_auto':>14}")
    print("-" * 76)

    all_pass = True
    failed_Ns = []

    for N in N_values:
        n_rows = [r for r in rows if r["N"] == N]
        nc_manual = [r["n_classes_manual_B"] for r in n_rows]
        nc_auto   = [r["n_classes_auto_B"]   for r in n_rows]
        auto_bs   = [r["auto_B"]             for r in n_rows]
        frac_manual = sum(r["correct_manual"] for r in n_rows) / len(n_rows)
        frac_auto   = sum(r["correct_auto"]   for r in n_rows) / len(n_rows)

        mean_m = statistics.mean(nc_manual)
        std_m  = statistics.stdev(nc_manual) if len(nc_manual) > 1 else 0.0
        mean_aB = statistics.mean(auto_bs)

        print(
            f"{N:>4}  {mean_m:>11.2f}  {std_m:>10.2f}  "
            f"{frac_manual:>15.0%}  {mean_aB:>11.1f}  {frac_auto:>13.0%}"
        )

        if frac_manual < 0.80:
            all_pass = False
            failed_Ns.append((N, frac_manual))

    print()

    # PASS / FAIL assertion on manual-B results
    if not failed_Ns:
        print("PASS: all N values achieved >= 80% correct quotient recovery (manual B=N).")
    else:
        for N, frac in failed_Ns:
            print(
                f"FAIL: N={N} achieved only {frac:.0%} correct with manual B=N "
                f"(threshold: 80%)."
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
