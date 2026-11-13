"""Tiger POMDP within-episode continuation diagnosis.

Hypothesis: the post-terminate observation (appended after open action ends
the episode) is an uninformative reset obs that dominates the marginal
continuation distribution, causing all histories to look the same (n=1).

Fix: set drop_terminal_obs=True in QuotientEstimator so the last obs of each
trajectory is excluded before extracting continuation windows.

Tests:
- Baseline (drop_terminal_obs=False) for comparison
- Within-episode (drop_terminal_obs=True)
- Multiple max_steps_per_episode values: 4, 5, 6, 8 (shorter episodes → higher
  fraction of listen observations → stronger within-episode signal)

Config: 5 seeds, N=5000, history_max_length=2, min_history_occurrences=80, alpha=0.01
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environments.layer1_discrete.tiger import TigerPOMDP
from quotient_learning.estimator import QuotientEstimator

SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_condition(
    label: str,
    max_steps: int,
    drop_terminal_obs: bool,
    n_trajectories: int = 5000,
    history_max_length: int = 2,
    min_history_occurrences: int = 80,
    alpha: float = 0.01,
    verbose: bool = True,
) -> dict:
    """Run 5 seeds for a given condition and return summary dict."""
    if verbose:
        print(f"\n  {label}  (max_steps={max_steps}, drop_terminal={drop_terminal_obs})")

    t0 = time.time()
    n_classes_per_seed = []

    for seed in SEEDS:
        env = TigerPOMDP()
        estimator = QuotientEstimator(
            env=env,
            n_trajectories=n_trajectories,
            window_length=2,
            history_max_length=history_max_length,
            min_history_occurrences=min_history_occurrences,
            alpha=alpha,
            max_steps_per_episode=max_steps,
            drop_terminal_obs=drop_terminal_obs,
        )
        quotient = estimator.run(seed=seed)
        n_classes_per_seed.append(quotient.n_classes)
        if verbose:
            print(f"    seed={seed}: n_classes={quotient.n_classes}")

    elapsed = time.time() - t0
    arr = np.array(n_classes_per_seed)
    mean_nc = float(arr.mean())
    std_nc = float(arr.std())
    seeds_2 = int(np.sum(arr == 2))

    if verbose:
        print(f"    -> mean={mean_nc:.2f} ± {std_nc:.2f}  n=2 on {seeds_2}/5 seeds  ({elapsed:.1f}s)")

    return {
        "label": label,
        "max_steps": max_steps,
        "drop_terminal_obs": drop_terminal_obs,
        "n_trajectories": n_trajectories,
        "history_max_length": history_max_length,
        "min_history_occurrences": min_history_occurrences,
        "alpha": alpha,
        "n_classes_per_seed": n_classes_per_seed,
        "mean_n_classes": mean_nc,
        "std_n_classes": std_nc,
        "seeds_recovering_2": seeds_2,
        "elapsed_s": elapsed,
    }


def main() -> None:
    print("=" * 65)
    print("Tiger POMDP: Within-Episode Continuation Sampling Diagnosis")
    print("=" * 65)
    print(f"Seeds: {SEEDS}  N=5000  history_max_length=2  min_occ=80  alpha=0.01")

    results = []

    # 1. Baseline at each max_steps value (drop_terminal_obs=False)
    print("\n--- BASELINE (drop_terminal_obs=False) ---")
    for max_steps in [4, 5, 6, 8, 50]:
        r = run_condition(
            label=f"baseline_steps{max_steps}",
            max_steps=max_steps,
            drop_terminal_obs=False,
        )
        results.append(r)

    # 2. Within-episode at each max_steps value (drop_terminal_obs=True)
    print("\n--- WITHIN-EPISODE (drop_terminal_obs=True) ---")
    for max_steps in [4, 5, 6, 8, 50]:
        r = run_condition(
            label=f"within_episode_steps{max_steps}",
            max_steps=max_steps,
            drop_terminal_obs=True,
        )
        results.append(r)

    # Summary table
    print("\n" + "=" * 75)
    print(f"{'Label':<38} {'max_steps':>9} {'drop_term':>9} {'mean_nc':>8} {'std':>6} {'n=2/5':>6}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['label']:<38} {r['max_steps']:>9} {str(r['drop_terminal_obs']):>9}"
            f" {r['mean_n_classes']:>8.2f} {r['std_n_classes']:>6.2f} {r['seeds_recovering_2']:>5}/5"
        )

    # Save CSV
    csv_path = RESULTS_DIR / "within_episode.csv"
    fieldnames = [
        "label", "max_steps", "drop_terminal_obs", "n_trajectories",
        "history_max_length", "min_history_occurrences", "alpha",
        "mean_n_classes", "std_n_classes", "seeds_recovering_2", "elapsed_s",
        "n_classes_seed0", "n_classes_seed1", "n_classes_seed2",
        "n_classes_seed3", "n_classes_seed4",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if k != "n_classes_per_seed"}
            for i, nc in enumerate(r["n_classes_per_seed"]):
                row[f"n_classes_seed{i}"] = nc
            writer.writerow(row)

    print(f"\nResults saved to: {csv_path}")

    # Diagnosis conclusion
    within_ep_rows = [r for r in results if r["drop_terminal_obs"]]
    best = max(within_ep_rows, key=lambda r: r["seeds_recovering_2"])
    any_recovery = any(r["seeds_recovering_2"] >= 4 for r in within_ep_rows)

    print("\n--- DIAGNOSIS ---")
    if any_recovery:
        print(f"CONFIRMED: drop_terminal_obs=True recovers n=2 on >=4/5 seeds.")
        print(f"Best config: {best['label']}  mean={best['mean_n_classes']:.2f}  {best['seeds_recovering_2']}/5 seeds")
        print("=> Post-terminate observation was the noise drowning the signal.")
    else:
        print(f"INCONCLUSIVE: drop_terminal_obs=True did not recover n=2 on >=4/5 seeds.")
        print(f"Best within-episode config: {best['label']}  mean={best['mean_n_classes']:.2f}  {best['seeds_recovering_2']}/5 seeds")
        print("=> Failure is deeper than the post-terminate observation.")


if __name__ == "__main__":
    main()
