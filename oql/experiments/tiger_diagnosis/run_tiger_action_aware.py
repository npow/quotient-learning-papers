"""Tiger POMDP action-aware history diagnosis.

Tests the hypothesis that the observation-only history representation (current
implementation) loses the action context that distinguishes lean-left from
lean-right after a listen.  The paper's Definition 1 says histories include
actions; the obs-only implementation is a simplification.

Runs 5 seeds of:
  - action_aware=False (obs-only, current): expected n=1
  - action_aware=True  (action-aware, new):  hypothesis n=2

Config: N=5000, history_max_length=2, min_history_occurrences=80, alpha=0.01

Results saved to experiments/tiger_diagnosis/results/action_aware.csv
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
N_TRAJECTORIES = 5000
HISTORY_MAX_LENGTH = 2
MIN_HISTORY_OCCURRENCES = 80
ALPHA = 0.01
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_tiger_seeds(action_aware: bool, verbose: bool = True) -> dict:
    """Run Tiger on all seeds with given action_aware setting. Returns result dict."""
    label = "action_aware" if action_aware else "obs_only"
    if verbose:
        print(f"\n{'='*60}")
        print(f"Tiger POMDP — {label}")
        print(f"  N={N_TRAJECTORIES}, history_max_length={HISTORY_MAX_LENGTH}")
        print(f"  min_history_occurrences={MIN_HISTORY_OCCURRENCES}, alpha={ALPHA}")
        print(f"  action_aware={action_aware}")

    t0 = time.time()
    n_classes_per_seed = []
    partition_details = []

    for seed in SEEDS:
        env = TigerPOMDP()
        est = QuotientEstimator(
            env=env,
            n_trajectories=N_TRAJECTORIES,
            window_length=2,
            history_max_length=HISTORY_MAX_LENGTH,
            min_history_occurrences=MIN_HISTORY_OCCURRENCES,
            alpha=ALPHA,
            max_steps_per_episode=50,
            action_aware=action_aware,
        )
        quotient = est.run(seed=seed)
        n_classes_per_seed.append(quotient.n_classes)
        partition_details.append({
            "seed": seed,
            "n_classes": quotient.n_classes,
            "partition_keys": [str(h) for h in quotient.partition.keys()],
            "class_assignments": {str(h): c for h, c in quotient.partition.items()},
        })
        if verbose:
            print(f"  seed={seed}: n_classes={quotient.n_classes}, "
                  f"partition={partition_details[-1]['class_assignments']}")

    elapsed = time.time() - t0
    n_arr = np.array(n_classes_per_seed)
    seeds_recovering_2 = int(np.sum(n_arr == 2))
    mean_n = float(n_arr.mean())
    std_n = float(n_arr.std())

    if verbose:
        print(f"  -> mean n_classes={mean_n:.2f} ± {std_n:.2f}, "
              f"seeds with n=2: {seeds_recovering_2}/{len(SEEDS)}, "
              f"time: {elapsed:.1f}s")

    return {
        "mode": label,
        "action_aware": action_aware,
        "n_trajectories": N_TRAJECTORIES,
        "history_max_length": HISTORY_MAX_LENGTH,
        "min_history_occurrences": MIN_HISTORY_OCCURRENCES,
        "alpha": ALPHA,
        "seeds": SEEDS,
        "n_classes_per_seed": n_classes_per_seed,
        "mean_n_classes": mean_n,
        "std_n_classes": std_n,
        "seeds_recovering_2": seeds_recovering_2,
        "time_seconds": elapsed,
        "partition_details": partition_details,
    }


def main():
    # Run observation-only (current, known-failing)
    result_obs = run_tiger_seeds(action_aware=False)

    # Run action-aware (new, hypothesis)
    result_aa = run_tiger_seeds(action_aware=True)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Tiger POMDP — obs-only vs action-aware histories")
    print("=" * 70)
    print(f"{'Mode':<20} {'Mean n_cls':>10} {'Std':>6} {'n=2 seeds':>10}")
    print("-" * 70)
    for r in [result_obs, result_aa]:
        print(f"{r['mode']:<20} {r['mean_n_classes']:>10.2f} {r['std_n_classes']:>6.2f} "
              f"{r['seeds_recovering_2']:>10}/{len(SEEDS)}")

    # Interpretation
    print("\n" + "=" * 70)
    if result_aa["seeds_recovering_2"] > result_obs["seeds_recovering_2"]:
        print("FINDING: Action-aware histories recover n=2 more often than obs-only.")
        print("DIAGNOSIS: The observation-only simplification WAS causing Tiger to fail.")
        print("  The bait-and-switch (obs-only instead of Definition 1) is the bug.")
    elif result_aa["seeds_recovering_2"] == result_obs["seeds_recovering_2"]:
        print("FINDING: Action-aware histories do NOT improve Tiger recovery.")
        print("DIAGNOSIS: The failure is fundamental to uniform-random exploration of Tiger.")
        print("  The quotient algorithm itself cannot separate states under random policy.")
    else:
        print("FINDING: Action-aware performs WORSE (unexpected).")

    # Write CSV results
    csv_path = RESULTS_DIR / "action_aware.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode", "action_aware",
            "seed", "n_classes",
            "mean_n_classes", "std_n_classes",
            "seeds_recovering_2",
            "n_trajectories", "history_max_length",
            "min_history_occurrences", "alpha",
        ])
        for r in [result_obs, result_aa]:
            for seed, n_cls in zip(r["seeds"], r["n_classes_per_seed"]):
                writer.writerow([
                    r["mode"], r["action_aware"],
                    seed, n_cls,
                    r["mean_n_classes"], r["std_n_classes"],
                    r["seeds_recovering_2"],
                    r["n_trajectories"], r["history_max_length"],
                    r["min_history_occurrences"], r["alpha"],
                ])

    print(f"\nResults saved to: {csv_path}")
    return result_obs, result_aa


if __name__ == "__main__":
    result_obs, result_aa = main()
