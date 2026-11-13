"""Tiger Continuous POMDP — Layer-1 Ground-Truth Experiment.

Runs ContinuousQuotientEstimator on TigerContinuousEnv across multiple seeds
and verifies that the estimated quotient recovers n=2 classes.

Usage:
    python experiments/layer1_ground_truth/run_tiger_continuous.py [--seeds N] ...

Expected result:
    n_classes == 2 in ≥ 90 % of seeds  (PASS).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make ``continuous_quotient`` importable when running from the
# project root (python experiments/layer1_ground_truth/run_tiger_continuous.py)
# without a prior ``pip install -e .``.
# ---------------------------------------------------------------------------
try:
    import continuous_quotient  # noqa: F401  — check if already on sys.path
except ModuleNotFoundError:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_PROJECT_ROOT))
    import continuous_quotient  # noqa: F401  — must succeed now

from continuous_quotient import (
    ContinuousEstimatorConfig,
    ContinuousQuotientEstimator,
)

# Env lives inside the experiments sub-package; add its parent to sys.path so
# the relative import works regardless of how the script is invoked.
_LAYER1_DIR = Path(__file__).resolve().parent
if str(_LAYER1_DIR) not in sys.path:
    sys.path.insert(0, str(_LAYER1_DIR))

from envs.tiger_continuous import TigerContinuousEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
_RESULTS_DIR = _LAYER1_DIR / "results"
_RESULTS_CSV = _RESULTS_DIR / "tiger_continuous.csv"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tiger-Continuous ground-truth quotient-recovery experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seeds", type=int, default=10,
                   help="Number of independent seeds to run.")
    p.add_argument("--n-trajectories", type=int, default=3000,
                   help="Number of trajectories (episodes) per seed.")
    p.add_argument("--sigma", type=float, default=0.5,
                   help="Gaussian observation noise std for TigerContinuousEnv.")
    p.add_argument("--alpha", type=float, default=0.01,
                   help="Base significance level for MMD tests (Benjamini-Hochberg over all pairs).")
    p.add_argument("--window-length", type=int, default=2,
                   help="Continuation window length k.")
    p.add_argument("--history-max-length", type=int, default=1,
                   help="Maximum history prefix length L.")
    p.add_argument("--min-occurrences", type=int, default=20,
                   help="Minimum occurrences for a history to be included.")
    p.add_argument("--n-permutations", type=int, default=199,
                   help="Permutation samples for the MMD null distribution.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    try:
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        raise SystemExit("tqdm is required: pip install tqdm") from exc

    config = ContinuousEstimatorConfig(
        n_trajectories=args.n_trajectories,
        max_steps_per_episode=20,   # matches TigerContinuousEnv default episode_length
        window_length=args.window_length,
        history_max_length=args.history_max_length,
        min_history_occurrences=args.min_occurrences,
        alpha=args.alpha,
        n_permutations=args.n_permutations,
    )

    print(
        f"Tiger-Continuous experiment | seeds={args.seeds} "
        f"n_traj={args.n_trajectories} sigma={args.sigma} "
        f"alpha={args.alpha} window={args.window_length} "
        f"history_max_len={args.history_max_length} "
        f"min_occ={args.min_occurrences} n_perm={args.n_permutations}"
    )

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    for seed in tqdm(range(args.seeds), desc="seeds", unit="seed"):
        env = TigerContinuousEnv(sigma=args.sigma, episode_length=20)
        estimator = ContinuousQuotientEstimator(config=config)

        t0 = time.perf_counter()
        quotient = estimator.fit_from_env(env, seed=seed, verbose=False)
        runtime_s = time.perf_counter() - t0

        records.append({
            "seed": seed,
            "n_classes": quotient.n_classes,
            "runtime_s": round(runtime_s, 4),
        })

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    fieldnames = ["seed", "n_classes", "runtime_s"]
    with _RESULTS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nResults written to {_RESULTS_CSV}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    import statistics

    n_classes_list = [r["n_classes"] for r in records]
    mean_nc = statistics.mean(n_classes_list)
    std_nc = statistics.pstdev(n_classes_list)  # population stdev (matches numpy default)
    fraction_correct = sum(1 for nc in n_classes_list if nc == 2) / len(n_classes_list)

    print(f"\n--- Summary ({args.seeds} seeds) ---")
    print(f"  n_classes : {mean_nc:.2f} ± {std_nc:.2f}")
    print(f"  n_classes per seed : {n_classes_list}")
    print(f"  Fraction recovering n=2 exactly : {fraction_correct:.2f} ({sum(nc == 2 for nc in n_classes_list)}/{len(n_classes_list)})")

    # ------------------------------------------------------------------
    # Pass/Fail gate
    # ------------------------------------------------------------------
    THRESHOLD = 0.9
    passed = fraction_correct >= THRESHOLD
    status = "PASS" if passed else "FAIL"
    print(f"\n{status}  (fraction_correct={fraction_correct:.2f}, threshold={THRESHOLD})")

    assert passed, (
        f"FAIL: only {fraction_correct:.2%} of seeds recovered n=2 "
        f"(threshold {THRESHOLD:.0%})"
    )


if __name__ == "__main__":
    main()
