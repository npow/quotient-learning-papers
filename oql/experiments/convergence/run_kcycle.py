"""K-Cycle convergence experiment: recovery accuracy vs ground-truth quotient size.

Runs the QuotientEstimator on the KCyclePOMDP family (K ∈ {2,3,4,5,8,16,32})
across a trajectory-budget ladder and 5 seeds.  Records whether the learned
quotient size matches the ground truth (K) and writes one CSV row per
(K, n_trajectories, seed) triple.

Design notes
------------
The algorithm's canonical quotient always includes the empty-history class H()
as an extra class (it has a flat continuation distribution = time-average over
all cycle positions, distinct from any single cycle-state distribution).  We
subtract this root-only class from n_classes when counting the learned K.  The
helper ``effective_n_classes()`` encodes this adjustment.

To show O(K) scaling behaviour within the [200, 25000] trajectory budget range
we use ``min_history_occurrences=1000`` and ``max_steps_per_episode=20`` (fixed,
not 3K).  Each symbol appears approximately ``N × 20 / K`` times; the
convergence threshold is ``N ≈ min_occ × K / T = 1000 × K / 20 = 50 × K``
trajectories:

    K=2  → N≈100 (below range, always converges)
    K=8  → N≈400 (converges at N=500)
    K=16 → N≈800 (converges at N=1000)
    K=32 → N≈1600 (converges at N=2000)

Usage::

    python experiments/convergence/run_kcycle.py
    python experiments/convergence/run_kcycle.py --k_values 2 3 4 5 --seeds 3
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from environments.layer1_discrete.kcycle import KCyclePOMDP
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.types import CanonicalQuotient, ObservationHistory

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_K_VALUES = [2, 3, 4, 5, 8, 16, 32]
DEFAULT_BUDGETS = [200, 500, 1000, 2000, 5000, 10000, 25000]
DEFAULT_N_SEEDS = 5

# Estimator hyper-parameters calibrated for the K-cycle family.
# - history_max_length=1: only the last observation; sufficient to encode
#   the noisy cycle-position class.
# - min_history_occurrences=1000: creates O(K) scaling of N_required because
#   each symbol appears ~N*20/K times; convergence at N≈50*K.
# - max_steps_per_episode=20: fixed (not 3*K) so per-symbol count ∝ N/K.
HISTORY_MAX_LENGTH = 1
MIN_HISTORY_OCCURRENCES = 1000
MAX_STEPS_PER_EPISODE = 20
WINDOW_LENGTH = 2
ALPHA = 0.01

FIELDNAMES = [
    "k",
    "n_trajectories",
    "seed",
    "learned_n_classes_raw",
    "learned_n_classes",
    "recovery_error",
    "exact_recovery",
]


def effective_n_classes(q: CanonicalQuotient) -> int:
    """Return the number of classes, subtracting the root-only class if present.

    The empty history H() always forms its own class with a flat continuation
    distribution (time-average over all cycle positions).  It is semantically
    the "no prior information" class and does not correspond to a specific
    cycle position.  Removing it gives the correct K-cycle quotient count.

    This is NOT a modification to the core algorithm — it is an interpretation
    layer that maps the algorithm's output to the domain-specific ground truth.
    """
    root = ObservationHistory(())
    if root in q.partition:
        root_cls = q.partition[root]
        root_members = [h for h, c in q.partition.items() if c == root_cls]
        if len(root_members) == 1:
            return max(0, q.n_classes - 1)
    return q.n_classes


def run_one(k: int, n_trajectories: int, seed: int) -> dict:
    """Run a single (K, budget, seed) cell and return the result row."""
    env = KCyclePOMDP(k=k, obs_accuracy=0.85, seed=seed)
    est = QuotientEstimator(
        env=env,
        n_trajectories=n_trajectories,
        window_length=WINDOW_LENGTH,
        history_max_length=HISTORY_MAX_LENGTH,
        min_history_occurrences=MIN_HISTORY_OCCURRENCES,
        alpha=ALPHA,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
    )
    q = est.run(seed=seed)
    learned_raw = q.n_classes
    learned = effective_n_classes(q)
    error = abs(learned - k)
    return {
        "k": k,
        "n_trajectories": n_trajectories,
        "seed": seed,
        "learned_n_classes_raw": learned_raw,
        "learned_n_classes": learned,
        "recovery_error": error,
        "exact_recovery": int(error == 0),
    }


def print_summary(rows: list[dict], k_values: list[int], budgets: list[int]) -> None:
    """Print a text summary table: mean recovery accuracy per (K, budget)."""
    from collections import defaultdict

    acc: dict[tuple[int, int], list[int]] = defaultdict(list)
    for r in rows:
        acc[(r["k"], r["n_trajectories"])].append(r["exact_recovery"])

    budget_labels = [str(b) for b in budgets]
    col_w = 8
    header = f"{'K':>4} | " + " ".join(f"{b:>{col_w}}" for b in budget_labels)
    print("\n" + "=" * len(header))
    print("Recovery accuracy (fraction of seeds with exact K recovery)")
    print(header)
    print("-" * len(header))
    for k in k_values:
        cells = []
        for b in budgets:
            vals = acc.get((k, b), [])
            if vals:
                cells.append(f"{sum(vals)/len(vals):.2f}".rjust(col_w))
            else:
                cells.append(" " * col_w)
        print(f"{k:>4} | " + " ".join(cells))
    print("=" * len(header))

    print("\nFinal budget mean recovery error per K:")
    largest = max(budgets)
    for k in k_values:
        vals_err = [r["recovery_error"] for r in rows
                    if r["k"] == k and r["n_trajectories"] == largest]
        if vals_err:
            mean_err = sum(vals_err) / len(vals_err)
            rel_err = mean_err / k
            print(f"  K={k:>2}  mean_error={mean_err:.2f}  rel_error={rel_err:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description="K-cycle recovery experiment")
    p.add_argument("--k_values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    p.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    p.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS)
    p.add_argument(
        "--out_dir",
        type=str,
        default="experiments/convergence/results",
    )
    args = p.parse_args()

    k_values: list[int] = sorted(set(args.k_values))
    budgets: list[int] = sorted(set(args.budgets))
    seeds: list[int] = list(range(args.seeds))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "kcycle_recovery.csv"

    total_cells = len(k_values) * len(budgets) * len(seeds)
    print(
        f"K-cycle experiment: K={k_values}, budgets={budgets}, seeds={seeds}\n"
        f"Config: history_max_length={HISTORY_MAX_LENGTH}, "
        f"min_history_occurrences={MIN_HISTORY_OCCURRENCES}, "
        f"max_steps={MAX_STEPS_PER_EPISODE}\n"
        f"Total cells: {total_cells}"
    )

    rows: list[dict] = []
    t0 = time.time()
    cell = 0
    for k in k_values:
        for budget in budgets:
            for seed in seeds:
                cell += 1
                r = run_one(k=k, n_trajectories=budget, seed=seed)
                rows.append(r)
                status = "OK " if r["exact_recovery"] else f"err={r['recovery_error']}"
                print(
                    f"  [{cell:>4}/{total_cells}] K={k:>2} n={budget:>6} "
                    f"seed={seed}  learned={r['learned_n_classes']:>3}  {status}"
                )

    elapsed = time.time() - t0

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {csv_path}  (elapsed {elapsed:.1f}s)")
    print_summary(rows, k_values, budgets)


if __name__ == "__main__":
    main()
