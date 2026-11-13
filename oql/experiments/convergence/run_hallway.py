"""Hallway POMDP quotient estimation experiment.

Runs QuotientEstimator on Cassandra's standard Hallway navigation POMDP
(60 states, 5 actions, 21 observations) across 5+ seeds with N=5000
trajectories and records the learned quotient size.

Reference: no published ground-truth quotient exists for Hallway.  With
history_max_length=1 the estimator partitions the 21 observation symbols, so
the structural ceiling is 21 (20 after the symbol without a full continuation
window is dropped); we report the learned count against that ceiling.

Design notes
------------
- history_max_length=1: single-observation windows.  The 21 observations
  are the primary discriminators; longer histories improve resolution at
  the cost of sample requirements that scale as 21^L.
- min_history_occurrences=50: ensures each candidate history has enough
  continuation samples for the G-test to be reliable.
- max_steps_per_episode=50: gives each episode enough steps for the agent
  to navigate between distinct hallway positions.
- alpha=0.01: standard significance level.
- window_length=2: 2-step continuation captures action-observation pairs.

Usage::

    python experiments/convergence/run_hallway.py
    python experiments/convergence/run_hallway.py --n_trajectories 8000 --n_seeds 10
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from environments.layer1_discrete.hallway import HallwayPOMDP
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.stats import bootstrap_ci
from quotient_learning.types import ObservationHistory

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_N_TRAJECTORIES = 5000
DEFAULT_N_SEEDS = 7  # paper reports 7 seeds: per-seed [19,19,19,20,20,19,19]
DEFAULT_HISTORY_MAX_LENGTH = 1
DEFAULT_MIN_HISTORY_OCCURRENCES = 50
DEFAULT_MAX_STEPS_PER_EPISODE = 50
DEFAULT_WINDOW_LENGTH = 2
DEFAULT_ALPHA = 0.01

# Structural ceiling: at history_max_length=1 the tested histories are the 21
# observation symbols, so the learned count cannot exceed 21.
PRIOR_BELIEF_CLUSTERS = 21
PRIOR_BELIEF_SOURCE = "structural ceiling: 21 observation symbols (no published quotient)"

FIELDNAMES = [
    "seed",
    "n_trajectories",
    "learned_n_classes_raw",
    "learned_n_classes",
    "prior_belief_clusters",
    "prior_source",
    "history_max_length",
    "min_history_occurrences",
    "max_steps_per_episode",
    "window_length",
    "alpha",
    "elapsed_s",
]


def _strip_root_class(q_raw: int, partition: dict) -> int:
    """Subtract root singleton class if present (matches KCycle convention)."""
    root = ObservationHistory(())
    if root in partition:
        root_cls = partition[root]
        root_members = [h for h, c in partition.items() if c == root_cls]
        if len(root_members) == 1:
            return max(0, q_raw - 1)
    return q_raw


def run_one(
    seed: int,
    n_trajectories: int,
    history_max_length: int,
    min_history_occurrences: int,
    max_steps_per_episode: int,
    window_length: int,
    alpha: float,
) -> dict:
    """Run one (seed) cell and return result row."""
    t0 = time.time()
    env = HallwayPOMDP(seed=seed, episodic=True)
    est = QuotientEstimator(
        env=env,
        n_trajectories=n_trajectories,
        window_length=window_length,
        history_max_length=history_max_length,
        min_history_occurrences=min_history_occurrences,
        alpha=alpha,
        max_steps_per_episode=max_steps_per_episode,
    )
    q = est.run(seed=seed)
    learned_raw = q.n_classes
    learned = _strip_root_class(learned_raw, q.partition)
    elapsed = time.time() - t0
    return {
        "seed": seed,
        "n_trajectories": n_trajectories,
        "learned_n_classes_raw": learned_raw,
        "learned_n_classes": learned,
        "prior_belief_clusters": PRIOR_BELIEF_CLUSTERS,
        "prior_source": PRIOR_BELIEF_SOURCE,
        "history_max_length": history_max_length,
        "min_history_occurrences": min_history_occurrences,
        "max_steps_per_episode": max_steps_per_episode,
        "window_length": window_length,
        "alpha": alpha,
        "elapsed_s": round(elapsed, 2),
    }


def print_summary(rows: list[dict], n_bootstrap: int = 1000) -> None:
    """Print text summary with mean, std, 95% CI."""
    learned_values = [r["learned_n_classes"] for r in rows]
    n = len(learned_values)
    mean_n = sum(learned_values) / n
    std_n = (sum((x - mean_n) ** 2 for x in learned_values) / max(1, n - 1)) ** 0.5

    lo, hi = bootstrap_ci(learned_values, n_resamples=n_bootstrap, ci=0.95, seed=0)
    prior = rows[0]["prior_belief_clusters"] if rows else PRIOR_BELIEF_CLUSTERS

    print("\n" + "=" * 60)
    print("Hallway POMDP (Cassandra 1998) — QuotientEstimator")
    print("=" * 60)
    print(f"  Specification:  60 states, 5 actions, 21 observations")
    print(f"  N trajectories: {rows[0]['n_trajectories']:,}")
    print(f"  N seeds:        {n}")
    print(f"  Learned n (mean ± std): {mean_n:.1f} ± {std_n:.2f}")
    print(f"  95% bootstrap CI:       [{lo:.1f}, {hi:.1f}]")
    print(f"  Structural ceiling:     {prior} observation symbols")
    print(f"  Ceiling source:         {rows[0]['prior_source'] if rows else PRIOR_BELIEF_SOURCE}")
    print(f"  Per-seed values:        {learned_values}")
    print()
    print(f"  Comparison to structural ceiling:")
    print(f"    Our n̄ = {mean_n:.1f}  vs.  ceiling {prior} (one symbol is dropped for lack of a")
    print(f"    full continuation window, so the effective ceiling is {prior - 1}).")
    print(f"    Learned count below the ceiling => some symbol pairs merged; no over-fragmentation")
    print(f"    is possible at history_max_length=1.  No exact-recovery claim is made.")
    print("=" * 60)


def main() -> None:
    p = argparse.ArgumentParser(description="Hallway POMDP quotient estimation")
    p.add_argument("--n_trajectories", type=int, default=DEFAULT_N_TRAJECTORIES)
    p.add_argument("--n_seeds", type=int, default=DEFAULT_N_SEEDS)
    p.add_argument("--history_max_length", type=int, default=DEFAULT_HISTORY_MAX_LENGTH)
    p.add_argument("--min_history_occurrences", type=int,
                   default=DEFAULT_MIN_HISTORY_OCCURRENCES)
    p.add_argument("--max_steps_per_episode", type=int,
                   default=DEFAULT_MAX_STEPS_PER_EPISODE)
    p.add_argument("--window_length", type=int, default=DEFAULT_WINDOW_LENGTH)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--out_dir", type=str,
                   default="experiments/convergence/results")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hallway.csv"

    seeds = list(range(args.n_seeds))
    print(
        f"Hallway experiment: N={args.n_trajectories:,}, seeds={seeds}\n"
        f"Config: history_max_length={args.history_max_length}, "
        f"min_history_occurrences={args.min_history_occurrences}, "
        f"max_steps={args.max_steps_per_episode}, "
        f"window_length={args.window_length}, alpha={args.alpha}"
    )

    rows: list[dict] = []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        print(f"  [{i+1}/{len(seeds)}] seed={seed} ... ", end="", flush=True)
        r = run_one(
            seed=seed,
            n_trajectories=args.n_trajectories,
            history_max_length=args.history_max_length,
            min_history_occurrences=args.min_history_occurrences,
            max_steps_per_episode=args.max_steps_per_episode,
            window_length=args.window_length,
            alpha=args.alpha,
        )
        rows.append(r)
        print(f"n={r['learned_n_classes']} (raw={r['learned_n_classes_raw']})  "
              f"[{r['elapsed_s']:.1f}s]")

    total_elapsed = time.time() - t0

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {csv_path}  (total elapsed {total_elapsed:.1f}s)")
    print_summary(rows, n_bootstrap=args.n_bootstrap)


if __name__ == "__main__":
    main()
