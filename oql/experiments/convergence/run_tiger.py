"""Tiger convergence experiment (T032).

Runs the QuotientEstimator on Tiger across a logarithmic trajectory ladder
and ≥ N seeds. Writes one row per (seed, budget) pair to CSV. Reproduces
the Scenario-1 acceptance criterion: learned quotient size ≈ 2 across seeds.

Usage:
    python experiments/convergence/run_tiger.py            # paper defaults (hml=1, N=3000, 10 seeds)
    python experiments/convergence/run_tiger.py --history_max_length 2  # collapse curve (n=1)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from environments.layer1_discrete.tiger import TigerPOMDP
from quotient_learning.config import ExperimentConfig, seed_everything
from quotient_learning.convergence import run_convergence_ladder


def _ladder(max_n: int) -> list[int]:
    budgets: list[int] = []
    b = 250
    while b <= max_n:
        budgets.append(b)
        b *= 2
    if max_n not in budgets:
        budgets.append(max_n)
    return sorted(set(budgets))


def main() -> None:
    p = argparse.ArgumentParser()
    # Defaults match the paper's calibrated Tiger config (Sec. "Layer 1"):
    # history_max_length=1 recovers the 2-class observation quotient on 10/10
    # seeds at N=3000; history_max_length=2 collapses to n=1 via the H(0,0)
    # union-find bridge (pass --history_max_length 2 to reproduce that curve).
    p.add_argument("--n_trajectories", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--window_length", type=int, default=2)
    p.add_argument("--history_max_length", type=int, default=1)
    p.add_argument("--min_history_occurrences", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--max_steps", type=int, default=15)
    p.add_argument("--out_dir", type=str,
                   default="experiments/convergence/results")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        name="tiger_convergence",
        environment_id=TigerPOMDP(seed=0).environment_id,
        n_trajectories=args.n_trajectories,
        window_length=args.window_length,
        equivalence_test="GTest",
        alpha=args.alpha,
        seeds=list(range(args.seeds)),
        extra={
            "history_max_length": args.history_max_length,
            "min_history_occurrences": args.min_history_occurrences,
            "max_steps_per_episode": args.max_steps,
        },
    )
    cfg.save(out_dir / "tiger_convergence.config.json")

    budgets = _ladder(args.n_trajectories)
    rows: list[dict] = []
    print(f"Running Tiger convergence: budgets={budgets} seeds={cfg.seeds}")
    t0 = time.time()
    for seed in cfg.seeds:
        seed_everything(seed)
        env = TigerPOMDP(seed=seed)
        checkpoints, q = run_convergence_ladder(
            env=env, budgets=budgets,
            estimator_kwargs=dict(
                window_length=args.window_length,
                history_max_length=args.history_max_length,
                min_history_occurrences=args.min_history_occurrences,
                alpha=args.alpha,
                max_steps_per_episode=args.max_steps,
            ),
            seed=seed,
        )
        for cp in checkpoints:
            rows.append({
                "seed": seed,
                "n_trajectories": cp.n_trajectories,
                "n_classes": cp.n_classes,
                "n_partition_entries": cp.n_partition_entries,
            })
        final = checkpoints[-1].n_classes
        print(f"  seed={seed}  final_n_classes={final}  "
              f"(ground_truth={env.metadata['ground_truth_n_classes']})")

    csv_path = out_dir / "tiger_convergence.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "wall_time_s": time.time() - t0,
        "n_rows": len(rows),
        "final_n_classes_per_seed": {
            seed: [r["n_classes"] for r in rows
                   if r["seed"] == seed and r["n_trajectories"] == budgets[-1]][0]
            for seed in cfg.seeds
        },
    }
    (out_dir / "tiger_convergence.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path} and summary; elapsed {summary['wall_time_s']:.1f}s")
    print(f"Final n_classes per seed: {summary['final_n_classes_per_seed']}")


if __name__ == "__main__":
    main()
