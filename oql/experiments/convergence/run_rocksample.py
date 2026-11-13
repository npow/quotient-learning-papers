"""RockSample convergence experiment (T033).

Default: RockSample(7,8). Can be re-pointed at (15,15) or (20,20) via flags
for the scaling-ladder experiments owned by US2 / T059.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from environments.layer1_discrete.rocksample import RockSamplePOMDP
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
    p.add_argument("--grid_size", type=int, default=7)
    p.add_argument("--n_rocks", type=int, default=8)
    p.add_argument("--n_trajectories", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--window_length", type=int, default=2)
    p.add_argument("--history_max_length", type=int, default=1)
    p.add_argument("--min_history_occurrences", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--out_dir", type=str,
                   default="experiments/convergence/results")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"rocksample_{args.grid_size}_{args.n_rocks}"

    cfg = ExperimentConfig(
        name=f"{name}_convergence",
        environment_id=f"RockSample({args.grid_size},{args.n_rocks})",
        n_trajectories=args.n_trajectories,
        window_length=args.window_length,
        equivalence_test="GTest",
        alpha=args.alpha,
        seeds=list(range(args.seeds)),
        extra={
            "grid_size": args.grid_size,
            "n_rocks": args.n_rocks,
            "history_max_length": args.history_max_length,
            "min_history_occurrences": args.min_history_occurrences,
            "max_steps_per_episode": args.max_steps,
        },
    )
    cfg.save(out_dir / f"{name}.config.json")

    budgets = _ladder(args.n_trajectories)
    rows: list[dict] = []
    t0 = time.time()
    print(f"Running RockSample({args.grid_size},{args.n_rocks}): budgets={budgets} seeds={cfg.seeds}")
    for seed in cfg.seeds:
        seed_everything(seed)
        env = RockSamplePOMDP(grid_size=args.grid_size, n_rocks=args.n_rocks, seed=seed)
        checkpoints, _ = run_convergence_ladder(
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
        print(f"  seed={seed}  final_n_classes={checkpoints[-1].n_classes}")

    csv_path = out_dir / f"{name}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"wall_time_s": time.time() - t0, "n_rows": len(rows)}
    (out_dir / f"{name}.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}; elapsed {summary['wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
