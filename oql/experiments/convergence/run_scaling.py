"""Scaling experiment (T059): empirical vs theoretical sample complexity.

Runs the QuotientEstimator across the RockSample(7,8) → (15,15) → (20,20)
ladder and a SyntheticPOMDP at S=200/500/1000. For each environment, finds
the smallest trajectory budget that produces a stable quotient and compares
against `complexity.trajectories_required`.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from environments.layer1_discrete.rocksample import RockSamplePOMDP
from environments.layer1_discrete.synthetic import SyntheticPOMDP
from quotient_learning.complexity import trajectories_required
from quotient_learning.config import seed_everything
from quotient_learning.convergence import run_convergence_ladder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--out_dir", type=str, default="experiments/convergence/results")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configurations = [
        ("RockSample(7,8)", lambda s: RockSamplePOMDP(7, 8, seed=s),
         [500, 1000, 2000], 20, 1, 30),
        ("RockSample(15,15)", lambda s: RockSamplePOMDP(15, 15, seed=s),
         [1000, 2500, 5000], 30, 1, 30),
        ("Synthetic(S=200)", lambda s: SyntheticPOMDP(n_states=200, n_obs=10, seed=s),
         [1000, 2000, 4000], 30, 1, 30),
        ("Synthetic(S=500)", lambda s: SyntheticPOMDP(n_states=500, n_obs=20, seed=s),
         [2000, 4000, 8000], 30, 1, 30),
    ]
    rows: list[dict] = []
    t0 = time.time()
    for name, make_env, budgets, max_steps, hist_len, min_occ in configurations:
        for seed in range(args.seeds):
            seed_everything(seed)
            env = make_env(seed)
            checkpoints, q = run_convergence_ladder(
                env=env, budgets=budgets,
                estimator_kwargs=dict(
                    window_length=2, history_max_length=hist_len,
                    min_history_occurrences=min_occ, alpha=0.01,
                    max_steps_per_episode=max_steps,
                ),
                seed=seed,
            )
            final = checkpoints[-1].n_classes
            theory = trajectories_required(
                epsilon=args.epsilon, delta=args.delta,
                n_classes=max(final, 2), window_length=2,
                episode_length=max_steps,
            )
            rows.append({
                "env": name,
                "seed": seed,
                "final_n_classes": final,
                "trajectories_used": budgets[-1],
                "theory_trajectories": theory,
                "within_bound": int(budgets[-1] >= theory),
            })
            print(f"  {name} seed={seed} n_classes={final} "
                  f"used={budgets[-1]} theory={theory}")

    csv_path = out_dir / "scaling.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"wall_time_s": time.time() - t0, "n_rows": len(rows)}
    (out_dir / "scaling.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}; elapsed {summary['wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
