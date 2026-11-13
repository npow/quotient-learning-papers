"""Main results table (T060): all envs × all baselines × seeds.

Produces the headline numbers for the paper: learned quotient size, R_crit
proxy (n_classes ratio), wall-clock time, and a same-or-different vs each
baseline. CSV + JSON output for direct ingestion by the figure scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from environments.base import POMDPEnvironment
from environments.layer1_discrete.rocksample import RockSamplePOMDP
from environments.layer1_discrete.synthetic import SyntheticPOMDP
from environments.layer1_discrete.tiger import TigerPOMDP
from environments.layer2_memory.tmaze import TMazePOMDP
from environments.layer4_realworld.robot_localization import RobotLocalizationPOMDP
from environments.layer4_realworld.sepsis import SepsisPOMDP
from experiments.baselines.bisimulation import bisimulation_quotient
from experiments.baselines.histogram_naive import histogram_naive_quotient
from experiments.baselines.pomdp_em import em_quotient
from experiments.baselines.psr_spectral import psr_quotient
from quotient_learning.config import seed_everything
from quotient_learning.estimator import QuotientEstimator


def _envs(name: str, seed: int) -> POMDPEnvironment:
    if name == "tiger":
        return TigerPOMDP(seed=seed)
    if name == "rocksample_7_8":
        return RockSamplePOMDP(7, 8, seed=seed)
    if name == "synthetic_200":
        return SyntheticPOMDP(n_states=200, n_obs=10, seed=seed)
    if name == "tmaze":
        return TMazePOMDP(corridor_length=5, seed=seed)
    if name == "sepsis":
        return SepsisPOMDP(seed=seed)
    if name == "robot_localization":
        return RobotLocalizationPOMDP(size=6, n_landmarks=4, seed=seed)
    raise KeyError(name)


def _baseline_runs(name: str, dataset, env: POMDPEnvironment, seed: int) -> dict[str, int]:
    """Return a dict of baseline_name -> n_classes."""
    out: dict[str, int] = {}
    try:
        out["psr"] = psr_quotient(dataset, n_obs=env.n_observations,
                                   rank=4, history_length=1, future_length=2,
                                   seed=seed).n_classes
    except Exception:
        out["psr"] = -1
    try:
        out["histogram_naive"] = histogram_naive_quotient(
            dataset, n_obs=env.n_observations, window_length=2,
            history_max_length=1, min_occurrences=30, epsilon=0.15,
        ).n_classes
    except Exception:
        out["histogram_naive"] = -1
    try:
        out["em"] = em_quotient(dataset,
                                  n_obs=env.n_observations,
                                  n_iterations=3, seed=seed).n_classes
    except Exception:
        out["em"] = -1
    if isinstance(env, SyntheticPOMDP):
        try:
            out["bisimulation_oracle"] = bisimulation_quotient(env).n_classes
        except Exception:
            out["bisimulation_oracle"] = -1
    else:
        out["bisimulation_oracle"] = -1  # not applicable
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="+",
                   default=["tiger", "rocksample_7_8", "tmaze",
                            "sepsis", "robot_localization", "synthetic_200"])
    p.add_argument("--n_trajectories", type=int, default=5000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=25)
    p.add_argument("--out_dir", type=str, default="experiments/main_table/results")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-environment overrides for calibrated defaults.
    # Tiger requires hml=1 to avoid transitive union-find bridges through H(0,0);
    # see paper §5.1.
    _env_overrides: dict[str, dict] = {
        "tiger": dict(
            n_trajectories=3000,
            window_length=2,
            history_max_length=1,
            min_history_occurrences=10,
            alpha=0.001,
            max_steps_per_episode=15,
        ),
    }

    rows: list[dict] = []
    t0 = time.time()
    for env_name in args.envs:
        for seed in range(args.seeds):
            seed_everything(seed)
            env = _envs(env_name, seed=seed)
            overrides = _env_overrides.get(env_name, {})
            est = QuotientEstimator(
                env=env,
                n_trajectories=overrides.get("n_trajectories", args.n_trajectories),
                window_length=overrides.get("window_length", 2),
                history_max_length=overrides.get("history_max_length", 2),
                min_history_occurrences=overrides.get("min_history_occurrences", 80),
                alpha=overrides.get("alpha", 0.01),
                max_steps_per_episode=overrides.get("max_steps_per_episode", args.max_steps),
            )
            dataset = est.collect(seed=seed)
            t_start = time.time()
            q = est.estimate(dataset)
            t_quotient = time.time() - t_start
            baselines = _baseline_runs(env_name, dataset, env, seed)
            row = {
                "env": env_name,
                "seed": seed,
                "quotient_estimator_n_classes": q.n_classes,
                "quotient_estimator_time_s": round(t_quotient, 3),
                "n_trajectories": args.n_trajectories,
                **{f"baseline_{k}": v for k, v in baselines.items()},
            }
            rows.append(row)
            print(f"  {env_name} seed={seed} ours={q.n_classes} "
                  f"baselines={baselines}")

    csv_path = out_dir / "main_table.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"wall_time_s": time.time() - t0, "n_rows": len(rows),
               "envs": args.envs, "seeds": args.seeds}
    (out_dir / "main_table.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}; elapsed {summary['wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
