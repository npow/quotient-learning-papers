"""Analysed-vs-reported configuration comparison.

Runs QuotientEstimator on each headline environment in two configurations:

  reported: stride 1 (all overlapping continuation windows), fixed per-pair
            level alpha -- the configuration used by every experiment table
            in the paper.
  analysed: window_stride = window_length * tau (tau = 1: these chains mix in
            O(1) steps), Bonferroni per-pair level alpha_0 = delta/(2 K_N^2)
            via the ``delta`` parameter -- the configuration covered by
            Theorems 3 and 5.

Writes results/analysed_config.csv and results/analysed_config_summary.md.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from environments.base import POMDPEnvironment
from environments.layer1_discrete.hallway import HallwayPOMDP
from environments.layer1_discrete.kcycle import KCyclePOMDP
from environments.layer1_discrete.rocksample import RockSamplePOMDP
from environments.layer1_discrete.synthetic import SyntheticPOMDP
from environments.layer1_discrete.tiger import TigerPOMDP
from environments.layer2_memory.tmaze import TMazePOMDP
from environments.layer4_realworld.robot_localization import RobotLocalizationPOMDP
from environments.layer4_realworld.sepsis import SepsisPOMDP
from quotient_learning.config import seed_everything
from quotient_learning.estimator import QuotientEstimator

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DELTA = 0.05  # family-wise budget for the analysed configuration
TAU = 1       # block length; all envs here mix in O(1) steps

# (env_name, constructor, estimator kwargs shared by both configs, seeds)
CONFIGS = [
    ("tiger",
     lambda seed: TigerPOMDP(seed=seed),
     dict(n_trajectories=3000, window_length=2, history_max_length=1,
          min_history_occurrences=10, alpha=0.001, max_steps_per_episode=15),
     10),
    ("kcycle_8",
     lambda seed: KCyclePOMDP(8, seed=seed),
     dict(n_trajectories=1000, window_length=2, history_max_length=1,
          min_history_occurrences=100, alpha=0.01, max_steps_per_episode=20),
     5),
    ("hallway",
     lambda seed: HallwayPOMDP(seed=seed),
     dict(n_trajectories=5000, window_length=2, history_max_length=1,
          min_history_occurrences=50, alpha=0.01, max_steps_per_episode=50),
     7),
    ("rocksample_7_8",
     lambda seed: RockSamplePOMDP(7, 8, seed=seed),
     dict(n_trajectories=5000, window_length=2, history_max_length=2,
          min_history_occurrences=80, alpha=0.01, max_steps_per_episode=25),
     5),
    ("tmaze",
     lambda seed: TMazePOMDP(corridor_length=5, seed=seed),
     dict(n_trajectories=5000, window_length=2, history_max_length=2,
          min_history_occurrences=80, alpha=0.01, max_steps_per_episode=25),
     5),
    ("sepsis",
     lambda seed: SepsisPOMDP(seed=seed),
     dict(n_trajectories=5000, window_length=2, history_max_length=2,
          min_history_occurrences=80, alpha=0.01, max_steps_per_episode=25),
     5),
    ("robot_localization",
     lambda seed: RobotLocalizationPOMDP(size=6, n_landmarks=4, seed=seed),
     dict(n_trajectories=5000, window_length=2, history_max_length=2,
          min_history_occurrences=80, alpha=0.01, max_steps_per_episode=25),
     5),
    ("synthetic_200",
     lambda seed: SyntheticPOMDP(n_states=200, n_obs=10, seed=seed),
     dict(n_trajectories=5000, window_length=2, history_max_length=2,
          min_history_occurrences=80, alpha=0.01, max_steps_per_episode=25),
     5),
]


def run_one(env: POMDPEnvironment, kwargs: dict, seed: int,
            analysed: bool) -> int:
    kw = dict(kwargs)
    if analysed:
        kw.pop("alpha", None)
        kw["delta"] = DELTA
        kw["window_stride"] = kw.get("window_length", 2) * TAU
    est = QuotientEstimator(env=env, **kw)
    dataset = est.collect(seed=seed)
    return est.estimate(dataset).n_classes


def main() -> None:
    rows: list[dict] = []
    t0 = time.time()
    for env_name, ctor, kwargs, seeds in CONFIGS:
        for seed in range(seeds):
            row: dict = {"env": env_name, "seed": seed}
            for label, analysed in (("reported", False), ("analysed", True)):
                seed_everything(seed)
                row[f"n_classes_{label}"] = run_one(ctor(seed), kwargs, seed,
                                                    analysed)
            rows.append(row)
            print(f"  {env_name} seed={seed} "
                  f"reported={row['n_classes_reported']} "
                  f"analysed={row['n_classes_analysed']}", flush=True)

    csv_path = RESULTS_DIR / "analysed_config.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["# Analysed vs reported configuration",
             "",
             f"delta={DELTA}, tau={TAU}, stride=window_length*tau; "
             "reported = stride 1 + per-pair alpha.",
             "",
             "| env | reported (mean ± std) | analysed (mean ± std) | seeds |",
             "|---|---|---|---|"]
    for env_name, _, _, seeds in CONFIGS:
        rep = [r["n_classes_reported"] for r in rows if r["env"] == env_name]
        ana = [r["n_classes_analysed"] for r in rows if r["env"] == env_name]
        lines.append(
            f"| {env_name} | {np.mean(rep):.1f} ± {np.std(rep, ddof=1) if len(rep) > 1 else 0:.2f} "
            f"| {np.mean(ana):.1f} ± {np.std(ana, ddof=1) if len(ana) > 1 else 0:.2f} | {seeds} |")
    (RESULTS_DIR / "analysed_config_summary.md").write_text(
        "\n".join(lines) + "\n")
    print(f"\nWrote {csv_path}; elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
