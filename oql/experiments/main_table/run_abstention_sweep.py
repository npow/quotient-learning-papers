"""Abstention-rate sweep for the certified path (kcycle_8).

Demonstrates the m* power gate binding: as the episode budget N falls below
the point where per-context inference counts reach m*, pairs abstain
(UNDETERMINED) instead of merging.  The safety property to verify is that
class recovery degrades only through abstention — no seed at any budget may
report a wrong merge (fewer than the 8 analytic classes).

Usage:
    python experiments/main_table/run_abstention_sweep.py \
        [--seeds 0-9] [--out experiments/main_table/results_abstention_sweep]

Outputs: sweep CSV, one certified sidecar per (N, seed), and
paper/figures/abstention_sweep.pdf.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from environments.layer1_discrete.kcycle import KCyclePOMDP
from experiments.main_table.run_certified import (
    BETA,
    DELTA,
    eps_min_for,
    parse_seeds,
)
from quotient_learning.config import CertifiedConfig, seed_everything
from quotient_learning.equivalence import TVThresholdEquivalenceTest
from quotient_learning.estimator import QuotientEstimator

ENV_NAME = "kcycle_8"
EXPECTED_CLASSES = 8
BUDGETS = [6000, 6500, 7000, 7250, 7500, 7750, 8000, 8500]


def run_one(n_trajectories: int, seed: int, out_dir: Path) -> dict:
    seed_everything(seed)
    env = KCyclePOMDP(8, seed=seed)
    eps_min = eps_min_for(ENV_NAME)
    alphabet = env.continuation_alphabet(1)
    est = QuotientEstimator(
        env=env,
        equivalence_test=TVThresholdEquivalenceTest(alphabet, eps_min),
        delta=DELTA,
        epsilon_min=eps_min,
        power_beta=BETA,
        certified_independent=True,
        selection_fraction=0.5,
        independent_context_end=0,
        include_root=False,
        n_trajectories=n_trajectories,
        window_length=1,
        history_max_length=1,
        min_history_occurrences=100,
        max_steps_per_episode=20,
    )
    q = est.run(seed=seed)
    row = {
        "env": ENV_NAME,
        "n_trajectories": n_trajectories,
        "seed": seed,
        "n_classes": q.n_classes,
        "expected": EXPECTED_CLASSES,
        "undetermined_rate": round(q.undetermined_rate, 4),
        "n_boundary_flags": len(q.boundary_flags),
    }
    try:
        recorded_out = out_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        recorded_out = out_dir.resolve()
    cfg = CertifiedConfig(
        k=1, tau=1, test="TV_FIXED_ALPHABET", alpha=DELTA,
        epsilon_min=eps_min, beta=BETA, seeds=[seed],
        selection_fraction=0.5, context_end=0,
        policy_id="uniform_random",
        n_trajectories=n_trajectories,
        history_max_length=1,
        min_history_occurrences=100,
        max_steps_per_episode=20,
        command=(
            "python experiments/main_table/run_abstention_sweep.py "
            f"--seeds {seed} --out {recorded_out}"
        ),
    )
    cfg.capture_env_pins()
    cfg.save_sidecar(out_dir / f"{ENV_NAME}_N{n_trajectories}_seed{seed}.certified.json")
    return row


def make_figure(rows: list[dict], fig_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = sorted({int(r["n_trajectories"]) for r in rows})
    means, mins, maxs = [], [], []
    for n in budgets:
        us = [float(r["undetermined_rate"]) for r in rows
              if int(r["n_trajectories"]) == n]
        means.append(float(np.mean(us)))
        mins.append(float(np.min(us)))
        maxs.append(float(np.max(us)))

    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.plot(budgets, means, marker="o", lw=1.5, color="tab:blue",
            label="mean over seeds")
    ax.fill_between(budgets, mins, maxs, alpha=0.25, color="tab:blue",
                    lw=0, label="min--max over seeds")
    ax.set_xlabel("episodes $N$")
    ax.set_ylabel("abstaining pair fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent / "results_abstention_sweep"),
    )
    ap.add_argument(
        "--fig",
        default=str(
            Path(__file__).resolve().parent.parent.parent
            / "paper" / "figures" / "abstention_sweep.pdf"
        ),
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    rows: list[dict] = []
    t0 = time.time()
    wrong_merges = 0
    for n in BUDGETS:
        for seed in seeds:
            row = run_one(n, seed, out_dir)
            rows.append(row)
            if row["n_classes"] < EXPECTED_CLASSES:
                wrong_merges += 1
            print(
                f"  N={n} seed={seed} n_classes={row['n_classes']} "
                f"und={row['undetermined_rate']}",
                flush=True,
            )

    csv_path = out_dir / "abstention_sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    make_figure(rows, Path(args.fig))
    print(f"\nWrote {csv_path} and {args.fig}; elapsed {time.time() - t0:.1f}s")
    if wrong_merges:
        print(f"SAFETY VIOLATION: {wrong_merges} runs merged below "
              f"{EXPECTED_CLASSES} classes")
        return 1
    print(f"safety check: no wrong merges in {len(rows)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
