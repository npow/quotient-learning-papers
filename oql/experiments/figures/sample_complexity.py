"""Empirical vs theoretical sample complexity figure.

Plots empirical trajectory count (x) vs the theoretical bound of Theorem thm:complexity (linear in n up to logs) from the
`scaling.csv` produced by `experiments/convergence/run_scaling.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from quotient_learning.complexity import theoretical_complexity_curve


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scaling_csv", type=str,
                   default="experiments/convergence/results/scaling.csv")
    p.add_argument("--out_dir", type=str, default="experiments/figures")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.scaling_csv)
    if not csv_path.exists():
        print(f"No scaling CSV at {csv_path}; skipping figure")
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["final_n_classes"], df["trajectories_used"],
               label="Empirical", color="C0", alpha=0.7)
    # Theoretical curve
    n_range = list(range(2, max(df["final_n_classes"]) + 5))
    curve = theoretical_complexity_curve(n_range, epsilon=0.1, delta=0.05,
                                         window_length=2, episode_length=30)
    ax.plot([n for n, _ in curve], [t for _, t in curve],
            "--", label="Theoretical bound", color="C3")
    ax.set_xlabel("Discovered quotient size")
    ax.set_ylabel("Trajectories used")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Sample complexity: empirical vs theoretical")
    fig.tight_layout()
    fig.savefig(out / "sample_complexity.png", dpi=200)
    fig.savefig(out / "sample_complexity.pdf")
    plt.close(fig)
    print(f"Wrote {out}/sample_complexity.{{png,pdf}}")


if __name__ == "__main__":
    main()
