"""Plot convergence curves (n_classes vs n_trajectories) per environment, per seed.

Reads `experiments/convergence/results/*.csv` and writes a PDF + PNG per env.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_one(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for seed, sub in df.groupby("seed"):
        sub = sub.sort_values("n_trajectories")
        ax.plot(sub["n_trajectories"], sub["n_classes"], marker="o", label=f"seed {seed}")
    ax.set_xlabel("Trajectories")
    ax.set_ylabel("Learned quotient size")
    ax.set_xscale("log")
    ax.set_title(csv_path.stem)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str,
                   default="experiments/convergence/results")
    p.add_argument("--out_dir", type=str, default="experiments/figures")
    args = p.parse_args()
    results = Path(args.results_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    found = 0
    for csv in results.glob("*.csv"):
        # Only the n_classes-vs-n_trajectories tables; skip wide-format CSVs.
        df = pd.read_csv(csv, nrows=0)
        if "n_classes" not in df.columns or "n_trajectories" not in df.columns:
            continue
        plot_one(csv, out / f"convergence_{csv.stem}.png")
        found += 1
    print(f"Wrote {found} convergence figures to {out}/")


if __name__ == "__main__":
    main()
