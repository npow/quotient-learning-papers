"""Main results table heatmap.

Reads `experiments/main_table/results/main_table.csv` and produces a grouped
bar chart of quotient size per (env, method).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str,
                   default="experiments/main_table/results/main_table.csv")
    p.add_argument("--out_dir", type=str, default="experiments/figures")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"No CSV at {csv_path}; skipping figure")
        return
    df = pd.read_csv(csv_path)
    method_cols = [c for c in df.columns if c.startswith("baseline_")] + ["quotient_estimator_n_classes"]
    avg = df.groupby("env")[method_cols].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    avg.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean learned quotient size")
    ax.set_xlabel("Environment")
    ax.set_title("Main results: quotient size by method × env")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out / "main_table.png", dpi=200)
    fig.savefig(out / "main_table.pdf")
    plt.close(fig)
    print(f"Wrote {out}/main_table.{{png,pdf}}")


if __name__ == "__main__":
    main()
