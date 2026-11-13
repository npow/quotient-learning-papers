"""Recovery accuracy vs K and trajectory budget — headline figure.

Reads ``experiments/convergence/results/kcycle_recovery.csv`` and writes:
  * ``experiments/figures/recovery_vs_k.png``  — dual-panel figure
  * ``experiments/figures/recovery_vs_k.pdf``  — vector version

Panel 1 (heatmap):
    Rows = K values, Columns = trajectory budgets.
    Cell colour = mean absolute recovery error |learned_K - K| across seeds.
    Green = 0 error (perfect), red = large error.

Panel 2 (line plot):
    X axis = n_trajectories (log scale).
    One line per K value.
    Y axis = mean exact-recovery accuracy across seeds (0–1).
    Shows the empirical "trajectories required" scaling with K.

Usage::

    python experiments/figures/recovery_vs_k.py
    python experiments/figures/recovery_vs_k.py --csv path/to/kcycle_recovery.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"k", "n_trajectories", "seed", "learned_n_classes", "recovery_error"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    # Ensure exact_recovery column exists.
    if "exact_recovery" not in df.columns:
        df["exact_recovery"] = (df["recovery_error"] == 0).astype(int)
    return df


def make_figure(df: pd.DataFrame, out_dir: Path) -> None:
    k_values = sorted(df["k"].unique())
    budgets = sorted(df["n_trajectories"].unique())

    # ------------------------------------------------------------------ #
    # Build pivot tables
    # ------------------------------------------------------------------ #
    # Mean recovery error per (K, budget)
    pivot_error = (
        df.groupby(["k", "n_trajectories"])["recovery_error"]
        .mean()
        .unstack("n_trajectories")
        .reindex(index=k_values, columns=budgets)
        .fillna(np.nan)
    )
    # Mean exact-recovery accuracy per (K, budget)
    pivot_acc = (
        df.groupby(["k", "n_trajectories"])["exact_recovery"]
        .mean()
        .unstack("n_trajectories")
        .reindex(index=k_values, columns=budgets)
        .fillna(np.nan)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ------------------------------------------------------------------ #
    # Panel 1: Heatmap of mean recovery error
    # ------------------------------------------------------------------ #
    ax = axes[0]
    data = pivot_error.values.astype(float)
    vmax = max(1.0, float(np.nanmax(data)))
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Mean |learned − K|", fontsize=9)
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([str(b) for b in budgets], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels([str(k) for k in k_values], fontsize=9)
    ax.set_xlabel("Trajectory budget", fontsize=10)
    ax.set_ylabel("K (ground-truth quotient size)", fontsize=10)
    ax.set_title("Mean recovery error (lower = better)", fontsize=11)

    # Annotate cells with rounded error values.
    for i, k in enumerate(k_values):
        for j, b in enumerate(budgets):
            val = pivot_error.loc[k, b]
            if not np.isnan(val):
                text_color = "white" if val > vmax * 0.6 else "black"
                ax.text(
                    j, i, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=7, color=text_color,
                )

    # ------------------------------------------------------------------ #
    # Panel 2: Line plot — exact recovery accuracy vs budget
    # ------------------------------------------------------------------ #
    ax2 = axes[1]
    cmap = plt.get_cmap("tab10")
    for idx, k in enumerate(k_values):
        row = pivot_acc.loc[k]
        valid = row.dropna()
        if valid.empty:
            continue
        xs = valid.index.to_numpy()
        ys = valid.values
        color = cmap(idx % 10)
        ax2.plot(xs, ys, marker="o", linewidth=1.8, markersize=5,
                 label=f"K={k}", color=color)

    ax2.set_xscale("log")
    ax2.set_xlim(left=budgets[0] * 0.8, right=budgets[-1] * 1.2)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Trajectory budget (log scale)", fontsize=10)
    ax2.set_ylabel("Exact recovery rate (across seeds)", fontsize=10)
    ax2.set_title("Recovery accuracy vs trajectory budget", fontsize=11)
    ax2.legend(title="K", loc="lower right", fontsize=8, title_fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle(
        "K-Cycle POMDP: Quotient Recovery Accuracy vs Ground Truth K",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "recovery_vs_k.png"
    pdf_path = out_dir / "recovery_vs_k.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=str,
        default="experiments/convergence/results/kcycle_recovery.csv",
    )
    p.add_argument("--out_dir", type=str, default="experiments/figures")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Run `python experiments/convergence/run_kcycle.py` first."
        )
    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    make_figure(df, Path(args.out_dir))


if __name__ == "__main__":
    main()
