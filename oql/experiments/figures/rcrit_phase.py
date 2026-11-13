"""R_crit phase-transition diagram.

Reads `experiments/rcrit_prediction/results/rcrit.csv` and plots
`predicted_feasible` vs `true_feasible` across the (n_classes, channel_capacity)
plane, highlighting agreement / disagreement regions.
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
                   default="experiments/rcrit_prediction/results/rcrit.csv")
    p.add_argument("--out_dir", type=str, default="experiments/figures")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"No CSV at {csv_path}; skipping figure")
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    correct = df[df["predicted_feasible"] == df["true_feasible"]]
    wrong = df[df["predicted_feasible"] != df["true_feasible"]]
    ax.scatter(correct["channel_capacity"], correct["r_crit"],
               c="C2", label=f"correct ({len(correct)})", alpha=0.6)
    ax.scatter(wrong["channel_capacity"], wrong["r_crit"],
               c="C3", label=f"wrong ({len(wrong)})", marker="x")
    diag = max(df["channel_capacity"].max(), df["r_crit"].max()) + 0.5
    ax.plot([0, diag], [0, diag], "--", color="grey", alpha=0.5,
            label="R_crit = capacity")
    ax.set_xlabel("Channel capacity (bits/step)")
    ax.set_ylabel("Learned R_crit (bits/step)")
    ax.set_title("R_crit phase-transition diagram")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "rcrit_phase.png", dpi=200)
    fig.savefig(out / "rcrit_phase.pdf")
    plt.close(fig)
    print(f"Wrote {out}/rcrit_phase.{{png,pdf}}")


if __name__ == "__main__":
    main()
