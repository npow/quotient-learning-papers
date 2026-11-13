"""Generate quotient size comparison figure (Figure 1 of paper draft).

Shows n_classes recovered per environment × method, with 95% CI across seeds.
Saves to experiments/figures/quotient_size_comparison.pdf and .png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
RESULTS = ROOT / "experiments"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

# Tiger
tiger = pd.read_csv(RESULTS / "layer1_ground_truth/results/tiger_continuous.csv")
tiger["env"] = "Tiger"
tiger["method"] = "MMD (ours)"
tiger["ground_truth"] = 2

# K-cycle
kcycle = pd.read_csv(RESULTS / "layer1_ground_truth/results/kcycle_continuous.csv")
kcycle["env"] = kcycle["K"].apply(lambda k: f"K-cycle (K={k})")
kcycle["method"] = "MMD (ours)"
kcycle["ground_truth"] = kcycle["K"]

# Ablation (Tiger only — shows method comparison)
ablation = pd.read_csv(RESULTS / "baselines/results/ablation_sample_efficiency.csv")
tiger_abl = ablation[ablation["env"] == "tiger_n2"].copy()
tiger_abl["env"] = "Tiger"
tiger_abl["ground_truth"] = 2
tiger_abl["method"] = tiger_abl["method"].map({
    "mmd_continuous": "MMD (ours)",
    "gtest_3bin": "G-test (3 bins)",
    "gtest_8bin": "G-test (8 bins)",
})

# ---------------------------------------------------------------------------
# Figure 1a: K-cycle quotient size recovery
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax1, ax2 = axes

K_vals = sorted(kcycle["K"].unique())
means = [kcycle[kcycle["K"] == K]["n_classes"].mean() for K in K_vals]
stds  = [kcycle[kcycle["K"] == K]["n_classes"].std() for K in K_vals]
ax1.errorbar(K_vals, means, yerr=stds, fmt='o-', color='steelblue', label='Estimated n̄', capsize=4, linewidth=2)
ax1.plot(K_vals, K_vals, 'k--', alpha=0.5, label='Ground truth n=K')
ax1.set_xlabel("K (cycle length)", fontsize=12)
ax1.set_ylabel("Estimated n_classes", fontsize=12)
ax1.set_title("K-Cycle: Quotient Recovery (5 seeds/K)", fontsize=13, fontweight='bold')
ax1.legend()
ax1.set_xticks(K_vals)
ax1.grid(True, alpha=0.3)

# Figure 1b: Tiger — method comparison at varying N
N_vals = sorted(tiger_abl["N"].unique())
methods = ["MMD (ours)", "G-test (8 bins)", "G-test (3 bins)"]
colors  = ["steelblue", "darkorange", "crimson"]
markers = ["o", "s", "^"]

for method, color, marker in zip(methods, colors, markers):
    df = tiger_abl[tiger_abl["method"] == method]
    means_m = [df[df["N"] == N]["correct"].mean() for N in N_vals]
    ax2.plot(N_vals, means_m, f'{marker}-', color=color, label=method, linewidth=2, markersize=6)

ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect accuracy')
ax2.set_xlabel("N (trajectories)", fontsize=12)
ax2.set_ylabel("Fraction correct (n=2)", fontsize=12)
ax2.set_title("Tiger: Method Comparison", fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_ylim(-0.05, 1.15)
ax2.set_xscale("log")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_dir = ROOT / "experiments" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "quotient_size_comparison.pdf", dpi=150, bbox_inches="tight")
plt.savefig(out_dir / "quotient_size_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved to {out_dir}/quotient_size_comparison.{{pdf,png}}")
