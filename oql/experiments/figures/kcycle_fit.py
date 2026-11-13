"""Generate regression figure: N_recover(K) with linear fit and confidence band.

Output: experiments/figures/kcycle_fit.png
        experiments/figures/kcycle_fit.pdf
"""

from __future__ import annotations

import csv
import pathlib
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CSV_PATH = REPO_ROOT / "experiments" / "convergence" / "results" / "kcycle_recovery.csv"
OUT_DIR = REPO_ROOT / "experiments" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
with open(CSV_PATH, newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        k = int(row["k"])
        n = int(row["n_trajectories"])
        err = float(row["recovery_error"])
        data[k][n].append(err)

k_to_n_recover: dict[int, int | None] = {}
for k in sorted(data):
    for n in sorted(data[k]):
        mean_err = sum(data[k][n]) / len(data[k][n])
        if mean_err == 0.0:
            k_to_n_recover[k] = n
            break
    else:
        k_to_n_recover[k] = None

K_vals = np.array(sorted(k for k, n in k_to_n_recover.items() if n is not None))
N_vals = np.array([k_to_n_recover[k] for k in K_vals], dtype=float)

# ---------------------------------------------------------------------------
# Linear fit and bootstrap confidence band
# ---------------------------------------------------------------------------

slope, intercept, r_val, _, _ = scipy_stats.linregress(K_vals, N_vals)
r2 = r_val ** 2

# Bootstrap for confidence band
N_BOOT = 5000
rng = np.random.default_rng(42)
n_data = len(K_vals)
boot_slopes: list[float] = []
boot_intercepts: list[float] = []

for _ in range(N_BOOT):
    idx = rng.integers(0, n_data, size=n_data)
    Ks = K_vals[idx]
    Ns = N_vals[idx]
    if len(np.unique(Ks)) < 2:
        continue
    s, i, _, _, _ = scipy_stats.linregress(Ks, Ns)
    boot_slopes.append(s)
    boot_intercepts.append(i)

boot_slopes_arr = np.array(boot_slopes)
boot_intercepts_arr = np.array(boot_intercepts)

K_fine = np.linspace(1.5, 34, 200)

# Compute per-K-point CI from bootstrap lines
preds = np.outer(boot_slopes_arr, K_fine) + boot_intercepts_arr[:, None]
ci_lo = np.percentile(preds, 2.5, axis=0)
ci_hi = np.percentile(preds, 97.5, axis=0)
fit_line = slope * K_fine + intercept

# 95% CI for slope
slope_sorted = sorted(boot_slopes)
n_boots = len(slope_sorted)
slope_lo = slope_sorted[int(0.025 * n_boots)]
slope_hi = slope_sorted[int(0.975 * n_boots) - 1]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

# Confidence band
ax.fill_between(K_fine, ci_lo, ci_hi, alpha=0.2, color="#1f77b4", label="95% bootstrap CI")

# Fit line
ax.plot(K_fine, fit_line, color="#1f77b4", linewidth=2,
        label=f"$N = {slope:.0f}K + {intercept:.0f}$, $R^2 = {r2:.3f}$")

# Data points (step function — show as scatter)
ax.scatter(K_vals, N_vals, color="black", zorder=5, s=70, marker="o",
           label="$N_{\\mathrm{recover}}(K)$ (observed)")

# Reference line: N = 100K (loose prior claim)
ax.plot(K_fine, 100 * K_fine, color="gray", linewidth=1.2, linestyle="--",
        label="$N = 100K$ (prior bound)")

ax.set_xlabel("$K$ (number of quotient classes)", fontsize=13)
ax.set_ylabel("$N_{\\mathrm{recover}}$ (trajectories)", fontsize=13)
ax.set_title("$K$-cycle: empirical recovery threshold vs.\ $K$", fontsize=13)
ax.set_xlim(0, 35)
ax.set_ylim(0, 3500)
ax.set_xticks([2, 4, 5, 8, 16, 32])
ax.legend(fontsize=10, loc="upper left")

# Annotate slope CI
ax.text(0.60, 0.08,
        f"slope = {slope:.0f}  [95% CI: {slope_lo:.0f}, {slope_hi:.0f}]",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()

for ext in ("png", "pdf"):
    out = OUT_DIR / f"kcycle_fit.{ext}"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

plt.close(fig)
