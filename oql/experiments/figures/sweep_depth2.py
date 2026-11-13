"""Generate the merge-bearing abstention sweep figure (S10).

Reads experiments/main_table/results_sweep_depth2/sweep_depth2.csv (produced
by experiments/main_table/run_sweep_depth2.py) and renders the two-panel
paper figure: (a) abstaining pair fraction vs budget, (b) reported class
count vs budget, showing that starving the m* gate over-splits the
merge-bearing quotient (3 reported classes instead of the analytic 2) and
never merges incorrectly.

Output: paper/figures/sweep_depth2.pdf
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

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CSV_PATH = (
    REPO_ROOT / "experiments" / "main_table" / "results_sweep_depth2"
    / "sweep_depth2.csv"
)
OUT_PATH = REPO_ROOT / "paper" / "figures" / "sweep_depth2.pdf"

EXPECTED_CLASSES = 2   # analytic r=0 quotient of the noisy depth-2 chain
OVERSPLIT_CLASSES = 3  # SAME pair abstained: all-singleton report

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

und: dict[int, list[float]] = defaultdict(list)
n_classes: dict[int, list[int]] = defaultdict(list)
incorrect = 0
with open(CSV_PATH, newline="") as fh:
    for row in csv.DictReader(fh):
        n = int(row["n_trajectories"])
        und[n].append(float(row["undetermined_rate"]))
        n_classes[n].append(int(row["n_classes"]))
        incorrect += int(row["incorrect_merges"])

assert incorrect == 0, f"{incorrect} incorrect merges in the sweep artifacts"
budgets = sorted(und)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9))

# (a) abstaining pair fraction, mean with min--max band over seeds.
u_mean = [float(np.mean(und[n])) for n in budgets]
u_min = [float(np.min(und[n])) for n in budgets]
u_max = [float(np.max(und[n])) for n in budgets]
ax_a.plot(budgets, u_mean, marker="o", lw=1.5, color="tab:blue",
          label="mean over seeds")
ax_a.fill_between(budgets, u_min, u_max, alpha=0.25, color="tab:blue",
                  lw=0, label="min--max over seeds")
ax_a.set_xlabel("episodes $N$")
ax_a.set_ylabel("abstaining pair fraction")
ax_a.set_ylim(-0.02, 1.02)
ax_a.legend(frameon=False, fontsize=8)
ax_a.set_title("(a) $m^*$ gate binding", fontsize=9)

# (b) reported class count, mean with min--max band over seeds.
c_mean = [float(np.mean(n_classes[n])) for n in budgets]
c_min = [float(np.min(n_classes[n])) for n in budgets]
c_max = [float(np.max(n_classes[n])) for n in budgets]
ax_b.axhline(OVERSPLIT_CLASSES, color="gray", lw=1.0, ls=":",
             label="fully split (3)")
ax_b.axhline(EXPECTED_CLASSES, color="gray", lw=1.0, ls="--",
             label="analytic quotient (2)")
ax_b.plot(budgets, c_mean, marker="s", lw=1.5, color="tab:red",
          label="reported classes (mean)")
ax_b.fill_between(budgets, c_min, c_max, alpha=0.25, color="tab:red", lw=0)
ax_b.set_xlabel("episodes $N$")
ax_b.set_ylabel("reported classes")
ax_b.set_ylim(1.7, 3.3)
ax_b.set_yticks([2, 3])
ax_b.legend(frameon=False, fontsize=8, loc="center right")
ax_b.set_title("(b) over-split, never merge", fontsize=9)

fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH)
plt.close(fig)
print(f"Saved {OUT_PATH}")
