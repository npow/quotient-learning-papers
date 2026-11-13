"""Generate Figure 2: R_crit vs. sigma_B sweep — two-panel figure.

Left panel  : Dec-Tiger (2-state binary environment)
Right panel : Continuous N-State Chain (4-state random-walk environment)

Both panels share the same axes format:
  X axis : sigma_B (noise std of agent B)
  Y axis : R_crit (bits)
  Blue line  : analytical R_crit = H(Q_A | Q_B) (closed-form)
  Red dots+errorbars : estimated R_crit (mean ± std across 5 seeds)

Reads:
  experiments/layer3_rcrit/results/rcrit_sweep.csv       (Dec-Tiger)
  experiments/layer3_rcrit/results/chain_rcrit.csv       (Chain)
Saves:
  experiments/figures/rcrit_sweep.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TIGER_CSV = ROOT / "experiments" / "layer3_rcrit" / "results" / "rcrit_sweep.csv"
CHAIN_CSV = ROOT / "experiments" / "layer3_rcrit" / "results" / "chain_rcrit.csv"
OUT_DIR = ROOT / "experiments" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Analytical formulas
# ---------------------------------------------------------------------------

def analytical_rcrit_tiger(sigma_a: float, sigma_b_arr: np.ndarray) -> np.ndarray:
    """R_crit for Dec-Tiger (2-state) at each sigma_b."""
    out = np.zeros_like(sigma_b_arr, dtype=float)
    for i, sigma_b in enumerate(sigma_b_arr):
        p_err_a = norm.cdf(-1.0 / sigma_a)
        p_err_b = norm.cdf(-1.0 / sigma_b)
        pLL = 0.5 * (1 - p_err_a) * (1 - p_err_b) + 0.5 * p_err_a * p_err_b
        pRR = pLL
        pLR = 0.5 * (1 - p_err_a) * p_err_b + 0.5 * p_err_a * (1 - p_err_b)
        pRL = pLR
        probs = [pLL, pRR, pLR, pRL]
        h_joint = -sum(p * np.log2(p) for p in probs if p > 0)
        pb0 = pLL + pRL
        pb1 = pRR + pLR
        h_qb = -(pb0 * np.log2(pb0) + pb1 * np.log2(pb1)) if (pb0 > 0 and pb1 > 0) else 0.0
        out[i] = h_joint - h_qb
    return out


def analytical_rcrit_chain(
    N: int,
    sigma_a: float,
    sigma_b_arr: np.ndarray,
) -> np.ndarray:
    """R_crit = H(Q_A | Q_B) for N-state chain at each sigma_b."""
    thresholds = [(2 * k + 1) / (2 * N) for k in range(N - 1)]
    bin_lo = [-np.inf] + thresholds
    bin_hi = thresholds + [np.inf]

    def p_class(sigma: float, s: int) -> np.ndarray:
        signal = s / N
        probs = np.array(
            [norm.cdf(hi, loc=signal, scale=sigma) - norm.cdf(lo, loc=signal, scale=sigma)
             for lo, hi in zip(bin_lo, bin_hi)],
            dtype=np.float64,
        )
        total = probs.sum()
        return probs / total if total > 0 else probs

    out = np.zeros_like(sigma_b_arr, dtype=float)
    for i, sigma_b in enumerate(sigma_b_arr):
        joint = np.zeros((N, N), dtype=np.float64)
        for s in range(N):
            pa = p_class(sigma_a, s)
            pb = p_class(float(sigma_b), s)
            joint += np.outer(pa, pb) / N
        h_joint = -float(np.sum(joint * np.log2(joint + 1e-300)))
        pb_marginal = joint.sum(axis=0)
        h_qb = -float(np.sum(pb_marginal * np.log2(pb_marginal + 1e-300)))
        out[i] = max(0.0, h_joint - h_qb)
    return out


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if not TIGER_CSV.exists():
    raise FileNotFoundError(
        f"Dec-Tiger results not found: {TIGER_CSV}\n"
        f"Run:  python experiments/layer3_rcrit/run_rcrit_sweep.py"
    )
if not CHAIN_CSV.exists():
    raise FileNotFoundError(
        f"Chain results not found: {CHAIN_CSV}\n"
        f"Run:  python experiments/layer3_rcrit/run_chain_rcrit.py"
    )

df_tiger = pd.read_csv(TIGER_CSV)
df_chain = pd.read_csv(CHAIN_CSV)


def _summarise(df: pd.DataFrame) -> tuple[list, list, list, list]:
    """Return (sigma_b_vals, means, stds, trues) lists."""
    sigma_b_vals = sorted(df["sigma_b"].unique())
    means, stds, trues = [], [], []
    for sb in sigma_b_vals:
        sub = df[df["sigma_b"] == sb]
        means.append(sub["rcrit_est"].mean())
        stds.append(sub["rcrit_est"].std(ddof=1) if len(sub) > 1 else 0.0)
        trues.append(sub["rcrit_true"].iloc[0])
    return sigma_b_vals, np.array(means), np.array(stds), np.array(trues)


sb_tiger, means_tiger, stds_tiger, trues_tiger = _summarise(df_tiger)
sb_chain, means_chain, stds_chain, trues_chain = _summarise(df_chain)

sigma_a_tiger = float(df_tiger["sigma_a"].iloc[0])
N_chain = 4  # from experiment definition

# Dense analytical curves for smooth plotting
dense_tiger = np.linspace(min(sb_tiger) * 0.8, max(sb_tiger) * 1.05, 200)
dense_chain = np.linspace(min(sb_chain) * 0.8, max(sb_chain) * 1.05, 200)

analytic_tiger = analytical_rcrit_tiger(sigma_a_tiger, dense_tiger)
analytic_chain = analytical_rcrit_chain(N_chain, sigma_a_tiger, dense_chain)

# ---------------------------------------------------------------------------
# Two-panel figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

panels = [
    (
        axes[0],
        dense_tiger, analytic_tiger,
        sb_tiger, means_tiger, stds_tiger,
        sigma_a_tiger,
        r"$R_{\mathrm{crit}}$ vs. Noise Asymmetry" "\n" r"Dec-Tiger (2-state)"
        + f"\n" + r"($\sigma_A=" + f"{sigma_a_tiger}" + r"$ fixed)",
    ),
    (
        axes[1],
        dense_chain, analytic_chain,
        sb_chain, means_chain, stds_chain,
        sigma_a_tiger,
        r"$R_{\mathrm{crit}}$ vs. Noise Asymmetry" "\n"
        r"4-State Chain ($N=4$)"
        + "\n" + r"($\sigma_A=" + f"{sigma_a_tiger}" + r"$ fixed)",
    ),
]

for ax, x_dense, y_dense, sb_vals, mns, sds, sigma_a, title in panels:
    # Analytical curve
    ax.plot(
        x_dense,
        y_dense,
        color="steelblue",
        linewidth=2.5,
        label=r"Analytical $R_{\mathrm{crit}}$",
        zorder=2,
    )
    # Estimated values with error bars
    ax.errorbar(
        sb_vals,
        mns,
        yerr=sds,
        fmt="o",
        color="crimson",
        ecolor="crimson",
        elinewidth=1.8,
        capsize=5,
        capthick=1.8,
        markersize=7,
        label=r"Estimated $\hat{R}_{\mathrm{crit}}$ (mean $\pm$ std, 5 seeds)",
        zorder=3,
    )
    ax.set_xlabel(r"$\sigma_B$ (observation noise of agent $B$)", fontsize=12)
    ax.set_ylabel(r"$R_{\mathrm{crit}}$ (bits)", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    # Annotate sigma_A
    ylim = ax.get_ylim()
    ax.axvline(x=sigma_a, color="steelblue", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.text(
        sigma_a + 0.02, ylim[1] * 0.03,
        rf"$\sigma_A={sigma_a}$",
        color="steelblue",
        fontsize=9,
        alpha=0.8,
    )

plt.tight_layout()

pdf_path = OUT_DIR / "rcrit_sweep.pdf"
png_path = OUT_DIR / "rcrit_sweep.png"
plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
plt.savefig(png_path, dpi=150, bbox_inches="tight")

print(f"Saved {pdf_path}")
print(f"Saved {png_path}")

# ---------------------------------------------------------------------------
# Print summary tables
# ---------------------------------------------------------------------------
for label, df_env, sigma_b_vals, trues, means, stds in [
    ("Dec-Tiger", df_tiger, sb_tiger, trues_tiger, means_tiger, stds_tiger),
    ("4-State Chain", df_chain, sb_chain, trues_chain, means_chain, stds_chain),
]:
    print(f"\n=== {label} ===")
    print(
        f"{'sigma_B':>8} | {'R_crit (true)':>14} | "
        f"{'R_crit (est mean±std)':>22} | {'Mean rel error':>14}"
    )
    print("-" * 70)
    for sb, t, m, s in zip(sigma_b_vals, trues, means, stds):
        sub = df_env[df_env["sigma_b"] == sb]
        mean_rel = sub["rel_error_pct"].mean()
        print(
            f"{sb:>8.2f} | {t:>14.4f} | "
            f"{m:>9.4f} ± {s:<9.4f} | {mean_rel:>10.1f}%"
        )
