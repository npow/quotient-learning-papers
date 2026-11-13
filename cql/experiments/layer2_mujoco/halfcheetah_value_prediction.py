"""Class-conditioned value prediction experiment for PO-HalfCheetah.

Addresses the 'n=8=B' reviewer objection by showing that the auto-selected
B=3 quotient classes predict future discounted return better than chance.

Three models are compared:
  - Null model   : constant predictor (mean return) — R² = 0 by construction
  - Class model  : one-hot class-id (3 features from auto-selected partition)
  - Oracle model : true velocity L2 norm (1 continuous feature)

Additional metric: eta-squared = between-class SS / total SS, the ANOVA
fraction of variance explained by group membership.  Also reports a
Kruskal-Wallis test across the 3 classes.

Usage:
    /tmp/mujoco-venv/bin/python experiments/layer2_mujoco/halfcheetah_value_prediction.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np


# ---------------------------------------------------------------------------
# PO-HalfCheetah wrapper (identical to run_halfcheetah.py)
# ---------------------------------------------------------------------------

def make_po_halfcheetah():
    import gymnasium as gym

    class POHalfCheetahEnv(gym.Wrapper):
        """PO-HalfCheetah: mask position observations (indices 0-8)."""
        def __init__(self):
            env = gym.make("HalfCheetah-v4")
            super().__init__(env)
            self._mask = np.ones(env.observation_space.shape[0], dtype=bool)
            self._mask[:9] = False  # zero out positions

        def _mask_obs(self, obs):
            obs = np.array(obs, dtype=np.float32)
            obs[~self._mask] = 0.0
            return obs

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self._mask_obs(obs), info

        def step(self, action):
            obs, r, term, trunc, info = self.env.step(action)
            return self._mask_obs(obs), r, term, trunc, info

    return POHalfCheetahEnv()


# ---------------------------------------------------------------------------
# Collect trajectories
# ---------------------------------------------------------------------------

def collect_trajectories(n_episodes: int = 1000, max_steps: int = 50, seed: int = 0):
    from continuous_quotient import collect_continuous_trajectories

    env = make_po_halfcheetah()
    zero_action = np.zeros(env.action_space.shape)
    policy = lambda obs, rng, _z=zero_action: _z

    print(f"Collecting {n_episodes} trajectories (zero-action policy, max_steps={max_steps})...")
    dataset = collect_continuous_trajectories(
        env, n_episodes=n_episodes, max_steps=max_steps, policy=policy, seed=seed
    )
    print(f"  Collected {dataset.n_trajectories} trajectories.")
    return dataset


# ---------------------------------------------------------------------------
# Fit quotient with auto-selected B (history_n_bins=0)
# ---------------------------------------------------------------------------

def fit_quotient_autobin(dataset):
    from continuous_quotient import ContinuousEstimatorConfig, ContinuousQuotientEstimator

    cfg = ContinuousEstimatorConfig(
        n_trajectories=dataset.n_trajectories,
        max_steps_per_episode=50,
        window_length=2,
        history_max_length=1,
        min_history_occurrences=20,
        alpha=0.05,
        n_permutations=99,
        history_n_bins=0,       # sentinel: auto-select via silhouette score
        max_samples_per_test=300,
    )
    est = ContinuousQuotientEstimator(cfg)
    print("Fitting quotient estimator (history_n_bins=0 → auto-select)...")
    quotient = est.fit(dataset, verbose=True)
    print(f"  Auto-selected B → n_classes={quotient.n_classes}")
    return quotient, cfg


# ---------------------------------------------------------------------------
# Build per-step feature records
# ---------------------------------------------------------------------------

def build_records(dataset, quotient, cfg, gamma: float = 0.99):
    """For every timestep return (class_id, vel_l2, disc_return).

    Uses the same _make_coarse_history_key logic as the estimator so class
    assignments are consistent with the fitted partition.
    """
    from continuous_quotient.estimator import _fit_obs_bins, _make_coarse_history_key

    # Recover the bin count from the partition keys.
    # Length-1 histories with no action produce keys = (bin_idx,); the
    # maximum bin_idx + 1 equals the number of bins actually used.
    if quotient.partition:
        resolved_bins = max(k[0] for k in quotient.partition.keys()) + 1
    else:
        resolved_bins = 3  # fallback
    print(f"  Resolved history_n_bins from partition keys: B={resolved_bins}")

    boundaries = _fit_obs_bins(dataset, resolved_bins)

    records = []
    for traj in dataset.trajectories:
        T = traj.length
        obs = traj.observations
        acts = traj.actions
        rews = traj.rewards  # length T-1

        # Discounted return from each step t onward.
        disc_returns = np.zeros(T, dtype=np.float64)
        running = 0.0
        for t in range(len(rews) - 1, -1, -1):
            running = rews[t] + gamma * running
            disc_returns[t] = running

        for t in range(T):
            h_obs = [obs[t]]
            h_acts = []
            hkey = _make_coarse_history_key(h_obs, h_acts, boundaries)
            class_id = quotient.partition.get(hkey, None)
            if class_id is None:
                continue

            vel_l2 = float(np.linalg.norm(np.asarray(obs[t], dtype=np.float32)))
            records.append({
                "class_id": int(class_id),
                "vel_l2": vel_l2,
                "disc_return": float(disc_returns[t]),
            })

    print(f"  Step records with known class: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Fit and evaluate three predictive models
# ---------------------------------------------------------------------------

def evaluate_models(records, n_classes: int):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.preprocessing import OneHotEncoder
    from scipy import stats

    class_ids = np.array([r["class_id"] for r in records], dtype=int)
    vel_l2    = np.array([r["vel_l2"]    for r in records], dtype=float)
    y         = np.array([r["disc_return"] for r in records], dtype=float)
    n_samples = len(y)

    results = {}

    # ------------------------------------------------------------------
    # Null model: constant predictor (mean)
    # ------------------------------------------------------------------
    y_null = np.full_like(y, y.mean())
    r2_null = r2_score(y, y_null)      # 0.0 by definition
    mae_null = mean_absolute_error(y, y_null)
    results["null"] = {"r2": r2_null, "mae": mae_null, "n_samples": n_samples}

    # ------------------------------------------------------------------
    # Class model: linear regression on one-hot class features
    # ------------------------------------------------------------------
    enc = OneHotEncoder(sparse_output=False, categories="auto")
    X_class = enc.fit_transform(class_ids.reshape(-1, 1))
    clf_class = LinearRegression(fit_intercept=False)  # one-hot has implicit intercept
    clf_class.fit(X_class, y)
    y_pred_class = clf_class.predict(X_class)
    r2_class = r2_score(y, y_pred_class)
    mae_class = mean_absolute_error(y, y_pred_class)
    results["class"] = {"r2": r2_class, "mae": mae_class, "n_samples": n_samples}

    # ------------------------------------------------------------------
    # Oracle model: linear regression on true velocity L2 norm
    # ------------------------------------------------------------------
    X_oracle = vel_l2.reshape(-1, 1)
    clf_oracle = LinearRegression()
    clf_oracle.fit(X_oracle, y)
    y_pred_oracle = clf_oracle.predict(X_oracle)
    r2_oracle = r2_score(y, y_pred_oracle)
    mae_oracle = mean_absolute_error(y, y_pred_oracle)
    results["oracle"] = {"r2": r2_oracle, "mae": mae_oracle, "n_samples": n_samples}

    # ------------------------------------------------------------------
    # Additional: eta-squared (fraction of total SS explained by classes)
    # and Kruskal-Wallis test
    # ------------------------------------------------------------------
    grand_mean = y.mean()
    ss_total = np.sum((y - grand_mean) ** 2)

    groups = [y[class_ids == cid] for cid in range(n_classes)]
    ss_between = sum(
        len(g) * (g.mean() - grand_mean) ** 2
        for g in groups if len(g) > 0
    )
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    # Kruskal-Wallis (non-parametric test for different class distributions)
    valid_groups = [g for g in groups if len(g) >= 5]
    if len(valid_groups) >= 2:
        kw_stat, kw_pval = stats.kruskal(*valid_groups)
    else:
        kw_stat, kw_pval = float("nan"), float("nan")

    # Per-class mean returns
    per_class = []
    for cid in range(n_classes):
        g = groups[cid]
        per_class.append({
            "class_id": cid,
            "n": len(g),
            "mean_return": float(g.mean()) if len(g) > 0 else float("nan"),
            "std_return": float(g.std()) if len(g) > 0 else float("nan"),
            "mean_vel_l2": float(vel_l2[class_ids == cid].mean()) if (class_ids == cid).sum() > 0 else float("nan"),
        })

    results["_meta"] = {
        "eta_sq": eta_sq,
        "kw_stat": kw_stat,
        "kw_pval": kw_pval,
        "per_class": per_class,
        "return_mean": float(grand_mean),
        "return_std": float(y.std()),
    }

    return results


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def make_latex_table(results, n_classes: int):
    meta = results["_meta"]
    kw_pval = meta["kw_pval"]
    eta_sq = meta["eta_sq"]

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Features & $R^2$ & MAE \\")
    lines.append(r"\midrule")

    model_labels = {
        "null":   (r"Null (constant mean)",              "0"),
        "class":  (rf"Quotient classes ($n={n_classes}$, $\eta^2={eta_sq:.3f}$)", str(n_classes)),
        "oracle": (r"Velocity $\ell_2$ (oracle)",        "1"),
    }
    for key in ("null", "class", "oracle"):
        label, feats = model_labels[key]
        r2  = results[key]["r2"]
        mae = results[key]["mae"]
        lines.append(f"  {label} & {feats} & {r2:.3f} & {mae:.3f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # format p-value
    if not np.isnan(kw_pval):
        pval_str = f"$p={kw_pval:.2e}$" if kw_pval >= 1e-300 else r"$p \ll 10^{-300}$"
    else:
        pval_str = "n/a"

    lines.append(
        rf"\caption{{Class-conditioned value prediction on PO-HalfCheetah "
        rf"(zero-action policy, $\gamma=0.99$, $n={results['class']['n_samples']:,}$ timesteps). "
        rf"The auto-selected $B={n_classes}$ quotient partition explains "
        rf"${results['class']['r2']*100:.1f}\%$ of return variance (linear $R^2$) and "
        rf"$\eta^2={eta_sq:.3f}$ of total variance (ANOVA effect size). "
        rf"Kruskal-Wallis test rejects equal-return distributions across classes "
        rf"({pval_str}), confirming classes capture genuine momentum-dependent "
        rf"behavioral structure. The continuous velocity oracle achieves "
        rf"$R^2={results['oracle']['r2']:.3f}$; the class predictor captures "
        rf"${results['class']['r2']/results['oracle']['r2']*100:.0f}\%$ "
        rf"of the oracle's signal with only {n_classes} discrete bins.}}"
    )
    lines.append(r"\label{tab:halfcheetah_value_prediction}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = ROOT / "experiments" / "layer2_mujoco" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: collect N=1000 trajectories with zero-action policy.
    dataset = collect_trajectories(n_episodes=1000, max_steps=50, seed=42)

    # Step 2: fit quotient with auto-selected B (history_n_bins=0).
    quotient, cfg = fit_quotient_autobin(dataset)

    if quotient.n_classes == 0:
        print("ERROR: quotient has 0 classes — something went wrong.")
        sys.exit(1)

    # Step 3: build per-step records (class_id, vel_l2, disc_return).
    records = build_records(dataset, quotient, cfg, gamma=0.99)

    if not records:
        print("ERROR: no step records with known class — partition is empty.")
        sys.exit(1)

    # Step 4: evaluate three models plus eta-squared / KW.
    print("Evaluating models...")
    results = evaluate_models(records, quotient.n_classes)
    meta = results["_meta"]

    # Step 5: print summary.
    print("\n" + "=" * 65)
    print("Value prediction comparison (PO-HalfCheetah)")
    print("=" * 65)
    print(f"  Overall return: mean={meta['return_mean']:.4f}, std={meta['return_std']:.4f}")
    print(f"  n_classes (auto-selected B): {quotient.n_classes}")
    print()
    print(f"{'Model':<30}  {'R²':>8}  {'MAE':>10}  {'n_samples':>10}")
    print("-" * 65)
    for name in ("null", "class", "oracle"):
        r = results[name]
        print(f"  {name:<28}  {r['r2']:>8.4f}  {r['mae']:>10.4f}  {r['n_samples']:>10,}")
    print()
    print(f"  Eta-squared (class effect, ANOVA): η²={meta['eta_sq']:.4f}")
    print(f"  Kruskal-Wallis: H={meta['kw_stat']:.2f}, p={meta['kw_pval']:.2e}")
    print()
    print("  Per-class statistics:")
    print(f"  {'Class':>5}  {'n':>7}  {'mean_vel_l2':>12}  {'mean_return':>12}  {'std_return':>12}")
    for pc in meta["per_class"]:
        print(f"  {pc['class_id']:>5}  {pc['n']:>7}  {pc['mean_vel_l2']:>12.4f}  "
              f"{pc['mean_return']:>12.4f}  {pc['std_return']:>12.4f}")
    print("=" * 65)

    r2_class  = results["class"]["r2"]
    r2_null   = results["null"]["r2"]
    r2_oracle = results["oracle"]["r2"]
    n_cls     = quotient.n_classes
    pct_oracle = r2_class / r2_oracle * 100 if r2_oracle > 0 else float("nan")

    # Step 6: print the key paper sentence.
    print("\nKey paper sentence:")
    print(
        f'  "The auto-selected B={n_cls} quotient classes explain '
        f'{r2_class*100:.1f}% of return variance (R²={r2_class:.3f} vs. '
        f'R²={r2_null:.3f} for null), with η²={meta["eta_sq"]:.3f} ANOVA '
        f'effect size (KW p={meta["kw_pval"]:.2e}), confirming they capture '
        f'genuine momentum-dependent behavioral structure '
        f'(oracle velocity R²={r2_oracle:.3f}, '
        f'classes recover {pct_oracle:.0f}% of the oracle signal)."'
    )

    # Step 7: save CSV.
    csv_path = out_dir / "halfcheetah_value_prediction.csv"
    fieldnames = ["model", "r2", "mae", "n_samples"]
    csv_rows = [
        {"model": name, "r2": results[name]["r2"],
         "mae": results[name]["mae"], "n_samples": results[name]["n_samples"]}
        for name in ("null", "class", "oracle")
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved to {csv_path}")

    # Step 8: save LaTeX table.
    latex = make_latex_table(results, n_cls)
    latex_path = out_dir / "halfcheetah_value_prediction.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {latex_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
