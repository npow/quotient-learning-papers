"""Compute per-class behavioral statistics for PO-HalfCheetah quotient.

Refutes the "n=8 = hyperparameter" objection by showing that the 8
quotient classes discovered by Algorithm 1 correspond to statistically
distinct behavioral regimes (different expected cumulative return).

Usage:
    /tmp/mujoco-venv/bin/python experiments/layer2_mujoco/halfcheetah_class_returns.py
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

def collect_trajectories(n_episodes: int = 500, max_steps: int = 50, seed: int = 0):
    """Collect episodes with zero-action policy (same as main experiment)."""
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
# Fit quotient and assign class ids
# ---------------------------------------------------------------------------

def fit_quotient(dataset, history_n_bins: int = 8):
    from continuous_quotient import ContinuousEstimatorConfig, ContinuousQuotientEstimator

    cfg = ContinuousEstimatorConfig(
        n_trajectories=dataset.n_trajectories,
        max_steps_per_episode=50,
        window_length=2,
        history_max_length=1,
        min_history_occurrences=20,
        alpha=0.05,
        n_permutations=99,
        history_n_bins=history_n_bins,
        max_samples_per_test=300,
    )
    est = ContinuousQuotientEstimator(cfg)
    print(f"Fitting quotient estimator (history_n_bins={history_n_bins})...")
    quotient = est.fit(dataset, verbose=True)
    print(f"  Discovered n_classes={quotient.n_classes}")
    return quotient, cfg


# ---------------------------------------------------------------------------
# Map each trajectory step to (class_id, velocity_l2, future_return)
# ---------------------------------------------------------------------------

def compute_step_records(dataset, quotient, cfg, gamma: float = 0.99):
    """For every timestep in the dataset, compute:
      - class_id: quotient class assigned to the current history
      - vel_l2: L2 norm of the current (masked) observation
      - disc_return: discounted cumulative reward from this step onward

    Returns list of dicts.
    """
    from continuous_quotient.estimator import _fit_obs_bins, _bin_obs, _make_coarse_history_key

    history_n_bins = cfg.history_n_bins
    boundaries = _fit_obs_bins(dataset, history_n_bins)

    records = []

    for traj in dataset.trajectories:
        T = traj.length
        obs = traj.observations
        acts = traj.actions
        rews = traj.rewards  # length T-1

        # Precompute discounted return from each step t onward.
        # disc_return[t] = sum_{i=t}^{T-2} gamma^(i-t) * rews[i]
        disc_returns = np.zeros(T, dtype=np.float64)
        running = 0.0
        for t in range(len(rews) - 1, -1, -1):
            running = rews[t] + gamma * running
            disc_returns[t] = running

        for t in range(T):
            # Build length-1 history (current obs only, no prior action).
            h_obs = [obs[t]]
            h_acts = []

            hkey = _make_coarse_history_key(h_obs, h_acts, boundaries)
            class_id = quotient.partition.get(hkey, None)

            if class_id is None:
                continue  # history not seen in training partition

            vel_l2 = float(np.linalg.norm(np.asarray(obs[t], dtype=np.float32)))
            dr = float(disc_returns[t])

            records.append({
                "class_id": class_id,
                "vel_l2": vel_l2,
                "disc_return": dr,
            })

    print(f"  Total step records with known class: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Aggregate per class
# ---------------------------------------------------------------------------

def aggregate_per_class(records, n_classes: int):
    from collections import defaultdict

    vel_by_class: dict[int, list] = defaultdict(list)
    ret_by_class: dict[int, list] = defaultdict(list)

    for r in records:
        cid = r["class_id"]
        vel_by_class[cid].append(r["vel_l2"])
        ret_by_class[cid].append(r["disc_return"])

    rows = []
    for cid in range(n_classes):
        vels = np.array(vel_by_class.get(cid, []))
        rets = np.array(ret_by_class.get(cid, []))
        rows.append({
            "class_id": cid,
            "mean_velocity_l2": float(np.mean(vels)) if len(vels) > 0 else float("nan"),
            "std_velocity_l2": float(np.std(vels)) if len(vels) > 0 else float("nan"),
            "mean_return": float(np.mean(rets)) if len(rets) > 0 else float("nan"),
            "std_return": float(np.std(rets)) if len(rets) > 0 else float("nan"),
            "n_samples": len(rets),
        })
    # Sort by mean velocity for interpretability.
    rows.sort(key=lambda r: r["mean_velocity_l2"] if not np.isnan(r["mean_velocity_l2"]) else -1)
    # Re-assign display class ids sorted by velocity.
    for i, row in enumerate(rows):
        row["sorted_rank"] = i
    return rows


# ---------------------------------------------------------------------------
# Statistical test: Kruskal-Wallis
# ---------------------------------------------------------------------------

def kruskal_wallis_test(records, n_classes: int):
    from collections import defaultdict
    from scipy import stats

    groups = defaultdict(list)
    for r in records:
        groups[r["class_id"]].append(r["disc_return"])

    group_arrays = [np.array(groups[cid]) for cid in range(n_classes)
                    if len(groups.get(cid, [])) >= 5]

    if len(group_arrays) < 2:
        print("  WARNING: fewer than 2 classes with >=5 samples; skipping KW test.")
        return None, None

    stat, pval = stats.kruskal(*group_arrays)
    return float(stat), float(pval)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def make_latex_table(rows):
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{crrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Rank & Class & $\bar{v}_{\ell_2}$ & $\sigma_{v}$ & $\bar{G}$ & $\sigma_G$ \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(
            f"  {row['sorted_rank']} & {row['class_id']} & "
            f"{row['mean_velocity_l2']:.3f} & {row['std_velocity_l2']:.3f} & "
            f"{row['mean_return']:.2f} & {row['std_return']:.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Per-class statistics for the $n{=}8$ quotient partition on")
    lines.append(r"PO-HalfCheetah. Classes are sorted by mean velocity $\ell_2$ norm.")
    lines.append(r"$\bar{G}$ = mean discounted return ($\gamma{=}0.99$) from each timestep")
    lines.append(r"assigned to that class. Kruskal-Wallis test across all 8 classes:}")
    lines.append(r"\label{tab:halfcheetah_class_returns}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = ROOT / "experiments" / "layer2_mujoco" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: collect trajectories (N=500 to stay within memory budget).
    dataset = collect_trajectories(n_episodes=500, max_steps=50, seed=42)

    # Step 2: fit quotient with history_n_bins=8 (same as paper experiment).
    quotient, cfg = fit_quotient(dataset, history_n_bins=8)

    if quotient.n_classes == 0:
        print("ERROR: quotient has 0 classes — something went wrong.")
        sys.exit(1)

    # Step 3: assign class ids and compute per-step discounted returns.
    records = compute_step_records(dataset, quotient, cfg, gamma=0.99)

    if not records:
        print("ERROR: no step records could be labelled — partition is empty.")
        sys.exit(1)

    # Step 4: aggregate per class.
    rows = aggregate_per_class(records, quotient.n_classes)

    # Step 5: Kruskal-Wallis test.
    kw_stat, kw_pval = kruskal_wallis_test(records, quotient.n_classes)

    # Step 6: print summary.
    print("\n" + "=" * 60)
    print("PO-HalfCheetah: per-class behavioral statistics")
    print("=" * 60)
    print(f"{'Rank':>4}  {'ClassID':>7}  {'mean_vel_L2':>11}  {'mean_return':>11}  {'n_samples':>9}")
    print("-" * 60)
    for row in rows:
        print(f"{row['sorted_rank']:>4}  {row['class_id']:>7}  "
              f"{row['mean_velocity_l2']:>11.3f}  {row['mean_return']:>11.2f}  "
              f"{row['n_samples']:>9}")
    print("=" * 60)

    ret_range_lo = min(r["mean_return"] for r in rows)
    ret_range_hi = max(r["mean_return"] for r in rows)
    print(f"\nMean return range: [{ret_range_lo:.2f}, {ret_range_hi:.2f}]")

    if kw_stat is not None:
        print(f"Kruskal-Wallis H={kw_stat:.2f}, p={kw_pval:.2e}")
        sig = "SIGNIFICANT (p < 0.001)" if kw_pval < 0.001 else (
              "SIGNIFICANT (p < 0.05)" if kw_pval < 0.05 else "NOT significant")
        print(f"Result: {sig}")
        print()
        print("Paper sentence:")
        print(f'  "The {quotient.n_classes} quotient classes discovered by Algorithm 1 correspond '
              f'to statistically distinct behavioral regimes '
              f'(Kruskal-Wallis p = {kw_pval:.2e}), with mean discounted return '
              f'($\\gamma=0.99$) varying from {ret_range_lo:.2f} to {ret_range_hi:.2f} '
              f'across classes (Table~\\ref{{tab:halfcheetah_class_returns}} in supplement)."')

    # Step 7: save CSV.
    csv_path = out_dir / "halfcheetah_class_returns.csv"
    fieldnames = ["class_id", "mean_velocity_l2", "std_velocity_l2",
                  "mean_return", "std_return", "n_samples"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"\nCSV saved to {csv_path}")

    # Step 8: save LaTeX snippet.
    if kw_stat is not None:
        latex = make_latex_table(rows)
        # Patch KW result into caption.
        latex = latex.replace(
            r"Kruskal-Wallis test across all 8 classes:}",
            f"Kruskal-Wallis $H={kw_stat:.1f}$, $p={kw_pval:.2e}$."
            r"\vspace{-4pt}}"
        )
        latex_path = out_dir / "halfcheetah_class_returns.tex"
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to {latex_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
