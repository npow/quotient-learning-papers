"""Auto-bin validation on PO-HalfCheetah (Layer 2).

Validates that auto_select_bins() recovers B=8 automatically on PO-HalfCheetah
without any user-specified hyperparameter, collapsing the reviewer objection
"why B=8 and not B=4 or B=16?".

Usage:
    /tmp/mujoco-venv/bin/python experiments/layer2_mujoco/run_halfcheetah_autobin.py

No torch import — runs entirely on the MuJoCo venv.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from continuous_quotient import (
    ContinuousEstimatorConfig,
    ContinuousQuotientEstimator,
    auto_select_bins,
    collect_continuous_trajectories,
)
from continuous_quotient.types import ContinuousTrajectoryDataset


# ---------------------------------------------------------------------------
# PO-HalfCheetah wrapper (mirrors run_halfcheetah.py)
# ---------------------------------------------------------------------------

def make_po_halfcheetah():
    """Wrap HalfCheetah-v4 to zero out position observations (PO variant)."""
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
# Main
# ---------------------------------------------------------------------------

def main():
    N_TRAJECTORIES = 500
    MAX_STEPS = 50
    N_SEEDS = 3

    out_dir = ROOT / "experiments" / "layer2_mujoco" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "halfcheetah_autobin.csv"

    # --- Step 1: collect dataset for auto_select_bins ---
    print("Collecting N=500 trajectories from PO-HalfCheetah for auto_select_bins ...")
    env = make_po_halfcheetah()
    zero_action = np.zeros(env.action_space.shape)
    policy = lambda obs, rng, _z=zero_action: _z

    dataset = collect_continuous_trajectories(
        env, n_episodes=N_TRAJECTORIES, max_steps=MAX_STEPS, policy=policy, seed=0
    )
    env.close()

    # --- Step 2: auto_select_bins ---
    print("Running auto_select_bins(b_min=2, b_max=32) ...")
    t_bin = time.time()
    auto_B = auto_select_bins(dataset, b_min=2, b_max=32)
    t_bin = time.time() - t_bin
    print(f"auto_select_bins on PO-HalfCheetah: B={auto_B}  ({t_bin:.1f}s)")

    # --- Step 3: run ContinuousQuotientEstimator with history_n_bins=0 (auto sentinel) ---
    rows = []
    n_classes_list = []

    for seed in range(N_SEEDS):
        print(f"\nSeed {seed}: fitting ContinuousQuotientEstimator (history_n_bins=0=auto) ...")
        env = make_po_halfcheetah()
        cfg = ContinuousEstimatorConfig(
            n_trajectories=N_TRAJECTORIES,
            max_steps_per_episode=MAX_STEPS,
            window_length=2,
            history_max_length=1,
            min_history_occurrences=20,
            alpha=0.05,
            n_permutations=99,
            history_n_bins=0,          # sentinel → auto_select_bins() at fit time
            max_samples_per_test=300,
        )
        est = ContinuousQuotientEstimator(cfg)
        t0 = time.time()
        q = est.fit_from_env(env, policy=policy, seed=seed, verbose=True)
        rt = round(time.time() - t0, 1)
        env.close()

        n_classes_list.append(q.n_classes)
        rows.append({
            "seed": seed,
            "auto_B": auto_B,
            "n_classes": q.n_classes,
            "runtime_s": rt,
        })
        print(f"  seed={seed}  auto_B={auto_B}  n_classes={q.n_classes}  runtime={rt}s")

    # --- Step 4: save CSV ---
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "auto_B", "n_classes", "runtime_s"])
        writer.writeheader()
        writer.writerows(rows)

    # --- Final report ---
    nc_str = ", ".join(
        f"n_classes={rows[i]['n_classes']} (seed {i})" for i in range(N_SEEDS)
    )
    print(f"\nauto_select_bins on PO-HalfCheetah: B={auto_B}")
    print(f"ContinuousQuotientEstimator (auto B): {nc_str}")
    print(f"\nResults saved to {out_csv}")

    if auto_B == 8:
        print("\n[PASS] B=8 selected automatically — confirms B=8 is data-driven, not arbitrary.")
    else:
        print(f"\n[NOTE] auto_select_bins chose B={auto_B} (not 8); "
              f"review silhouette scores or consider widening b_max.")


if __name__ == "__main__":
    main()
