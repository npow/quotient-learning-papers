"""PO-HalfCheetah sinusoidal-policy experiment (Layer 2).

Evaluates ContinuousQuotientEstimator under a structured sinusoidal locomotion
policy rather than the zero-action (passive) baseline.  This addresses the
reviewer question: "does the algorithm find non-trivial structure under a
task-relevant policy that actually makes the cheetah move?"

Usage:
    python experiments/layer2_mujoco/run_halfcheetah_sinusoidal.py
    python experiments/layer2_mujoco/run_halfcheetah_sinusoidal.py --seeds 5 --verbose
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from continuous_quotient import (
    ContinuousEstimatorConfig,
    ContinuousQuotientEstimator,
)


def make_po_halfcheetah():
    """Wrap HalfCheetah-v4 to zero out position observations (PO variant)."""
    try:
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
    except Exception as e:
        print(f"ERROR: Could not create HalfCheetah-v4: {e}")
        print("Install: pip install 'gymnasium[mujoco]'")
        sys.exit(1)


def sinusoidal_policy(obs, rng, freq=2.0, amp=0.5):
    """Sinusoidal torque pattern that creates periodic locomotion.

    Frequency and phase are fixed; amplitude is small to avoid extreme states.
    This creates structured velocity histories unlike both zero-action and random.
    HalfCheetah action space: 6 joints.
    """
    t = getattr(sinusoidal_policy, "_t", 0)
    sinusoidal_policy._t = t + 0.01
    action = amp * np.sin(2 * np.pi * freq * np.arange(6) / 6 + t)
    return action.astype(np.float32)


def run_sinusoidal(args):
    out = ROOT / "experiments" / "layer2_mujoco" / "results" / "halfcheetah_sinusoidal.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(args.seeds):
        # Reset the sinusoidal phase counter so each seed starts from t=0
        sinusoidal_policy._t = 0

        env = make_po_halfcheetah()
        cfg = ContinuousEstimatorConfig(
            n_trajectories=args.n_trajectories,
            max_steps_per_episode=args.max_steps,
            window_length=2,
            history_max_length=1,
            min_history_occurrences=20,
            alpha=0.05,
            n_permutations=99,
            history_n_bins=8,
            max_samples_per_test=300,
        )
        est = ContinuousQuotientEstimator(cfg)
        t0 = time.time()
        q = est.fit_from_env(env, policy=sinusoidal_policy, seed=seed, verbose=args.verbose)
        rt = time.time() - t0
        rows.append({
            "seed": seed,
            "policy": "sinusoidal",
            "n_classes": q.n_classes,
            "runtime_s": round(rt, 1),
        })
        print(f"  seed={seed} policy=sinusoidal n_classes={q.n_classes} runtime={rt:.1f}s")

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "policy", "n_classes", "runtime_s"])
        writer.writeheader()
        writer.writerows(rows)

    ns = [r["n_classes"] for r in rows]
    std = float(np.std(ns))
    mean = float(np.mean(ns))
    print(f"\nPO-HalfCheetah sinusoidal: n_classes = {mean:.1f} ± {std:.1f}")
    print(f"  std ≤ 2: {'PASS' if std <= 2 else 'FAIL'}")
    print(f"  n_classes ≥ 2: {'PASS' if min(ns) >= 2 else 'FAIL'}")
    print(f"Results saved to {out}")

    return rows


def compare_results(sinusoidal_rows):
    """Print side-by-side comparison with existing zero-action results."""
    zero_csv = ROOT / "experiments" / "layer2_mujoco" / "results" / "halfcheetah_continuous.csv"
    if not zero_csv.exists():
        print("\n(zero-action results not found; skipping comparison)")
        return

    zero_rows = []
    with open(zero_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            zero_rows.append(row)

    zero_ns = [int(r["n_classes"]) for r in zero_rows]
    sin_ns = [r["n_classes"] for r in sinusoidal_rows]

    print("\n" + "=" * 55)
    print(f"{'Policy':<22} {'seeds':>5}  {'mean n_classes':>14}  {'std':>5}")
    print("-" * 55)
    print(f"{'zero-action':<22} {len(zero_ns):>5}  {np.mean(zero_ns):>14.1f}  {np.std(zero_ns):>5.2f}")
    print(f"{'sinusoidal':<22} {len(sin_ns):>5}  {np.mean(sin_ns):>14.1f}  {np.std(sin_ns):>5.2f}")
    print("=" * 55)


def main():
    p = argparse.ArgumentParser(description="PO-HalfCheetah sinusoidal-policy experiment")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-trajectories", type=int, default=3000)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"Running sinusoidal-policy experiment: {args.seeds} seeds, "
          f"{args.n_trajectories} trajectories, max_steps={args.max_steps}")
    rows = run_sinusoidal(args)
    compare_results(rows)


if __name__ == "__main__":
    main()
