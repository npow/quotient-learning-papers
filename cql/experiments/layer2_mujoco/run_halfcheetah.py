"""PO-HalfCheetah experiment (Layer 2): continuous vs. discretized baseline.

Runs ContinuousQuotientEstimator on Partially-Observable HalfCheetah
(position observations zeroed out, velocity observations only).

Usage:
    python experiments/layer2_mujoco/run_halfcheetah.py --mode continuous
    python experiments/layer2_mujoco/run_halfcheetah.py --mode discretized
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from continuous_quotient import (
    ContinuousEstimatorConfig,
    ContinuousQuotientEstimator,
    collect_continuous_trajectories,
)
from continuous_quotient.types import ContinuousTrajectoryDataset


def make_po_halfcheetah():
    """Wrap HalfCheetah-v4 to zero out position observations (PO variant)."""
    try:
        import gymnasium as gym
        import numpy as np

        class POHalfCheetahEnv(gym.Wrapper):
            """PO-HalfCheetah: mask position observations (indices 0-8)."""
            def __init__(self):
                env = gym.make("HalfCheetah-v4")
                super().__init__(env)
                # Velocity-only obs: keep qvel (indices 9-17 in 18-dim obs)
                self._mask = np.ones(env.observation_space.shape[0], dtype=bool)
                self._mask[:9] = False  # zero out positions

            def _mask_obs(self, obs):
                import numpy as np
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


def run_continuous(args):
    import numpy as np
    out = ROOT / "experiments" / "layer2_mujoco" / "results" / "halfcheetah_continuous.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(args.seeds):
        env = make_po_halfcheetah()
        cfg = ContinuousEstimatorConfig(
            n_trajectories=args.n_trajectories,
            max_steps_per_episode=args.max_steps,
            window_length=2,
            history_max_length=1,
            min_history_occurrences=args.min_occurrences,
            alpha=args.alpha,
            n_permutations=args.n_permutations,
            history_n_bins=args.history_bins,
            max_samples_per_test=args.max_samples,
        )
        est = ContinuousQuotientEstimator(cfg)
        t0 = time.time()
        # Use zero-action policy: under random continuous actions the 8-dim
        # action noise swamps the velocity signal and the MMD cannot detect
        # differences between velocity bins. A zero action reveals the passive
        # dynamics (velocity decay from friction) which ARE velocity-dependent,
        # producing a non-trivial quotient structure.
        zero_action = np.zeros(env.action_space.shape)
        policy = lambda obs, rng, _z=zero_action: _z
        q = est.fit_from_env(env, policy=policy, seed=seed, verbose=args.verbose)
        rt = time.time() - t0
        rows.append({"seed": seed, "n_classes": q.n_classes, "runtime_s": round(rt, 1)})
        print(f"  seed={seed} n_classes={q.n_classes} runtime={rt:.1f}s")

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "n_classes", "runtime_s"])
        writer.writeheader()
        writer.writerows(rows)

    import numpy as np
    ns = [r["n_classes"] for r in rows]
    print(f"\nPO-HalfCheetah continuous: n_classes = {np.mean(ns):.1f} ± {np.std(ns):.1f} "
          f"(std ≤ 2: {'PASS' if np.std(ns) <= 2 else 'FAIL'})")
    print(f"Results saved to {out}")


def run_discretized(args):
    """Baseline: quantile-discretize observations, then run upstream G-test estimator."""
    out = ROOT / "experiments" / "layer2_mujoco" / "results" / "halfcheetah_discretized.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from quotient_learning.estimator import QuotientEstimator
        from quotient_learning.equivalence import GTestEquivalenceTest
        from quotient_learning.dataset import Trajectory, TrajectoryDataset
        import numpy as np
    except ImportError:
        print("ERROR: quotient_learning not installed. Run: pip install -e /root/code/online-quotient-learning")
        sys.exit(1)

    rows = []
    for seed in range(args.seeds):
        env = make_po_halfcheetah()
        t0 = time.time()
        ds = collect_continuous_trajectories(env, args.n_trajectories, args.max_steps, seed=seed)

        # Quantile-discretize each observation dimension independently.
        all_obs = np.stack([o for traj in ds.trajectories for o in traj.observations])
        n_bins = args.bins
        boundaries = np.percentile(all_obs, np.linspace(0, 100, n_bins + 1)[1:-1], axis=0)

        def disc_obs(o):
            return int(np.sum(np.searchsorted(boundaries.T, o)))  # collapsed scalar

        disc_trajs = []
        for traj in ds.trajectories:
            obs_int = tuple(disc_obs(o) for o in traj.observations)
            acts_int = tuple(int(a[0]) if hasattr(a, '__len__') else int(a) for a in traj.actions)
            disc_trajs.append(Trajectory(observations=obs_int, actions=acts_int,
                                          rewards=tuple(traj.rewards)))
        dataset = TrajectoryDataset(trajectories=disc_trajs, policy_id="uniform_random")

        est = QuotientEstimator(env=None, n_trajectories=len(disc_trajs),
                                window_length=2, history_max_length=1,
                                min_history_occurrences=args.min_occurrences,
                                alpha=args.alpha)
        try:
            q = est.estimate(dataset)
            n_cls = q.n_classes
        except Exception as ex:
            print(f"  seed={seed}: estimator failed: {ex}")
            n_cls = -1

        rt = time.time() - t0
        rows.append({"seed": seed, "bins": n_bins, "n_classes": n_cls, "runtime_s": round(rt, 1)})
        print(f"  seed={seed} bins={n_bins} n_classes={n_cls} runtime={rt:.1f}s")

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "bins", "n_classes", "runtime_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {out}")


def main():
    p = argparse.ArgumentParser(description="PO-HalfCheetah quotient experiment")
    p.add_argument("--mode", choices=["continuous", "discretized"], default="continuous")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-trajectories", type=int, default=5000)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--min-occurrences", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-permutations", type=int, default=99)
    p.add_argument("--history-bins", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--bins", type=int, default=3, help="discretization bins (baseline only)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.mode == "continuous":
        run_continuous(args)
    else:
        run_discretized(args)


if __name__ == "__main__":
    main()
