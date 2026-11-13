"""PO-Ant experiment (Layer 2): confirm saturation artifact removed.

PO-Ant: Ant-v4 with velocity observations only.
Under discretization (3 bins per dim), the estimated quotient saturates
at the code cap (64 classes = 2^6). This experiment shows the continuous
algorithm produces an unsaturated, stable quotient.

Usage:
    python experiments/layer2_mujoco/run_ant.py --mode continuous
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


def make_po_ant():
    try:
        import gymnasium as gym
        import numpy as np

        class POAntEnv(gym.Wrapper):
            """PO-Ant: keep only velocity observations (indices 13-26 of 27-dim obs)."""
            def __init__(self):
                env = gym.make("Ant-v4")
                super().__init__(env)
                # Ant obs: 13 position qpos + 14 velocity qvel = 27 total
                self._mask = np.zeros(env.observation_space.shape[0], dtype=bool)
                self._mask[13:] = True  # velocity only

            def _mask_obs(self, obs):
                return np.asarray(obs, dtype=np.float32)[self._mask]

            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                return self._mask_obs(obs), info

            def step(self, action):
                obs, r, term, trunc, info = self.env.step(action)
                return self._mask_obs(obs), r, term, trunc, info

        return POAntEnv()
    except Exception as e:
        print(f"ERROR: Could not create Ant-v4: {e}")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="PO-Ant quotient experiment")
    p.add_argument("--mode", choices=["continuous", "discretized"], default="continuous")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-trajectories", type=int, default=5000)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--min-occurrences", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-permutations", type=int, default=99)
    p.add_argument("--history-bins", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--bins", type=int, default=3)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    out_name = f"ant_{args.mode}.csv"
    out = ROOT / "experiments" / "layer2_mujoco" / "results" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    if args.mode == "continuous":
        import numpy as np
        for seed in range(args.seeds):
            env = make_po_ant()
            cfg = ContinuousEstimatorConfig(
                n_trajectories=args.n_trajectories, max_steps_per_episode=args.max_steps,
                window_length=2, history_max_length=1,
                min_history_occurrences=args.min_occurrences,
                alpha=args.alpha, n_permutations=args.n_permutations,
                history_n_bins=args.history_bins, max_samples_per_test=args.max_samples,
            )
            est = ContinuousQuotientEstimator(cfg)
            t0 = time.time()
            # Use zero-action policy: random 8-dim actions swamp velocity signal.
            # Passive (zero-torque) dynamics are velocity-dependent and reveal
            # the true quotient structure without action noise.
            zero_action = np.zeros(env.action_space.shape)
            policy = lambda obs, rng, _z=zero_action: _z
            q = est.fit_from_env(env, policy=policy, seed=seed, verbose=args.verbose)
            rt = time.time() - t0
            rows.append({"seed": seed, "mode": "continuous", "n_classes": q.n_classes, "runtime_s": round(rt, 1)})
            print(f"  seed={seed} n_classes={q.n_classes} runtime={rt:.1f}s")

        import numpy as np
        ns = [r["n_classes"] for r in rows]
        print(f"\nPO-Ant continuous: n_classes = {np.mean(ns):.1f} ± {np.std(ns):.1f}")
        cap = 64
        saturated = sum(n >= cap for n in ns)
        print(f"Saturation check (n ≥ {cap}): {saturated}/{args.seeds} seeds saturated "
              f"({'FAIL: artifact persists' if saturated > 0 else 'PASS: artifact removed'})")

    else:
        # Discretized baseline to show the saturation artifact.
        from continuous_quotient import collect_continuous_trajectories
        import numpy as np
        try:
            from quotient_learning.estimator import QuotientEstimator
            from quotient_learning.dataset import Trajectory, TrajectoryDataset
        except ImportError:
            print("ERROR: quotient_learning not installed")
            sys.exit(1)

        for seed in range(args.seeds):
            env = make_po_ant()
            ds = collect_continuous_trajectories(env, args.n_trajectories, args.max_steps, seed=seed)
            all_obs = np.stack([o for traj in ds.trajectories for o in traj.observations])
            n_bins = args.bins
            boundaries = np.percentile(all_obs, np.linspace(0, 100, n_bins + 1)[1:-1], axis=0)

            def disc_obs(o):
                return int(np.sum(np.searchsorted(boundaries.T, o)))

            disc_trajs = []
            for traj in ds.trajectories:
                obs_int = tuple(disc_obs(o) for o in traj.observations)
                acts_int = tuple(0 for _ in traj.actions)  # continuous actions → dummy
                disc_trajs.append(Trajectory(observations=obs_int, actions=acts_int,
                                              rewards=tuple(traj.rewards)))
            dataset = TrajectoryDataset(trajectories=disc_trajs, policy_id="uniform_random")

            est = QuotientEstimator(env=None, n_trajectories=len(disc_trajs),
                                    window_length=2, history_max_length=1,
                                    min_history_occurrences=args.min_occurrences, alpha=args.alpha)
            try:
                q = est.estimate(dataset)
                n_cls = q.n_classes
            except Exception as ex:
                n_cls = -1
            rows.append({"seed": seed, "mode": f"discretized_{n_bins}bin", "n_classes": n_cls, "runtime_s": 0})
            print(f"  seed={seed} bins={n_bins} n_classes={n_cls}")

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "mode", "n_classes", "runtime_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
