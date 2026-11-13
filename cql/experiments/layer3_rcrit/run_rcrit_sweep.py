"""R_crit Sweep Experiment — noise asymmetry in Dec-Tiger.

Fixes sigma_A = 0.1 and sweeps sigma_B over [0.1, 0.2, 0.5, 1.0, 2.0].
For each (sigma_A, sigma_B) pair and 5 seeds we record:
  - rcrit_true : analytical value H(Q_A | Q_B)
  - rcrit_est  : our estimator's value

The analytical prediction is monotonically increasing in sigma_B because
a more confused agent B has higher conditional entropy H(Q̂_A | Q̂_B).

Algorithm per (sigma_A, sigma_B, seed):
  1. Collect joint trajectories from DecTigerContinuousEnv with listen-only
     policy (action=0) so tiger state is stable and the bimodal signal
     accumulates clearly in the observation distribution.
  2. Fit independent ContinuousQuotientEstimator for each agent.
  3. Assign each trajectory step to its quotient class via coarse-binned
     history keys (matching the estimator's internal representation).
  4. Plugin-estimate R_crit = H(Q̂_A | Q̂_B) = H(Q̂_A, Q̂_B) - H(Q̂_B).
  5. Compare to analytical value.

Usage:
    python experiments/layer3_rcrit/run_rcrit_sweep.py

Output:
    experiments/layer3_rcrit/results/rcrit_sweep.csv
"""

from __future__ import annotations

import csv
import sys
import time
from collections import Counter
from math import log2
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
try:
    import continuous_quotient  # noqa: F401
except ModuleNotFoundError:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_PROJECT_ROOT))
    import continuous_quotient  # noqa: F401

from continuous_quotient import (
    ContinuousEstimatorConfig,
    ContinuousQuotientEstimator,
    ContinuousTrajectory,
    ContinuousTrajectoryDataset,
)
from continuous_quotient.estimator import _fit_obs_bins, _make_coarse_history_key

_LAYER3_DIR = Path(__file__).resolve().parent
if str(_LAYER3_DIR) not in sys.path:
    sys.path.insert(0, str(_LAYER3_DIR))

from envs.dec_tiger_continuous import DecTigerContinuousEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_RESULTS_DIR = _LAYER3_DIR / "results"
_RESULTS_CSV = _RESULTS_DIR / "rcrit_sweep.csv"

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
SIGMA_A = 0.1
SIGMA_B_VALUES = [0.1, 0.2, 0.5, 1.0, 2.0]
N_SEEDS = 5

# ContinuousEstimatorConfig matching the task specification.
_CONFIG = ContinuousEstimatorConfig(
    n_trajectories=1000,
    history_n_bins=4,
    alpha=0.05,
    n_permutations=99,
    max_samples_per_test=300,
    # Keep defaults for remaining fields (window_length=2, history_max_length=1,
    # min_history_occurrences=20, max_steps_per_episode=50).
)


# ---------------------------------------------------------------------------
# Analytical R_crit
# ---------------------------------------------------------------------------

def analytical_rcrit(sigma_a: float, sigma_b: float) -> float:
    """Compute analytical R_crit = H(Q_A | Q_B) for independent Gaussian Dec-Tiger.

    Each agent classifies tiger side by sgn(obs), obs ~ N(±1, sigma²).
    The per-agent error probability: p_err = Φ(-1/sigma).

    R_crit = H(Q̂_A, Q̂_B) - H(Q̂_B)
    """
    p_err_a = norm.cdf(-1.0 / sigma_a)
    p_err_b = norm.cdf(-1.0 / sigma_b)

    # Joint distribution over (Q_A, Q_B) ∈ {L, R}²
    # P(tiger=Left) = P(tiger=Right) = 0.5
    pLL = 0.5 * (1 - p_err_a) * (1 - p_err_b) + 0.5 * p_err_a * p_err_b
    pRR = pLL
    pLR = 0.5 * (1 - p_err_a) * p_err_b + 0.5 * p_err_a * (1 - p_err_b)
    pRL = pLR

    probs = [pLL, pRR, pLR, pRL]
    h_joint = -sum(p * np.log2(p) for p in probs if p > 0)

    pb0 = pLL + pRL   # P(Q_B = Left)
    pb1 = pRR + pLR   # P(Q_B = Right)
    h_qb = -(pb0 * np.log2(pb0) + pb1 * np.log2(pb1)) if (pb0 > 0 and pb1 > 0) else 0.0

    return h_joint - h_qb


# ---------------------------------------------------------------------------
# Trajectory collection — listen-only policy
# ---------------------------------------------------------------------------

def collect_dec_tiger_trajs(
    env: DecTigerContinuousEnv,
    n_trajectories: int,
    seed: int,
) -> tuple[ContinuousTrajectoryDataset, ContinuousTrajectoryDataset]:
    """Collect episodes with listen-only policy; return per-agent datasets."""
    rng = np.random.default_rng(seed)
    trajs_A: list[ContinuousTrajectory] = []
    trajs_B: list[ContinuousTrajectory] = []

    max_steps = 20  # matches env.episode_length default

    for _ in range(n_trajectories):
        ep_seed = int(rng.integers(0, 2**31))
        obs_A, obs_B = env.reset(seed=ep_seed)

        obs_A_list = [np.array([float(obs_A[0])], dtype=np.float32)]
        obs_B_list = [np.array([float(obs_B[0])], dtype=np.float32)]
        act_A_list: list[int] = []
        act_B_list: list[int] = []

        for _ in range(max_steps):
            obs_A, obs_B, _reward, done = env.step(0, 0)  # listen-only
            obs_A_list.append(np.array([float(obs_A[0])], dtype=np.float32))
            obs_B_list.append(np.array([float(obs_B[0])], dtype=np.float32))
            act_A_list.append(0)
            act_B_list.append(0)
            if done:
                break

        trajs_A.append(ContinuousTrajectory(
            observations=obs_A_list,
            actions=act_A_list,
            rewards=[0.0] * len(act_A_list),
        ))
        trajs_B.append(ContinuousTrajectory(
            observations=obs_B_list,
            actions=act_B_list,
            rewards=[0.0] * len(act_B_list),
        ))

    return (
        ContinuousTrajectoryDataset(trajectories=trajs_A),
        ContinuousTrajectoryDataset(trajectories=trajs_B),
    )


# ---------------------------------------------------------------------------
# Class assignment — must use coarse-binned keys, matching the estimator
# ---------------------------------------------------------------------------

def assign_classes(
    dataset: ContinuousTrajectoryDataset,
    quotient,
    history_max_length: int,
    history_n_bins: int,
) -> list[int]:
    """Map each trajectory step to its quotient class id."""
    boundaries = _fit_obs_bins(dataset, history_n_bins)
    assignments: list[int] = []
    for traj in dataset.trajectories:
        obs = traj.observations
        acts = traj.actions
        T = len(obs)
        for start in range(T):
            hlen = min(history_max_length - 1, start)
            h_start = start - hlen
            h_obs = obs[h_start: start + 1]
            h_acts = acts[h_start: start]
            key = _make_coarse_history_key(list(h_obs), list(h_acts), boundaries)
            cid = quotient.class_of(key)
            if cid is not None:
                assignments.append(cid)
    return assignments


# ---------------------------------------------------------------------------
# Plugin R_crit estimator
# ---------------------------------------------------------------------------

def _plugin_entropy(labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return -sum((c / total) * log2(c / total) for c in counts.values() if c > 0)


def _estimate_rcrit(assignments_A: list[int], assignments_B: list[int]) -> float:
    """Estimate R_crit = H(Q̂_A | Q̂_B) = H(Q̂_A, Q̂_B) - H(Q̂_B) in bits."""
    n = min(len(assignments_A), len(assignments_B))
    if n == 0:
        return 0.0
    A = assignments_A[:n]
    B = assignments_B[:n]
    AB = [a * 1024 + b for a, b in zip(A, B)]
    return max(0.0, _plugin_entropy(AB) - _plugin_entropy(B))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        raise SystemExit("tqdm is required: pip install tqdm") from exc

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"R_crit sweep experiment")
    print(f"  sigma_A (fixed) = {SIGMA_A}")
    print(f"  sigma_B values  = {SIGMA_B_VALUES}")
    print(f"  seeds           = {N_SEEDS}")
    print(f"  n_trajectories  = {_CONFIG.n_trajectories}")
    print(f"  history_n_bins  = {_CONFIG.history_n_bins}")
    print(f"  alpha           = {_CONFIG.alpha}")
    print(f"  n_permutations  = {_CONFIG.n_permutations}")
    print(f"  max_samples/test= {_CONFIG.max_samples_per_test}")
    print()

    # Pre-compute all analytical values.
    true_values = {sigma_b: analytical_rcrit(SIGMA_A, sigma_b) for sigma_b in SIGMA_B_VALUES}
    print("Analytical R_crit values:")
    for sigma_b, val in true_values.items():
        print(f"  sigma_B={sigma_b:.1f} → {val:.4f} bits")
    print()

    records: list[dict] = []
    fieldnames = ["sigma_a", "sigma_b", "seed", "n_A", "n_B",
                  "rcrit_true", "rcrit_est", "rel_error_pct"]

    total_runs = len(SIGMA_B_VALUES) * N_SEEDS
    pbar = tqdm(total=total_runs, desc="sweep", unit="run")

    for sigma_b in SIGMA_B_VALUES:
        rcrit_true = true_values[sigma_b]
        for seed in range(N_SEEDS):
            t0 = time.perf_counter()

            env = DecTigerContinuousEnv(
                sigma_a=SIGMA_A,
                sigma_b=sigma_b,
                episode_length=20,
            )

            # Collect joint trajectories.
            dataset_A, dataset_B = collect_dec_tiger_trajs(
                env,
                n_trajectories=_CONFIG.n_trajectories,
                seed=seed,
            )

            # Fit independent quotient estimators.
            estimator_A = ContinuousQuotientEstimator(config=_CONFIG)
            estimator_B = ContinuousQuotientEstimator(config=_CONFIG)

            quotient_A = estimator_A.fit(dataset_A, env_id="dec_tiger_A", verbose=False)
            quotient_B = estimator_B.fit(dataset_B, env_id="dec_tiger_B", verbose=False)

            n_A = quotient_A.n_classes
            n_B = quotient_B.n_classes

            # Assign classes using coarse-binned keys.
            assignments_A = assign_classes(
                dataset_A, quotient_A,
                history_max_length=_CONFIG.history_max_length,
                history_n_bins=_CONFIG.history_n_bins,
            )
            assignments_B = assign_classes(
                dataset_B, quotient_B,
                history_max_length=_CONFIG.history_max_length,
                history_n_bins=_CONFIG.history_n_bins,
            )

            rcrit_est = _estimate_rcrit(assignments_A, assignments_B)
            runtime_s = time.perf_counter() - t0

            rel_error_pct = (
                abs(rcrit_est - rcrit_true) / max(abs(rcrit_true), 1e-9) * 100.0
            )

            records.append({
                "sigma_a": SIGMA_A,
                "sigma_b": sigma_b,
                "seed": seed,
                "n_A": n_A,
                "n_B": n_B,
                "rcrit_true": round(rcrit_true, 6),
                "rcrit_est": round(rcrit_est, 6),
                "rel_error_pct": round(rel_error_pct, 4),
            })

            pbar.set_postfix(
                sigma_b=f"{sigma_b:.1f}",
                seed=seed,
                n_A=n_A,
                n_B=n_B,
                rcrit_est=f"{rcrit_est:.3f}",
                true=f"{rcrit_true:.3f}",
            )
            pbar.update(1)

    pbar.close()

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    with _RESULTS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nResults written to {_RESULTS_CSV}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    import statistics

    print(
        f"\n{'sigma_B':>8} | {'R_crit (true)':>14} | "
        f"{'R_crit (est mean±std)':>22} | {'Mean rel error':>14}"
    )
    print("-" * 70)

    for sigma_b in SIGMA_B_VALUES:
        rows = [r for r in records if r["sigma_b"] == sigma_b]
        rcrit_true = true_values[sigma_b]
        ests = [r["rcrit_est"] for r in rows]
        rel_errs = [r["rel_error_pct"] for r in rows]

        mean_est = statistics.mean(ests)
        std_est = statistics.stdev(ests) if len(ests) > 1 else 0.0
        mean_rel = statistics.mean(rel_errs)

        print(
            f"{sigma_b:>8.1f} | {rcrit_true:>14.4f} | "
            f"{mean_est:>9.4f} ± {std_est:<9.4f} | {mean_rel:>10.1f}%"
        )

    print()
    print("Expected trend: R_crit (est) increases monotonically with sigma_B.")

    # Check monotonicity of true values (sanity).
    true_list = [true_values[s] for s in SIGMA_B_VALUES]
    monotone = all(true_list[i] <= true_list[i + 1] for i in range(len(true_list) - 1))
    print(f"Analytical values monotone: {monotone}  {[round(v, 4) for v in true_list]}")

    # Check estimator means follow the same trend.
    mean_ests = []
    for sigma_b in SIGMA_B_VALUES:
        rows = [r for r in records if r["sigma_b"] == sigma_b]
        mean_ests.append(statistics.mean(r["rcrit_est"] for r in rows))
    est_monotone = all(mean_ests[i] <= mean_ests[i + 1] for i in range(len(mean_ests) - 1))
    print(f"Estimated means monotone:   {est_monotone}  {[round(v, 4) for v in mean_ests]}")


if __name__ == "__main__":
    main()
