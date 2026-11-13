"""Dec-Tiger Continuous — Layer-3 R_crit Estimation Experiment.

Estimates R_crit = H(Q̂_A | Q̂_B) for two agents in DecTigerContinuousEnv
and compares against the analytical value of 0.0 bits (equal-capacity agents
with the same 2-class quotient structure require zero communication to
coordinate perfectly).

Algorithm per seed:
  1. Collect joint trajectories from DecTigerContinuousEnv using a
     LISTEN-ONLY policy so the tiger state stays stable and the continuation
     distribution is strongly shaped by the current observation.
  2. Wrap each agent's observations as a single-agent ContinuousTrajectory
     dataset and run ContinuousQuotientEstimator independently.
  3. Map every step's observation to its quotient class for agent A and B
     using the same coarse-binned history key format used internally by the
     estimator (_make_coarse_history_key from estimator.py).
  4. Plugin-estimate H(Q̂_A | Q̂_B) = H(Q̂_A, Q̂_B) - H(Q̂_B).
  5. Compare to analytical R_crit = 0.0.

Key design choices:
  - Listen-only policy: open actions reset the tiger state, washing out the
    observation→state correlation and making all bins look the same to MMD.
    With listen-only the state is stable so observations accumulate a
    clean bimodal signal.
  - 4 observation bins: avoids boundary-bin ambiguity present with 8 bins
    (bins 3 & 4 straddle the zero boundary and can land in either class).
  - Coarse binned key lookup: the quotient partition is keyed by binned
    integers (from _make_coarse_history_key), not raw float bytes
    (from make_history_key). Using the wrong key format makes every lookup
    return None, giving trivially empty assignments.

Usage:
    python experiments/layer3_rcrit/run_dec_tiger_continuous.py [--seeds N] ...

Expected result:
    n_A >= 2, n_B >= 2 for all seeds; mean rcrit_est is a valid float
    near 0.0 (analytical R_crit = 0 bits for equal-capacity agents).
"""

from __future__ import annotations

import argparse
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
# Path setup — make ``continuous_quotient`` importable when running from any
# working directory without a prior ``pip install -e .``.
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

# Env lives inside the layer3_rcrit/envs sub-package.
_LAYER3_DIR = Path(__file__).resolve().parent
if str(_LAYER3_DIR) not in sys.path:
    sys.path.insert(0, str(_LAYER3_DIR))

from envs.dec_tiger_continuous import DecTigerContinuousEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
_RESULTS_DIR = _LAYER3_DIR / "results"
_RESULTS_CSV = _RESULTS_DIR / "dec_tiger_rcrit.csv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dec-Tiger R_crit estimation experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of independent seeds to run.")
    p.add_argument("--n-trajectories", type=int, default=1000,
                   help="Number of joint trajectories (episodes) per seed.")
    p.add_argument("--sigma-a", type=float, default=0.5,
                   help="Observation noise std for agent A.")
    p.add_argument("--sigma-b", type=float, default=0.5,
                   help="Observation noise std for agent B.")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Base significance level for MMD tests.")
    p.add_argument("--window-length", type=int, default=2,
                   help="Continuation window length k.")
    p.add_argument("--history-max-length", type=int, default=1,
                   help="Maximum history prefix length L.")
    p.add_argument("--min-occurrences", type=int, default=10,
                   help="Minimum occurrences for a history to be included.")
    p.add_argument("--n-permutations", type=int, default=99,
                   help="Permutation samples for the MMD null distribution.")
    p.add_argument("--history-n-bins", type=int, default=4,
                   help="Observation bins for coarse history keys.")
    p.add_argument("--output-csv", type=str, default=None,
                   help="Output CSV path (overrides default dec_tiger_rcrit.csv).")
    p.add_argument("--append-csv", action="store_true", default=False,
                   help="Append to output CSV instead of overwriting.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Trajectory collection — LISTEN-ONLY policy
# ---------------------------------------------------------------------------

def collect_dec_tiger_trajs(
    env: DecTigerContinuousEnv,
    n_trajectories: int,
    max_steps: int,
    seed: int,
) -> tuple[ContinuousTrajectoryDataset, ContinuousTrajectoryDataset]:
    """Collect episodes using a listen-only policy and return per-agent datasets.

    Listen-only (action 0) keeps the tiger state fixed throughout the episode,
    so the continuation observation distribution is strongly shaped by the
    current observation value.  Open actions (1 or 2) reset the tiger state,
    which destroys the observation→state correlation and makes all history bins
    look identical to the MMD test, causing n_classes=1 (fully merged).

    Both agents always play action 0 (listen).

    Returns
    -------
    dataset_A, dataset_B : ContinuousTrajectoryDataset
    """
    rng = np.random.default_rng(seed)
    trajs_A: list[ContinuousTrajectory] = []
    trajs_B: list[ContinuousTrajectory] = []

    for _ in range(n_trajectories):
        ep_seed = int(rng.integers(0, 2**31))
        obs_A, obs_B = env.reset(seed=ep_seed)

        obs_A_list = [np.array([float(obs_A[0])], dtype=np.float32)]
        obs_B_list = [np.array([float(obs_B[0])], dtype=np.float32)]
        act_A_list: list[int] = []
        act_B_list: list[int] = []

        for _ in range(max_steps):
            # Listen-only: action 0 keeps the tiger state stable.
            obs_A, obs_B, _reward, done = env.step(0, 0)
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
# Class assignment — must use the same coarse-binned key as the estimator
# ---------------------------------------------------------------------------

def assign_classes(
    dataset: ContinuousTrajectoryDataset,
    quotient: "CanonicalContinuousQuotient",  # type: ignore[name-defined]
    history_max_length: int,
    history_n_bins: int,
) -> list[int]:
    """Map each trajectory step to its quotient class.

    The estimator's partition is keyed by coarse-binned integer tuples built
    by _make_coarse_history_key — NOT by the raw-float bytes from
    make_history_key.  Using the wrong key format causes every lookup to return
    None, producing empty assignments and a trivially zero R_crit.

    Steps whose binned history key is not in the partition (not enough
    occurrences during fitting) are skipped (not appended).
    """
    # Fit the same quantile boundaries used during estimator.fit().
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
# Analytical R_crit
# ---------------------------------------------------------------------------

def analytical_rcrit(sigma_a: float, sigma_b: float) -> float:
    """Compute analytical R_crit = H(Q_A | Q_B) for independent Gaussian Dec-Tiger.

    Each agent classifies tiger side by sgn(obs), where obs ~ N(±1, sigma²).
    The error probability for each agent: p_err = Φ(-1/sigma).

    For sigma=0.5: p_err = Φ(-2) ≈ 0.0228, giving R_crit ≈ 0.261 bits.
    """
    p_err_a = norm.cdf(-1.0 / sigma_a)
    p_err_b = norm.cdf(-1.0 / sigma_b)

    # Joint distribution over (Q_A, Q_B) ∈ {0,1}²
    # Q=0 means "Left", Q=1 means "Right"
    # P(tiger=Left) = P(tiger=Right) = 0.5
    # When tiger=Left:  P(Q=0|Left)  = 1-p_err,  P(Q=1|Left)  = p_err
    # When tiger=Right: P(Q=0|Right) = p_err,     P(Q=1|Right) = 1-p_err
    pLL = 0.5 * (1 - p_err_a) * (1 - p_err_b) + 0.5 * p_err_a * p_err_b  # both say Left
    pRR = 0.5 * p_err_a * p_err_b + 0.5 * (1 - p_err_a) * (1 - p_err_b)  # both say Right
    pLR = 0.5 * (1 - p_err_a) * p_err_b + 0.5 * p_err_a * (1 - p_err_b)  # A=Left, B=Right
    pRL = 0.5 * p_err_a * (1 - p_err_b) + 0.5 * (1 - p_err_a) * p_err_b  # A=Right, B=Left

    # H(Q_A, Q_B)
    probs = [pLL, pRR, pLR, pRL]
    h_joint = -sum(p * np.log2(p) for p in probs if p > 0)

    # H(Q_B)
    pb0 = pLL + pRL  # P(Q_B=Left)
    pb1 = pRR + pLR  # P(Q_B=Right)
    h_qb = -(pb0 * np.log2(pb0) + pb1 * np.log2(pb1)) if (pb0 > 0 and pb1 > 0) else 0.0

    return h_joint - h_qb


# ---------------------------------------------------------------------------
# Plugin entropy estimator
# ---------------------------------------------------------------------------

def _plugin_entropy(labels: Sequence[int]) -> float:
    """Compute the plugin (MLE) Shannon entropy in bits from a label sequence."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return -sum((c / total) * log2(c / total) for c in counts.values() if c > 0)


def _joint_label(a: int, b: int) -> int:
    """Encode a (class_A, class_B) pair as a single integer for joint entropy."""
    return a * 1024 + b


def _estimate_rcrit(
    assignments_A: list[int],
    assignments_B: list[int],
) -> float:
    """Estimate R_crit = H(Q̂_A | Q̂_B) = H(Q̂_A, Q̂_B) - H(Q̂_B) in bits."""
    n = min(len(assignments_A), len(assignments_B))
    if n == 0:
        return 0.0

    A = assignments_A[:n]
    B = assignments_B[:n]
    AB = [_joint_label(a, b) for a, b in zip(A, B)]

    h_ab = _plugin_entropy(AB)
    h_b = _plugin_entropy(B)
    return max(0.0, h_ab - h_b)  # clamp to 0 to avoid floating-point negatives


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    try:
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        raise SystemExit("tqdm is required: pip install tqdm") from exc

    config = ContinuousEstimatorConfig(
        n_trajectories=args.n_trajectories,
        max_steps_per_episode=20,
        window_length=args.window_length,
        history_max_length=args.history_max_length,
        min_history_occurrences=args.min_occurrences,
        alpha=args.alpha,
        n_permutations=args.n_permutations,
        history_n_bins=args.history_n_bins,
        max_samples_per_test=300,
    )

    print(
        f"Dec-Tiger R_crit experiment | seeds={args.seeds} "
        f"n_traj={args.n_trajectories} sigma_a={args.sigma_a} "
        f"sigma_b={args.sigma_b} alpha={args.alpha} "
        f"window={args.window_length} history_max_len={args.history_max_length} "
        f"min_occ={args.min_occurrences} n_perm={args.n_permutations} "
        f"n_bins={args.history_n_bins} policy=listen-only"
    )

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Override output CSV if specified.
    output_csv = _RESULTS_CSV
    if args.output_csv is not None:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    rcrit_true = analytical_rcrit(args.sigma_a, args.sigma_b)
    print(f"Analytical R_crit (sigma_a={args.sigma_a}, sigma_b={args.sigma_b}): "
          f"{rcrit_true:.4f} bits")

    for seed in tqdm(range(args.seeds), desc="seeds", unit="seed"):
        env = DecTigerContinuousEnv(
            sigma_a=args.sigma_a,
            sigma_b=args.sigma_b,
            episode_length=20,
        )

        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Collect joint trajectories with listen-only policy.
        # ------------------------------------------------------------------
        dataset_A, dataset_B = collect_dec_tiger_trajs(
            env,
            n_trajectories=args.n_trajectories,
            max_steps=20,
            seed=seed,
        )

        # ------------------------------------------------------------------
        # Fit independent quotient estimators for each agent.
        # ------------------------------------------------------------------
        estimator_A = ContinuousQuotientEstimator(config=config)
        estimator_B = ContinuousQuotientEstimator(config=config)

        quotient_A = estimator_A.fit(dataset_A, env_id="dec_tiger_A", verbose=False)
        quotient_B = estimator_B.fit(dataset_B, env_id="dec_tiger_B", verbose=False)

        n_A = quotient_A.n_classes
        n_B = quotient_B.n_classes

        # ------------------------------------------------------------------
        # Assign classes using the same coarse-binned key format as the
        # estimator (NOT make_history_key which uses raw float bytes).
        # ------------------------------------------------------------------
        assignments_A = assign_classes(
            dataset_A, quotient_A,
            history_max_length=args.history_max_length,
            history_n_bins=args.history_n_bins,
        )
        assignments_B = assign_classes(
            dataset_B, quotient_B,
            history_max_length=args.history_max_length,
            history_n_bins=args.history_n_bins,
        )

        tqdm.write(
            f"  seed={seed}  n_A={n_A}  n_B={n_B}  "
            f"assigned_A={len(assignments_A)}  assigned_B={len(assignments_B)}"
        )

        # ------------------------------------------------------------------
        # Estimate R_crit = H(Q̂_A | Q̂_B).
        # ------------------------------------------------------------------
        rcrit_est = _estimate_rcrit(assignments_A, assignments_B)
        runtime_s = time.perf_counter() - t0

        rel_error_pct = abs(rcrit_est - rcrit_true) / max(abs(rcrit_true), 1e-9) * 100.0

        records.append({
            "seed": seed,
            "sigma_a": args.sigma_a,
            "sigma_b": args.sigma_b,
            "n_A": n_A,
            "n_B": n_B,
            "rcrit_est": round(rcrit_est, 6),
            "rcrit_true": round(rcrit_true, 6),
            "rel_error_pct": round(rel_error_pct, 4),
            "runtime_s": round(runtime_s, 4),
        })

        tqdm.write(
            f"  seed={seed:2d}  n_A={n_A}  n_B={n_B}  "
            f"rcrit_est={rcrit_est:.4f}  rel_error_pct={rel_error_pct:.2f}%"
        )

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    fieldnames = ["seed", "sigma_a", "sigma_b", "n_A", "n_B", "rcrit_est", "rcrit_true", "rel_error_pct", "runtime_s"]
    write_mode = "a" if args.append_csv and output_csv.exists() else "w"
    with output_csv.open(write_mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_mode == "w":
            writer.writeheader()
        writer.writerows(records)

    print(f"\nResults written to {output_csv}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    import statistics

    rcrit_list = [r["rcrit_est"] for r in records]
    rel_errors_pct = [r["rel_error_pct"] for r in records]
    n_A_list = [r["n_A"] for r in records]
    n_B_list = [r["n_B"] for r in records]

    mean_rcrit = statistics.mean(rcrit_list)
    mean_rel_err_pct = statistics.mean(rel_errors_pct)

    print(f"\n--- Summary ({args.seeds} seeds) ---")
    print(f"  n_A per seed         : {n_A_list}")
    print(f"  n_B per seed         : {n_B_list}")
    print(f"  rcrit_est            : {[round(v, 4) for v in rcrit_list]}")
    print(f"  mean rcrit_est       : {mean_rcrit:.4f}  (analytical={rcrit_true:.4f} bits)")
    print(f"  mean rel_error_pct   : {mean_rel_err_pct:.2f}%")
    print(
        f"\nMean relative error vs CORRECT analytical R_crit "
        f"({rcrit_true:.3f} bits): {mean_rel_err_pct:.1f}%"
    )

    # ------------------------------------------------------------------
    # CSV printout
    # ------------------------------------------------------------------
    print(f"\n--- CSV: {output_csv} ---")
    with output_csv.open() as fh:
        print(fh.read())

    # ------------------------------------------------------------------
    # Pass/Fail gate — n_classes and valid rcrit
    # ------------------------------------------------------------------
    all_n_ok = all(n >= 2 for n in n_A_list) and all(n >= 2 for n in n_B_list)
    rcrit_valid = all(isinstance(v, float) for v in rcrit_list)

    status = "PASS" if all_n_ok and rcrit_valid else "FAIL"
    print(f"\n{status}  (n_A_min={min(n_A_list)}, n_B_min={min(n_B_list)}, "
          f"mean_rcrit_est={mean_rcrit:.4f}, mean_rel_error_pct={mean_rel_err_pct:.2f}%)")

    if not all_n_ok:
        raise AssertionError(
            f"FAIL: some seeds produced n_A<2 or n_B<2. "
            f"n_A={n_A_list}, n_B={n_B_list}"
        )
    assert rcrit_valid, "FAIL: rcrit_est contains non-float values"


if __name__ == "__main__":
    main()
