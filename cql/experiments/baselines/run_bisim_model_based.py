"""Bisimulation metric baseline for Tiger and K-cycle continuous environments.

Two baselines are implemented and compared against the MMD-permutation method:

1. Oracle bisim (model-based):
   - Tiger: exactly 2 classes (L vs R), because the true observation
     distributions N(-1, 0.5²) and N(+1, 0.5²) are distinct.
   - K-cycle-K=K: exactly K classes (each state has a unique continuation).
   These are written directly without running an algorithm.

2. Approximate (model-free) Wasserstein-ε bisim:
   - Uses the same continuation matrices as the MMD method.
   - Test statistic: W₁(first-dim continuation projections).
   - Threshold ε ∈ {0.1, 0.2, 0.5, 1.0} replaces the permutation p-value.
   - Two histories are merged (same class) if W₁ < ε.

Output: experiments/baselines/results/bisim_comparison.csv
Columns: env, method, threshold_or_alpha, seed, n_classes, correct, runtime_s
"""

from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
LAYER1_DIR = ROOT / "experiments" / "layer1_ground_truth"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAYER1_DIR))

from continuous_quotient import ContinuousEstimatorConfig
from continuous_quotient.estimator import (
    collect_continuous_trajectories,
    _fit_obs_bins,
    _extract_histories_and_continuations,
)
from continuous_quotient.types import ContinuousTrajectoryDataset
from envs.tiger_continuous import TigerContinuousEnv
from envs.kcycle_continuous import KCycleContinuousEnv

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
RESULTS_DIR = ROOT / "experiments" / "baselines" / "results"
RESULTS_CSV = RESULTS_DIR / "bisim_comparison.csv"

# ---------------------------------------------------------------------------
# Hyperparameters (mirror the main experiments)
# ---------------------------------------------------------------------------
TIGER_N = 1000          # trajectories per seed
TIGER_SEEDS = 10
TIGER_SIGMA = 0.5

KCYCLE_K = 4
KCYCLE_N = 3000         # trajectories per seed
KCYCLE_SEEDS = 5
KCYCLE_NOISE = 0.1

# Shared continuation / history config
WINDOW_LENGTH = 2
HISTORY_MAX_LEN = 1
MIN_OCCURRENCES = 20
HISTORY_N_BINS = 8

# Wasserstein thresholds to sweep
W_THRESHOLDS = [0.1, 0.2, 0.5, 1.0]


# ---------------------------------------------------------------------------
# Union-Find (mirrors estimator.py — reproduced here for independence)
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, items: list) -> None:
        self._parent = {h: h for h in items}

    def find(self, h):
        root = h
        while self._parent[root] != root:
            root = self._parent[root]
        cur = h
        while self._parent[cur] != root:
            nxt = self._parent[cur]
            self._parent[cur] = root
            cur = nxt
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def n_groups(self) -> int:
        return len({self.find(h) for h in self._parent})


# ---------------------------------------------------------------------------
# Wasserstein-ε test
# ---------------------------------------------------------------------------

def wasserstein_test(samples_a: np.ndarray, samples_b: np.ndarray, threshold: float) -> bool:
    """Return True (same class) if W₁ distance on first dimension < threshold."""
    a = samples_a[:, 0]
    b = samples_b[:, 0]
    d = wasserstein_distance(a, b)
    return d < threshold   # True → merge (same class)


# ---------------------------------------------------------------------------
# Run Wasserstein bisim on one (env, seed) pair
# ---------------------------------------------------------------------------

def run_wasserstein_bisim(
    dataset: ContinuousTrajectoryDataset,
    threshold: float,
) -> int:
    """Return estimated n_classes using Wasserstein-ε clustering."""
    history_continuations = _extract_histories_and_continuations(
        dataset,
        history_max_length=HISTORY_MAX_LEN,
        window_length=WINDOW_LENGTH,
        min_occurrences=MIN_OCCURRENCES,
        history_n_bins=HISTORY_N_BINS,
    )

    viable = list(history_continuations.keys())
    if not viable:
        return 0

    uf = _UnionFind(viable)
    for i in range(len(viable)):
        for j in range(i + 1, len(viable)):
            ha, hb = viable[i], viable[j]
            same = wasserstein_test(
                history_continuations[ha],
                history_continuations[hb],
                threshold=threshold,
            )
            if same:
                uf.union(ha, hb)

    return uf.n_groups()


# ---------------------------------------------------------------------------
# Load existing MMD-permutation results from layer1 CSVs
# ---------------------------------------------------------------------------

def load_mmd_tiger_results() -> list[dict]:
    """Load tiger results from layer1_ground_truth CSV."""
    csv_path = ROOT / "experiments" / "layer1_ground_truth" / "results" / "tiger_continuous.csv"
    records = []
    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found — skipping MMD tiger results")
        return records
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append({
                "env": "tiger_n2",
                "method": "mmd_permutation",
                "threshold_or_alpha": 0.01,
                "seed": int(row["seed"]),
                "n_classes": int(row["n_classes"]),
                "correct": 1 if int(row["n_classes"]) == 2 else 0,
                "runtime_s": float(row["runtime_s"]),
            })
    return records


def load_mmd_kcycle_results(K: int = 4) -> list[dict]:
    """Load K-cycle K=4 results from layer1_ground_truth CSV."""
    csv_path = ROOT / "experiments" / "layer1_ground_truth" / "results" / "kcycle_continuous.csv"
    records = []
    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found — skipping MMD kcycle results")
        return records
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if int(row["K"]) != K:
                continue
            records.append({
                "env": f"kcycle_k{K}",
                "method": "mmd_permutation",
                "threshold_or_alpha": 0.01,
                "seed": int(row["seed"]),
                "n_classes": int(row["n_classes"]),
                "correct": int(row["correct"]),
                "runtime_s": float(row["runtime_s"]),
            })
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    # -----------------------------------------------------------------------
    # 1. Oracle bisim (model-based, no data required)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("1. Oracle bisim (model-based ground truth)")
    print("=" * 60)

    # Tiger: exactly 2 classes because N(-1, 0.25) ≠ N(+1, 0.25)
    for seed in range(TIGER_SEEDS):
        all_records.append({
            "env": "tiger_n2",
            "method": "oracle_bisim",
            "threshold_or_alpha": "N/A",
            "seed": seed,
            "n_classes": 2,
            "correct": 1,
            "runtime_s": 0.0,
        })
    print(f"  Tiger oracle bisim: n=2 (all {TIGER_SEEDS} seeds correct by construction)")

    # K-cycle K=4: exactly K classes
    for seed in range(KCYCLE_SEEDS):
        all_records.append({
            "env": f"kcycle_k{KCYCLE_K}",
            "method": "oracle_bisim",
            "threshold_or_alpha": "N/A",
            "seed": seed,
            "n_classes": KCYCLE_K,
            "correct": 1,
            "runtime_s": 0.0,
        })
    print(f"  K-cycle-K={KCYCLE_K} oracle bisim: n={KCYCLE_K} (all {KCYCLE_SEEDS} seeds correct by construction)")

    # -----------------------------------------------------------------------
    # 2. Load MMD-permutation results
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("2. MMD-permutation results (from existing CSVs)")
    print("=" * 60)

    mmd_tiger = load_mmd_tiger_results()
    mmd_kcycle = load_mmd_kcycle_results(K=KCYCLE_K)
    all_records.extend(mmd_tiger)
    all_records.extend(mmd_kcycle)
    print(f"  Loaded {len(mmd_tiger)} tiger MMD rows, {len(mmd_kcycle)} kcycle MMD rows")

    # -----------------------------------------------------------------------
    # 3. Wasserstein-ε baseline on Tiger
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"3. Wasserstein-ε baseline — Tiger (N={TIGER_N}, {TIGER_SEEDS} seeds)")
    print("=" * 60)

    for threshold in W_THRESHOLDS:
        method_name = f"wasserstein_eps{threshold}"
        seed_results = []
        for seed in range(TIGER_SEEDS):
            env = TigerContinuousEnv(sigma=TIGER_SIGMA, episode_length=20)
            dataset = collect_continuous_trajectories(
                env=env,
                n_episodes=TIGER_N,
                max_steps=20,
                seed=seed,
            )
            t0 = time.perf_counter()
            n_classes = run_wasserstein_bisim(dataset, threshold=threshold)
            runtime_s = time.perf_counter() - t0
            correct = 1 if n_classes == 2 else 0
            seed_results.append(n_classes)
            all_records.append({
                "env": "tiger_n2",
                "method": method_name,
                "threshold_or_alpha": threshold,
                "seed": seed,
                "n_classes": n_classes,
                "correct": correct,
                "runtime_s": round(runtime_s, 4),
            })

        acc = sum(1 for nc in seed_results if nc == 2) / len(seed_results)
        print(f"  ε={threshold:4.1f} | n_classes={seed_results} | acc={acc:.2f}")

    # -----------------------------------------------------------------------
    # 4. Wasserstein-ε baseline on K-cycle K=4
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"4. Wasserstein-ε baseline — K-cycle K={KCYCLE_K} (N={KCYCLE_N}, {KCYCLE_SEEDS} seeds)")
    print("=" * 60)

    for threshold in W_THRESHOLDS:
        method_name = f"wasserstein_eps{threshold}"
        seed_results = []
        for seed in range(KCYCLE_SEEDS):
            env = KCycleContinuousEnv(K=KCYCLE_K, obs_noise=KCYCLE_NOISE, episode_length=50)
            dataset = collect_continuous_trajectories(
                env=env,
                n_episodes=KCYCLE_N,
                max_steps=50,
                seed=seed,
            )
            t0 = time.perf_counter()
            n_classes = run_wasserstein_bisim(dataset, threshold=threshold)
            runtime_s = time.perf_counter() - t0
            correct = 1 if n_classes == KCYCLE_K else 0
            seed_results.append(n_classes)
            all_records.append({
                "env": f"kcycle_k{KCYCLE_K}",
                "method": method_name,
                "threshold_or_alpha": threshold,
                "seed": seed,
                "n_classes": n_classes,
                "correct": correct,
                "runtime_s": round(runtime_s, 4),
            })

        acc = sum(1 for nc in seed_results if nc == KCYCLE_K) / len(seed_results)
        print(f"  ε={threshold:4.1f} | n_classes={seed_results} | acc={acc:.2f}")

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    fieldnames = ["env", "method", "threshold_or_alpha", "seed", "n_classes", "correct", "runtime_s"]
    with RESULTS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nResults written to {RESULTS_CSV}")

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)

    # Group records by (env, method, threshold_or_alpha)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in all_records:
        key = (row["env"], row["method"], row["threshold_or_alpha"])
        groups[key].append(row)

    # Sort: env first, then oracle, then mmd, then wasserstein
    def sort_key(k):
        env, method, thresh = k
        method_order = {"oracle_bisim": 0, "mmd_permutation": 1}
        mo = method_order.get(method, 2)
        thresh_val = 0.0 if thresh == "N/A" else float(thresh)
        return (env, mo, thresh_val)

    sorted_keys = sorted(groups.keys(), key=sort_key)

    print(f"{'env':<15} {'method':<25} {'threshold':<12} {'mean_n_classes':<16} {'acc':<8} {'mean_rt_s'}")
    print("-" * 90)

    for key in sorted_keys:
        env, method, thresh = key
        rows = groups[key]
        n_classes_list = [r["n_classes"] for r in rows]
        correct_list = [r["correct"] for r in rows]
        rt_list = [r["runtime_s"] for r in rows]

        mean_nc = np.mean(n_classes_list)
        acc = np.mean(correct_list)
        mean_rt = np.mean(rt_list)

        thresh_str = str(thresh)
        print(f"{env:<15} {method:<25} {thresh_str:<12} {mean_nc:<16.2f} {acc:<8.2f} {mean_rt:.3f}")

    print()
    print("Notes:")
    print("  oracle_bisim  — ground truth from known model (Tiger: n=2, K-cycle: n=K)")
    print("  mmd_permutation — our MMD + permutation test (loaded from layer1_ground_truth CSVs)")
    print("  wasserstein_epsX — W₁ < ε threshold (model-free bisim approximation, no calibration)")


if __name__ == "__main__":
    main()
