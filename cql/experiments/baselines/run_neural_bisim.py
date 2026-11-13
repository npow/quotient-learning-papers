"""DBC-style neural bisimulation baseline for Tiger and K-cycle continuous environments.

Trains a small MLP encoder E: observation_history → R^d with a bisimulation-style
objective, then clusters embeddings via k-means + silhouette to estimate quotient size.

Bisimulation loss for pair (h1, h2):
    L = (d_embed(h1,h2) - W1(T(h1), T(h2)))^2
where W1 is approximated by the Wasserstein-1 distance between k-step continuation
observation distributions (first-dim projection, estimated from the same continuation
matrices used by the MMD method).

Comparison: loads existing MMD results from layer1_ground_truth CSVs.

Output: experiments/baselines/results/dbc_comparison.csv
Columns: env, seed, dbc_n_classes, mmd_n_classes, ground_truth, dbc_correct, mmd_correct
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Must be set before any parallel torch work starts.
# Containerized environments have severe overhead for tiny-matrix threading.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
LAYER1_DIR = ROOT / "experiments" / "layer1_ground_truth"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAYER1_DIR))

from continuous_quotient import collect_continuous_trajectories
from continuous_quotient.estimator import _extract_histories_and_continuations
from continuous_quotient.types import ContinuousTrajectoryDataset
from envs.tiger_continuous import TigerContinuousEnv
from envs.kcycle_continuous import KCycleContinuousEnv

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
RESULTS_DIR = ROOT / "experiments" / "baselines" / "results"
RESULTS_CSV = RESULTS_DIR / "dbc_comparison.csv"

# ---------------------------------------------------------------------------
# Hyperparameters — mirror the main experiments for fair comparison
# ---------------------------------------------------------------------------
TIGER_N_TRAJ = 1000
TIGER_MAX_STEPS = 20
TIGER_SEEDS = list(range(5))
TIGER_SIGMA = 0.5
TIGER_GT = 2

KCYCLE_N_TRAJ = 3000
KCYCLE_MAX_STEPS = 50
KCYCLE_SEEDS = list(range(3))
KCYCLE_NOISE = 0.1
KCYCLE_K_VALUES = [2, 3, 4]

# Shared continuation / history config (must match MMD baseline for fair comparison)
WINDOW_LENGTH = 2
HISTORY_MAX_LEN = 1
MIN_OCCURRENCES = 10
HISTORY_N_BINS = 8

# DBC encoder hyperparameters
EMB_DIM = 16
N_EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
GAMMA = 0.9


# ---------------------------------------------------------------------------
# DBC Encoder
# ---------------------------------------------------------------------------

class DBCEncoder(nn.Module):
    """Small MLP: obs_feature_dim → R^emb_dim."""

    def __init__(self, obs_dim: int, emb_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Core DBC training + quotient estimation
# ---------------------------------------------------------------------------

def run_dbc_baseline(
    env,
    n_traj: int,
    max_steps: int,
    seed: int,
    history_n_bins: int = HISTORY_N_BINS,
    n_epochs: int = N_EPOCHS,
    emb_dim: int = EMB_DIM,
    gamma: float = GAMMA,
) -> int:
    """Train DBC encoder and estimate quotient via k-means + silhouette.

    Returns estimated n_classes (int). Returns 1 if too few histories found.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = collect_continuous_trajectories(env, n_traj, max_steps, seed=seed)

    hc = _extract_histories_and_continuations(
        ContinuousTrajectoryDataset(trajectories=ds.trajectories),
        history_max_length=HISTORY_MAX_LEN,
        window_length=WINDOW_LENGTH,
        min_occurrences=MIN_OCCURRENCES,
        history_n_bins=history_n_bins,
    )

    if len(hc) < 2:
        return 1

    keys = list(hc.keys())
    n_hist = len(keys)

    # Input features: mean of continuation vectors as a proxy for history state.
    # This is a reasonable approximation since E[continuation | history] ≈ T(history).
    X = np.stack([hc[k].mean(axis=0) for k in keys], axis=0)  # (n_hist, d_cont)
    d = X.shape[1]

    # Pre-compute full pairwise W1 matrix once (O(n_hist^2) calls, not per epoch).
    # With n_hist typically < 20, this is fast and avoids repeated wasserstein_distance
    # calls inside the training loop.
    w1_matrix = np.zeros((n_hist, n_hist), dtype=np.float32)
    cont_1d = [hc[k][:, 0] for k in keys]  # first-dim projection per history
    for ia in range(n_hist):
        for ib in range(ia + 1, n_hist):
            d_w1 = wasserstein_distance(cont_1d[ia], cont_1d[ib])
            w1_matrix[ia, ib] = d_w1
            w1_matrix[ib, ia] = d_w1
    w1_tensor = torch.tensor(w1_matrix, dtype=torch.float32)  # (n_hist, n_hist)

    # Train encoder
    encoder = DBCEncoder(d, emb_dim)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
    xt = torch.tensor(X, dtype=torch.float32)

    rng = np.random.default_rng(seed)
    for epoch in range(n_epochs):
        idx_a = rng.integers(0, n_hist, BATCH_SIZE).astype(np.int64)
        idx_b = rng.integers(0, n_hist, BATCH_SIZE).astype(np.int64)

        # torch.from_numpy avoids a copy and is much faster than torch.tensor()
        ta = torch.from_numpy(idx_a)
        tb = torch.from_numpy(idx_b)

        za = encoder(xt[ta])
        zb = encoder(xt[tb])
        d_emb = torch.norm(za - zb, dim=-1)

        # Bisimulation targets: index pre-computed W1 matrix (no Python loop)
        bisim_target = w1_tensor[ta, tb]

        loss = ((d_emb - bisim_target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Extract embeddings and cluster via k-means + silhouette
    with torch.no_grad():
        embeddings = encoder(xt).numpy()  # (n_hist, emb_dim)

    best_k = 2
    best_score = -1.0
    max_k = min(n_hist, 9)

    for k in range(2, max_k):
        try:
            km = KMeans(n_clusters=k, n_init=3, random_state=seed)
            labels = km.fit_predict(embeddings)
            if len(set(labels)) < k:
                # KMeans collapsed — fewer distinct clusters than requested
                continue
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_k


# ---------------------------------------------------------------------------
# Load existing MMD results
# ---------------------------------------------------------------------------

def load_mmd_tiger(seeds: list[int]) -> dict[int, int]:
    """Return {seed: n_classes} for tiger from layer1 CSV."""
    csv_path = LAYER1_DIR / "results" / "tiger_continuous.csv"
    result: dict[int, int] = {}
    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found — MMD tiger results unavailable")
        return result
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            s = int(row["seed"])
            if s in seeds:
                result[s] = int(row["n_classes"])
    return result


def load_mmd_kcycle(K: int, seeds: list[int]) -> dict[int, int]:
    """Return {seed: n_classes} for K-cycle K=K from layer1 CSV."""
    csv_path = LAYER1_DIR / "results" / "kcycle_continuous.csv"
    result: dict[int, int] = {}
    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found — MMD kcycle results unavailable")
        return result
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if int(row["K"]) == K:
                s = int(row["seed"])
                if s in seeds:
                    result[s] = int(row["n_classes"])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    # ------------------------------------------------------------------
    # Tiger (5 seeds, ground truth = 2)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Tiger-Continuous  (ground truth: n_classes=2)")
    print("=" * 60)
    mmd_tiger = load_mmd_tiger(TIGER_SEEDS)

    for seed in TIGER_SEEDS:
        print(f"  Tiger seed={seed} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        env = TigerContinuousEnv(sigma=TIGER_SIGMA, episode_length=TIGER_MAX_STEPS)
        dbc_k = run_dbc_baseline(
            env, TIGER_N_TRAJ, TIGER_MAX_STEPS, seed=seed, history_n_bins=HISTORY_N_BINS
        )
        elapsed = time.perf_counter() - t0
        mmd_k = mmd_tiger.get(seed, None)
        record = {
            "env": "tiger_n2",
            "seed": seed,
            "dbc_n_classes": dbc_k,
            "mmd_n_classes": mmd_k if mmd_k is not None else "",
            "ground_truth": TIGER_GT,
            "dbc_correct": int(dbc_k == TIGER_GT),
            "mmd_correct": int(mmd_k == TIGER_GT) if mmd_k is not None else "",
        }
        records.append(record)
        print(f"dbc={dbc_k}  mmd={mmd_k}  gt={TIGER_GT}  ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # K-cycle K=2,3,4 (3 seeds each)
    # ------------------------------------------------------------------
    for K in KCYCLE_K_VALUES:
        print()
        print("=" * 60)
        print(f"K-cycle K={K}  (ground truth: n_classes={K})")
        print("=" * 60)
        mmd_kc = load_mmd_kcycle(K, KCYCLE_SEEDS)

        for seed in KCYCLE_SEEDS:
            print(f"  KCycle-K{K} seed={seed} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            env = KCycleContinuousEnv(K=K, obs_noise=KCYCLE_NOISE, episode_length=KCYCLE_MAX_STEPS)
            dbc_k = run_dbc_baseline(
                env, KCYCLE_N_TRAJ, KCYCLE_MAX_STEPS, seed=seed,
                history_n_bins=K,  # match MMD method: use K bins for K-cycle
            )
            elapsed = time.perf_counter() - t0
            mmd_k = mmd_kc.get(seed, None)
            record = {
                "env": f"kcycle_k{K}",
                "seed": seed,
                "dbc_n_classes": dbc_k,
                "mmd_n_classes": mmd_k if mmd_k is not None else "",
                "ground_truth": K,
                "dbc_correct": int(dbc_k == K),
                "mmd_correct": int(mmd_k == K) if mmd_k is not None else "",
            }
            records.append(record)
            print(f"dbc={dbc_k}  mmd={mmd_k}  gt={K}  ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    fieldnames = [
        "env", "seed", "dbc_n_classes", "mmd_n_classes",
        "ground_truth", "dbc_correct", "mmd_correct",
    ]
    with RESULTS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nResults written to {RESULTS_CSV}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("SUMMARY: DBC (neural bisimulation) vs MMD (this work)")
    print("=" * 65)
    print(f"{'Environment':<18}  {'DBC correct':>12}  {'MMD correct':>12}")
    print("-" * 48)

    envs_order = ["tiger_n2"] + [f"kcycle_k{K}" for K in KCYCLE_K_VALUES]
    dbc_total_correct = 0
    dbc_total = 0
    mmd_total_correct = 0
    mmd_total = 0

    for env_name in envs_order:
        env_recs = [r for r in records if r["env"] == env_name]
        if not env_recs:
            continue

        dbc_correct = sum(int(r["dbc_correct"]) for r in env_recs)
        dbc_n = len(env_recs)
        dbc_frac = dbc_correct / dbc_n

        mmd_vals = [r for r in env_recs if r["mmd_correct"] != ""]
        if mmd_vals:
            mmd_correct = sum(int(r["mmd_correct"]) for r in mmd_vals)
            mmd_n = len(mmd_vals)
            mmd_frac = mmd_correct / mmd_n
            mmd_str = f"{mmd_correct}/{mmd_n} ({mmd_frac:.0%})"
            mmd_total_correct += mmd_correct
            mmd_total += mmd_n
        else:
            mmd_str = "N/A"

        dbc_str = f"{dbc_correct}/{dbc_n} ({dbc_frac:.0%})"
        dbc_total_correct += dbc_correct
        dbc_total += dbc_n

        print(f"{env_name:<18}  {dbc_str:>12}  {mmd_str:>12}")

    print("-" * 48)
    dbc_overall = dbc_total_correct / dbc_total if dbc_total else 0.0
    mmd_overall = mmd_total_correct / mmd_total if mmd_total else 0.0
    print(f"{'OVERALL':<18}  "
          f"{dbc_total_correct}/{dbc_total} ({dbc_overall:.0%})  "
          f"{mmd_total_correct}/{mmd_total} ({mmd_overall:.0%})")
    print("=" * 65)
    print()
    print(f"MMD method  : {mmd_overall:.0%} accuracy  ({mmd_total_correct}/{mmd_total})")
    print(f"DBC baseline: {dbc_overall:.0%} accuracy  ({dbc_total_correct}/{dbc_total})")


if __name__ == "__main__":
    main()
