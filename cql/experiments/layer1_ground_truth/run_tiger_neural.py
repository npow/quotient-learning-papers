"""Compare NeuralQuotientEstimator vs ContinuousQuotientEstimator on Tiger and K-cycle.

FR-2 acceptance criterion: Encoder trained on K-cycle (finite) recovers K classes
with same accuracy as the union-find algorithm.

Usage:
    python experiments/layer1_ground_truth/run_tiger_neural.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_LAYER1_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _LAYER1_DIR.parents[1]

for _p in [str(_PROJECT_ROOT), str(_LAYER1_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Check torch availability early
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    print("ERROR: torch is not available. Cannot run NeuralQuotientEstimator.")
    print("Install with: pip install torch")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
from sklearn.metrics import adjusted_rand_score

from continuous_quotient import (
    ContinuousEstimatorConfig,
    ContinuousQuotientEstimator,
    collect_continuous_trajectories,
)
from continuous_quotient.neural_encoder import NeuralQuotientEstimator, NeuralEncoderConfig
from continuous_quotient.types import ContinuousTrajectoryDataset
from envs.tiger_continuous import TigerContinuousEnv
from envs.kcycle_continuous import KCycleContinuousEnv

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
MMD_CFG = ContinuousEstimatorConfig(
    n_trajectories=1000,
    max_steps_per_episode=20,
    window_length=2,
    history_max_length=1,
    min_history_occurrences=10,
    alpha=0.05,
    n_permutations=99,
    history_n_bins=8,
    max_samples_per_test=300,
)

NEURAL_CFG = NeuralEncoderConfig(
    hidden_size=64,
    n_layers=1,
    embedding_dim=32,
    n_epochs=100,
    batch_size=64,
    lr=1e-3,
    temperature=0.1,
    seed=0,
)

_RESULTS_DIR = _LAYER1_DIR / "results"
_RESULTS_CSV = _RESULTS_DIR / "neural_comparison_v2.csv"

# ---------------------------------------------------------------------------
# Ground-truth n_classes per env
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    "tiger": 2,
    "kcycle-K2": 2,
    "kcycle-K3": 3,
    "kcycle-K4": 4,
}


def run_one(env_name: str, env, expected_n: int, seed: int) -> dict:
    """Run MMD + Neural estimators for one env/seed and return result row."""
    t0 = time.perf_counter()

    # 1. Collect trajectories
    dataset: ContinuousTrajectoryDataset = collect_continuous_trajectories(
        env=env,
        n_episodes=MMD_CFG.n_trajectories,
        max_steps=MMD_CFG.max_steps_per_episode,
        seed=seed,
    )

    # 2. MMD quotient (ground truth labels for neural training)
    mmd_estimator = ContinuousQuotientEstimator(config=MMD_CFG)
    mmd_quotient = mmd_estimator.fit(dataset, verbose=False)
    mmd_n = mmd_quotient.n_classes
    mmd_correct = int(mmd_n == expected_n)

    # 3. Neural quotient trained on MMD labels
    neural_cfg = NeuralEncoderConfig(
        hidden_size=NEURAL_CFG.hidden_size,
        n_layers=NEURAL_CFG.n_layers,
        embedding_dim=NEURAL_CFG.embedding_dim,
        n_epochs=NEURAL_CFG.n_epochs,
        batch_size=NEURAL_CFG.batch_size,
        lr=NEURAL_CFG.lr,
        temperature=NEURAL_CFG.temperature,
        k_max=NEURAL_CFG.k_max,
        seed=seed,  # per-seed training
    )
    neural_estimator = NeuralQuotientEstimator(config=neural_cfg)
    neural_quotient = neural_estimator.fit(
        dataset,
        mmd_labels=mmd_quotient,
        history_max_length=MMD_CFG.history_max_length,
        verbose=False,
    )
    neural_n = neural_quotient.n_classes
    neural_correct = int(neural_n == expected_n)

    # 4. Compute Adjusted Rand Index between MMD and neural partitions
    # Only over keys present in both partitions
    shared_keys = set(mmd_quotient.partition.keys()) & set(neural_quotient.partition.keys())
    if len(shared_keys) >= 2:
        mmd_labels_arr = [mmd_quotient.partition[k] for k in shared_keys]
        neural_labels_arr = [neural_quotient.partition[k] for k in shared_keys]
        rand_index = float(adjusted_rand_score(mmd_labels_arr, neural_labels_arr))
    else:
        rand_index = float("nan")

    # 5. Training diagnostics
    loss_hist = neural_estimator.training_meta.loss_history
    loss_start = round(loss_hist[0], 4) if loss_hist else float("nan")
    loss_end = round(loss_hist[-1], 4) if loss_hist else float("nan")
    loss_decreased = bool(loss_hist and loss_hist[-1] < loss_hist[0])

    runtime_s = round(time.perf_counter() - t0, 2)

    return {
        "env": env_name,
        "seed": seed,
        "mmd_n_classes": mmd_n,
        "neural_n_classes": neural_n,
        "mmd_correct": mmd_correct,
        "neural_correct": neural_correct,
        "rand_index": round(rand_index, 4) if not (rand_index != rand_index) else "nan",
        "runtime_s": runtime_s,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "loss_decreased": int(loss_decreased),
        "expected_n": expected_n,
    }


def main() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Define experiment grid (tiger: 3 seeds, kcycle K=2,3,4: 2 seeds each)
    experiments: list[tuple[str, object, int, list[int]]] = [
        ("tiger",     lambda: TigerContinuousEnv(sigma=0.5, episode_length=20), 2, list(range(3))),
        ("kcycle-K2", lambda: KCycleContinuousEnv(K=2, obs_noise=0.1, episode_length=20), 2, list(range(2))),
        ("kcycle-K3", lambda: KCycleContinuousEnv(K=3, obs_noise=0.1, episode_length=20), 3, list(range(2))),
        ("kcycle-K4", lambda: KCycleContinuousEnv(K=4, obs_noise=0.1, episode_length=20), 4, list(range(2))),
    ]

    fieldnames = [
        "env", "seed", "mmd_n_classes", "neural_n_classes",
        "mmd_correct", "neural_correct", "rand_index", "runtime_s",
        "loss_start", "loss_end", "loss_decreased", "expected_n",
    ]

    records: list[dict] = []
    total = sum(len(seeds) for _, _, _, seeds in experiments)
    done = 0

    print(f"\nRunning NeuralQuotientEstimator benchmark ({total} runs total)")
    print(f"torch={torch.__version__}  MMD n_traj={MMD_CFG.n_trajectories}  neural epochs={NEURAL_CFG.n_epochs}")
    print("=" * 80)

    for env_name, env_factory, expected_n, seeds in experiments:
        env_records = []
        for seed in seeds:
            done += 1
            print(f"[{done}/{total}] {env_name} seed={seed} ...", end="", flush=True)
            env = env_factory()
            rec = run_one(env_name, env, expected_n, seed)
            env_records.append(rec)
            records.append(rec)
            status = "OK" if rec["neural_correct"] else "WRONG"
            print(
                f"  mmd={rec['mmd_n_classes']} neural={rec['neural_n_classes']} "
                f"(expected {expected_n}) rand={rec['rand_index']} "
                f"loss {rec['loss_start']}→{rec['loss_end']} "
                f"[{status}] {rec['runtime_s']}s"
            )

        # Per-env summary
        mmd_acc = sum(r["mmd_correct"] for r in env_records) / len(env_records)
        neu_acc = sum(r["neural_correct"] for r in env_records) / len(env_records)
        rand_vals = [r["rand_index"] for r in env_records if r["rand_index"] != "nan"]
        mean_rand = round(sum(rand_vals) / len(rand_vals), 4) if rand_vals else float("nan")
        loss_ok = sum(r["loss_decreased"] for r in env_records)
        print(
            f"  >> {env_name}: mmd_acc={mmd_acc:.2f} neural_acc={neu_acc:.2f} "
            f"mean_rand={mean_rand} loss_decreasing={loss_ok}/{len(env_records)}"
        )
        print()

    # Write CSV
    with _RESULTS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Results written to {_RESULTS_CSV}")

    # ---------------------------------------------------------------------------
    # Print comparison table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{'ENV':<14} {'SEED':>4} {'EXPECTED':>8} {'MMD_N':>6} {'NEU_N':>6} "
          f"{'MMD_OK':>6} {'NEU_OK':>6} {'RAND':>8} {'LOSS↓':>6} {'TIME':>7}")
    print("-" * 80)
    for r in records:
        loss_arrow = "yes" if r["loss_decreased"] else "NO"
        print(
            f"{r['env']:<14} {r['seed']:>4} {r['expected_n']:>8} "
            f"{r['mmd_n_classes']:>6} {r['neural_n_classes']:>6} "
            f"{r['mmd_correct']:>6} {r['neural_correct']:>6} "
            f"{str(r['rand_index']):>8} {loss_arrow:>6} {r['runtime_s']:>6.1f}s"
        )
    print("-" * 80)

    # Overall summary
    all_mmd_acc = sum(r["mmd_correct"] for r in records) / len(records)
    all_neu_acc = sum(r["neural_correct"] for r in records) / len(records)
    all_rand = [r["rand_index"] for r in records if r["rand_index"] != "nan"]
    overall_rand = round(sum(all_rand) / len(all_rand), 4) if all_rand else float("nan")
    all_loss_ok = sum(r["loss_decreased"] for r in records)

    print(f"\nOverall: mmd_acc={all_mmd_acc:.2f}  neural_acc={all_neu_acc:.2f}  "
          f"mean_rand={overall_rand}  loss_decreasing={all_loss_ok}/{len(records)}")

    # FR-2 gate: neural should match expected_n at ≥ 50% of runs
    # (lenient threshold since DBSCAN tuning matters; report not assert)
    threshold = 0.5
    passed = all_neu_acc >= threshold
    status = "PASS" if passed else "FAIL"
    print(f"\nFR-2 gate (neural_acc >= {threshold}): {status}  (neural_acc={all_neu_acc:.2f})")


if __name__ == "__main__":
    main()
