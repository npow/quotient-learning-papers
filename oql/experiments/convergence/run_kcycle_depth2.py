"""Depth-2 exact-recovery benchmark: deterministic-emission K-cycle at L=2.

Every other exact-recovery experiment in the paper uses history_max_length=1,
where the tested histories are single observation symbols and the estimator
cannot over-fragment (each symbol starts in its own component; the only
possible error is a false merge).  This benchmark exercises the depth
dimension with a target that is still known analytically.

Construction: KCyclePOMDP with obs_accuracy=1.0.  Emissions are then
deterministic, so the belief after any history is a point mass on the state
of the last observation, and the continuation law of a history depends only
on its final symbol.  The quotient of H_{<=2} therefore has exactly K classes
(class i = {(i), (i-1, i)}), while the estimator tests 2K+1 histories
(root + K singletons + K consecutive pairs).  Exact recovery now requires
K cross-length merges -- (i-1, i) with (i) -- and admits both failure modes:
a false split leaves a pair history in its own class (over-fragmentation,
impossible at L=1), and a false merge joins two different cycle positions.

Sweep: K in {4, 8, 16} x N in {1000, 2000, 5000} x 5 seeds,
history_max_length=2, window_length=2, min_history_occurrences=100.

Writes results/kcycle_depth2.csv and results/kcycle_depth2.summary.json.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from environments.layer1_discrete.kcycle import KCyclePOMDP
from quotient_learning.config import seed_everything
from quotient_learning.estimator import QuotientEstimator

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

KS = [4, 8, 16]
NS = [1000, 2000, 5000]
N_SEEDS = 5
MAX_STEPS = 20


def main() -> None:
    rows: list[dict] = []
    t0 = time.time()
    for k in KS:
        for n_traj in NS:
            for seed in range(N_SEEDS):
                seed_everything(seed)
                env = KCyclePOMDP(k, obs_accuracy=1.0, seed=seed)
                est = QuotientEstimator(
                    env=env,
                    n_trajectories=n_traj,
                    window_length=2,
                    history_max_length=2,
                    min_history_occurrences=100,
                    alpha=0.01,
                    max_steps_per_episode=MAX_STEPS,
                )
                dataset = est.collect(seed=seed)
                q = est.estimate(dataset)
                n_tested = len(q.partition)
                rows.append({
                    "k": k,
                    "n_trajectories": n_traj,
                    "seed": seed,
                    "learned_n_classes": q.n_classes,
                    "n_tested_histories": n_tested,
                    "exact_recovery": int(q.n_classes == k),
                })
                print(f"  K={k} N={n_traj} seed={seed}: "
                      f"n={q.n_classes} (target {k}), "
                      f"tested={n_tested}", flush=True)

    csv_path = RESULTS_DIR / "kcycle_depth2.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    exact = {}
    for k in KS:
        for n_traj in NS:
            sub = [r for r in rows if r["k"] == k
                   and r["n_trajectories"] == n_traj]
            exact[f"K={k},N={n_traj}"] = (
                sum(r["exact_recovery"] for r in sub), len(sub))
    summary = {
        "wall_time_s": time.time() - t0,
        "n_rows": len(rows),
        "exact_recovery_per_config": {k: f"{a}/{b}"
                                      for k, (a, b) in exact.items()},
    }
    (RESULTS_DIR / "kcycle_depth2.summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}; elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
