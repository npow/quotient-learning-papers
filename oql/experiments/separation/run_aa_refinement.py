"""Action-aware refinement demonstration (paper Remark 1, certified).

Runs QuotientEstimator on AASeparationPOMDP in both variants under the
certified configuration: the observation-only quotient must merge the two
marker contexts (identical next-observation marginals), the action-aware
quotient must separate them (disjoint (a, o') supports, TV = 1). This is
the empirical exercise of the division argument that connects the suffix
quotient to probe-exact equivalence.

Usage:
    python experiments/separation/run_aa_refinement.py --seeds 0-4
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from environments.layer1_discrete.aa_separation import AASeparationPOMDP
from experiments.scoring import merge_result_rows
from quotient_learning.config import CertifiedConfig, seed_everything
from quotient_learning.equivalence import GTestEquivalenceTest
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.types import ObservationHistory

DELTA = 0.05
BETA = 0.05
EPS_MIN = 0.5
K, L, TAU = 1, 1, 2  # k*tau >= L + k
N_TRAJ = 4000
T_STEPS = 20


def run_one(seed: int, action_aware: bool, out_dir: Path) -> dict:
    seed_everything(seed)
    env = AASeparationPOMDP(seed=seed)
    alphabet = env.continuation_alphabet(K, action_aware=action_aware)
    est = QuotientEstimator(
        env=env,
        n_trajectories=N_TRAJ,
        window_length=K,
        history_max_length=L,
        min_history_occurrences=30,
        max_steps_per_episode=T_STEPS,
        equivalence_test=GTestEquivalenceTest(alphabet=alphabet),
        delta=DELTA,
        window_stride=K * TAU,
        epsilon_min=EPS_MIN,
        power_beta=BETA,
        action_aware=action_aware,
    )
    q = est.run(seed=seed)
    h2, h3 = ObservationHistory((2,)), ObservationHistory((3,))
    variant = "action_aware" if action_aware else "obs_only"
    expected = env.metadata[f"ground_truth_n_classes_{variant}"]
    row = {
        "variant": variant,
        "seed": seed,
        "n_classes": q.n_classes,
        "expected": expected,
        "markers_separated": int(q.partition.get(h2) != q.partition.get(h3)),
        "undetermined_rate": round(q.undetermined_rate, 4),
    }
    cfg = CertifiedConfig(
        k=K, tau=TAU, test="G_RAW_FIXED_ALPHABET", alpha=DELTA,
        epsilon_min=EPS_MIN, beta=BETA, seeds=[seed],
        actual_stride=K * TAU,
        command=(
            f"python experiments/separation/run_aa_refinement.py --seeds {seed}"
        ),
    )
    cfg.capture_env_pins()
    cfg.save_sidecar(out_dir / f"aa_refinement_{variant}_seed{seed}.certified.json")
    return row


def parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results_certified"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for action_aware in (False, True):
        for seed in parse_seeds(args.seeds):
            row = run_one(seed, action_aware, out_dir)
            rows.append(row)
            print(f"  {row['variant']} seed={seed} n={row['n_classes']}/"
                  f"{row['expected']} sep={row['markers_separated']}", flush=True)

    csv_path = out_dir / "aa_refinement.csv"
    all_rows = merge_result_rows(csv_path, rows, ("variant", "seed"))
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {csv_path}; elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
