"""Noisy depth-2 benchmark sweep (review finding E1).

Sweeps the noise level eta of NoisyDepth2KCycle under the certified
configuration, scoring exact recovery against the analytically known suffix
partition and comparing the empirical sample threshold against the m* bound.

Usage:
    python experiments/separation/run_noisy_depth2.py \
        --eta 0.0 0.1 0.2 0.3 --seeds 0-4 [--out .../results_certified]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from environments.layer2_memory.noisy_depth2_kcycle import NoisyDepth2KCycle
from experiments.scoring import merge_result_rows, pair_disagreement_rate, partitions_agree
from quotient_learning.config import CertifiedConfig, seed_everything
from quotient_learning.equivalence import GTestEquivalenceTest
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.stats import required_samples

DELTA = 0.05
BETA = 0.05
M = 2
K = 1          # window length; the exact laws are next-symbol laws
L = 2          # depth-2 contexts required by construction
TAU = L + K    # smallest tau with K*tau >= L + K (non-overlapping blocks)
N_TRAJ = 16000
T_STEPS = 60


def run_one(eta: float, seed: int, out_dir: Path) -> dict:
    seed_everything(seed)
    env = NoisyDepth2KCycle(m=M, eta=eta, seed=seed)
    eps = env.epsilon_min(max_length=L)
    alphabet = env.continuation_alphabet(K)
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
        epsilon_min=eps,
        power_beta=BETA,
    )
    q = est.run(seed=seed)
    truth = env.target_partition(max_length=L)
    row = {
        "eta": eta,
        "seed": seed,
        "eps_min": round(eps, 6),
        "m_star": required_samples(eps, DELTA, BETA, alphabet)
        if eps > 0 else "",
        "n_classes": q.n_classes,
        "n_true_classes": len(set(truth.values())),
        "exact_recovery": int(partitions_agree(q.partition, truth)),
        "pair_disagreement": round(pair_disagreement_rate(q.partition, truth), 4),
        "undetermined_rate": round(q.undetermined_rate, 4),
    }
    cfg = CertifiedConfig(
        k=K, tau=TAU, test="G_RAW_FIXED_ALPHABET", alpha=DELTA,
        actual_stride=K * TAU,
        epsilon_min=eps if eps > 0 else 1.0, beta=BETA, seeds=[seed],
        command=(
            f"python experiments/separation/run_noisy_depth2.py "
            f"--eta {eta} --seeds {seed}"
        ),
    )
    cfg.capture_env_pins()
    cfg.save_sidecar(out_dir / f"noisy_depth2_eta{eta}_seed{seed}.certified.json")
    return row


def parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eta", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5])
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results_certified"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for eta in args.eta:
        for seed in parse_seeds(args.seeds):
            row = run_one(eta, seed, out_dir)
            rows.append(row)
            print(f"  eta={eta} seed={seed} exact={row['exact_recovery']} "
                  f"n={row['n_classes']}/{row['n_true_classes']} "
                  f"und={row['undetermined_rate']}", flush=True)

    csv_path = out_dir / "noisy_depth2_sweep.csv"
    all_rows = merge_result_rows(csv_path, rows, ('eta', 'seed'))
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    lines = [
        "# Noisy depth-2 sweep (certified configuration)",
        "",
        f"m={M}, k={K}, L={L}, N={N_TRAJ}, T={T_STEPS}, delta={DELTA}, "
        f"beta={BETA}, gate eps = exact eps_min(eta).",
        "",
        "| eta | eps_min | exact recovery | pair disagreement (mean) | undetermined (mean) |",
        "|---|---|---|---|---|",
    ]
    summary_etas = sorted({str(r["eta"]) for r in all_rows}, key=float)
    for eta in summary_etas:
        sub = [r for r in all_rows if str(r["eta"]) == eta]
        rec = np.mean([float(r["exact_recovery"]) for r in sub])
        dis = np.mean([float(r["pair_disagreement"]) for r in sub])
        und = np.mean([float(r["undetermined_rate"]) for r in sub])
        lines.append(
            f"| {eta} | {sub[0]['eps_min']} | {rec:.2f} ({int(sum(r['exact_recovery'] for r in sub))}/{len(sub)}) "
            f"| {dis:.4f} | {und:.3f} |")
    (out_dir / "noisy_depth2_summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {csv_path}; elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
