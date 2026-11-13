"""Certified-vs-exploratory head-to-head on the position-flip chain (B4).

The position-flip chain's two length-one contexts have continuation laws
Bernoulli(0.9) and Bernoulli(0.1) at episode position r = 0 (TV gap 0.8),
but identical position-pooled laws (both exactly Bernoulli(0.5)).  The
exploratory mode scans overlapping windows across all positions, so it
estimates the pooled law and merges the pair; the merge is not a
calibration accident and does not vanish with more data --- pooling
positions changes the estimand.  The certified path at fixed r = 0
separates the pair under its finite-sample certificate.

Usage:
    python experiments/main_table/run_headtohead.py [--seeds 0-9] \
        [--out experiments/main_table/results_headtohead]
"""

from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path

from environments.layer1_discrete.position_flip import PositionFlipChain
from experiments.main_table.run_certified import BETA, DELTA, parse_seeds
from quotient_learning.config import CertifiedConfig, seed_everything
from quotient_learning.equivalence import (
    GTestEquivalenceTest,
    NotCertifiedWarning,
    TVThresholdEquivalenceTest,
)
from quotient_learning.estimator import QuotientEstimator

EPS_MIN = 0.75          # <= analytic r=0 TV gap 0.8
EXPECTED_R0_CLASSES = 2  # episode-position estimand at r=0
N_TRAJ_CERTIFIED = 4000
N_TRAJ_EXPLORATORY = 400  # ~400 x 40 overlapping windows ~ same order of blocks
MAX_STEPS = 40


def run_certified_path(seed: int, out_dir: Path) -> dict:
    seed_everything(seed)
    env = PositionFlipChain(seed=seed)
    alphabet = env.continuation_alphabet(1)
    est = QuotientEstimator(
        env=env,
        equivalence_test=TVThresholdEquivalenceTest(alphabet, EPS_MIN),
        delta=DELTA, epsilon_min=EPS_MIN, power_beta=BETA,
        certified_independent=True, selection_fraction=0.5,
        independent_context_end=0, include_root=False,
        n_trajectories=N_TRAJ_CERTIFIED, window_length=1,
        history_max_length=1, min_history_occurrences=10,
        max_steps_per_episode=MAX_STEPS,
    )
    q = est.run(seed=seed)
    try:
        recorded_out = out_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        recorded_out = out_dir.resolve()
    cfg = CertifiedConfig(
        k=1, tau=1, test="TV_FIXED_ALPHABET", alpha=DELTA,
        epsilon_min=EPS_MIN, beta=BETA, seeds=[seed],
        selection_fraction=0.5, context_end=0,
        policy_id="uniform_random",
        n_trajectories=N_TRAJ_CERTIFIED, history_max_length=1,
        min_history_occurrences=10, max_steps_per_episode=MAX_STEPS,
        command=(
            "python experiments/main_table/run_headtohead.py "
            f"--seeds {seed} --out {recorded_out}"
        ),
    )
    cfg.capture_env_pins()
    cfg.save_sidecar(out_dir / f"position_flip_seed{seed}.certified.json")
    return {
        "path": "certified_r0",
        "seed": seed,
        "n_classes": q.n_classes,
        "undetermined_rate": round(q.undetermined_rate, 4),
    }


def run_exploratory_path(seed: int) -> dict:
    seed_everything(seed)
    env = PositionFlipChain(seed=seed)
    est = QuotientEstimator(
        env=env,
        equivalence_test=GTestEquivalenceTest(),
        n_trajectories=N_TRAJ_EXPLORATORY, window_length=1,
        history_max_length=1, min_history_occurrences=50,
        alpha=0.05, max_steps_per_episode=MAX_STEPS,
        include_root=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotCertifiedWarning)
        q = est.run(seed=seed)
    return {
        "path": "exploratory_pooled",
        "seed": seed,
        "n_classes": q.n_classes,
        "undetermined_rate": round(q.undetermined_rate, 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent / "results_headtohead"),
    )
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    rows: list[dict] = []
    t0 = time.time()
    for seed in seeds:
        for row in (run_certified_path(seed, out_dir), run_exploratory_path(seed)):
            rows.append(row)
            print(f"  {row['path']} seed={seed} n_classes={row['n_classes']} "
                  f"und={row['undetermined_rate']}", flush=True)

    csv_path = out_dir / "headtohead.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    cert = [r for r in rows if r["path"] == "certified_r0"]
    expl = [r for r in rows if r["path"] == "exploratory_pooled"]
    cert_ok = sum(r["n_classes"] == EXPECTED_R0_CLASSES for r in cert)
    expl_merged = sum(r["n_classes"] < EXPECTED_R0_CLASSES for r in expl)
    print(f"\ncertified: {cert_ok}/{len(cert)} seeds recover the r=0 quotient "
          f"({EXPECTED_R0_CLASSES} classes)")
    print(f"exploratory: {expl_merged}/{len(expl)} seeds merge the pair "
          "(pooled estimand)")
    print(f"Wrote {csv_path}; elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
