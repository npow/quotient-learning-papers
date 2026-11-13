"""Merge-bearing abstention sweep on the noisy depth-2 chain (S10).

The existing sweep (run_abstention_sweep.py) uses the 8-cycle, whose
quotient is all-singleton -- there, full abstention is observationally
identical to correct recovery.  This sweep uses a merge-bearing quotient:
at r = 0, k = L = 1 on NoisyDepth2KCycle(m=2, eta=0.1, nu=0.2), the three
length-one contexts induce exactly TWO classes -- (1,) and (2,) share the
L_odd row [0.8, 0.1, 0.1] verbatim (a genuine SAME pair), while (0,) has
the posterior-mixture law [1/21, 10/21, 10/21] at TV ~0.7524 from L_odd.

Degradation therefore has a visible signature: when the budget starves the
m* gate, the SAME pair (1,)~(2,) abstains instead of merging and the
reported partition OVER-SPLITS to 3 classes instead of the analytic 2.
The safety property is that no budget and seed ever reports an incorrect
merge (a class containing (0,) together with (1,) or (2,)).

Gate: K = 3 contexts, alpha_0 = delta/(2 K^2) = 2.778e-3, eps = 0.5
(<= 0.7523, the recorded analytic gap of run_certified.py), beta = 0.05,
d = 3, m* = 278.  Context masses are (0,) ~0.4565 and (1,), (2,) ~0.2717
each, so the last contexts to reach the gate are (1,) and (2,): expected
transition near N ~ 2 m* / 0.2717 ~ 2,050 episodes.  The budget grid spans
full abstention (N = 1,000) to zero abstention (N = 2,800).

Usage:
    python experiments/main_table/run_sweep_depth2.py \
        [--seeds 0-9] [--out experiments/main_table/results_sweep_depth2]

Outputs: sweep CSV, one certified sidecar per (N, seed), and summary.md.
The paper figure is generated separately by experiments/figures/sweep_depth2.py.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from collections import Counter
from pathlib import Path

from environments.layer2_memory.noisy_depth2_kcycle import NoisyDepth2KCycle
from experiments.main_table.run_certified import (
    BETA,
    DELTA,
    eps_min_for,
    parse_seeds,
)
from quotient_learning.config import CertifiedConfig, seed_everything
from quotient_learning.equivalence import TVThresholdEquivalenceTest
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.stats import required_samples
from quotient_learning.types import ObservationHistory

ENV_NAME = "noisy_depth2"
M_SYMBOLS, ETA, NU = 2, 0.1, 0.2
EXPECTED_CLASSES = 2           # analytic r=0 quotient: {(1,),(2,)} and {(0,)}
OVERSPLIT_CLASSES = 3          # SAME pair abstained: all-singleton report
N_CONTEXTS = 3
ALPHABET_SIZE = 3
# Analytic truth partition at r=0, k=1 (see run_certified.py noisy_depth2
# entry and the module docstring): (1,) and (2,) share L_odd exactly.
TRUTH = {
    ObservationHistory((0,)): 0,
    ObservationHistory((1,)): 1,
    ObservationHistory((2,)): 1,
}
BUDGETS = [1000, 1400, 1800, 2000, 2100, 2200, 2400, 2800]
MAX_STEPS = 4                  # r + k = 1 step needed; small budget-friendly
Q = 30                         # selection threshold, as in run_certified.py


def m_star() -> int:
    """The gate value at the sweep configuration (K=3 contexts)."""
    return required_samples(
        eps_min_for(ENV_NAME), DELTA / (2.0 * N_CONTEXTS**2), BETA, ALPHABET_SIZE
    )


def incorrect_merges(partition: dict[ObservationHistory, int]) -> int:
    """Count context pairs merged by the estimator but distinct in truth."""
    common = [h for h in partition if h in TRUTH]
    return sum(
        1
        for a, b in itertools.combinations(common, 2)
        if partition[a] == partition[b] and TRUTH[a] != TRUTH[b]
    )


def run_one(n_trajectories: int, seed: int, out_dir: Path) -> dict:
    seed_everything(seed)
    env = NoisyDepth2KCycle(m=M_SYMBOLS, eta=ETA, nu=NU, seed=seed)
    eps_min = eps_min_for(ENV_NAME)
    alphabet = env.continuation_alphabet(1)
    est = QuotientEstimator(
        env=env,
        equivalence_test=TVThresholdEquivalenceTest(alphabet, eps_min),
        delta=DELTA, epsilon_min=eps_min, power_beta=BETA,
        certified_independent=True, selection_fraction=0.5,
        independent_context_end=0, include_root=False,
        n_trajectories=n_trajectories, window_length=1,
        history_max_length=1, min_history_occurrences=Q,
        max_steps_per_episode=MAX_STEPS,
    )
    q = est.run(seed=seed)
    same_pair = (
        q.partition.get(ObservationHistory((1,)))
        == q.partition.get(ObservationHistory((2,)))
        and q.partition.get(ObservationHistory((1,))) is not None
    )
    row = {
        "env": ENV_NAME,
        "n_trajectories": n_trajectories,
        "seed": seed,
        "n_classes": q.n_classes,
        "expected": EXPECTED_CLASSES,
        "undetermined_rate": round(q.undetermined_rate, 4),
        "n_boundary_flags": len(q.boundary_flags),
        "incorrect_merges": incorrect_merges(q.partition),
        "same_pair_merged": int(same_pair),
    }
    try:
        recorded_out = out_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        recorded_out = out_dir.resolve()
    cfg = CertifiedConfig(
        k=1, tau=1, test="TV_FIXED_ALPHABET", alpha=DELTA,
        epsilon_min=eps_min, beta=BETA, seeds=[seed],
        selection_fraction=0.5, context_end=0,
        policy_id="uniform_random",
        n_trajectories=n_trajectories, history_max_length=1,
        min_history_occurrences=Q, max_steps_per_episode=MAX_STEPS,
        command=(
            "python experiments/main_table/run_sweep_depth2.py "
            f"--seeds {seed} --out {recorded_out}"
        ),
    )
    cfg.capture_env_pins()
    cfg.save_sidecar(out_dir / f"{ENV_NAME}_N{n_trajectories}_seed{seed}.certified.json")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent / "results_sweep_depth2"),
    )
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    rows: list[dict] = []
    t0 = time.time()
    wrong = 0
    for n in BUDGETS:
        for seed in seeds:
            row = run_one(n, seed, out_dir)
            rows.append(row)
            wrong += row["incorrect_merges"]
            print(
                f"  N={n} seed={seed} n_classes={row['n_classes']} "
                f"und={row['undetermined_rate']} "
                f"bad_merges={row['incorrect_merges']}",
                flush=True,
            )

    csv_path = out_dir / "sweep_depth2.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Merge-bearing abstention sweep (noisy depth-2 chain, S10)",
        "",
        f"r=0, k=L=1, {N_CONTEXTS} contexts, analytic quotient "
        f"{EXPECTED_CLASSES} classes ((1,) ~ (2,) is a genuine SAME pair); "
        f"eps_min={eps_min_for(ENV_NAME)} <= analytic gap 0.7523; "
        f"delta={DELTA}, beta={BETA}, alpha_0=delta/(2K^2)="
        f"{DELTA / (2 * N_CONTEXTS**2):.3e}, d={ALPHABET_SIZE}, "
        f"m*={m_star()}; q={Q}; {len(seeds)} seeds per budget; context "
        "masses (0,)~0.4565, (1,)=(2,)~0.2717 -> gate transition expected "
        "near N~2,050.",
        "",
        "Degradation signature: the SAME pair abstaining OVER-SPLITS the "
        "report to 3 classes; an incorrect merge would join (0,) with "
        "(1,) or (2,).",
        "",
        "| N | class counts {n: seeds} | mean und. | min-max und. | "
        "incorrect merges |",
        "|---|---|---|---|---|",
    ]
    for n in BUDGETS:
        sub = [r for r in rows if r["n_trajectories"] == n]
        dist = Counter(r["n_classes"] for r in sub)
        dist_s = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
        us = [r["undetermined_rate"] for r in sub]
        bad = sum(r["incorrect_merges"] for r in sub)
        lines.append(
            f"| {n} | {dist_s} | {sum(us) / len(us):.3f} | "
            f"{min(us):.3f}-{max(us):.3f} | {bad} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"\nWrote {csv_path}; elapsed {time.time() - t0:.1f}s")
    if wrong:
        print(f"SAFETY VIOLATION: {wrong} incorrect merges across the sweep")
        return 1
    print(f"safety check: no incorrect merges in {len(rows)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
