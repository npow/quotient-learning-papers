"""Nested-context certified recovery on the noisy depth-2 chain (S4).

Exercises the analysis's distinctive nested case: TWO eligible context
lengths at one endpoint.  At r = 1 with L = 2 and k = 1 on
NoisyDepth2KCycle(m=2, eta=0.1, nu=0.2), the candidate set holds the three
length-1 contexts (b,) ending at index 1 AND the nine positive-mass
length-2 contexts (a, b) spanning indices 0..1.  Every trajectory matching
(a, b) also feeds its length-1 suffix (b,) -- the nesting Theorem 1's
one-count-at-a-time conditioning is built for -- and the paper treats the
twelve strings as distinct elements of the candidate set.

Analytic quotient (exact laws; the chain starts at the stationary pair
distribution, so suffix laws equal endpoint laws at every r):

  - law(a, b) = transition_row(a, b) exactly (two symbols pin the pair
    state), and law(b,) mixes rows over the stationary posterior of the
    hidden predecessor.
  - L_odd class (8 members): (1,), (2,), and every (a, b) with b != 0 --
    all share the row [0.8, 0.1, 0.1] verbatim, including genuine
    length-1-with-length-2 merges such as (1,) with (0, 1).
  - (0,)   -> [1/21, 10/21, 10/21] (posterior mixture)   -- singleton.
  - (0, 0) -> uniform [1/3, 1/3, 1/3]                    -- singleton.
  - (1, 0) -> [eta/3, eta/3, 1-2eta/3] = [.033, .033, .933] -- singleton.
  - (2, 0) -> [.033, .933, .033]                          -- singleton.

  Five classes over twelve contexts.  The minimal between-class TV over ALL
  eligible context pairs is TV(law(0,), law(0,0)) = 2/7 ~= 0.2857, so eps
  must satisfy eps <= 2/7; this runner uses eps = 0.28.

Gate: K = 12, alpha_0 = delta/(2 K^2) = 1.736e-4, beta = 0.05, d = 3,
m* = 1167.  The rarest context (0, 0) has visitation mass ~0.02174, so the
inference half must be large: with N = 140,000 episodes its expected count
is ~1,522 (9+ standard deviations above m*), sized for zero abstention at
q = 30 <= m*.

Usage:
    python experiments/main_table/run_nested.py [--seeds 0-19] \
        [--out experiments/main_table/results_nested]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

from environments.layer2_memory.noisy_depth2_kcycle import NoisyDepth2KCycle
from experiments.main_table.run_certified import BETA, DELTA, parse_seeds
from experiments.scoring import partitions_agree
from quotient_learning.config import CertifiedConfig, seed_everything
from quotient_learning.equivalence import TVThresholdEquivalenceTest
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.stats import required_samples
from quotient_learning.types import ObservationHistory

ENV_NAME = "noisy_depth2"
M_SYMBOLS, ETA, NU = 2, 0.1, 0.2
R_ENDPOINT = 1                 # both lengths 1 and 2 eligible (r >= 1)
L_MAX = 2                      # nested candidate lengths {1, 2}
EXPECTED_CLASSES = 5
N_CONTEXTS = 12                # 3 length-1 + 9 positive-mass length-2 words
MIN_GAP = 2.0 / 7.0            # analytic minimal between-class TV
EPS_MIN = 0.28                 # <= 2/7 = 0.285714...
ALPHABET_SIZE = 3              # d = |O|^k, k = 1
M_STAR = required_samples(
    EPS_MIN, DELTA / (2.0 * N_CONTEXTS**2), BETA, ALPHABET_SIZE
)
N_TRAJ = 140_000               # rarest context (0,0), mass ~0.0217: inference
                               # count ~1522 >> m* = 1167 (zero abstention)
MAX_STEPS = 3                  # needs r + k = 2 steps; one step of headroom
Q = 30                         # selection threshold (q <= m*)

_TOL = 1e-9


def analytic_partition(env: NoisyDepth2KCycle) -> dict[ObservationHistory, int]:
    """Exact quotient over the 12 eligible contexts at r=1, from exact laws."""
    part = env.target_partition(L_MAX)
    assert part is not None
    return part


def check_analytic_quotient(env: NoisyDepth2KCycle) -> None:
    """Assert the derivation in the module docstring against the exact laws."""
    part = analytic_partition(env)
    assert len(part) == N_CONTEXTS, f"expected {N_CONTEXTS} contexts, got {len(part)}"
    assert len(set(part.values())) == EXPECTED_CLASSES
    # The L_odd class contains both length-1 and length-2 contexts (nested
    # merge across lengths), e.g. (1,) with (0, 1).
    assert part[ObservationHistory((1,))] == part[ObservationHistory((0, 1))]
    assert part[ObservationHistory((1,))] == part[ObservationHistory((2,))]
    # Singletons.
    for lone in ((0,), (0, 0), (1, 0), (2, 0)):
        cid = part[ObservationHistory(lone)]
        assert sum(1 for c in part.values() if c == cid) == 1, f"{lone} not singleton"
    # Minimal between-class TV over ALL eligible contexts, and the eps bound.
    words = [h.observations for h in part]
    laws = {w: env.exact_continuation_law(w) for w in words}
    min_gap = min(
        env._tv(laws[a], laws[b])
        for a, b in itertools.combinations(words, 2)
        if part[ObservationHistory(a)] != part[ObservationHistory(b)]
    )
    assert abs(min_gap - MIN_GAP) < _TOL, f"min between-class TV {min_gap} != 2/7"
    assert abs(env.epsilon_min(L_MAX) - MIN_GAP) < _TOL
    assert EPS_MIN <= min_gap, f"eps_min={EPS_MIN} exceeds analytic gap {min_gap}"


def run_one(seed: int, out_dir: Path) -> dict:
    seed_everything(seed)
    env = NoisyDepth2KCycle(m=M_SYMBOLS, eta=ETA, nu=NU, seed=seed)
    alphabet = env.continuation_alphabet(1)
    est = QuotientEstimator(
        env=env,
        equivalence_test=TVThresholdEquivalenceTest(alphabet, EPS_MIN),
        delta=DELTA, epsilon_min=EPS_MIN, power_beta=BETA,
        certified_independent=True, selection_fraction=0.5,
        independent_context_end=R_ENDPOINT, include_root=False,
        n_trajectories=N_TRAJ, window_length=1,
        history_max_length=L_MAX, min_history_occurrences=Q,
        max_steps_per_episode=MAX_STEPS,
    )
    q = est.run(seed=seed)
    truth = analytic_partition(env)
    exact = partitions_agree(q.partition, truth)
    try:
        recorded_out = out_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        recorded_out = out_dir.resolve()
    cfg = CertifiedConfig(
        k=1, tau=1, test="TV_FIXED_ALPHABET", alpha=DELTA,
        epsilon_min=EPS_MIN, beta=BETA, seeds=[seed],
        selection_fraction=0.5, context_end=R_ENDPOINT,
        policy_id="uniform_random",
        n_trajectories=N_TRAJ, history_max_length=L_MAX,
        min_history_occurrences=Q, max_steps_per_episode=MAX_STEPS,
        command=(
            "python experiments/main_table/run_nested.py "
            f"--seeds {seed} --out {recorded_out}"
        ),
    )
    cfg.capture_env_pins()
    cfg.save_sidecar(out_dir / f"{ENV_NAME}_r{R_ENDPOINT}_seed{seed}.certified.json")
    return {
        "env": ENV_NAME,
        "r": R_ENDPOINT,
        "seed": seed,
        "n_contexts": len(q.partition),
        "n_classes": q.n_classes,
        "expected": EXPECTED_CLASSES,
        "exact_partition": int(exact),
        "undetermined_rate": round(q.undetermined_rate, 4),
        "n_boundary_flags": len(q.boundary_flags),
        "m_star": M_STAR,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent / "results_nested"),
    )
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    check_analytic_quotient(NoisyDepth2KCycle(m=M_SYMBOLS, eta=ETA, nu=NU))

    rows: list[dict] = []
    t0 = time.time()
    for seed in seeds:
        row = run_one(seed, out_dir)
        rows.append(row)
        print(f"  seed={seed} contexts={row['n_contexts']} "
              f"n_classes={row['n_classes']} exact={row['exact_partition']} "
              f"und={row['undetermined_rate']}", flush=True)

    csv_path = out_dir / "nested.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    n_fail = sum(
        row["n_classes"] != EXPECTED_CLASSES
        or not row["exact_partition"]
        or row["undetermined_rate"] > 0
        for row in rows
    )
    exact_ok = sum(row["exact_partition"] for row in rows)
    lines = [
        "# Nested-context certified recovery (noisy depth-2 chain, S4)",
        "",
        f"Endpoint r={R_ENDPOINT}, k=1, L={L_MAX}: {N_CONTEXTS} eligible "
        f"contexts (3 length-1 nested under 9 length-2), analytic quotient "
        f"{EXPECTED_CLASSES} classes; minimal between-class TV over all "
        f"eligible contexts = 2/7 ~= {MIN_GAP:.6f} (attained by (0,) vs "
        f"(0,0)); eps_min={EPS_MIN} <= 2/7; delta={DELTA}, beta={BETA}, "
        f"K={N_CONTEXTS}, alpha_0=delta/(2K^2)={DELTA / (2 * N_CONTEXTS**2):.3e}, "
        f"d={ALPHABET_SIZE}, m*={M_STAR}; N={N_TRAJ} episodes, 50/50 split, "
        f"q={Q}; rarest context (0,0) mass ~0.0217 -> ~1522 expected "
        "inference blocks.",
        "",
        "| seeds | exact nested partition | zero abstention |",
        "|---|---|---|",
        f"| {len(rows)} | {exact_ok}/{len(rows)} | "
        f"{sum(row['undetermined_rate'] == 0 for row in rows)}/{len(rows)} |",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"\nWrote {csv_path}; elapsed {time.time() - t0:.1f}s")
    if n_fail:
        print(f"EXPECTATION VIOLATED: {n_fail} seeds failed exact "
              "no-abstention nested recovery")
        return 1
    print(f"exact nested-partition recovery with zero abstention in "
          f"{len(rows)}/{len(rows)} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
