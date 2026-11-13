"""Multi-endpoint certified recovery on the position-flip chain (REV-2).

The paper prices a family of endpoints r_1, ..., r_T at one extra
union-bound level: split the family-wise failure budget delta across the T
endpoints (delta/T each) and run the single-endpoint certified procedure at
each endpoint on the SAME collected trajectories.  This runner executes that
recipe at T = 2 endpoints, r in {0, 1}, on the position-flip chain of
run_headtohead.py, whose transition kernel flips with episode-position
parity.

Analytic quotient per endpoint (kernel at position t, k = L = 1):

  r = 0: contexts (0,) and (1,) have continuation laws Ber(0.9) and Ber(0.1)
         (the even-position kernel), TV gap 0.8 -- two singleton classes.
  r = 1: the chain is Markov in (current symbol, position parity), so
         conditioning on O_1 alone determines the law of O_2 via the ODD
         kernel: (0,) -> Ber(0.1) and (1,) -> Ber(0.9).  Same two singleton
         classes, laws swapped, same TV gap 0.8.

Each endpoint runs at per-endpoint budget delta/T = 0.025, with the m* gate
recomputed at that level: K = 2 contexts, alpha_0 = (delta/T)/(2 K^2)
= 3.125e-3, eps = 0.75 <= 0.8, beta = 0.05, d = 2, giving m* = 112.  With
N = 4,000 episodes each context collects about N/4 = 1,000 inference blocks,
so no pair abstains.

Usage:
    python experiments/main_table/run_multiendpoint.py [--seeds 0-9] \
        [--out experiments/main_table/results_multiendpoint]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from environments.layer1_discrete.position_flip import PositionFlipChain
from experiments.main_table.run_certified import BETA, DELTA, parse_seeds
from quotient_learning.collection import collect_trajectories, uniform_random_policy
from quotient_learning.config import CertifiedConfig, seed_everything
from quotient_learning.equivalence import TVThresholdEquivalenceTest
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.stats import required_samples

ENDPOINTS: tuple[int, ...] = (0, 1)
DELTA_PER_ENDPOINT = DELTA / len(ENDPOINTS)  # 0.025: delta/T union-bound split
EPS_MIN = 0.75                # <= analytic TV gap 0.8 at BOTH endpoints
EXPECTED_CLASSES = 2          # two singleton classes at each endpoint
K_CONTEXTS = 2                # candidate contexts per endpoint (binary alphabet)
ALPHABET_SIZE = 2             # d = |O|^k with k = 1
# Per-endpoint gate: alpha_0 = (delta/T) / (2 K^2); beta stays at the
# per-pair Type-II budget of the certified pipeline.
M_STAR = required_samples(
    EPS_MIN, DELTA_PER_ENDPOINT / (2.0 * K_CONTEXTS**2), BETA, ALPHABET_SIZE
)
N_TRAJ = 4000                 # ~N/4 = 1000 inference blocks per context >> m*
MAX_STEPS = 8                 # needs r + k = 2 steps; headroom is harmless
Q = 10                        # selection threshold, as in run_headtohead.py


def analytic_law_next_one(context_symbol: int, r: int) -> float:
    """P(O_{r+1} = 1 | O_r = context_symbol): the endpoint-r continuation law."""
    return PositionFlipChain.transition_prob_next_one(context_symbol, r)


def check_analytic_quotient() -> None:
    """Assert the derived quotient at every endpoint before running."""
    for r in ENDPOINTS:
        p0 = analytic_law_next_one(0, r)
        p1 = analytic_law_next_one(1, r)
        gap = abs(p0 - p1)
        # Two distinct singleton classes with TV gap 0.8 at each endpoint.
        assert gap == 0.8, f"r={r}: analytic TV gap {gap} != 0.8"
        assert p0 != p1, f"r={r}: laws coincide; quotient would have 1 class"
        assert EPS_MIN <= gap, f"r={r}: eps_min={EPS_MIN} exceeds gap {gap}"
    # The r=1 laws are the r=0 laws swapped (parity flip of the kernel).
    assert analytic_law_next_one(0, 1) == analytic_law_next_one(1, 0)
    assert analytic_law_next_one(1, 1) == analytic_law_next_one(0, 0)


def run_endpoint(env, dataset, r: int, seed: int, out_dir: Path) -> dict:
    """Single-endpoint certified procedure at level delta/T on shared data."""
    alphabet = env.continuation_alphabet(1)
    est = QuotientEstimator(
        env=env,
        equivalence_test=TVThresholdEquivalenceTest(alphabet, EPS_MIN),
        delta=DELTA_PER_ENDPOINT, epsilon_min=EPS_MIN, power_beta=BETA,
        certified_independent=True, selection_fraction=0.5,
        independent_context_end=r, include_root=False,
        n_trajectories=N_TRAJ, window_length=1,
        history_max_length=1, min_history_occurrences=Q,
        max_steps_per_episode=MAX_STEPS,
    )
    q = est.estimate(dataset)
    try:
        recorded_out = out_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        recorded_out = out_dir.resolve()
    cfg = CertifiedConfig(
        k=1, tau=1, test="TV_FIXED_ALPHABET", alpha=DELTA_PER_ENDPOINT,
        epsilon_min=EPS_MIN, beta=BETA, seeds=[seed],
        selection_fraction=0.5, context_end=r,
        policy_id="uniform_random",
        n_trajectories=N_TRAJ, history_max_length=1,
        min_history_occurrences=Q, max_steps_per_episode=MAX_STEPS,
        command=(
            "python experiments/main_table/run_multiendpoint.py "
            f"--seeds {seed} --out {recorded_out}"
        ),
    )
    cfg.capture_env_pins()
    cfg.save_sidecar(out_dir / f"position_flip_r{r}_seed{seed}.certified.json")
    return {
        "env": "position_flip",
        "r": r,
        "seed": seed,
        "n_classes": q.n_classes,
        "expected": EXPECTED_CLASSES,
        "undetermined_rate": round(q.undetermined_rate, 4),
        "n_boundary_flags": len(q.boundary_flags),
        "delta_endpoint": DELTA_PER_ENDPOINT,
        "m_star": M_STAR,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent / "results_multiendpoint"),
    )
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    check_analytic_quotient()

    rows: list[dict] = []
    t0 = time.time()
    for seed in seeds:
        seed_everything(seed)
        env = PositionFlipChain(seed=seed)
        # One trajectory pool per seed, shared by every endpoint: the whole
        # point of the endpoint family is to reuse the same long
        # trajectories at log T (here: delta/2 per endpoint) cost.
        dataset = collect_trajectories(
            env=env, policy=uniform_random_policy(env.n_actions),
            n_episodes=N_TRAJ, max_steps=MAX_STEPS, seed=seed,
            policy_id="uniform_random",
        )
        for r in ENDPOINTS:
            row = run_endpoint(env, dataset, r, seed, out_dir)
            rows.append(row)
            print(f"  r={r} seed={seed} n_classes={row['n_classes']} "
                  f"und={row['undetermined_rate']}", flush=True)

    csv_path = out_dir / "multiendpoint.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    n_fail = sum(
        r["n_classes"] != EXPECTED_CLASSES or r["undetermined_rate"] > 0
        for r in rows
    )
    lines = [
        "# Multi-endpoint certified recovery (position-flip chain, REV-2)",
        "",
        f"Family delta={DELTA} split across T={len(ENDPOINTS)} endpoints "
        f"r in {list(ENDPOINTS)} (delta/T={DELTA_PER_ENDPOINT} each); "
        f"eps_min={EPS_MIN} <= analytic TV gap 0.8 at both endpoints; "
        f"beta={BETA}; K={K_CONTEXTS} contexts per endpoint, d={ALPHABET_SIZE}, "
        f"m*={M_STAR}; N={N_TRAJ} episodes per seed, one shared trajectory "
        f"pool per seed across endpoints; 50/50 split; q={Q}.",
        "",
        "Analytic laws P(next=1 | context): "
        "r=0: (0,)->0.9, (1,)->0.1; r=1: (0,)->0.1, (1,)->0.9 "
        "(parity-flipped kernel). Two singleton classes at each endpoint.",
        "",
        "| endpoint r | recovered 2 classes | zero abstention | seeds |",
        "|---|---|---|---|",
    ]
    for r in ENDPOINTS:
        sub = [row for row in rows if row["r"] == r]
        ok = sum(row["n_classes"] == EXPECTED_CLASSES for row in sub)
        no_abst = sum(row["undetermined_rate"] == 0 for row in sub)
        lines.append(f"| {r} | {ok}/{len(sub)} | {no_abst}/{len(sub)} | {len(sub)} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"\nWrote {csv_path}; elapsed {time.time() - t0:.1f}s")
    if n_fail:
        print(f"EXPECTATION VIOLATED: {n_fail} endpoint-runs failed exact "
              "no-abstention recovery")
        return 1
    print(f"exact recovery with zero abstention at both endpoints in "
          f"{len(rows)} endpoint-runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
