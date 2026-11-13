"""Certified headline regeneration (single entry point, review finding E2).

Runs QuotientEstimator under the exact finite-sample configuration:

  - disjoint trajectory halves for context selection and inference,
  - one prespecified block per inference trajectory,
  - the fixed-alphabet empirical-TV threshold test,
  - Bonferroni per-pair level alpha_0 = delta / (2 K_N^2),
  - the m*(eps_min, alpha_0, beta) power gate with UNDETERMINED pairs
    reported instead of merged.

Every run writes its result row and a validated CertifiedConfig JSON sidecar.
Exploratory overlapping-window results are kept separate.

Usage:
    python experiments/main_table/run_certified.py --seeds 0-4 \
        [--envs tiger kcycle_8 ...] [--out experiments/main_table/results_certified]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from environments.layer1_discrete.hallway import HallwayPOMDP
from environments.layer1_discrete.kcycle import KCyclePOMDP
from environments.layer1_discrete.rocksample import RockSamplePOMDP
from environments.layer1_discrete.synthetic import SyntheticPOMDP
from environments.layer1_discrete.tiger import TigerPOMDP
from environments.layer2_memory.noisy_depth2_kcycle import NoisyDepth2KCycle
from environments.layer2_memory.tmaze import TMazePOMDP
from environments.layer4_realworld.robot_localization import RobotLocalizationPOMDP
from environments.layer4_realworld.sepsis import SepsisPOMDP
from experiments.scoring import merge_result_rows
from quotient_learning.config import CertifiedConfig, seed_everything
from quotient_learning.equivalence import TVThresholdEquivalenceTest
from quotient_learning.estimator import QuotientEstimator

DELTA = 0.05    # family-wise failure budget
BETA = 0.05     # per-pair Type-II budget for the power gate
EPS_MIN = 0.5   # default TV-gap parameter for the gate

# Analytic minimum pairwise TV gaps between distinct context laws, where
# known in closed form.  tiger (always-listen, r=1, k=1, accuracy a=0.85):
# TV = (2a-1)^2 = 0.49.  kcycle_8 (k=1, r=0): min pairwise TV = 0.7225.
# The gap model (Eq. 2 of the paper) requires eps_min <= this gap; a larger
# eps_min shrinks m* and widens the SAME region, which is anti-conservative
# for the Type-II certificate.
ANALYTIC_TV_GAPS: dict[str, float] = {
    "tiger": 0.49,
    "kcycle_8": 0.7225,
    # noisy_depth2 (m=2, eta=0.1, nu=0.2) at r=0, k=1: contexts (1,) and
    # (2,) share the L_odd row exactly; TV between the (0,) law and L_odd
    # is 0.752380... (exact_continuation_law), rounded DOWN — recorded gaps
    # bound eps_min from above, so rounding up would weaken the assertion.
    "noisy_depth2": 0.7523,
}

# Per-environment eps_min overrides keeping eps_min <= the analytic gap.
EPS_OVERRIDES: dict[str, float] = {
    "tiger": 0.45,
}


def eps_min_for(name: str) -> float:
    """Certified eps_min for an environment, checked against its known gap."""
    eps = EPS_OVERRIDES.get(name, EPS_MIN)
    gap = ANALYTIC_TV_GAPS.get(name)
    if gap is not None and eps > gap:
        raise ValueError(
            f"{name}: eps_min={eps} exceeds the analytic TV gap {gap}; "
            "the gap model of the certificate would be violated"
        )
    return eps


def certified_tau(k: int, max_len: int) -> int:
    """Deprecated sidecar field; independent trajectory blocks use tau=1."""
    return 1

# name -> (constructor, estimator kwargs, default seed count, expected classes or None)
CONFIGS: dict = {
    # Exact-recovery environments target the length-one quotient (k=1;
    # "What recovery can mean at L=1" in the paper), which keeps the fixed
    # alphabet small enough for the m* gate to be satisfiable at realistic
    # budgets; the L=2 stability environments keep k=1 for the same reason.
    "tiger": (
        lambda seed: TigerPOMDP(seed=seed),
        # The fixed block ends at observation index 1. Episodes too short for
        # a continuation contribute no sample; no episode contributes twice.
        dict(n_trajectories=20000, window_length=1, history_max_length=1,
             min_history_occurrences=10, max_steps_per_episode=15,
             independent_context_end=1, policy_id="always_listen"),
        10, 2),
    "kcycle_8": (
        lambda seed: KCyclePOMDP(8, seed=seed),
        dict(n_trajectories=8000, window_length=1, history_max_length=1,
             min_history_occurrences=100, max_steps_per_episode=20),
        5, 8),
    "tmaze": (
        lambda seed: TMazePOMDP(corridor_length=5, seed=seed),
        dict(n_trajectories=5000, window_length=2, history_max_length=2,
             min_history_occurrences=80, max_steps_per_episode=25),
        5, 4),
    "hallway": (
        lambda seed: HallwayPOMDP(seed=seed),
        # 4x the supplementary budget so the m* gate can resolve most of
        # the 21-symbol alphabet under the certified configuration.
        dict(n_trajectories=20000, window_length=1, history_max_length=1,
             min_history_occurrences=50, max_steps_per_episode=50),
        7, None),
    "rocksample_7_8": (
        lambda seed: RockSamplePOMDP(7, 8, seed=seed),
        dict(n_trajectories=5000, window_length=1, history_max_length=2,
             min_history_occurrences=80, max_steps_per_episode=25),
        5, None),
    "sepsis": (
        lambda seed: SepsisPOMDP(seed=seed),
        dict(n_trajectories=5000, window_length=1, history_max_length=2,
             min_history_occurrences=80, max_steps_per_episode=25),
        5, None),
    "robot_localization": (
        lambda seed: RobotLocalizationPOMDP(size=6, n_landmarks=4, seed=seed),
        dict(n_trajectories=5000, window_length=1, history_max_length=2,
             min_history_occurrences=80, max_steps_per_episode=25),
        5, None),
    "synthetic_200": (
        lambda seed: SyntheticPOMDP(n_states=200, n_obs=10, seed=seed),
        dict(n_trajectories=5000, window_length=1, history_max_length=2,
             min_history_occurrences=80, max_steps_per_episode=25),
        5, None),
    "noisy_depth2": (
        lambda seed: NoisyDepth2KCycle(m=2, eta=0.1, seed=seed),
        # Episode-position estimand at r=0, k=1: eligible contexts are the
        # three length-1 suffixes (0,), (1,), (2,); (1,) and (2,) share the
        # L_odd law exactly, so the analytic quotient has 2 classes and the
        # run exercises the SAME/merge path (unlike tiger and kcycle_8,
        # whose quotients are all-singleton).  NOTE: env.metadata's
        # ground_truth_n_classes describes the full depth-2 suffix quotient,
        # a different estimand — do not use it here.
        dict(n_trajectories=16000, window_length=1, history_max_length=2,
             min_history_occurrences=30, max_steps_per_episode=60),
        5, 2),
}


def run_one(name: str, seed: int, out_dir: Path) -> dict:
    ctor, kwargs, _, expected = CONFIGS[name]
    seed_everything(seed)
    env = ctor(seed)
    kwargs = dict(kwargs)
    k = kwargs["window_length"]
    tau = certified_tau(k, kwargs["history_max_length"])
    context_end = kwargs.pop("independent_context_end", 0)
    policy_id = kwargs.pop("policy_id", "uniform_random")
    policy = (lambda _obs, _rng: 0) if policy_id == "always_listen" else None
    alphabet = env.continuation_alphabet(k)
    eps_min = eps_min_for(name)
    est = QuotientEstimator(
        env=env,
        equivalence_test=TVThresholdEquivalenceTest(alphabet, eps_min),
        delta=DELTA,
        epsilon_min=eps_min,
        power_beta=BETA,
        certified_independent=True,
        selection_fraction=0.5,
        independent_context_end=context_end,
        include_root=False,
        policy=policy,
        policy_id=policy_id,
        **kwargs,
    )
    q = est.run(seed=seed)
    if expected is None:
        # CAUTION: metadata ground truths generally describe the full-suffix
        # quotient, not the episode-position estimand at the configured r;
        # rows relying on this fallback must not be published without a
        # per-estimand re-derivation (see noisy_depth2 above).
        expected = env.metadata.get("ground_truth_n_classes")
    row = {
        "env": name,
        "seed": seed,
        "n_classes": q.n_classes,
        "expected": expected if expected is not None else "",
        "undetermined_rate": round(q.undetermined_rate, 4),
        "n_boundary_flags": len(q.boundary_flags),
    }
    try:
        recorded_out = out_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        recorded_out = out_dir.resolve()
    cfg = CertifiedConfig(
        k=k, tau=tau, test="TV_FIXED_ALPHABET", alpha=DELTA,
        epsilon_min=eps_min, beta=BETA, seeds=[seed],
        selection_fraction=0.5, context_end=context_end,
        policy_id=policy_id,
        n_trajectories=kwargs["n_trajectories"],
        history_max_length=kwargs["history_max_length"],
        min_history_occurrences=kwargs["min_history_occurrences"],
        max_steps_per_episode=kwargs["max_steps_per_episode"],
        command=(
            "python experiments/main_table/run_certified.py "
            f"--envs {name} --seeds {seed} --out {recorded_out}"
        ),
    )
    cfg.capture_env_pins()
    cfg.save_sidecar(out_dir / f"{name}_seed{seed}.certified.json")
    return row


def parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default=None,
                    help="e.g. 0-4 or 0,1,2; default: per-env seed counts")
    ap.add_argument("--envs", nargs="*", default=list(CONFIGS.keys()))
    # Default deliberately avoids the quarantined "results_certified" name
    # (legacy G-test artifacts live in results_legacy_gtest); published
    # tables regenerate into results_independent_smoke explicitly.
    ap.add_argument("--out", default=str(Path(__file__).parent / "results_scratch"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    for name in args.envs:
        if name not in CONFIGS:
            print(f"unknown env {name!r}; choices: {list(CONFIGS)}", file=sys.stderr)
            return 2
        seeds = parse_seeds(args.seeds) if args.seeds else list(range(CONFIGS[name][2]))
        for seed in seeds:
            row = run_one(name, seed, out_dir)
            rows.append(row)
            print(f"  {name} seed={seed} n={row['n_classes']} "
                  f"und={row['undetermined_rate']} flags={row['n_boundary_flags']}",
                  flush=True)

    csv_path = out_dir / "certified_main.csv"
    all_rows = merge_result_rows(csv_path, rows, ('env', 'seed'))
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(all_rows[0].keys()), lineterminator="\n"
        )
        w.writeheader()
        w.writerows(all_rows)

    lines = [
        "# Certified headline results",
        "",
        f"delta={DELTA}, disjoint 50/50 trajectory split, one fixed block per "
        f"inference trajectory; eps_min={EPS_MIN} (per-env overrides: "
        f"{EPS_OVERRIDES}), beta={BETA}; "
        "fixed-alphabet empirical-TV test; "
        "power-gated union-find (UNDETERMINED pairs never merge).",
        "",
        "| env | n_classes (mean ± sd) | undetermined rate (mean) | seeds |",
        "|---|---|---|---|",
    ]
    summary_envs = [e for e in CONFIGS if any(r["env"] == e for r in all_rows)]
    for name in summary_envs:
        ns = [float(r["n_classes"]) for r in all_rows if r["env"] == name]
        us = [float(r["undetermined_rate"]) for r in all_rows if r["env"] == name]
        sd = np.std(ns, ddof=1) if len(ns) > 1 else 0.0
        lines.append(
            f"| {name} | {np.mean(ns):.1f} ± {sd:.2f} | {np.mean(us):.3f} | {len(ns)} |")
    (out_dir / "certified_summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {csv_path}; elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
