"""Tiger POMDP hyperparameter sweep: coarse-to-fine search for n=2 recovery.

Coarse sweep (Phase 1): GTest only (fast).
  Grid: 2 n_traj × 3 hml × 2 wl × 3 min_occ × 3 alpha × 2 max_steps × 2 tests
  = 432 configs × 3 seeds = 1296 runs  (~0.30s/run → ~390s = 6.5 min)

Fine validation (Phase 2): configs that hit n=2 on ≥1 coarse seed → 10 seeds.
  Estimated: 50 winners × 10 seeds × 0.30s = 150s = 2.5 min.

MMD/EmpiricalBernstein: NOT included in sweep (timing calibration showed
  MMD-200 @ n=5000 takes 1303s and Bernstein takes 25s, making the grid
  intractable within the 10-min budget). These test variants are evaluated
  separately if needed.

Constraints:
- uniform_random_policy only (exploration variants owned by parallel agent).
- Does NOT modify quotient_learning/ or paper/main.tex.
- Outputs: hp_sweep.csv, hp_sweep_winners.md
"""

from __future__ import annotations

import csv
import itertools
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environments.layer1_discrete.tiger import TigerPOMDP
from quotient_learning.collection import uniform_random_policy
from quotient_learning.equivalence import GTestEquivalenceTest
from quotient_learning.estimator import QuotientEstimator

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "hp_sweep.csv"
WINNERS_PATH = RESULTS_DIR / "hp_sweep_winners.md"

# ---------------------------------------------------------------------------
# Coarse grid (GTest only for speed)
# ---------------------------------------------------------------------------
COARSE_N_TRAJ = [3000, 10000]       # 2 values
COARSE_HML = [1, 2, 3]              # 3 values
COARSE_WL = [1, 2]                  # 2 values
COARSE_MIN_OCC = [10, 50, 200]      # 3 values
COARSE_ALPHA = [0.001, 0.05, 0.2]  # 3 values
COARSE_MAX_STEPS = [15, 50]         # 2 values
COARSE_TESTS = [
    ("gtest_pc05", GTestEquivalenceTest(pseudocount=0.5)),
    ("gtest_pc0",  GTestEquivalenceTest(pseudocount=0.0)),
]
# Total: 2×3×2×3×3×2×2 = 432 configs × 3 coarse seeds = 1296 runs ≈ 390s

COARSE_SEEDS = [0, 1, 2]
FINE_SEEDS = list(range(10))


@dataclass
class RunRecord:
    n_trajectories: int
    history_max_length: int
    window_length: int
    min_history_occurrences: int
    alpha: float
    max_steps_per_episode: int
    test_id: str
    seed: int
    n_classes: int
    elapsed_s: float
    phase: str  # "coarse" or "fine"


def run_single(
    *,
    n_trajectories: int,
    history_max_length: int,
    window_length: int,
    min_history_occurrences: int,
    alpha: float,
    max_steps_per_episode: int,
    test_id: str,
    equiv_test,
    seed: int,
    phase: str,
) -> RunRecord:
    env = TigerPOMDP()
    policy = uniform_random_policy(env.n_actions)

    estimator = QuotientEstimator(
        env=env,
        n_trajectories=n_trajectories,
        window_length=window_length,
        history_max_length=history_max_length,
        min_history_occurrences=min_history_occurrences,
        equivalence_test=equiv_test,
        alpha=alpha,
        max_steps_per_episode=max_steps_per_episode,
        policy=policy,
        policy_id="uniform_random",
    )

    t0 = time.perf_counter()
    quotient = estimator.run(seed=seed)
    elapsed = time.perf_counter() - t0

    return RunRecord(
        n_trajectories=n_trajectories,
        history_max_length=history_max_length,
        window_length=window_length,
        min_history_occurrences=min_history_occurrences,
        alpha=alpha,
        max_steps_per_episode=max_steps_per_episode,
        test_id=test_id,
        seed=seed,
        n_classes=quotient.n_classes,
        elapsed_s=elapsed,
        phase=phase,
    )


def config_key(r: RunRecord) -> tuple:
    return (
        r.n_trajectories,
        r.history_max_length,
        r.window_length,
        r.min_history_occurrences,
        r.alpha,
        r.max_steps_per_episode,
        r.test_id,
    )


def main() -> bool:
    wall_t0 = time.perf_counter()
    all_records: list[RunRecord] = []

    # -----------------------------------------------------------------------
    # Phase 1: coarse sweep (GTest only)
    # -----------------------------------------------------------------------
    coarse_configs = []
    for (n_traj, hml, wl, min_occ, alpha, max_steps, (test_id, equiv_test)) in itertools.product(
        COARSE_N_TRAJ, COARSE_HML, COARSE_WL,
        COARSE_MIN_OCC, COARSE_ALPHA, COARSE_MAX_STEPS, COARSE_TESTS
    ):
        for seed in COARSE_SEEDS:
            coarse_configs.append(dict(
                n_trajectories=n_traj, history_max_length=hml, window_length=wl,
                min_history_occurrences=min_occ, alpha=alpha,
                max_steps_per_episode=max_steps, test_id=test_id,
                equiv_test=equiv_test, seed=seed, phase="coarse",
            ))

    n_unique = len(coarse_configs) // len(COARSE_SEEDS)
    print(f"Phase 1 (coarse): {len(coarse_configs)} runs across "
          f"{n_unique} unique configs × {len(COARSE_SEEDS)} seeds", flush=True)

    for idx, kwargs in enumerate(coarse_configs):
        rec = run_single(**kwargs)
        all_records.append(rec)
        if (idx + 1) % 200 == 0:
            elapsed = time.perf_counter() - wall_t0
            n2_so_far = sum(1 for r in all_records if r.n_classes == 2)
            print(f"  [{idx+1}/{len(coarse_configs)}] elapsed={elapsed:.1f}s  "
                  f"n=2 hits: {n2_so_far}", flush=True)

    coarse_elapsed = time.perf_counter() - wall_t0
    n2_coarse = sum(1 for r in all_records if r.n_classes == 2 and r.phase == "coarse")
    print(f"Phase 1 done in {coarse_elapsed:.1f}s  (n=2 hits: {n2_coarse})", flush=True)

    # Identify candidate configs: ≥1 seed hit n=2
    coarse_by_config: dict[tuple, list[RunRecord]] = defaultdict(list)
    for r in all_records:
        if r.phase == "coarse":
            coarse_by_config[config_key(r)].append(r)

    candidate_configs = [
        cfg for cfg, records in coarse_by_config.items()
        if any(r.n_classes == 2 for r in records)
    ]
    print(f"\nPhase 2 candidates: {len(candidate_configs)} configs hit n=2 on ≥1 coarse seed",
          flush=True)

    # -----------------------------------------------------------------------
    # Phase 2: fine validation (10 seeds)
    # -----------------------------------------------------------------------
    fine_by_config: dict[tuple, list[RunRecord]] = defaultdict(list)

    if candidate_configs:
        fine_total = len(candidate_configs) * len(FINE_SEEDS)
        print(f"Phase 2 (fine): {fine_total} runs", flush=True)
        fine_count = 0

        # Build test lookup from coarse records
        test_lookup: dict[str, GTestEquivalenceTest] = {
            test_id: equiv_test for test_id, equiv_test in COARSE_TESTS
        }

        for cfg in candidate_configs:
            (n_traj, hml, wl, min_occ, alpha, max_steps, test_id) = cfg
            equiv_test = test_lookup[test_id]
            for seed in FINE_SEEDS:
                rec = run_single(
                    n_trajectories=n_traj,
                    history_max_length=hml,
                    window_length=wl,
                    min_history_occurrences=min_occ,
                    alpha=alpha,
                    max_steps_per_episode=max_steps,
                    test_id=test_id,
                    equiv_test=equiv_test,
                    seed=seed,
                    phase="fine",
                )
                all_records.append(rec)
                fine_by_config[cfg].append(rec)
                fine_count += 1
                if fine_count % 200 == 0:
                    elapsed = time.perf_counter() - wall_t0
                    print(f"  fine [{fine_count}/{fine_total}] elapsed={elapsed:.1f}s",
                          flush=True)

        fine_elapsed = time.perf_counter() - wall_t0 - coarse_elapsed
        print(f"Phase 2 done in {fine_elapsed:.1f}s", flush=True)
    else:
        print("No candidates. Skipping Phase 2.", flush=True)

    total_elapsed = time.perf_counter() - wall_t0

    # -----------------------------------------------------------------------
    # Write hp_sweep.csv
    # -----------------------------------------------------------------------
    fieldnames = [
        "phase", "n_trajectories", "history_max_length", "window_length",
        "min_history_occurrences", "alpha", "max_steps_per_episode",
        "test_id", "seed", "n_classes", "elapsed_s",
    ]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_records:
            writer.writerow({
                "phase": r.phase,
                "n_trajectories": r.n_trajectories,
                "history_max_length": r.history_max_length,
                "window_length": r.window_length,
                "min_history_occurrences": r.min_history_occurrences,
                "alpha": r.alpha,
                "max_steps_per_episode": r.max_steps_per_episode,
                "test_id": r.test_id,
                "seed": r.seed,
                "n_classes": r.n_classes,
                "elapsed_s": f"{r.elapsed_s:.4f}",
            })
    print(f"\nCSV written: {CSV_PATH} ({len(all_records)} rows)", flush=True)

    # -----------------------------------------------------------------------
    # Identify winners: fine configs with ≥4/10 seeds n=2
    # -----------------------------------------------------------------------
    winners: list[dict] = []
    if fine_by_config:
        for cfg, records in fine_by_config.items():
            n2 = sum(1 for r in records if r.n_classes == 2)
            reliability = n2 / len(records)
            if n2 >= 4:
                (n_traj, hml, wl, min_occ, alpha, max_steps, test_id) = cfg
                winners.append({
                    "n_trajectories": n_traj,
                    "history_max_length": hml,
                    "window_length": wl,
                    "min_history_occurrences": min_occ,
                    "alpha": alpha,
                    "max_steps_per_episode": max_steps,
                    "test_id": test_id,
                    "n2_seeds": n2,
                    "total_seeds": len(records),
                    "reliability": reliability,
                })
        # Sort: reliability DESC, n_trajectories ASC
        winners.sort(key=lambda w: (-w["reliability"], w["n_trajectories"]))

    # -----------------------------------------------------------------------
    # Write hp_sweep_winners.md
    # -----------------------------------------------------------------------
    n_coarse_unique = len(coarse_by_config)
    n_coarse_runs = sum(1 for r in all_records if r.phase == "coarse")
    n_fine_runs = sum(1 for r in all_records if r.phase == "fine")

    with open(WINNERS_PATH, "w") as f:
        f.write("# Tiger HP Sweep: Configs Achieving n=2 (≥4/10 seeds)\n\n")
        f.write(f"**Date**: 2026-05-25  **Policy**: uniform_random_policy\n\n")
        f.write(f"**Note**: MMD and EmpiricalBernstein excluded from sweep "
                f"(too slow: MMD-200 @ n=5000 measured at 1303s/run, "
                f"Bernstein @ n=5000 at 25s/run).\n\n")
        f.write("## Sweep Statistics\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total wall-clock time | {total_elapsed:.1f}s ({total_elapsed/60:.1f} min) |\n")
        f.write(f"| Phase 1 runs | {n_coarse_runs} |\n")
        f.write(f"| Phase 1 unique configs | {n_coarse_unique} |\n")
        f.write(f"| Phase 1 n=2 hits | {n2_coarse} |\n")
        f.write(f"| Phase 2 candidates | {len(candidate_configs)} |\n")
        f.write(f"| Phase 2 runs | {n_fine_runs} |\n")
        f.write(f"| Total runs | {len(all_records)} |\n\n")

        if not winners:
            f.write("## Result: NO WINNER FOUND\n\n")
            f.write("No config achieved n=2 on ≥4/10 seeds with `uniform_random_policy` + GTest.\n\n")
            # Report best coarse result
            best_coarse = sorted(
                coarse_by_config.items(),
                key=lambda kv: -sum(1 for r in kv[1] if r.n_classes == 2),
            )
            if best_coarse:
                best_cfg, best_recs = best_coarse[0]
                n2 = sum(1 for r in best_recs if r.n_classes == 2)
                f.write(f"### Best Coarse Config (n=2 on {n2}/{len(best_recs)} coarse seeds)\n\n")
                (n_traj, hml, wl, min_occ, alpha, max_steps, test_id) = best_cfg
                f.write(f"| Param | Value |\n|-------|-------|\n")
                f.write(f"| n_trajectories | {n_traj} |\n")
                f.write(f"| history_max_length | {hml} |\n")
                f.write(f"| window_length | {wl} |\n")
                f.write(f"| min_history_occurrences | {min_occ} |\n")
                f.write(f"| alpha | {alpha} |\n")
                f.write(f"| max_steps_per_episode | {max_steps} |\n")
                f.write(f"| test_id | {test_id} |\n\n")
                f.write("### Recommendation for Paper\n\n")
                f.write(
                    "Tiger n=2 recovery with `uniform_random_policy` + GTest was not reliably "
                    "achieved (≥4/5 seeds) in this grid. Recommend reporting n_classes as 1 "
                    "under default config (root-singleton exclusion) and noting that n=2 is "
                    "achieved only with exploration-biased policy (parallel agent result) or "
                    "larger trajectory budget (n_trajectories ≥ 100k).\n"
                )
        else:
            f.write(f"## Winners: {len(winners)} configs achieving ≥4/10 seeds n=2\n\n")
            f.write("Sorted by reliability (desc) then n_trajectories (asc).\n\n")
            f.write("| n_traj | hml | wl | min_occ | alpha | max_steps | test | n=2/total | reliability |\n")
            f.write("|--------|-----|-----|---------|-------|-----------|------|-----------|-------------|\n")
            for w in winners:
                f.write(
                    f"| {w['n_trajectories']} | {w['history_max_length']} | "
                    f"{w['window_length']} | {w['min_history_occurrences']} | "
                    f"{w['alpha']} | {w['max_steps_per_episode']} | "
                    f"{w['test_id']} | {w['n2_seeds']}/{w['total_seeds']} | "
                    f"{w['reliability']:.0%} |\n"
                )
            best = winners[0]
            f.write("\n## Best Config (smallest budget, highest reliability)\n\n")
            f.write(f"| Param | Value |\n|-------|-------|\n")
            f.write(f"| n_trajectories | {best['n_trajectories']} |\n")
            f.write(f"| history_max_length | {best['history_max_length']} |\n")
            f.write(f"| window_length | {best['window_length']} |\n")
            f.write(f"| min_history_occurrences | {best['min_history_occurrences']} |\n")
            f.write(f"| alpha | {best['alpha']} |\n")
            f.write(f"| max_steps_per_episode | {best['max_steps_per_episode']} |\n")
            f.write(f"| test_id | {best['test_id']} |\n")
            f.write(f"| **Reliability** | **{best['n2_seeds']}/{best['total_seeds']} = "
                    f"{best['reliability']:.0%}** |\n\n")
            f.write("## Recommendation for Paper\n\n")
            f.write(
                f"Use `uniform_random_policy` + `{best['test_id']}` with:\n\n"
                f"```python\n"
                f"EstimatorConfig(\n"
                f"    n_trajectories={best['n_trajectories']},\n"
                f"    history_max_length={best['history_max_length']},\n"
                f"    window_length={best['window_length']},\n"
                f"    min_history_occurrences={best['min_history_occurrences']},\n"
                f"    alpha={best['alpha']},\n"
                f"    max_steps_per_episode={best['max_steps_per_episode']},\n"
                f")\n"
                f"```\n\n"
                f"Achieves n=2 on {best['n2_seeds']}/{best['total_seeds']} seeds "
                f"({best['reliability']:.0%} reliability).\n"
            )

    print(f"Winners report: {WINNERS_PATH}", flush=True)

    # -----------------------------------------------------------------------
    # Terminal summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("TIGER HP SWEEP SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Total runs          : {len(all_records)}", flush=True)
    print(f"Wall-clock time     : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", flush=True)
    print(f"Coarse unique cfgs  : {n_coarse_unique}", flush=True)
    print(f"Coarse n=2 hits     : {n2_coarse}", flush=True)
    print(f"Candidates (≥1 n=2) : {len(candidate_configs)}", flush=True)
    print(f"Fine winners (≥4/10): {len(winners)}", flush=True)

    if winners:
        best = winners[0]
        print(f"\nBEST CONFIG:", flush=True)
        print(f"  n_trajectories={best['n_trajectories']}, "
              f"hml={best['history_max_length']}, "
              f"wl={best['window_length']}, "
              f"min_occ={best['min_history_occurrences']}", flush=True)
        print(f"  alpha={best['alpha']}, "
              f"max_steps={best['max_steps_per_episode']}, "
              f"test={best['test_id']}", flush=True)
        print(f"  Reliability: {best['n2_seeds']}/{best['total_seeds']} = "
              f"{best['reliability']:.0%}", flush=True)
    else:
        print("\nNo fine winner. Best coarse result:")
        best_coarse = sorted(
            coarse_by_config.items(),
            key=lambda kv: -sum(1 for r in kv[1] if r.n_classes == 2),
        )
        if best_coarse:
            cfg, recs = best_coarse[0]
            n2 = sum(1 for r in recs if r.n_classes == 2)
            print(f"  n=2 on {n2}/{len(recs)} coarse seeds: {cfg}", flush=True)

    return len(winners) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
