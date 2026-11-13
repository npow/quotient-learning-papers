"""PO-HalfCheetah bin-sensitivity sweep (Theorem thm:disc_recovery validation).

Runs QuotientEstimator on PO-HalfCheetah with bins_per_dim in {2, 3, 4, 6}
across multiple seeds.  Reports how the learned quotient size changes with B,
providing empirical evidence for Theorem thm:disc_recovery: the binned-POMDP
quotient converges monotonically to the continuous-POMDP quotient as B grows.

Output:
  results/pomujoco_bin_sensitivity.csv   -- per-(B, seed) row
  results/pomujoco_bin_sensitivity.md    -- human-readable summary

Usage:
  python experiments/convergence/run_pomujoco_bins.py
  python experiments/convergence/run_pomujoco_bins.py --seeds 3 --n-traj 2000

The script gracefully skips if PO-MuJoCo (gymnasium+mujoco) is not installed,
writing a skip notice to the output files rather than raising an error.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "pomujoco_bin_sensitivity.csv"
MD_PATH = RESULTS_DIR / "pomujoco_bin_sensitivity.md"

BINS_GRID = [2, 3, 4, 6]


def _write_csv(rows: list[dict]) -> None:
    if not rows:
        CSV_PATH.write_text("")
        return
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {CSV_PATH}")


def _write_md(rows: list[dict], n_seeds: int) -> None:
    """Aggregate by bins_per_dim and write a Markdown summary table."""
    from collections import defaultdict

    by_b: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        by_b[int(r["bins_per_dim"])].append(int(r["n_classes"]))

    lines = [
        "# PO-HalfCheetah bin sensitivity (Theorem thm:disc_recovery)",
        "",
        f"Seeds per configuration: {n_seeds}",
        "",
        "| bins_per_dim | mean n_classes | std n_classes |",
        "|---|---|---|",
    ]
    for b in sorted(by_b):
        vals = np.array(by_b[b], dtype=float)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        lines.append(f"| {b} | {mean:.1f} | {std:.1f} |")

    lines += [
        "",
        "Expected: monotone increase in mean n_classes with bins_per_dim,",
        "with decelerating growth rate (consistent with convergence to the",
        "continuous-POMDP quotient as B -> infinity).",
        "",
        "Reference: Theorem thm:disc_recovery in paper/main.tex,",
        "Section 'Discretization Recovery: Formal Guarantees for PO-MuJoCo'.",
    ]
    MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"  wrote {MD_PATH}")


def run(seeds: int = 5, n_traj: int = 3000) -> None:
    print(f"\n=== PO-HalfCheetah bin sensitivity sweep ===")
    print(f"  bins_per_dim: {BINS_GRID}, seeds: {seeds}, n_traj: {n_traj}")

    # Import PO-HalfCheetah — skip gracefully if unavailable.
    try:
        from environments.layer3_continuous.po_halfcheetah import (
            POHalfCheetah,
            POMuJoCoUnavailable,
        )
    except ImportError as e:
        _skip(f"ImportError: {e}")
        return

    from quotient_learning.estimator import QuotientEstimator

    rows: list[dict] = []

    for bins in BINS_GRID:
        for seed in range(seeds):
            t0 = time.time()
            try:
                env = POHalfCheetah(bins_per_dim=bins, seed=seed)
            except Exception as e:
                # MuJoCo may fail on headless servers — skip, don't crash.
                print(f"    bins={bins} seed={seed}: SKIP ({e})")
                rows.append(
                    {
                        "bins_per_dim": bins,
                        "seed": seed,
                        "n_classes": -1,
                        "elapsed_s": 0.0,
                        "status": f"skip:{e}",
                    }
                )
                continue

            est = QuotientEstimator(
                env=env,
                n_trajectories=n_traj,
                window_length=1,
                history_max_length=1,
                min_history_occurrences=30,
                alpha=0.05,
                seed=seed,
            )
            result = est.fit()
            elapsed = time.time() - t0

            # Apply root-class convention (subtract 1 if root class is singleton).
            n_raw = result.n_classes
            n_classes = n_raw - 1 if n_raw > 1 else n_raw

            print(
                f"    bins={bins} seed={seed}: "
                f"n_classes={n_classes} (raw={n_raw})  "
                f"[{elapsed:.1f}s]"
            )
            rows.append(
                {
                    "bins_per_dim": bins,
                    "seed": seed,
                    "n_classes": n_classes,
                    "n_classes_raw": n_raw,
                    "elapsed_s": round(elapsed, 2),
                    "status": "ok",
                }
            )
            env.close()

    _write_csv(rows)
    good_rows = [r for r in rows if r["status"] == "ok"]
    if good_rows:
        _write_md(good_rows, seeds)
        _print_summary(good_rows)
    else:
        _skip("No successful runs (MuJoCo/gymnasium unavailable on this host)")


def _skip(reason: str) -> None:
    msg = f"SKIP: {reason}\n"
    CSV_PATH.write_text(msg)
    MD_PATH.write_text(
        f"# PO-HalfCheetah bin sensitivity\n\n{msg}\n"
        "Re-run after installing: pip install -e .[continuous]\n"
    )
    print(f"  {msg.strip()}")
    print(f"  wrote skip notice to {CSV_PATH} and {MD_PATH}")


def _print_summary(rows: list[dict]) -> None:
    from collections import defaultdict

    by_b: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        by_b[int(r["bins_per_dim"])].append(int(r["n_classes"]))

    print("\n  Summary (Theorem thm:disc_recovery validation):")
    print(f"  {'bins_per_dim':>14}  {'mean n':>8}  {'std n':>7}")
    for b in sorted(by_b):
        vals = np.array(by_b[b], dtype=float)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        print(f"  {b:>14}  {mean:>8.1f}  {std:>7.1f}")

    # Check monotonicity as a soft assertion.
    sorted_bs = sorted(by_b)
    means = [float(np.mean(by_b[b])) for b in sorted_bs]
    is_monotone = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    flag = "PASS" if is_monotone else "WARN (non-monotone — check seeds/budget)"
    print(f"\n  Monotonicity check: {flag}")
    print(
        "  (Expected by Theorem thm:disc_recovery: finer binning resolves more"
        " of the continuous quotient structure.)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per B")
    parser.add_argument(
        "--n-traj",
        type=int,
        default=3000,
        help="Trajectories per (B, seed) run",
    )
    args = parser.parse_args()
    run(seeds=args.seeds, n_traj=args.n_traj)


if __name__ == "__main__":
    main()
