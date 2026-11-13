"""MISSPECIFICATION PROBE — NOT a certified configuration (REV-5).

This run INTENTIONALLY sets eps_min = 0.5 ABOVE the true total-variation
distance (tau = 0.10) of one engineered context pair of MisspecChain: the
TV gap model (Eq. 2 of the paper) is deliberately violated, because the
pair ((0,), (1,)) is neither equal nor eps_min-separated.  The purpose is
to make the resulting failure mode VISIBLE: run outside its gap assumption,
the certified machinery issues a confident SAME verdict on the sub-gap pair
(empirical TV concentrates near 0.10 < eps_min/2 = 0.25) in every seed — a
silent, uncertified merge with no abstention and no flag.  Every other
component is the certified configuration verbatim: disjoint 50/50
trajectory split, one prespecified block per inference trajectory,
fixed-alphabet empirical-TV threshold test, Bonferroni per-pair level
alpha_0 = delta / (2 K^2), and the m*(eps_min, alpha_0, beta) power gate.

THIS IS NOT A CERTIFIED CONFIGURATION AND ITS RESULTS CERTIFY NOTHING.
JSON sidecars are written with "certified": false and
"misspecification_probe": true, so certified table generation refuses them
by construction.  This script deliberately does NOT import or bypass the
``eps_min_for`` guard of run_certified.py — that guard exists precisely to
keep certified runs out of this regime; the probe stands alone.

Per-pair margin ("near-threshold") reporting: for every tested pair the run
records the empirical TV, the decision threshold eps_min/2, the margin
|empirical TV - eps_min/2|, the verdict, and a near_threshold flag with the
rule margin < eps_min/8.  Rationale: at the gate count m*, each empirical
histogram lies within eps_min/4 of its law at the configured confidence
(the concentration behind Theorem 1), so a margin below eps_min/8 — half
that per-histogram tolerance — marks verdicts that could plausibly flip
under resampling.  Note the probe's headline finding is that the sub-gap
merge is NOT near-threshold: its margin (~0.15) exceeds eps_min/8 = 0.0625,
i.e. the wrong merge is confident, not a borderline call.

Usage:
    uv run python experiments/main_table/run_misspec.py --seeds 0-9 \
        [--out experiments/main_table/results_misspec]

Outputs (in --out): misspec_pairs.csv (one row per seed per tested pair),
one JSON sidecar per run, and summary.md with the headline numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from environments.layer2_memory.misspec_chain import (
    SEPARATED_TV,
    MisspecChain,
)
from experiments.scoring import merge_result_rows
from quotient_learning.config import seed_everything
from quotient_learning.equivalence import TVThresholdEquivalenceTest
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.stats import required_samples
from quotient_learning.types import Verdict

ENV_NAME = "misspec_chain"

# ----------------------------------------------------------------------
# Probe configuration.  DELTA/BETA match the certified harness; EPS_MIN is
# the harness default, kept DELIBERATELY above the true sub-gap TV so the
# gap model is violated.  These constants are defined locally on purpose:
# the probe must not import run_certified's eps_min_for guard.
# ----------------------------------------------------------------------
DELTA = 0.05          # family-wise failure budget (as in the certified runs)
BETA = 0.05           # per-pair Type-II budget for the power gate
EPS_MIN = 0.5         # MISSPECIFIED: true TV of the sub-gap pair is TAU < EPS_MIN/2
TAU = 0.10            # engineered sub-gap TV(P_(0,), P_(1,)) — see MisspecChain

EXPECTED_CONTEXTS = 3           # (0,), (1,), (2,) at endpoint r=0
WINDOW_LENGTH = 1               # k = 1
HISTORY_MAX_LENGTH = 1          # L = 1
CONTEXT_END = 0                 # r = 0
SELECTION_FRACTION = 0.5        # certified 50/50 split
MIN_HISTORY_OCCURRENCES = 10    # selection threshold q (q <= m*)
MAX_STEPS_PER_EPISODE = 3       # only obs[0] (context) and obs[1] (block) needed
BUDGET_SAFETY = 1.5             # head-room so every inference count clears m*

# Bonferroni per-pair level for the expected candidate count K = 3, as in
# Theorem 2: alpha_0 = delta / (2 K^2).  The estimator recomputes this from
# the realized selection-half count; run_one() checks they agree.
ALPHA_0 = DELTA / (2.0 * EXPECTED_CONTEXTS**2)


def probe_budget() -> tuple[int, int]:
    """Return (m_star, n_trajectories) with the budget sized so that every
    per-context inference count clears the m* gate.

    Sizing follows the certified-harness practice (run_certified /
    run_abstention_sweep): the inference half holds N * (1 - selection
    fraction) trajectories and each of the 3 contexts receives ~1/3 of them
    (the initial symbol is uniform), so per-context counts concentrate near
    N/6.  N is chosen as BUDGET_SAFETY times the break-even budget, rounded
    up to the next hundred.
    """
    m_star = required_samples(EPS_MIN, ALPHA_0, BETA, EXPECTED_CONTEXTS)
    per_traj_share = (1.0 - SELECTION_FRACTION) / EXPECTED_CONTEXTS
    n = math.ceil(BUDGET_SAFETY * m_star / per_traj_share / 100.0) * 100
    return m_star, n


M_STAR, N_TRAJECTORIES = probe_budget()


# ----------------------------------------------------------------------
# Margin computation (pure helpers, unit-tested in
# tests/unit/test_misspec_probe.py)
# ----------------------------------------------------------------------

def empirical_tv(counts_a, counts_b, categories) -> float:
    """Total-variation distance between two normalized count histograms over
    a FIXED category set — the same statistic as TVThresholdEquivalenceTest.

    ``counts_a`` / ``counts_b`` map category -> count; missing categories
    count zero.  Both totals must be positive.
    """
    n_a = sum(counts_a.values())
    n_b = sum(counts_b.values())
    if n_a <= 0 or n_b <= 0:
        raise ValueError("both histograms need at least one sample")
    return 0.5 * sum(
        abs(counts_a.get(c, 0) / n_a - counts_b.get(c, 0) / n_b)
        for c in sorted(categories)
    )


def margin_fields(tv: float, eps_min: float) -> dict:
    """Threshold / margin / near_threshold fields for one tested pair.

    near_threshold rule: margin < eps_min / 8 (see module docstring for the
    rationale: half the per-histogram eps_min/4 TV tolerance at the gate
    count m*).  The comparison is strict, so a margin exactly at eps_min/8
    is not flagged.
    """
    threshold = eps_min / 2.0
    margin = abs(tv - threshold)
    return {
        "threshold": threshold,
        "margin": margin,
        "near_threshold": margin < eps_min / 8.0,
    }


# ----------------------------------------------------------------------
# Recording estimator: identical clustering, plus per-pair outcome records
# ----------------------------------------------------------------------

class RecordingQuotientEstimator(QuotientEstimator):
    """QuotientEstimator that additionally records every pairwise outcome.

    Purely additive: clustering is delegated verbatim to the parent class;
    the (deterministic) TV threshold test is then re-evaluated per pair for
    reporting only, so the recorded verdicts are exactly the verdicts that
    drove union-find.  No core-package file is modified.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_records: list[dict] = []
        self.recorded_alpha0: float | None = None
        self.recorded_m_star: int | None = None

    def _pairwise_cluster(self, viable, samples, alpha):
        uf, undetermined, n_pairs = super()._pairwise_cluster(viable, samples, alpha)
        self.recorded_alpha0 = alpha
        self.recorded_m_star = self._gate_threshold(alpha)
        undetermined_set = set(undetermined)
        for i, ha in enumerate(viable):
            for hb in viable[i + 1:]:
                rec = {
                    "context_a": ha,
                    "context_b": hb,
                    "n_a": samples[ha].n_samples,
                    "n_b": samples[hb].n_samples,
                }
                if (ha, hb) in undetermined_set:
                    rec["verdict"] = Verdict.UNDETERMINED.value
                    rec["empirical_tv"] = None
                else:
                    out = self.equivalence_test.test(
                        samples[ha], samples[hb], alpha=alpha
                    )
                    rec["verdict"] = out.verdict.value
                    rec["empirical_tv"] = float(out.test_statistic)
                self.pair_records.append(rec)
        return uf, undetermined, n_pairs


# ----------------------------------------------------------------------
# Provenance pins (standalone; CertifiedConfig is not used because its
# sidecars assert "certified": true, which this probe must never claim)
# ----------------------------------------------------------------------

def _is_artifact_path(path: str) -> bool:
    parts = path.split("/")
    if parts[-1] == "ARTIFACTS_MANIFEST.json":
        return True
    for i, seg in enumerate(parts[:-1]):
        if seg == "experiments" and any(
            p.startswith("results") for p in parts[i + 1:-1]
        ):
            return True
        if seg == "paper" and parts[i + 1] == "figures":
            return True
    return False


def _capture_pins() -> dict:
    import contextlib
    import platform
    import subprocess

    pins: dict = {"python": platform.python_version(), "packages": {}}
    for mod in ("numpy", "scipy"):
        with contextlib.suppress(ImportError):
            pins["packages"][mod] = __import__(mod).__version__
    try:
        cwd = Path(__file__).resolve().parent
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=cwd,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, cwd=cwd,
        ).stdout.strip()
        dirty_paths = [ln[3:].strip() for ln in porcelain.splitlines() if ln.strip()]
        non_artifact = [p for p in dirty_paths if not _is_artifact_path(p)]
        pins["commit"] = commit + ("-dirty" if non_artifact else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pins["commit"] = "unknown"
    return pins


PROBE_NOTE = (
    "MISSPECIFICATION PROBE: eps_min=0.5 is deliberately set ABOVE the true "
    f"TV ({TAU}) of the engineered pair ((0,), (1,)) of MisspecChain, "
    "violating the gap model (paper Eq. 2). This run exhibits the "
    "uncertified silent-merge failure mode and certifies NOTHING."
)


def _write_sidecar(seed: int, out_dir: Path, pins: dict) -> None:
    try:
        recorded_out = out_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        recorded_out = out_dir.resolve()
    sidecar = {
        "certified": False,                 # this probe certifies nothing
        "misspecification_probe": True,
        "probe_note": PROBE_NOTE,
        "stride": 0,
        "k": WINDOW_LENGTH,
        "tau": 1,   # deprecated block-gap field kept for sidecar layout parity
        "test": "TV_FIXED_ALPHABET",
        "alpha": DELTA,
        "epsilon_min": EPS_MIN,
        "beta": BETA,
        "seeds": [seed],
        "level_correction": "BONFERRONI",
        "command": (
            "python experiments/main_table/run_misspec.py "
            f"--seeds {seed} --out {recorded_out}"
        ),
        "commit": pins["commit"],
        "python": pins["python"],
        "packages": pins["packages"],
        "sampling_scheme": "INDEPENDENT_TRAJECTORY_BLOCK",
        "selection_fraction": SELECTION_FRACTION,
        "context_end": CONTEXT_END,
        "policy_id": "uniform_random",
        "n_trajectories": N_TRAJECTORIES,
        "history_max_length": HISTORY_MAX_LENGTH,
        "min_history_occurrences": MIN_HISTORY_OCCURRENCES,
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "m_star": M_STAR,
        "env_tau_subgap_tv": TAU,
        "true_separated_tv": SEPARATED_TV,
        "subgap_pair": ["(0,)", "(1,)"],
    }
    path = out_dir / f"{ENV_NAME}_seed{seed}.misspec.json"
    path.write_text(json.dumps(sidecar, indent=2, default=str) + "\n")


# ----------------------------------------------------------------------
# One probe run
# ----------------------------------------------------------------------

def run_one(seed: int, out_dir: Path, pins: dict) -> tuple[dict, list[dict]]:
    """Run one seed; return (seed summary row, per-pair rows)."""
    seed_everything(seed)
    env = MisspecChain(tau=TAU, seed=seed)
    alphabet = env.continuation_alphabet(WINDOW_LENGTH)
    est = RecordingQuotientEstimator(
        env=env,
        equivalence_test=TVThresholdEquivalenceTest(alphabet, EPS_MIN),
        delta=DELTA,
        epsilon_min=EPS_MIN,
        power_beta=BETA,
        certified_independent=True,
        selection_fraction=SELECTION_FRACTION,
        independent_context_end=CONTEXT_END,
        include_root=False,
        n_trajectories=N_TRAJECTORIES,
        window_length=WINDOW_LENGTH,
        history_max_length=HISTORY_MAX_LENGTH,
        min_history_occurrences=MIN_HISTORY_OCCURRENCES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
    )
    q = est.run(seed=seed)

    if est.recorded_alpha0 is None or abs(est.recorded_alpha0 - ALPHA_0) > 1e-15:
        print(
            f"  WARNING seed={seed}: realized alpha_0={est.recorded_alpha0} "
            f"differs from the K={EXPECTED_CONTEXTS} budget value {ALPHA_0} "
            "(candidate count deviated from 3)",
            file=sys.stderr,
        )

    true_tvs = env.true_pairwise_tvs()
    pair_rows: list[dict] = []
    for rec in est.pair_records:
        wa = rec["context_a"].observations
        wb = rec["context_b"].observations
        if wb < wa:  # stable lexicographic ordering in the CSV
            wa, wb = wb, wa
            n_a, n_b = rec["n_b"], rec["n_a"]
        else:
            n_a, n_b = rec["n_a"], rec["n_b"]
        true_tv = true_tvs[frozenset((wa, wb))]
        row = {
            "env": ENV_NAME,
            "seed": seed,
            "context_a": str(wa),
            "context_b": str(wb),
            "n_a": n_a,
            "n_b": n_b,
            "m_star": est.recorded_m_star,
            "verdict": rec["verdict"],
            "true_tv": round(true_tv, 6),
            "is_subgap_pair": 0.0 < true_tv < EPS_MIN / 2.0,
        }
        if rec["empirical_tv"] is None:
            row.update(empirical_tv="", threshold=EPS_MIN / 2.0,
                       margin="", near_threshold="")
        else:
            mf = margin_fields(rec["empirical_tv"], EPS_MIN)
            row.update(
                empirical_tv=round(rec["empirical_tv"], 6),
                threshold=mf["threshold"],
                margin=round(mf["margin"], 6),
                near_threshold=mf["near_threshold"],
            )
        pair_rows.append(row)
    pair_rows.sort(key=lambda r: (r["context_a"], r["context_b"]))

    subgap = [r for r in pair_rows if r["is_subgap_pair"]]
    counts = [c for r in pair_rows for c in (r["n_a"], r["n_b"])]
    summary = {
        "seed": seed,
        "n_classes_reported": q.n_classes,
        "n_classes_true": len(set(env.target_partition(HISTORY_MAX_LENGTH).values())),
        "subgap_verdict": subgap[0]["verdict"] if subgap else "",
        "subgap_empirical_tv": subgap[0]["empirical_tv"] if subgap else "",
        "subgap_margin": subgap[0]["margin"] if subgap else "",
        "undetermined_rate": q.undetermined_rate,
        "n_boundary_flags": len(q.boundary_flags),
        "min_inference_count": min(counts) if counts else 0,
        "m_star": est.recorded_m_star,
    }
    _write_sidecar(seed, out_dir, pins)
    return summary, pair_rows


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def write_summary_md(path: Path, summaries: list[dict], pair_rows: list[dict]) -> None:
    n_seeds = len(summaries)
    subgap = [r for r in pair_rows if r["is_subgap_pair"]]
    separated = [r for r in pair_rows if not r["is_subgap_pair"]]
    subgap_same = sum(1 for r in subgap if r["verdict"] == Verdict.SAME.value)
    sep_diff = sum(1 for r in separated if r["verdict"] == Verdict.DIFFERENT.value)
    near_flags = sum(1 for r in pair_rows if r["near_threshold"] is True)
    undetermined = sum(1 for r in pair_rows if r["verdict"] == Verdict.UNDETERMINED.value)
    sub_tvs = [float(r["empirical_tv"]) for r in subgap if r["empirical_tv"] != ""]
    sub_margins = [float(r["margin"]) for r in subgap if r["margin"] != ""]
    min_count = min((s["min_inference_count"] for s in summaries), default=0)
    reported = sorted({s["n_classes_reported"] for s in summaries})
    honest_m_star = required_samples(TAU, ALPHA_0, BETA, EXPECTED_CONTEXTS)
    honest_n = math.ceil(
        honest_m_star * EXPECTED_CONTEXTS / (1.0 - SELECTION_FRACTION)
    )
    lines = [
        "# Misspecification probe (REV-5) — NOT certified; results certify nothing",
        "",
        PROBE_NOTE,
        "",
        "## Configuration",
        "",
        f"- Environment: MisspecChain(tau={TAU}); contexts (0,), (1,), (2,) at "
        f"r={CONTEXT_END}, k={WINDOW_LENGTH}; true pairwise TVs: "
        f"TV((0,),(1,)) = {TAU} (sub-gap), TV((0,),(2,)) = TV((1,),(2,)) = "
        f"{SEPARATED_TV}.",
        f"- eps_min = {EPS_MIN} (SAME threshold eps_min/2 = {EPS_MIN / 2}), "
        f"delta = {DELTA}, beta = {BETA}, Bonferroni alpha_0 = delta/(2 K^2) = "
        f"{ALPHA_0:.6f} at K = {EXPECTED_CONTEXTS}.",
        f"- m* = {M_STAR} per context; budget N = {N_TRAJECTORIES} episodes "
        f"(50/50 split; ~N/6 = {N_TRAJECTORIES // 6} inference blocks per "
        "context).",
        f"- Estimator: certified machinery verbatim (disjoint split, one block "
        "per inference trajectory, fixed-alphabet empirical-TV test, m* gate); "
        "only the gap assumption is violated.",
        f"- near_threshold rule: margin < eps_min/8 = {EPS_MIN / 8}.",
        "",
        "## Headline results",
        "",
        f"- Silent merge on the sub-gap pair ((0,), (1,)): {subgap_same}/{n_seeds} "
        "seeds returned a confident SAME verdict (no abstention, no flag).",
        f"- Mean empirical TV on the sub-gap pair: {_mean(sub_tvs):.4f} "
        f"(true {TAU}); mean margin to threshold: {_mean(sub_margins):.4f}.",
        f"- near_threshold flags: {near_flags}/{len(pair_rows)} pair rows — the "
        "wrong merge is NOT a borderline call under the eps_min/8 rule.",
        f"- Separated pairs: {sep_diff}/{len(separated)} DIFFERENT verdicts "
        "(all correct).",
        f"- Reported class count: {reported} in every seed "
        f"(true quotient: 3) — the merge silently deletes one class.",
        f"- UNDETERMINED pairs: {undetermined}; minimum inference count "
        f"{min_count} >= m* = {M_STAR} (the abstention gate never triggered).",
        f"- A correctly specified run (eps_min = tau = {TAU}) would require "
        f"m* = {honest_m_star} blocks per context, i.e. roughly N >= "
        f"{honest_n} episodes at this layout — the honest price of resolving "
        "the pair.",
        "",
        "## Files",
        "",
        "- misspec_pairs.csv: one row per seed per tested pair "
        "(empirical TV, threshold, margin, verdict, near_threshold, true TV).",
        f"- {ENV_NAME}_seed<seed>.misspec.json: per-run sidecars "
        '("certified": false, "misspecification_probe": true).',
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,1,2")
    ap.add_argument(
        "--out", default=str(Path(__file__).parent / "results_misspec")
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    pins = _capture_pins()

    print(
        f"misspecification probe: eps_min={EPS_MIN} vs true sub-gap TV={TAU}; "
        f"alpha_0={ALPHA_0:.6f}, m*={M_STAR}, N={N_TRAJECTORIES}"
    )
    summaries: list[dict] = []
    all_pairs: list[dict] = []
    t0 = time.time()
    for seed in seeds:
        summary, pair_rows = run_one(seed, out_dir, pins)
        summaries.append(summary)
        all_pairs.extend(pair_rows)
        print(
            f"  seed={seed} reported={summary['n_classes_reported']} "
            f"(true {summary['n_classes_true']}) "
            f"subgap={summary['subgap_verdict']} "
            f"tv={summary['subgap_empirical_tv']} "
            f"margin={summary['subgap_margin']} "
            f"min_count={summary['min_inference_count']} (m*={summary['m_star']})",
            flush=True,
        )

    csv_path = out_dir / "misspec_pairs.csv"
    merged = merge_result_rows(
        csv_path, all_pairs, ("env", "seed", "context_a", "context_b")
    )
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(merged[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(merged)

    write_summary_md(out_dir / "summary.md", summaries, all_pairs)
    print(f"\nWrote {csv_path} and {out_dir / 'summary.md'}; "
          f"elapsed {time.time() - t0:.1f}s")

    # Probe expectation check: the run is only useful if it exhibits the
    # failure mode cleanly at a budget where the gate never abstains.
    failures: list[str] = []
    subgap_rows = [r for r in all_pairs if r["is_subgap_pair"]]
    if not all(r["verdict"] == Verdict.SAME.value for r in subgap_rows):
        failures.append("sub-gap pair not merged (SAME) in every seed")
    if not all(
        r["verdict"] == Verdict.DIFFERENT.value
        for r in all_pairs if not r["is_subgap_pair"]
    ):
        failures.append("a separated pair was not returned DIFFERENT")
    if any(s["min_inference_count"] < s["m_star"] for s in summaries):
        failures.append("an inference count fell below m* (budget undersized)")
    if any(s["undetermined_rate"] > 0 for s in summaries):
        failures.append("unexpected UNDETERMINED pairs")
    if any(s["n_classes_reported"] != 2 for s in summaries):
        failures.append("reported class count deviated from the expected merge (2)")
    if failures:
        print("PROBE EXPECTATION CHECK FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print(
        "probe expectation check: silent merge exhibited in "
        f"{len(summaries)}/{len(summaries)} seeds; separated pairs correct; "
        "no abstention"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
