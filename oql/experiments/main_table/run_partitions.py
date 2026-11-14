"""Per-seed partition outputs for every reported certified configuration.

Concern addressed: the paper reported class COUNTS only.  This script re-runs
each reported configuration with the SAME seeds, captures the quotient that
``QuotientEstimator.estimate`` returns (by wrapping that method, so no core or
runner code is modified), and writes, for every (experiment, endpoint, seed):

  * the recovered partition as a canonical string (classes separated by " | ",
    members by ","; a context is its observation word joined by "-");
    contexts the estimator dropped before testing are listed after "unassigned:";
  * the ground-truth partition (exact, from the environment's closed-form or
    brute-force laws);
  * ``exact_match`` (same/different agreement on every common pair) and
    ``pair_error`` = number of common context pairs whose same/different
    label disagrees with the truth, out of ``n_pairs``.

It also cross-checks the re-run class count against the committed CSV of the
original runner and records the result in the ``count_matches_committed``
column, so that the per-seed table is demonstrably the same run.

Usage (from the project root):
    uv run python experiments/main_table/run_partitions.py \
        [--out experiments/main_table/results_partitions]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from environments.layer1_discrete.kcycle import KCyclePOMDP  # noqa: E402
from environments.layer1_discrete.position_flip import PositionFlipChain  # noqa: E402
from environments.layer1_discrete.posthoc_random import PosthocRandomHMM  # noqa: E402
from environments.layer1_discrete.tiger import TigerPOMDP  # noqa: E402
from environments.layer2_memory.misspec_chain import MisspecChain  # noqa: E402
from environments.layer2_memory.noisy_depth2_kcycle import NoisyDepth2KCycle  # noqa: E402
from quotient_learning.collection import collect_trajectories, uniform_random_policy  # noqa: E402
from quotient_learning.config import seed_everything  # noqa: E402
from quotient_learning.estimator import QuotientEstimator  # noqa: E402
from quotient_learning.types import ObservationHistory  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_oql_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# capture the quotient returned by the estimator
# ----------------------------------------------------------------------
_CAPTURED: list = []
_orig_estimate = QuotientEstimator.estimate


def _capturing_estimate(self, dataset):
    q = _orig_estimate(self, dataset)
    _CAPTURED.append(q)
    return q


QuotientEstimator.estimate = _capturing_estimate


def _take() -> object:
    assert len(_CAPTURED) == 1, f"expected one estimate() call, got {len(_CAPTURED)}"
    return _CAPTURED.pop()


# ----------------------------------------------------------------------
# partition helpers
# ----------------------------------------------------------------------
def _ctx(h) -> tuple[int, ...]:
    return tuple(h.observations) if hasattr(h, "observations") else tuple(h)


def _ctx_str(c: tuple[int, ...]) -> str:
    return "-".join(str(x) for x in c)


def _canon(partition: dict) -> dict[tuple[int, ...], int]:
    return {_ctx(h): int(c) for h, c in partition.items()}


def _fmt(partition: dict[tuple[int, ...], int], universe=()) -> str:
    blocks: dict[int, list] = {}
    for c, cid in partition.items():
        blocks.setdefault(cid, []).append(c)
    ordered = sorted((sorted(b) for b in blocks.values()), key=lambda b: b[0])
    s = " | ".join(",".join(_ctx_str(c) for c in b) for b in ordered)
    missing = sorted(set(universe) - set(partition))
    if missing:
        s += " ; unassigned: " + ",".join(_ctx_str(c) for c in missing)
    return s


def _pair_error(got: dict, truth: dict) -> tuple[int, int]:
    common = sorted(c for c in got if c in truth)
    pairs = list(itertools.combinations(common, 2))
    bad = sum((got[a] == got[b]) != (truth[a] == truth[b]) for a, b in pairs)
    return bad, len(pairs)


def _truth_from_blocks(blocks: list[list[tuple[int, ...]]]) -> dict:
    return {c: i for i, b in enumerate(blocks) for c in b}


def _row(experiment: str, r: int, seed: int, q, truth: dict, committed_n) -> dict:
    got = _canon(q.partition)
    bad, npairs = _pair_error(got, truth)
    missing = [c for c in truth if c not in got]
    return {
        "experiment": experiment,
        "endpoint_r": r,
        "seed": seed,
        "n_contexts_truth": len(truth),
        "n_classes_truth": len(set(truth.values())),
        "n_classes_reported": q.n_classes,
        "partition_reported": _fmt(got, universe=truth),
        "partition_truth": _fmt(truth),
        "n_unassigned": len(missing),
        "n_pairs": npairs,
        "pair_error": bad,
        "exact_match": int(bad == 0 and not missing),
        "undetermined_rate": round(q.undetermined_rate, 4),
        "count_matches_committed": (
            "" if committed_n is None else int(int(committed_n) == q.n_classes)
        ),
    }


def _committed(csv_path: Path, key_cols: dict, col: str) -> dict:
    """{key tuple: committed value of `col`} from a committed runner CSV."""
    out = {}
    with csv_path.open() as f:
        for rec in csv.DictReader(f):
            if all(rec[k] == str(v) for k, v in key_cols.items()):
                out[int(rec["seed"])] = rec[col]
    return out


# ----------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------
def collect_rows(tmp: Path, quick: bool = False) -> list[dict]:
    rows: list[dict] = []
    mt = HERE

    def seeds(n):
        return range(2 if quick else n)

    # --- Table 1: tiger, kcycle_8, noisy_depth2 (20 seeds) ---------------
    rc = _load("run_certified")
    smoke_csv = mt / "results_independent_smoke" / "certified_main.csv"
    truths = {
        "tiger": _truth_from_blocks([[(0,)], [(1,)]]),
        "kcycle_8": _truth_from_blocks([[(i,)] for i in range(8)]),
    }
    d2 = NoisyDepth2KCycle(m=2, eta=0.1, nu=0.2, seed=0)
    truths["noisy_depth2"] = {
        _ctx(h): c for h, c in d2.target_partition(1).items()
    }
    for name in ("tiger", "kcycle_8", "noisy_depth2"):
        comm = _committed(smoke_csv, {"env": name}, "n_classes")
        for seed in seeds(20):
            rc.run_one(name, seed, tmp)
            rows.append(_row(name, 1 if name == "tiger" else 0, seed, _take(),
                             truths[name], comm.get(seed)))
            print(f"  {name} seed={seed}", flush=True)

    # --- position-flip head-to-head (certified arm, 10 seeds) ------------
    hh = _load("run_headtohead")
    comm = _committed(mt / "results_headtohead" / "headtohead.csv",
                      {"path": "certified_r0"}, "n_classes")
    truth = _truth_from_blocks([[(0,)], [(1,)]])
    for seed in seeds(10):
        hh.run_certified_path(seed, tmp)
        rows.append(_row("position_flip", 0, seed, _take(), truth, comm.get(seed)))
        print(f"  position_flip seed={seed}", flush=True)

    # --- multi-endpoint (shared pool, r in {0,1}) -------------------------
    me = _load("run_multiendpoint")
    for seed in seeds(10):
        seed_everything(seed)
        env = PositionFlipChain(seed=seed)
        ds = collect_trajectories(
            env=env, policy=uniform_random_policy(env.n_actions),
            n_episodes=me.N_TRAJ, max_steps=me.MAX_STEPS, seed=seed,
            policy_id="uniform_random",
        )
        for r in me.ENDPOINTS:
            me.run_endpoint(env, ds, r, seed, tmp)
            comm = _committed(mt / "results_multiendpoint" / "multiendpoint.csv",
                              {"r": r}, "n_classes")
            rows.append(_row("position_flip_multi", r, seed, _take(), truth,
                             comm.get(seed)))
        print(f"  multiendpoint seed={seed}", flush=True)

    # --- nested L=2 -------------------------------------------------------
    ne = _load("run_nested")
    env = NoisyDepth2KCycle(m=ne.M_SYMBOLS, eta=ne.ETA, nu=ne.NU, seed=0)
    truth = {_ctx(h): c for h, c in ne.analytic_partition(env).items()}
    comm = _committed(mt / "results_nested" / "nested.csv", {}, "n_classes")
    for seed in seeds(20):
        ne.run_one(seed, tmp)
        rows.append(_row("noisy_depth2_nested_L2", 1, seed, _take(), truth,
                         comm.get(seed)))
        print(f"  nested seed={seed}", flush=True)

    # --- stochastic-policy Tiger -----------------------------------------
    nd = _load("run_nondegenerate")
    laws, _ = nd.tiger_closed_loop_laws()
    part, _gap = nd.true_partition_and_gap(laws)
    truth_blocks = nd.write_ground_truth(tmp)  # also validates eps <= gap
    comm = _committed(mt / "results_nondegenerate" / "nondegenerate_main.csv",
                      {}, "n_classes")
    for seed in seeds(20):
        nd.run_one(seed, truth_blocks, tmp)
        rows.append(_row("tiger_stochastic", 1, seed, _take(), dict(part),
                         comm.get(seed)))
        print(f"  tiger_stochastic seed={seed}", flush=True)

    # --- post-hoc Dirichlet HMM ------------------------------------------
    ph = _load("run_posthoc")
    inst = ph.select_instance_seed()
    henv = PosthocRandomHMM(ph.N_STATES, ph.N_OBS, instance_seed=inst,
                            concentration=ph.CONCENTRATION, seed=0)
    ptruth = ph.write_ground_truth(henv, tmp)
    truth = {_ctx(c): cid for c, cid in ptruth["partition"].items()}
    comm = _committed(mt / "results_posthoc" / "posthoc_main.csv", {}, "n_classes")
    for seed in seeds(20):
        row = ph.run_one(seed, inst, ptruth, tmp)
        if row["abstain_all"]:
            raise RuntimeError(f"posthoc seed {seed}: abstain-all fallback fired")
        rows.append(_row("posthoc_hmm", 0, seed, _take(), truth, comm.get(seed)))
        print(f"  posthoc seed={seed}", flush=True)

    # --- misspecification probe ------------------------------------------
    ms = _load("run_misspec")
    pins = ms._capture_pins()
    menv = MisspecChain(tau=ms.TAU, seed=0)
    truth = {_ctx(h): c for h, c in menv.target_partition(1).items()}
    comm = _committed(mt / "results_misspec" / "summary.csv", {}, "n_classes_reported") \
        if (mt / "results_misspec" / "summary.csv").exists() else {}
    for seed in seeds(10):
        ms.run_one(seed, tmp, pins)
        rows.append(_row("misspec_chain", 0, seed, _take(), truth, comm.get(seed)))
        print(f"  misspec seed={seed}", flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "results_partitions"))
    ap.add_argument("--quick", action="store_true", help="2 seeds per experiment")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        rows = collect_rows(Path(td), quick=args.quick)
    path = out / "partitions_per_seed.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Per-seed partition outputs",
        "",
        "| experiment | r | seeds | exact match | max pair error | distinct reported partitions |",
        "|---|---|---|---|---|---|",
    ]
    keys = sorted({(r["experiment"], r["endpoint_r"]) for r in rows},
                  key=lambda k: [x["experiment"] for x in rows].index(k[0]))
    for exp, r in keys:
        sub = [x for x in rows if x["experiment"] == exp and x["endpoint_r"] == r]
        ex = sum(x["exact_match"] for x in sub)
        mx = max(x["pair_error"] for x in sub)
        dp = len({x["partition_reported"] for x in sub})
        lines.append(f"| {exp} | {r} | {len(sub)} | {ex}/{len(sub)} | {mx} | {dp} |")
    bad = [x for x in rows if x["count_matches_committed"] == 0]
    lines += ["", f"Rows whose re-run class count differs from the committed CSV: {len(bad)}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {path}; elapsed {time.time() - t0:.1f}s")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
