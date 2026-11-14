"""Verify that the paper's independent-trajectory table matches artifacts.

The table in ``paper/main.tex`` is deliberately generated from the committed
CSV and its per-run certified configuration sidecars.  This checker fails if
the displayed numbers drift or if a sidecar no longer describes the sampling
scheme and policy stated in the paper.  ``--fix`` rewrites only the marked
AUTOGEN block.

Rows aggregate over all committed seeds per environment: ``recovered`` must
be identical across seeds (the checker fails on a mixed column), and the
undetermined rate shown is the maximum over seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "main.tex"
RESULTS = ROOT / "experiments" / "main_table" / "results_independent_smoke"
CSV_PATH = RESULTS / "certified_main.csv"

ENV_SPECS = {
    "tiger": {
        "label": "Tiger (always listen)",
        "policy_id": "always_listen",
        "context_end": 1,
        "n_trajectories": 20_000,
        "history_max_length": 1,
        "max_steps_per_episode": 15,
        "epsilon_min": 0.45,   # <= analytic TV gap 0.49
    },
    "kcycle_8": {
        "label": "noisy 8-cycle",
        "policy_id": "uniform_random",
        "context_end": 0,
        "n_trajectories": 8_000,
        "history_max_length": 1,
        "max_steps_per_episode": 20,
        "epsilon_min": 0.5,    # <= analytic TV gap 0.7225
    },
    "noisy_depth2": {
        "label": "noisy depth-2 (merge)",
        "policy_id": "uniform_random",
        "context_end": 0,
        "n_trajectories": 16_000,
        "history_max_length": 2,
        "max_steps_per_episode": 60,
        "epsilon_min": 0.5,    # <= analytic TV gap 0.7524
    },
}


def _load_and_validate() -> dict[str, list[dict[str, str]]]:
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_env: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_env.setdefault(row["env"], []).append(row)
    if set(by_env) != set(ENV_SPECS):
        raise SystemExit(
            f"expected exactly {sorted(ENV_SPECS)}, found {sorted(by_env)}"
        )

    for env, spec in ENV_SPECS.items():
        for row in by_env[env]:
            seed = int(row["seed"])
            sidecar_path = RESULTS / f"{env}_seed{seed}.certified.json"
            data = json.loads(sidecar_path.read_text())
            expected = {
                "certified": True,
                "test": "TV_FIXED_ALPHABET",
                "sampling_scheme": "INDEPENDENT_TRAJECTORY_BLOCK",
                "selection_fraction": 0.5,
                "policy_id": spec["policy_id"],
                "context_end": spec["context_end"],
                "n_trajectories": spec["n_trajectories"],
                "history_max_length": spec["history_max_length"],
                "max_steps_per_episode": spec["max_steps_per_episode"],
                "epsilon_min": spec["epsilon_min"],
            }
            mismatches = {
                key: (data.get(key), value)
                for key, value in expected.items()
                if data.get(key) != value
            }
            if mismatches:
                raise SystemExit(f"invalid sidecar {sidecar_path}: {mismatches}")
    return by_env


def render() -> str:
    by_env = _load_and_validate()
    lines = []
    for env, spec in ENV_SPECS.items():
        rows = by_env[env]
        episodes = spec["n_trajectories"]
        n_seeds = len(rows)
        expected_vals = {int(r["expected"]) for r in rows}
        recovered_vals = {int(r["n_classes"]) for r in rows}
        if len(expected_vals) != 1:
            raise SystemExit(f"{env}: inconsistent expected column {expected_vals}")
        if len(recovered_vals) != 1:
            raise SystemExit(
                f"{env}: recovered classes differ across seeds {recovered_vals}; "
                "the aggregated table cannot represent this"
            )
        und_max = max(float(r["undetermined_rate"]) for r in rows)
        lines.append(
            f"    {spec['label']} & {episodes:,} & {n_seeds} "
            f"& {expected_vals.pop()} & {recovered_vals.pop()} "
            f"& {und_max:.2f}\\\\"
        )
    return "\n".join(lines)


PART_CSV = (ROOT / "experiments" / "main_table" / "results_partitions"
            / "partitions_per_seed.csv")
PART_LABELS = {
    "tiger": "Tiger (always listen)",
    "kcycle_8": "noisy 8-cycle",
    "noisy_depth2": "noisy depth-2 (merge)",
    "position_flip": "position-flip chain",
    "position_flip_multi": "position-flip, two endpoints",
    "noisy_depth2_nested_L2": "noisy depth-2, nested $L=2$",
    "tiger_stochastic": "Tiger, stochastic policy",
    "posthoc_hmm": "random HMM (data-driven $\\eps$)",
    "misspec_chain": "misspecification probe $\\dagger$",
}
PART_SEEDS = {
    "tiger": 20, "kcycle_8": 20, "noisy_depth2": 20, "position_flip": 10,
    "position_flip_multi": 10, "noisy_depth2_nested_L2": 20,
    "tiger_stochastic": 20, "posthoc_hmm": 20, "misspec_chain": 10,
}


def _tex_partition(text: str) -> str:
    blocks = [b.strip() for b in text.split("|")]
    return " $\\mid$ ".join(blocks)


def render_partitions() -> str:
    with PART_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    lines = []
    keys: list[tuple[str, str]] = []
    for r in rows:
        k = (r["experiment"], r["endpoint_r"])
        if k not in keys:
            keys.append(k)
    if {k[0] for k in keys} != set(PART_LABELS):
        raise SystemExit(f"partition experiments mismatch: {sorted({k[0] for k in keys})}")
    for exp, r_ in sorted(keys, key=lambda k: (list(PART_LABELS).index(k[0]), k[1])):
        sub = [x for x in rows if x["experiment"] == exp and x["endpoint_r"] == r_]
        if len(sub) != PART_SEEDS[exp]:
            raise SystemExit(f"{exp} r={r_}: {len(sub)} seeds != {PART_SEEDS[exp]}")
        bad = [x for x in sub if x["count_matches_committed"] == "0"]
        if bad:
            raise SystemExit(f"{exp}: re-run class count differs from committed CSV")
        parts = {x["partition_reported"] for x in sub}
        if len(parts) != 1:
            raise SystemExit(f"{exp} r={r_}: partitions differ across seeds ({len(parts)}); "
                             "the table caption assumes identical partitions")
        exact = sum(int(x["exact_match"]) for x in sub)
        worst = max(int(x["pair_error"]) for x in sub)
        npairs = {int(x["n_pairs"]) for x in sub}.pop()
        lines.append(
            f"    {PART_LABELS[exp]} & {r_} & {len(sub)} & {exact}/{len(sub)} "
            f"& {worst}/{npairs} & {_tex_partition(parts.pop())}\\\\"
        )
    return "\n".join(lines)


def check_block(tex: str, name: str, expected: str, fix: bool) -> tuple[str, int]:
    pattern = re.compile(
        rf"(% BEGIN AUTOGEN: {name}\n)(.*?)(% END AUTOGEN: {name})", re.DOTALL)
    m = pattern.search(tex)
    if not m:
        print(f"missing {name} AUTOGEN block", file=sys.stderr)
        return tex, 1
    expected += "\n"
    if m.group(2) == expected:
        print(f"{name} block: OK")
        return tex, 0
    if fix:
        print(f"fixed {name} block")
        return pattern.sub(lambda mm: mm.group(1) + expected + mm.group(3),
                           tex, count=1), 0
    print(f"{name} block differs from artifacts:\n--- paper ---\n{m.group(2)}"
          f"--- artifacts ---\n{expected}", file=sys.stderr)
    return tex, 1


def main_partitions(fix: bool) -> int:
    tex = TEX.read_text()
    tex, rc = check_block(tex, "partitions", render_partitions(), fix)
    if fix:
        TEX.write_text(tex)
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args(argv)

    tex = TEX.read_text()
    pattern = re.compile(
        r"(% BEGIN AUTOGEN: independent_smoke\n)(.*?)(% END AUTOGEN: independent_smoke)",
        re.DOTALL,
    )
    match = pattern.search(tex)
    if not match:
        print("missing independent_smoke AUTOGEN block", file=sys.stderr)
        return 1
    expected = render() + "\n"
    if match.group(2) == expected:
        print("independent_smoke block: OK")
        return main_partitions(args.fix)
    if args.fix:
        tex = pattern.sub(
            lambda m: m.group(1) + expected + m.group(3), tex, count=1
        )
        TEX.write_text(tex)
        print("fixed independent_smoke block")
        return main_partitions(True)
    print(
        "independent_smoke block differs from artifacts:\n"
        f"--- paper ---\n{match.group(2)}--- artifacts ---\n{expected}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
