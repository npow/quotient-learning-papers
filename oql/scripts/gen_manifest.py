"""Artifact manifest: SHA-256 registry of the paper and reported pipeline.

Writes experiments/ARTIFACTS_MANIFEST.json mapping repo-relative paths to
SHA-256 hashes for all committed certified artifacts (result CSVs, summary
markdown, per-run sidecars) plus the dependency lockfile. `--check` verifies
the committed manifest against the working tree and exits 1 on any drift,
giving reviewers a one-command integrity check:

    python scripts/gen_manifest.py          # (re)generate
    python scripts/gen_manifest.py --check  # verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "experiments" / "ARTIFACTS_MANIFEST.json"

GLOBS = [
    "experiments/main_table/results_independent_smoke/*",
    "experiments/main_table/results_abstention_sweep/*",
    "experiments/main_table/results_headtohead/*",
    "experiments/main_table/results_sweep_depth2/*",
    "experiments/main_table/results_nested/*",
    "experiments/main_table/results_multiendpoint/*",
    "experiments/main_table/results_nondegenerate/*",
    "experiments/main_table/results_posthoc/*",
    "experiments/main_table/results_misspec/*",
    "experiments/main_table/results_partitions/*",
    "experiments/main_table/run_partitions.py",
    "experiments/main_table/check_tiger_stochastic.py",
    "experiments/main_table/run_certified.py",
    "experiments/main_table/run_abstention_sweep.py",
    "experiments/main_table/run_headtohead.py",
    "experiments/main_table/run_sweep_depth2.py",
    "experiments/main_table/run_nested.py",
    "experiments/main_table/run_multiendpoint.py",
    "experiments/main_table/run_nondegenerate.py",
    "experiments/main_table/run_posthoc.py",
    "experiments/main_table/run_misspec.py",
    "experiments/figures/sweep_depth2.py",
    "environments/layer1_discrete/position_flip.py",
    "environments/layer1_discrete/posthoc_random.py",
    "environments/layer2_memory/misspec_chain.py",
    "experiments/scoring.py",
    "environments/base.py",
    "environments/layer1_discrete/kcycle.py",
    "environments/layer1_discrete/tiger.py",
    "environments/layer2_memory/noisy_depth2_kcycle.py",
    "quotient_learning/*.py",
    "scripts/check_paper_numbers.py",
    "scripts/gen_manifest.py",
    "paper/main.tex",
    "paper/references.bib",
    "paper/checklist.tex",
    "paper/neurips_2026.sty",
    "pyproject.toml",
    "uv.lock",
]


def collect() -> dict[str, str]:
    entries: dict[str, str] = {}
    for pattern in GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                entries[str(path.relative_to(ROOT))] = digest
    return entries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify committed manifest against working tree")
    args = ap.parse_args(argv)

    current = collect()
    if args.check:
        if not MANIFEST.exists():
            print("manifest missing; run scripts/gen_manifest.py", file=sys.stderr)
            return 1
        committed = json.loads(MANIFEST.read_text())["artifacts"]
        drift = sorted(
            set(committed.items()) ^ set(current.items())
        )
        if drift:
            for path, digest in drift:
                print(f"DRIFT {path} {digest[:12]}", file=sys.stderr)
            return 1
        print(f"manifest OK ({len(current)} artifacts)")
        return 0

    MANIFEST.write_text(json.dumps(
        {"schema": 1, "hash": "sha256", "artifacts": current}, indent=2
    ) + "\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(current)} artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
