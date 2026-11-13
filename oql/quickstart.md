# Quickstart: reproduce the paper

Minimal walkthrough to verify and regenerate every number reported in
`paper/main.tex` (*Finite-Sample Recovery of Predictive History Quotients
from Independent POMDP Trajectories*). All runs are CPU-only and seeded.

## 0. Install

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/). From the repo
root:

```bash
uv sync --extra dev
```

## 1. Verify the committed artifacts (seconds, no compute)

The paper's Table 1 is an AUTOGEN block generated from the committed CSV and
per-run sidecars; this checker fails if the displayed numbers drift or if any
sidecar stops describing the sampling scheme stated in the paper:

```bash
uv run python scripts/check_paper_numbers.py
# expected: "independent_smoke block: OK", exit 0
```

Check SHA-256 integrity of the certified artifacts, runner sources, core
library, and paper against the committed manifest:

```bash
uv run python scripts/gen_manifest.py --check
# expected on a clean checkout: "manifest OK (<n> artifacts)", exit 0;
# any modified file is reported as "DRIFT <path> <hash>" with exit 1
```

## 2. Regenerate the main table (~3 minutes)

Three environments × 20 seeds, exactly the configuration of Table 1 (Tiger
always-listen with eps_min=0.45, noisy 8-cycle, noisy depth-2 merge case):

```bash
uv run python experiments/main_table/run_certified.py \
    --envs tiger kcycle_8 noisy_depth2 --seeds 0-19 \
    --out experiments/main_table/results_independent_smoke
```

Expected: every seed prints `n=<expected classes> und=0.0` (tiger and
noisy_depth2 recover 2 classes, kcycle_8 recovers 8), and
`certified_main.csv` + one `*.certified.json` sidecar per run are written.
The runner refuses to start if any environment's eps_min exceeds its recorded
analytic TV gap.

## 3. Regenerate the abstention sweep (~1 minute)

8 budgets × 10 seeds on the noisy 8-cycle (Figure 1); also renders the figure
to `paper/figures/abstention_sweep.pdf` (override with `--fig`):

```bash
uv run python experiments/main_table/run_abstention_sweep.py --seeds 0-9 \
    --out experiments/main_table/results_abstention_sweep
```

Expected: abstention collapses from total to zero across N in
[6,500, 8,000]; the run aborts if any (budget, seed) reports a wrong merge
(fewer than 8 classes). Final line: `safety check: no wrong merges in 80
runs`.

## 4. Regenerate the head-to-head (seconds)

Certified vs. exploratory overlapping-window mode on the position-flip chain:

```bash
uv run python experiments/main_table/run_headtohead.py --seeds 0-9 \
    --out experiments/main_table/results_headtohead
```

Expected: `certified: 10/10 seeds recover the r=0 quotient (2 classes)` and
`exploratory: 10/10 seeds merge the pair (pooled estimand)`.

## 5. Regenerate the additional checks (~2 minutes total)

Six further checks reported in Section 5 of the paper (merge-bearing sweep =
Figure 2, nested contexts at L=2, multi-endpoint T=2, nondegenerate
stochastic policy, post-hoc ground truth with data-driven eps, and the
deliberately uncertified misspecification probe):

```bash
uv run python experiments/main_table/run_sweep_depth2.py --seeds 0-9
uv run python experiments/figures/sweep_depth2.py   # renders paper/figures/sweep_depth2.pdf from the CSV
uv run python experiments/main_table/run_nested.py --seeds 0-19
uv run python experiments/main_table/run_multiendpoint.py --seeds 0-9
uv run python experiments/main_table/run_nondegenerate.py --seeds 0-19
uv run python experiments/main_table/run_posthoc.py --seeds 0-19
uv run python experiments/main_table/run_misspec.py --seeds 0-9
```

Each writes a `summary.md` with the headline numbers quoted in the paper.
Expected headlines: sweep_depth2 brackets its gate transition in
N ∈ [2,000, 2,400] with zero incorrect merges; nested, multiendpoint,
nondegenerate, and posthoc all report exact recovery with zero abstention
on every seed; the misspecification probe reports a silent 10/10 SAME merge
on its engineered sub-gap pair (that failure is the finding — its sidecars
are `*.misspec.json` with `"certified": false`).

## 6. Re-check after regenerating

Result rows are deterministic per seed and must match the committed CSVs, so
`scripts/check_paper_numbers.py` must still pass. Sidecar provenance fields
(`commit`, `python`, `packages`, recorded `command`) reflect your checkout,
so the byte-level manifest will report drift on regenerated sidecars; refresh
it and re-verify:

```bash
uv run python scripts/check_paper_numbers.py     # must still print OK
uv run python scripts/gen_manifest.py            # regenerate manifest
uv run python scripts/gen_manifest.py --check    # must now pass
```

## 7. Run the tests (~40 seconds)

```bash
uv run pytest
# expected: 237 passed, 1 skipped (PO-MuJoCo needs the optional mujoco extra)
```

## Scope note

Only the nine runners above (plus the two `scripts/` checkers) constitute
the paper's certified pipeline. Everything else under `experiments/` —
including the PSR/EM/histogram baseline table in
`experiments/main_table/results/` — is exploratory or legacy and is not
comparable to the certified results; see `experiments/LEGACY.md`.
