# Finite-Sample Recovery of Predictive History Quotients

Code and certified artifacts for the paper *Finite-Sample Recovery of
Predictive History Quotients from Independent POMDP Trajectories*
(`paper/main.tex`).

The paper studies a deliberately narrow, identifiable abstraction problem:
under one fixed data-collection policy, which bounded observation contexts
have the same finite-horizon continuation law? It defines the
**episode-position predictive quotient** — an explicitly policy- and
position-dependent estimand — and an estimator that selects contexts on one
half of independently reset trajectories, takes at most one prespecified
continuation block per trajectory from the other half, compares
fixed-alphabet histograms by empirical total variation, and **abstains**
whenever a pair lacks the samples required by its own concentration bound.
Under an epsilon-gap and a per-context visitation floor, exact recovery holds
with probability 1 − delta from
O((d + log(K/delta)) / (mu_min · eps²)) trajectories, with a matching
two-point lower bound in mu_min and eps. Implementation checks on three
analytically solvable environments recover the exact quotients on all 20
seeds each; two budget sweeps show the power gate degrading through
abstention — pure abstention on the 8-cycle, over-splitting on the
merge-bearing chain, never an incorrect merge; a head-to-head shows
the overlapping-window mode systematically merging contexts the
episode-position quotient separates (pooling positions changes the
estimand); and further checks exercise nested contexts (L=2), simultaneous
multi-endpoint recovery (T=2), a nondegenerate stochastic policy, a
data-driven eps recipe validated against post-hoc brute-force ground truth,
and a deliberate gap-model violation that exhibits the uncertified
silent-merge regime.

## Certified path (what the paper reports)

Exactly nine experiment families back the numbers in `paper/main.tex`. Each
run writes a result row plus a validated sidecar recording the full
configuration, seed, package pins, and git commit (`*.certified.json`; the
misspecification probe writes `*.misspec.json` with `"certified": false`
because it deliberately violates the gap model).

| Paper item | Runner | Committed artifacts |
|---|---|---|
| Table 1 (main table, 3 envs × 20 seeds) | `experiments/main_table/run_certified.py` | `experiments/main_table/results_independent_smoke/` |
| Figure 1 (abstention sweep, 8 budgets × 10 seeds) | `experiments/main_table/run_abstention_sweep.py` | `experiments/main_table/results_abstention_sweep/` |
| Figure 2 (merge-bearing sweep, 8 budgets × 10 seeds) | `experiments/main_table/run_sweep_depth2.py` | `experiments/main_table/results_sweep_depth2/` |
| Nested contexts at L=2 (20 seeds) | `experiments/main_table/run_nested.py` | `experiments/main_table/results_nested/` |
| Head-to-head (certified vs. exploratory, 10 seeds) | `experiments/main_table/run_headtohead.py` | `experiments/main_table/results_headtohead/` |
| Two endpoints at once (T=2, 10 seeds) | `experiments/main_table/run_multiendpoint.py` | `experiments/main_table/results_multiendpoint/` |
| Nondegenerate stochastic policy (20 seeds) | `experiments/main_table/run_nondegenerate.py` | `experiments/main_table/results_nondegenerate/` |
| Post-hoc ground truth + data-driven eps (20 seeds) | `experiments/main_table/run_posthoc.py` | `experiments/main_table/results_posthoc/` |
| Misspecification probe (uncertified by design, 10 seeds) | `experiments/main_table/run_misspec.py` | `experiments/main_table/results_misspec/` |

Everything else under `experiments/` is exploratory or legacy and is **not**
part of the paper's claim set — see `experiments/LEGACY.md` for the
directory-by-directory classification, including why the implemented
PSR/EM/histogram baselines are not reported.

## Quickstart

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/). From the repo
root:

```bash
cd 2026-05-online-quotient-learning
uv sync --extra dev     # install pinned dependencies (incl. pytest) from uv.lock
```

**Verify the committed artifacts (no compute, seconds):**

```bash
# Paper table must match the committed CSV + sidecars (exits non-zero on drift)
uv run python scripts/check_paper_numbers.py

# SHA-256 integrity of all certified artifacts, sources, and the paper
uv run python scripts/gen_manifest.py --check
```

**Regenerate the certified results from scratch:**

```bash
# Main table: tiger, kcycle_8, noisy_depth2 × seeds 0-19 (~3 min total, CPU)
uv run python experiments/main_table/run_certified.py \
    --envs tiger kcycle_8 noisy_depth2 --seeds 0-19 \
    --out experiments/main_table/results_independent_smoke

# Abstention sweep: 8 budgets × seeds 0-9 (~1 min); also renders the figure
uv run python experiments/main_table/run_abstention_sweep.py --seeds 0-9 \
    --out experiments/main_table/results_abstention_sweep

# Head-to-head on the position-flip chain: seeds 0-9 (seconds)
uv run python experiments/main_table/run_headtohead.py --seeds 0-9 \
    --out experiments/main_table/results_headtohead
```

Result rows are deterministic given the seed and must reproduce the committed
CSVs exactly. Sidecar provenance fields (`commit`, `python`, `packages`,
recorded `command`) reflect *your* checkout, so after regenerating, refresh
the manifest with `uv run python scripts/gen_manifest.py` before re-running
`--check`. See `quickstart.md` for the step-by-step walkthrough.

## Run the tests

```bash
uv run pytest           # ~40 s; 237 pass, PO-MuJoCo skips without the mujoco extra
```

Tests cover the TV calculation, the m* sample bound, split disjointness, the
one-block-per-trajectory invariant, the Bonferroni level, action-aware
alignment, root exclusion, and sidecar validation.

## Repository layout

- `quotient_learning/` — core library: `QuotientEstimator`
  (`certified_independent=True` is the theorem-facing mode),
  `TVThresholdEquivalenceTest`, `CertifiedConfig` sidecars. The default
  `GTestEquivalenceTest` and the overlapping-window scan are the
  **exploratory** mode, not covered by the paper's theorems; legacy modules
  (e.g. `rcrit.py`) from an earlier project phase remain for test coverage.
- `environments/` — synthetic POMDPs. The paper uses `tiger`, `kcycle`,
  `noisy_depth2_kcycle`, and `position_flip`; the rest serve exploratory
  scripts.
- `experiments/main_table/` — certified runners and their committed result
  directories (`results_independent_smoke/`, `results_abstention_sweep/`,
  `results_headtohead/`). `results/` and `results_legacy_gtest/` are
  exploratory/legacy — see the README inside each.
- `experiments/` (other subdirectories) — exploratory and legacy experiments;
  classified in `experiments/LEGACY.md`.
- `experiments/ARTIFACTS_MANIFEST.json` — SHA-256 registry of every certified
  artifact, the runner sources, the core library, and the paper.
- `scripts/` — `check_paper_numbers.py` (paper table ↔ artifacts),
  `gen_manifest.py` (manifest generate/`--check`).
- `paper/` — LaTeX manuscript (`main.tex`), references, checklist.
- `tests/` — unit + integration tests (`uv run pytest`).
- Historical spec-kit documents (`plan.md`, `spec.md`, `research.md`,
  `tasks.md`, `data-model.md`) describe an earlier project phase and do not
  reflect the current paper's claims.

## License

MIT — see `LICENSE`. The submission uses only synthetic environments
implemented in this repository; no external datasets, GPUs, or human data.
