# Main table — certified runners and result directories

The paper's reported numbers come from the **certified** runners in
this directory (finite-sample path: disjoint 50/50 trajectory split, one
fixed block per inference trajectory, fixed-alphabet empirical-TV test,
Bonferroni budget, m* power gate with abstention):

| Runner | Output directory | Paper item |
|---|---|---|
| `run_certified.py` | `results_independent_smoke/` | Table 1 (3 envs × 20 seeds) |
| `run_abstention_sweep.py` | `results_abstention_sweep/` + `paper/figures/abstention_sweep.pdf` | Figure 1 |
| `run_sweep_depth2.py` | `results_sweep_depth2/` (Figure 2 PDF is rendered separately by `experiments/figures/sweep_depth2.py`) | Figure 2 (merge-bearing sweep) |
| `run_nested.py` | `results_nested/` | "Nested contexts at L=2" (20 seeds) |
| `run_headtohead.py` | `results_headtohead/` | certified-vs-exploratory head-to-head |
| `run_multiendpoint.py` | `results_multiendpoint/` | "Two endpoints at once" (Corollary: multi-endpoint, T=2) |
| `run_nondegenerate.py` | `results_nondegenerate/` | "A nondegenerate stochastic policy" (20 seeds) |
| `run_posthoc.py` | `results_posthoc/` | "Post-hoc ground truth and a data-driven eps" (20 seeds) |
| `run_misspec.py` | `results_misspec/` | misspecification probe — deliberately violates the gap model; sidecars are `*.misspec.json` with `"certified": false` |

## Reproduce

```bash
uv run python experiments/main_table/run_certified.py \
    --envs tiger kcycle_8 noisy_depth2 --seeds 0-19 \
    --out experiments/main_table/results_independent_smoke

uv run python experiments/main_table/run_abstention_sweep.py --seeds 0-9 \
    --out experiments/main_table/results_abstention_sweep

uv run python experiments/main_table/run_headtohead.py --seeds 0-9 \
    --out experiments/main_table/results_headtohead

uv run python experiments/main_table/run_sweep_depth2.py --seeds 0-9 \
    --out experiments/main_table/results_sweep_depth2

uv run python experiments/main_table/run_nested.py --seeds 0-19 \
    --out experiments/main_table/results_nested

uv run python experiments/main_table/run_multiendpoint.py --seeds 0-9 \
    --out experiments/main_table/results_multiendpoint

uv run python experiments/main_table/run_nondegenerate.py --seeds 0-19 \
    --out experiments/main_table/results_nondegenerate

uv run python experiments/main_table/run_posthoc.py --seeds 0-19 \
    --out experiments/main_table/results_posthoc

uv run python experiments/main_table/run_misspec.py --seeds 0-9 \
    --out experiments/main_table/results_misspec
```

Then verify `uv run python scripts/check_paper_numbers.py` (paper table ↔
artifacts) and `uv run python scripts/gen_manifest.py --check` (SHA-256
integrity). Every run writes a `*.certified.json` sidecar recording its full
configuration, seed, package pins, and commit. `run_certified.py` supports
additional environments beyond the three published ones; rows relying on
`env.metadata` ground truths must not be published without a per-estimand
re-derivation (see the caution in the script).

## Output schema (`certified_main.csv`)

| Column | Notes |
|--------|-------|
| `env`, `seed` | run identity |
| `n_classes` | recovered quotient classes |
| `expected` | per-estimand analytic class count (blank if unknown) |
| `undetermined_rate` | abstaining pair fraction |
| `n_boundary_flags` | class pairs whose separation rests on an abstention |

## Legacy content in this directory

`run_all.py`, `run_analysed_config.py`, and `run_layer_experiments.py` are
exploratory drivers (overlapping windows, G-test); their outputs in
`results/` — including the PSR/EM/histogram baseline columns — and the
quarantined `results_legacy_gtest/` are **not** part of the paper's claim
set and are not comparable to the certified tables. See the README inside
each directory and `../LEGACY.md`.
