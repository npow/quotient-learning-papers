# Certified vs. exploratory/legacy experiments

The paper (`paper/main.tex`) reports results from exactly nine experiment
families, all driven by the certified estimator path
(`QuotientEstimator(certified_independent=True)` +
`TVThresholdEquivalenceTest`: disjoint 50/50 trajectory split, one fixed
block per inference trajectory, fixed-alphabet empirical-TV test, Bonferroni
budget, m* power gate with abstention). The misspecification probe runs that
same machinery verbatim but deliberately violates the gap model; its
sidecars are marked `"certified": false`:

| Runner | Committed artifacts | Paper item |
|---|---|---|
| `main_table/run_certified.py` | `main_table/results_independent_smoke/` | Table 1 |
| `main_table/run_abstention_sweep.py` | `main_table/results_abstention_sweep/` | Figure 1 |
| `main_table/run_sweep_depth2.py` | `main_table/results_sweep_depth2/` | Figure 2 (merge-bearing sweep) |
| `main_table/run_nested.py` | `main_table/results_nested/` | "Nested contexts at L=2" |
| `main_table/run_headtohead.py` | `main_table/results_headtohead/` | head-to-head paragraph |
| `main_table/run_multiendpoint.py` | `main_table/results_multiendpoint/` | "Two endpoints at once" |
| `main_table/run_nondegenerate.py` | `main_table/results_nondegenerate/` | "A nondegenerate stochastic policy" |
| `main_table/run_posthoc.py` | `main_table/results_posthoc/` | "Post-hoc ground truth and a data-driven eps" |
| `main_table/run_misspec.py` | `main_table/results_misspec/` | misspecification probe (uncertified by design) |

These, plus `experiments/scoring.py`, the environments they import, and the
two `scripts/` checkers, are hash-pinned in `experiments/ARTIFACTS_MANIFEST.json`.
Everything below is **not** part of the paper's certified pipeline. Nothing
here is deleted, because tests and the manifest reference some paths; treat
these directories as read-only history.

## Legacy / exploratory directories

Most of these predate the paper's current estimand (the episode-position
predictive quotient) and come from an earlier project phase (Myhill–Nerode
quotients, R_crit communication bounds, benchmark suites). They run the
exploratory mode — overlapping-window scans with the asymptotic G-test or
kernel tests — which the paper explicitly states is not covered by its
theorems (Section 5, "Implementation checks").

- `convergence/` — trajectory-ladder convergence sweeps (Tiger, RockSample,
  k-cycle, T-Maze, scaling) from the earlier project phase, exploratory mode.
  Not in the paper: uncertified estimator configuration and full-suffix
  ground truths, a different estimand from the paper's Table 1.
- `rcrit_prediction/` — Dec-POMDP "R_crit" communication phase-transition
  pipeline from the earlier project. Not in the paper: the current paper
  makes no multi-agent or communication-rate claims at all.
- `tiger_diagnosis/` — ad-hoc Tiger configuration/hyperparameter sweeps used
  while diagnosing the exploratory estimator. Not in the paper: diagnostic
  scratch work on the uncertified path.
- `separation/` — `run_separation.py` + `results/` are the earlier project's
  "coarser than PSR" demonstration; the paper explicitly makes no
  smaller-than-PSR claim (Related work). `run_noisy_depth2.py` and
  `run_aa_refinement.py` + `results_certified/` run certified-style
  configurations but are supplementary material not cited by `main.tex`.
- `baselines/` — implementations of PSR, EM, naive-histogram, bisimulation,
  and CSSR baselines plus `results_certified/` (CSSR comparison). Not in the
  paper: see the rationale section below.
- `mmd_tuning/` — MMD kernel-test ablation on Tiger (exploratory mode;
  `results/` is empty). Not in the paper: the kernel test is exploratory-only.
- `nixon_comparison/` — comparison against the probe-exact quotient of an
  earlier external manuscript. Not in the paper: that framing belongs to the
  earlier project phase; `main.tex` does not cite it.
- `theory/` — Le Cam lower-bound simulation for the earlier project's
  N = Omega(n/(T eps^2)) claim. Not in the paper: the current lower bound
  (Proposition 4) is proved analytically, with a different construction.
- `figures/` — mostly figure scripts (convergence curves, R_crit phase
  plots, main table figure) for the earlier project, EXCEPT
  `figures/sweep_depth2.py`, which renders the paper's Figure 2 from
  `main_table/results_sweep_depth2/sweep_depth2.csv` and is hash-pinned in
  the manifest. The paper's Figure 1 (`abstention_sweep.pdf`) is rendered
  directly by `main_table/run_abstention_sweep.py`.
- `main_table/results/` — exploratory G-test results including the
  baseline-comparison table `main_table.csv` and layer-experiment outputs
  produced by `run_all.py`, `run_analysed_config.py`, and
  `run_layer_experiments.py`. Not in the paper: exploratory estimand and
  protocol; see `main_table/results/README.md`.
- `main_table/results_legacy_gtest/` — quarantined results of an earlier
  G-test run of `run_certified.py`'s environment set, kept for provenance.
  Not in the paper: superseded by `results_independent_smoke/`; see
  `main_table/results_legacy_gtest/README.md`.

The runners `main_table/run_all.py`, `main_table/run_analysed_config.py`, and
`main_table/run_layer_experiments.py` are exploratory drivers retained for
the legacy outputs above.

## Why implemented baselines are not reported in the paper

The repository implements external baselines — PSR spectral
(`baselines/psr_spectral.py`), tabular POMDP EM (`baselines/pomdp_em.py`),
naive histogram clustering (`baselines/histogram_naive.py`), and a
bisimulation oracle (`baselines/bisimulation.py`) — and
`main_table/results/main_table.csv` contains their class counts. They are
not reported in the paper for two verifiable reasons:

1. **They run only on the exploratory path.** Their driver,
   `main_table/run_all.py`, constructs `QuotientEstimator` without
   `certified_independent=True` and without an equivalence test, so it falls
   back to the default `GTestEquivalenceTest` over overlapping windows
   (`quotient_learning/estimator.py`, `equivalence_test or
   GTestEquivalenceTest()`), and the baselines consume that same
   overlapping-window dataset (`dataset = est.collect(...)` in
   `run_all.py`). The paper states this mode "is not covered by
   Theorems 1–2" (Section 5), and the head-to-head experiment shows it can
   estimate a different quantity (the position-pooled law) than the paper's
   estimand.
2. **There is no per-estimand ground truth to score them against.**
   `main_table.csv` carries no expected-classes column, and the only
   available reference — `env.metadata["ground_truth_n_classes"]` —
   describes the full-suffix quotient, generally a *different estimand* from
   the episode-position quotient at the configured endpoint r (see the
   explicit caution in `main_table/run_certified.py` and caveat 1 of
   `main_table/results_legacy_gtest/README.md`). Reporting baseline class
   counts without a per-estimand reference would compare methods on an
   unscoreable target.

The CSSR comparison (`baselines/run_cssr_comparison.py`,
`baselines/results_certified/`) is the one baseline run under a
certified-style same-target protocol; it is likewise not cited by
`main.tex`, and its `noisy_depth2` rows target the depth-2 suffix quotient
(expected 5 classes) rather than the paper's r=0 episode-position estimand
(expected 2), so it is not comparable to Table 1 either.
