# LEGACY NOTICE — deprecated G-test sweep (NOT the certified path)

Exploratory-mode results under a different estimand and protocol; **not
comparable to the certified tables** in `../results_independent_smoke/`.
See `../../LEGACY.md` for the full certified/exploratory classification.
Details:

These results were produced by an earlier pipeline using the asymptotic
G-test with strided overlapping-window sampling
(`"test": "G_RAW_FIXED_ALPHABET"` in every sidecar, commit `01cf4f1`).
That code path is explicitly **not covered** by the finite-sample theorems in
the paper (see "Implementation checks": the exploratory mode "is not covered
by Theorems 1-2").

The sidecars carry `certified: true` because the field predates the current
`CertifiedConfig` validation; today's loader would reject them.  Do **not**
cite these numbers as certified results.

Two additional caveats for anyone re-running the wider environment set:

1. The `expected` column values here were read from
   `env.metadata["ground_truth_n_classes"]`, which describes the full-suffix
   quotient — generally a *different estimand* from the episode-position
   quotient at the configured endpoint `r`.  Per-estimand ground truths must
   be re-derived before comparison (see the `noisy_depth2` entry in
   `run_certified.py` for a worked example).
2. The high abstention rates in the harder environments (tmaze, sepsis,
   robot_localization, synthetic_200) were measured under the legacy
   sampling scheme and do not quantify the current TV-path estimator.

The current certified artifacts live in `../results_independent_smoke/`
(main table) and `../results_abstention_sweep/` (power-gate sweep).
