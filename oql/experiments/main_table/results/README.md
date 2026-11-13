# LEGACY NOTICE — exploratory-mode results (NOT the paper's tables)

Everything in this directory was produced by the **exploratory** pipeline
(`run_all.py`, `run_analysed_config.py`, `run_layer_experiments.py`):
overlapping-window scans scored with the asymptotic G-test or kernel tests,
under the default `QuotientEstimator` configuration
(`certified_independent=False`). That mode estimates a different quantity
than the paper's episode-position predictive quotient — it pools episode
positions and treats overlapping windows as samples — and is explicitly not
covered by the paper's theorems (Section 5, "Implementation checks").

**These numbers are not comparable to the certified tables.** In particular,
the `baseline_*` columns of `main_table.csv` (PSR, EM, naive histogram,
bisimulation oracle) are exploratory-mode outputs with no per-estimand ground
truth; see "Why implemented baselines are not reported in the paper" in
`../../LEGACY.md`.

Some generated files here (e.g. `all_results.md`) carry self-descriptions
from the earlier project phase ("cites this directory for reproducibility");
those statements are stale. The paper's reported artifacts live in the
`../results_*` directories listed in `../README.md` (main table, two
abstention sweeps, nested, head-to-head, multi-endpoint, nondegenerate
policy, post-hoc, misspecification probe), verified by
`scripts/check_paper_numbers.py` and `scripts/gen_manifest.py --check`.
