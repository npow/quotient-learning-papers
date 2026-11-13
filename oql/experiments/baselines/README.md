# Baselines — implemented, NOT reported in the paper

> **NOTICE:** these baselines run on the exploratory path (via
> `../main_table/run_all.py`) with no per-estimand ground truth, so the paper
> does not report them; see "Why implemented baselines are not reported in
> the paper" in `../LEGACY.md`. The CSSR comparison
> (`run_cssr_comparison.py`, `results_certified/`) uses a certified-style
> same-target protocol but is likewise not cited by `paper/main.tex`.
> The description below is kept for historical provenance.

Four baselines from research.md Q3 — each isolates a methodological difference
from our equivalence-test approach.

| File | Method | What it tests |
|------|--------|---------------|
| `psr_spectral.py` | Predictive State Representations (Boots et al. 2011) | Sample complexity vs spectral methods |
| `bisimulation.py` | Exact bisimulation partition (Ferns et al. 2004) | Oracle upper bound — needs model access |
| `pomdp_em.py` | Tabular POMDP Baum-Welch EM | Naive model-learning baseline |
| `histogram_naive.py` | L¹-distance histogram clustering | Equivalence-test ablation |

## Running

Baselines are imported by `experiments/main_table/run_all.py`. They expose
free functions returning `CanonicalQuotient` for direct comparison against
`QuotientEstimator` output.

```python
from experiments.baselines.psr_spectral import psr_quotient
from experiments.baselines.histogram_naive import histogram_naive_quotient
from experiments.baselines.pomdp_em import em_quotient
from experiments.baselines.bisimulation import bisimulation_quotient
```

## Acceptance criteria

- All four baselines run to completion on Tiger and produce a finite quotient.
- On at least one Layer-2 or Layer-3 environment, the equivalence-test learner
  produces a finite quotient where ≥ 1 baseline fails (degenerate quotient,
  or non-convergence within budget).
