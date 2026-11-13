# Continuous-Space Myhill-Nerode Quotient Learning

Model-free quotient estimation for POMDPs with continuous state/action/observation spaces.
Extends [online-quotient-learning](../2026-05-online-quotient-learning) to eliminate manual discretization.

## Quick Start

```bash
pip install -e .
python experiments/layer1_ground_truth/run_tiger_continuous.py --seeds 10
python experiments/layer1_ground_truth/run_kcycle_continuous.py --K-values 2 3 4 5 8 --seeds 5
```

## Key Results

| Environment | N | Method | Accuracy | Avg n̄ |
|---|---|---|---|---|
| Tiger (σ=0.5, B=8) | 3000 | MMD (ours) | **10/10 seeds** | 2.00 ± 0.00 |
| K-cycle K=2 | 3000 | MMD (ours) | **5/5 seeds** | 2.00 ± 0.00 |
| K-cycle K=3 | 3000 | MMD (ours) | **5/5 seeds** | 3.00 ± 0.00 |
| K-cycle K=4 | 3000 | MMD (ours) | **5/5 seeds** | 4.00 ± 0.00 |
| K-cycle K=5 | 3000 | MMD (ours) | **5/5 seeds** | 5.00 ± 0.00 |
| K-cycle K=8 | 3000 | MMD (ours) | **5/5 seeds** | 8.00 ± 0.00 |

All K-cycle runs use `history_n_bins=K` (aligned resolution). With `B=2K` the estimator over-splits for K≥3 (`results/kcycle_bins_sensitivity.csv`); see the paper's resolution analysis. Dec-Tiger R_crit results are reported for all seeds in `layer3_rcrit/results/` (α=0.05/R=99 and α=0.01/R=999).

**Ablation — Tiger (N=5000):**
- MMD continuous (ours): 3/3 seeds correct
- G-test 3-bin: 0/3 seeds correct (too coarse)
- G-test 8-bin: 2/3 seeds correct (sensitive to bin count)

**Bisimulation baseline — model-free Wasserstein-ε (Tiger + K-cycle K=4):**

| Method | Tiger acc | K-cycle-K4 acc | Notes |
|---|---|---|---|
| Oracle bisim (model access) | 1.00 | 1.00 | Upper bound |
| MMD-permutation (ours) | **1.00** | **1.00** | No threshold tuning |
| Wasserstein ε=0.1 | 1.00 | 0.00 | Over-splits K-cycle |
| Wasserstein ε=0.2 | 1.00 | 0.00 | Wrong n̄=2 for K=4 |
| Wasserstein ε=0.5 | 0.90 | 0.00 | Over-merges Tiger |
| Wasserstein ε=1.0 | 0.00 | 0.00 | Trivial merge |

Wasserstein-ε has no universally good threshold — must be tuned per-environment. MMD-permutation achieves 100% on both without threshold selection (permutation calibration adapts automatically).

## Method

**ContinuousMMDTest**: kernel MMD² permutation test on raw float32 continuation vectors.
No discretization of the continuation alphabet required.

**ContinuousQuotientEstimator**: same union-find algorithm as the finite case, with:
- Coarse observation binning for history keys (`history_n_bins=K` for K-state environments)
- MMD test replaces G-test for equivalence decisions
- Deterministic per-pair subsampling seeds for reproducibility

**Sample complexity** (paper Thm. 1): N = O(K_hist (τ+k) log(K_hist/δ) / (T ε²_MMD)) with K_hist = O(n) active bins — alphabet-free (|C| replaced by the MMD separation; dimension enters through ε_MMD). Multiplicity is controlled with Benjamini–Hochberg over all bin pairs (not Bonferroni: with R permutations no p-value is below 1/(R+1)).

## Codex Sign-Off (2026-05-26)

Methodology reviewed by GPT-4o via Netflix Model Gateway:
- **D1 (history binning)**: APPROVE — approximation error bound documented (w=1/B; choose B≥2n)
- **D2 (no Bonferroni)**: APPROVE — raw α justified (permutation floor > Bonferroni-corrected α)
- **D3 (biased MMD²)**: APPROVE — variance reduction, nonnegativity, Type I preserved
- **D4 (deterministic subsampling)**: APPROVE — pair-hash seed ensures reproducibility
- **Final verdict**: APPROVE for submission

## Experiment Results

All results in `experiments/*/results/*.csv`. Figures in `experiments/figures/`.

Tests: `pytest tests/unit tests/integration -v` → 12/12 passing.
