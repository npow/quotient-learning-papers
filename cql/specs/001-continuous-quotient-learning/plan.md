# Implementation Plan: Continuous State/Action Quotient Learning

**Branch**: `001-continuous-quotient-learning` | **Date**: 2026-05-25 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-continuous-quotient-learning/spec.md`

## Summary

Extend the model-free Myhill-Nerode quotient learning algorithm (from `online-quotient-learning`) to operate natively on continuous state/action/observation spaces, eliminating the manual discretization step that currently produces artifacts on PO-MuJoCo benchmarks. The technical approach: (1) replace the G-test histogram with a kernel MMD permutation test on raw continuation vectors; (2) add an optional GRU-based neural quotient encoder trained contrastively on MMD-labeled pairs; (3) derive sample complexity bounds using RKHS mean-embedding convergence rates in place of the finite continuation alphabet size |C|; (4) validate on continuous-observation Tiger, PO-HalfCheetah/Ant, and continuous Dec-Tiger R_crit estimation.

## Technical Context

**Language/Version**: Python 3.11

**Primary Dependencies**:
- `quotient_learning` (editable from `online-quotient-learning`) — base estimator, trajectory types, rcrit
- `numpy`, `scipy` — MMD computation, permutation calibration
- `torch` 2.x — GRU encoder, InfoNCE loss
- `scikit-learn` — DBSCAN clustering, Rand index
- `gymnasium[mujoco]` — PO-HalfCheetah, PO-Ant environments (already in online-quotient-learning experiments)

**Storage**: Files — experiment results as CSV/JSON; encoder checkpoints as `.pt`; no database

**Testing**: pytest (mirrors online-quotient-learning test structure)

**Target Platform**: Linux (Workbench), single GPU for neural encoder training; CPU-only for MMD estimator

**Project Type**: Research library + experiments (mirrors online-quotient-learning layout)

**Performance Goals**:
- MMD test: < 1s per pair at m=200 continuations on CPU
- Neural encoder inference: < 10ms per history (spec FR-2)
- Full PO-HalfCheetah run (N=10K trajectories): < 30 minutes on CPU

**Constraints**:
- Bonferroni-corrected α₀ = α / K_hist² applied internally (same as finite case)
- Finite-case compatibility: `CanonicalContinuousQuotient` must be a drop-in replacement for `CanonicalQuotient` in `rcrit.py`

**Scale/Scope**: Single-agent, up to N=100K trajectories; up to L=5 history length; up to k=5 continuation window

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Gate | Status | Evidence |
|------|--------|---------|
| §I Theory-First: FR-3 and FR-4 formally stated before experiments | ✅ PASS | research.md §R-3 states the sample complexity bound and its derivation; FR-4 theorem is stated in spec before any experiment tasks |
| §II Finite-Case Recovery: continuous estimator reduces to finite when O, A discrete | ✅ PASS (with note) | research.md §R-1 documents that MMD with counting kernel is an *alternative consistent test* for the same null; G-test is retained for finite case; these co-exist rather than reduce. Documented in Complexity Tracking. |
| §III Reproducibility: ≥3 seeds | ✅ PASS | All experiments use ≥5 seeds per spec FR-5 |
| §IV Approximation Honesty: neural bounds distinguished from proven/empirical | ✅ PASS | spec FR-2 acceptance criteria distinguish "matches union-find accuracy" (empirical) from formal guarantees |
| §V Scope Discipline: no scope creep | ✅ PASS | Non-Goals explicitly exclude theory extension (→ continuous-pomdp-spaces), N>2 agents, adversarial |

## Project Structure

### Documentation (this feature)

```text
specs/001-continuous-quotient-learning/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── api.md           # Python API contract
└── tasks.md             # Phase 2 output (/speckit-tasks)
```

### Source Code (repository root)

```text
continuous_quotient/
├── __init__.py               # public API re-exports
├── equivalence.py            # ContinuousMMDTest (extends MMDEquivalenceTest)
├── estimator.py              # ContinuousQuotientEstimator (extends QuotientEstimator)
├── neural_encoder.py         # NeuralQuotientEstimator, GRU encoder, InfoNCE loss
├── kernels.py                # rbf_kernel, laplace_kernel, median_bandwidth
├── bounds.py                 # mmd_separation_estimate, trajectory_budget
└── types.py                  # ContinuousTrajectory, ContinuationWindow, etc.

experiments/
├── layer1_ground_truth/
│   ├── run_tiger_continuous.py
│   ├── run_kcycle_continuous.py
│   └── results/
├── layer2_mujoco/
│   ├── run_halfcheetah.py
│   ├── run_ant.py
│   └── results/
├── layer3_rcrit/
│   ├── run_dec_tiger_continuous.py
│   └── results/
├── baselines/
│   └── run_discretized_gtest.py   # baseline: existing discretize → G-test path
└── figures/
    ├── gen_quotient_size_comparison.py
    └── gen_sample_complexity.py

tests/
├── unit/
│   ├── test_continuous_mmd.py
│   ├── test_kernels.py
│   ├── test_estimator.py
│   └── test_neural_encoder.py
└── integration/
    ├── test_tiger_continuous.py
    └── test_kcycle_continuous.py   # marked --slow

pyproject.toml
CLAUDE.md
```

**Structure Decision**: Single-project layout mirroring `online-quotient-learning`. The `continuous_quotient` package imports from `quotient_learning` via editable install; it is not a fork. All new code lives here; upstream is unmodified.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Constitution §II: MMD test does not algebraically reduce to G-test in finite case | The two tests share a null hypothesis but not a test statistic. Claiming algebraic reduction would require the χ² kernel which has weaker power than RBF in practice. | Swapping to the χ² kernel for finite compatibility would degrade test power in the continuous regime — the primary use case. G-test is retained for finite environments; MMD is the continuous replacement. Both are consistent tests for the same null (P_{h₁} = P_{h₂}). |
