# Tasks: Continuous State/Action Quotient Learning

**Input**: Design documents from `specs/001-continuous-quotient-learning/`

**Prerequisites**: plan.md ✅ | spec.md ✅ | research.md ✅ | data-model.md ✅ | contracts/api.md ✅

**Tests**: Unit and integration tests included per spec (research artifact with reproducibility gate from constitution §III).

**Organization**: Tasks grouped by user story. Each story is independently testable.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no blocking dependencies)
- **[US1/2/3]**: User story label (maps to spec.md Scenarios 1–3)

---

## Phase 1: Setup

**Purpose**: Project initialization — pyproject.toml, package skeleton, test and experiment directory structure.

- [ ] T001 Create `pyproject.toml` with dependencies: numpy, scipy, torch, scikit-learn, gymnasium[mujoco]; editable dependency on `online-quotient-learning`
- [ ] T002 Create `continuous_quotient/__init__.py` re-exporting public API per `contracts/api.md`
- [ ] T003 [P] Create `tests/unit/` and `tests/integration/` directory scaffolding with `conftest.py` files
- [ ] T004 [P] Create `experiments/layer1_ground_truth/`, `experiments/layer2_mujoco/`, `experiments/layer3_rcrit/`, `experiments/baselines/`, `experiments/figures/` directories each with a `README.md` stating purpose and expected outputs

**Checkpoint**: `pip install -e .` succeeds; `pytest tests/` collects 0 tests with no errors.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core types, kernel functions, and base equivalence test that ALL user stories require.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T005 Implement `ContinuousTrajectory`, `ContinuationWindow`, `ContinuousEquivalenceResult`, `CanonicalContinuousQuotient`, and `ContinuousEstimatorConfig` in `continuous_quotient/types.py` per `data-model.md`
- [ ] T006 Implement `rbf_kernel`, `laplace_kernel`, and `median_bandwidth` (median heuristic on pooled samples) in `continuous_quotient/kernels.py` per research.md §R-1
- [ ] T007 [P] Implement `ContinuousMMDTest` in `continuous_quotient/equivalence.py`: biased MMD² estimator over raw float32 continuation vectors + permutation calibration; inherits `EquivalenceTest` protocol from `online-quotient-learning`
- [ ] T008 [P] Unit test `ContinuousMMDTest` in `tests/unit/test_continuous_mmd.py`: (a) Type I error ≤ α on 500 permutations at two Gaussian samples with equal mean; (b) Type II error → 0 on well-separated Gaussians; (c) bandwidth fallback to median heuristic when `bandwidth=None`

**Checkpoint**: `pytest tests/unit/test_continuous_mmd.py -v` passes; `ContinuousMMDTest` importable.

---

## Phase 3: User Story 1 — Continuous-Control POMDP Researcher 🎯 MVP

**Goal**: A researcher can run `ContinuousQuotientEstimator.fit()` on continuous-observation environments and recover correct quotients — no discretization step required.

**Independent Test**: `python experiments/layer1_ground_truth/run_tiger_continuous.py --seeds 10` recovers n=2 on ≥9/10 seeds; `python experiments/layer2_mujoco/run_ant.py --mode continuous` produces a quotient size < 64 (saturation artifact removed).

### Implementation for User Story 1

- [ ] T009 [US1] Implement `ContinuousQuotientEstimator.fit()` in `continuous_quotient/estimator.py`: enumerate histories from `ContinuousTrajectory` list, extract `ContinuationWindow` samples per history, run pairwise `ContinuousMMDTest` with Bonferroni-corrected α, union-find merge → `CanonicalContinuousQuotient`
- [ ] T010 [P] [US1] Implement `ContinuousQuotientEstimator.fit_from_env()` convenience wrapper in `continuous_quotient/estimator.py`: collect trajectories via `collect_trajectories` from `online-quotient-learning`, then call `fit()`
- [ ] T011 [P] [US1] Implement Gaussian-observation Tiger environment in `experiments/layer1_ground_truth/envs/tiger_continuous.py`: state ∈ {Left, Right}, observation = state_float + N(0, σ²), discrete actions {listen, open-left, open-right}; ground-truth n=2
- [ ] T012 [P] [US1] Implement continuous K-cycle environment in `experiments/layer1_ground_truth/envs/kcycle_continuous.py`: state = continuous angle θ ∈ [0, 2π) advancing by 2π/K per step; observation = sin(θ) + N(0, 0.01); ground-truth n=K
- [ ] T013 [US1] Write experiment script `experiments/layer1_ground_truth/run_tiger_continuous.py`: run `ContinuousQuotientEstimator` on Gaussian Tiger for 10 seeds; log `(seed, n_classes, runtime_s)` to `results/tiger_continuous.csv`; assert n=2 on ≥9/10 seeds
- [ ] T014 [P] [US1] Write experiment script `experiments/layer1_ground_truth/run_kcycle_continuous.py`: sweep K ∈ {2,3,4,5,8} for 5 seeds each; log to `results/kcycle_continuous.csv`; assert recovered class count = K on all seeds
- [ ] T015 [US1] Adapt PO-HalfCheetah experiment from `online-quotient-learning/experiments/` to use `ContinuousQuotientEstimator` (remove `discretize_trajectory` call) in `experiments/layer2_mujoco/run_halfcheetah.py`; log `(seed, n_classes, runtime_s)` to `results/halfcheetah_continuous.csv`
- [ ] T016 [US1] Adapt PO-Ant experiment similarly in `experiments/layer2_mujoco/run_ant.py`; log to `results/ant_continuous.csv`; primary assertion: `n_classes < 64` (saturation artifact removal)
- [ ] T017 [US1] Integration test `tests/integration/test_tiger_continuous.py`: run 5 seeds with N=2000 trajectories; assert ≥4/5 seeds recover n=2; mark `@pytest.mark.integration`

**Checkpoint**: Scenarios 1 acceptance criteria met: stable quotient on PO-HalfCheetah (std ≤ 2 across seeds); PO-Ant no longer saturates.

---

## Phase 4: User Story 2 — R_crit Pipeline for Neural Agents

**Goal**: A practitioner can estimate R_crit between two agents with continuous sensor streams — end-to-end from trajectories to bandwidth threshold, with a neural quotient encoder option.

**Independent Test**: `python experiments/layer3_rcrit/run_dec_tiger_continuous.py --seeds 10` reports R_crit within ±10% of analytical value.

### Implementation for User Story 2

- [ ] T018 [US2] Implement `GRUHistoryEncoder` (2-layer GRU, hidden=128, input=d_O+d_A → L2-normalised 64-dim projection) in `continuous_quotient/neural_encoder.py` per `data-model.md` §NeuralQuotientEncoder
- [ ] T019 [US2] Implement InfoNCE contrastive training loop in `continuous_quotient/neural_encoder.py`: generate positive/negative pairs from MMD-labeled history partition, train for N epochs, log `TrainingMeta.val_rand_index` per epoch
- [ ] T020 [P] [US2] Implement DBSCAN cluster extraction and `NeuralQuotientEstimator.fit()` in `continuous_quotient/neural_encoder.py`: Stage 1 pretrain option + Stage 2 fine-tune; output `CanonicalContinuousQuotient` from embedding clusters
- [ ] T021 [P] [US2] Unit test neural encoder in `tests/unit/test_neural_encoder.py`: (a) forward pass shape check; (b) contrastive loss decreases over 10 training steps on synthetic same/different pairs; (c) `EncoderNotFittedError` raised before `.fit()`
- [ ] T022 [P] [US2] Implement continuous Dec-Tiger environment in `experiments/layer3_rcrit/envs/dec_tiger_continuous.py`: two-agent Dec-POMDP, each agent receives independent Gaussian-perturbed observation; expose `reset()` / `step()` returning `(obs_A, obs_B, reward, done)`
- [ ] T023 [US2] Write R_crit estimation experiment in `experiments/layer3_rcrit/run_dec_tiger_continuous.py`: run `ContinuousQuotientEstimator` for both agents, pass quotient labels to `rcrit.classify_phase_transition()` (reused from `online-quotient-learning`), log `(seed, rcrit_estimated, rcrit_analytical, rel_error_pct)` to `results/dec_tiger_rcrit.csv`; assert mean rel error ≤ 10%

**Checkpoint**: US2 acceptance criteria met: R_crit pipeline runs end-to-end without discretization; ±10% accuracy on Dec-Tiger continuous.

---

## Phase 5: User Story 3 — Algorithm Developer Benchmarking

**Goal**: An algorithm developer can compare kernel MMD vs. discretized G-test and inspect sample complexity bounds — with a full ablation table and theoretical bound computation.

**Independent Test**: `python experiments/baselines/run_ablation.py` produces an ablation table showing MMD requires fewer trajectories than 3-bin G-test at matching quotient accuracy on continuous Tiger.

### Implementation for User Story 3

- [ ] T024 [US3] Implement discretized G-test baseline experiment in `experiments/baselines/run_discretized_gtest.py`: apply `quantile_discretize(n_bins=3)` and `quantile_discretize(n_bins=8)` then `QuotientEstimator` from `online-quotient-learning`; run on the same continuous Tiger and K-cycle environments; log to `results/baseline_discretized.csv`
- [ ] T025 [P] [US3] Implement `mmd_separation_estimate()` and `trajectory_budget()` in `continuous_quotient/bounds.py` per `contracts/api.md` §ε-Approximate Quotient Utilities; `trajectory_budget` implements N = O(n² kτ log(n/δ) / (Tε²_MMD))
- [ ] T026 [P] [US3] Unit test `bounds.py` in `tests/unit/test_bounds.py`: (a) `trajectory_budget` is monotone increasing in n, decreasing in ε_mmd; (b) reduces to finite-case Theorem 5 formula when |C| substituted for kernel term
- [ ] T027 [US3] Write ablation experiment `experiments/baselines/run_ablation.py`: sweep N ∈ {500, 1000, 2000, 5000} trajectories; for each N compare {MMD, G-test-3bin, G-test-8bin} on continuous Tiger; log Rand index vs. ground truth to `results/ablation_sample_efficiency.csv`
- [ ] T028 [P] [US3] Integration test `tests/integration/test_kcycle_continuous.py`: run 5 seeds on K-cycle K=5; assert ≥4/5 seeds recover n=5; mark `@pytest.mark.integration --slow`
- [ ] T029 [US3] Generate quotient size comparison figure `experiments/figures/gen_quotient_size_comparison.py`: bar chart (continuous MMD vs. discretized G-test) on Tiger, K-cycle-8, PO-HalfCheetah, PO-Ant; save to `experiments/figures/quotient_size_comparison.pdf`
- [ ] T030 [P] [US3] Generate sample complexity figure `experiments/figures/gen_sample_complexity.py`: plot theoretical O(n²) bound from `trajectory_budget` vs. empirical N required on Layer 1 environments; save to `experiments/figures/sample_complexity.pdf`

**Checkpoint**: All three user stories independently functional; ablation table shows MMD advantage for continuous Tiger.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Reproducibility compliance (constitution §III), documentation, final validation.

- [ ] T031 [P] Write `continuous_quotient/README.md` with package overview, install instructions, and links to `specs/001-continuous-quotient-learning/`
- [ ] T032 [P] Write `experiments/README.md` summarising all five layers, expected runtimes, and seed/config reproducibility log format
- [ ] T033 Validate quickstart.md: run every command from `specs/001-continuous-quotient-learning/quickstart.md` in a clean virtualenv; fix any divergence between docs and code
- [ ] T034 [P] Add `pytest.ini` / `pyproject.toml` markers: `slow` (MuJoCo integration tests), `integration`; document in contributing notes
- [ ] T035 [P] Verify constitution §II compliance: add a `tests/unit/test_finite_compat.py` that instantiates `ContinuousMMDTest` on discrete integer-encoded observations and confirms it accepts the same trajectory input format as the finite `QuotientEstimator` (interface compatibility)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — BLOCKS all user story phases
- **US1 (Phase 3)**: Depends on Phase 2; no dependency on US2 or US3
- **US2 (Phase 4)**: Depends on Phase 2 and **T009** (ContinuousQuotientEstimator needed to generate MMD labels for encoder training); otherwise independent of US3
- **US3 (Phase 5)**: Depends on Phase 2 and **T009**, **T011**, **T012** (needs continuous environments and estimator for ablation); parallel with US2 after T009
- **Polish (Phase 6)**: Depends on all desired story phases complete

### Within-Story Dependencies

```
Phase 3:
  T005 → T009 → T013 → T015, T016
  T006 → T007 → T009
  T011, T012 → T013, T014   (environments before experiment scripts)
  T009 → T017               (estimator before integration test)

Phase 4:
  T009 → T018 → T019 → T020 → T023
  T022 → T023               (environment before experiment script)

Phase 5:
  T009, T011, T012 → T024, T027
  T025 → T026               (bounds before unit test)
  T027 → T029, T030         (results before figures)
```

### Parallel Opportunities Within Stories

**Phase 3 (US1)** — after T009 is done, launch in parallel:
```
Task T011: tiger_continuous.py env
Task T012: kcycle_continuous.py env
Task T015: run_halfcheetah.py
Task T016: run_ant.py
```

**Phase 4 (US2)** — after T019 is done, launch in parallel:
```
Task T020: NeuralQuotientEstimator.fit() + DBSCAN
Task T021: neural encoder unit tests
Task T022: dec_tiger_continuous.py env
```

**Phase 5 (US3)** — after T009 is done, launch in parallel:
```
Task T024: baseline discretized experiment
Task T025: bounds.py implementation
Task T028: kcycle integration test
```

---

## Implementation Strategy

### MVP (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational — **blocks everything**
3. Complete Phase 3: US1 — continuous estimator + Layer 1/2 validation
4. **STOP and VALIDATE**: `python experiments/layer1_ground_truth/run_tiger_continuous.py --seeds 10` passes; PO-Ant quotient size < 64

### Incremental Delivery

1. Setup + Foundational → `ContinuousMMDTest` importable and tested
2. US1 → Continuous quotient estimation works, no discretization required (MVP paper experiment)
3. US2 → Neural encoder + R_crit pipeline for continuous Dec-Tiger
4. US3 → Ablation table, sample complexity figures, full comparison

---

## Notes

- [P] tasks have no file conflicts and can run simultaneously
- [US1/2/3] labels trace each task back to its user story acceptance criteria
- All experiment scripts log seed, config, and results to CSV for reproducibility (constitution §III)
- MuJoCo integration tests are marked `--slow`; run separately from unit tests in CI
- `online-quotient-learning` is a read-only upstream dependency — no modifications to it
