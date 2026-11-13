---

description: "Implementation task list for Online Learning of Quotients"
---

# Tasks: Online Learning of Quotients

**Input**: Design documents from `/root/code/online-quotient-learning/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/equivalence_test.md, quickstart.md

**Tests**: Tests INCLUDED — the spec's Constitution gates (Finite-Case Recovery, Theory-First, Reproducibility) and FR acceptance criteria (FR-1/FR-2 correctness on Tiger/RockSample; FR-3 convergence; FR-4 sample-complexity bounds) require executable correctness checks.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Exact file paths are included in every task description

## Path Conventions

Single-project research codebase rooted at `/root/code/online-quotient-learning/`. Library code in `quotient_learning/`, environment wrappers in `environments/`, experiment scripts in `experiments/`, tests in `tests/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project skeleton, dependency manifest, dev tooling.

- [X] T001 Create directory skeleton (`quotient_learning/`, `environments/{base,layer1_discrete,layer2_memory,layer3_continuous,layer4_realworld,layer5_marl}/`, `experiments/{convergence,rcrit_prediction,baselines,main_table,figures}/`, `tests/{unit,integration}/`, `paper/`) with empty `__init__.py` files where Python packages are needed at /root/code/online-quotient-learning/
- [X] T002 Create /root/code/online-quotient-learning/pyproject.toml declaring Python 3.11 project, runtime deps (numpy, scipy, pomdp-py, gymnasium, h5py, joblib, tqdm, matplotlib, seaborn) and `[dev]` extras (pytest, pytest-cov, ruff, mypy, black)
- [X] T003 [P] Create /root/code/online-quotient-learning/.ruff.toml and configure ruff + black formatter
- [X] T004 [P] Create /root/code/online-quotient-learning/tests/conftest.py with shared fixtures (rng seed, tmp HDF5 path, canonical-POMDP fixtures placeholder)
- [X] T005 [P] Create /root/code/online-quotient-learning/pytest.ini with markers (unit, integration, slow) and coverage config
- [X] T006 [P] Create /root/code/online-quotient-learning/README.md mirroring quickstart.md entry points

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Data model, environment interface, dataset I/O, exception types — required by every user story.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T007 Create /root/code/online-quotient-learning/quotient_learning/__init__.py exporting public API symbols (placeholders, filled by later tasks)
- [X] T008 [P] Implement core dataclasses `ObservationHistory`, `QuotientClass`, `CanonicalQuotient`, `ContinuationSample`, `EquivalenceTestResult`, `RCritEstimate` in /root/code/online-quotient-learning/quotient_learning/types.py per data-model.md (immutable, hashable where appropriate)
- [X] T009 [P] Implement custom exceptions `InsufficientSamplesError`, `WindowMismatchError`, `NonErgodicError`, `ConvergenceTimeoutError` in /root/code/online-quotient-learning/quotient_learning/errors.py
- [X] T010 [P] Implement `Trajectory` and `TrajectoryDataset` (immutable container with `environment_id`, `policy_id`, seed metadata) in /root/code/online-quotient-learning/quotient_learning/dataset.py
- [X] T011 [P] Implement abstract `POMDPEnvironment` interface (`reset`, `step`, `n_observations`, `n_actions`, `metadata`) in /root/code/online-quotient-learning/environments/base.py
- [X] T012 [P] Implement HDF5 trajectory I/O (`save_dataset`, `load_dataset`) in /root/code/online-quotient-learning/quotient_learning/io.py
- [X] T013 [P] Implement config logging + seeding utilities (`ExperimentConfig`, `seed_everything`, JSON dump/load) in /root/code/online-quotient-learning/quotient_learning/config.py
- [X] T014 [P] Implement `extract_continuation_samples(dataset, history, window_length)` helper in /root/code/online-quotient-learning/quotient_learning/sampling.py
- [X] T015 Implement abstract `EquivalenceTest` base class (signature from contracts/equivalence_test.md, preconditions check) in /root/code/online-quotient-learning/quotient_learning/equivalence.py
- [X] T016 [P] Add unit tests for data model invariants (history hashability, dataset immutability, exception messages) in /root/code/online-quotient-learning/tests/unit/test_types.py

**Checkpoint**: Foundation ready — user story implementation can now begin in parallel.

---

## Phase 3: User Story 1 — Validate quotient learning on a known POMDP (Priority: P1) 🎯 MVP

**Goal**: A researcher can run the estimator on Tiger / RockSample(7,8) and confirm the learned quotient matches the ground-truth canonical quotient with monotonically improving convergence.

**Independent Test**: `python experiments/convergence/run_tiger.py --n_trajectories 5000 --seeds 3` prints learned quotient size = 2 across all seeds; `pytest tests/integration/test_tiger_recovery.py` passes.

### Tests for User Story 1 ⚠️

**Write these tests FIRST and confirm they FAIL before implementing the corresponding production code.**

- [X] T017 [P] [US1] Contract test `GTestEquivalenceTest` Type I/II error ≤ 0.05 on Tiger continuation samples (100 trials, 1000 samples each) per contracts/equivalence_test.md correctness gate in /root/code/online-quotient-learning/tests/unit/test_gtest.py
- [X] T018 [P] [US1] Contract test `MMDEquivalenceTest` same-distribution vs. different-distribution discrimination on synthetic categorical data in /root/code/online-quotient-learning/tests/unit/test_mmd.py
- [X] T019 [P] [US1] Unit test continuation-sample extraction respects window length and history prefix in /root/code/online-quotient-learning/tests/unit/test_sampling.py
- [X] T020 [P] [US1] Integration test Tiger quotient recovery: `n_classes == 2` within 10k trajectories across ≥3 seeds in /root/code/online-quotient-learning/tests/integration/test_tiger_recovery.py
- [X] T021 [P] [US1] Integration test RockSample(7,8) quotient recovery within 50k trajectories in /root/code/online-quotient-learning/tests/integration/test_rocksample_recovery.py
- [X] T022 [P] [US1] Unit test convergence curve is monotonic-in-expectation (rolling-window check, ≥3 seeds) in /root/code/online-quotient-learning/tests/unit/test_convergence_monotonicity.py

### Implementation for User Story 1

- [X] T023 [P] [US1] Implement `TigerPOMDP` environment (2 hidden states, 2 obs, 3 actions, standard reward) in /root/code/online-quotient-learning/environments/layer1_discrete/tiger.py
- [X] T024 [P] [US1] Implement `RockSamplePOMDP(grid_size, n_rocks)` parameterized for (7,8), (15,15), (20,20) in /root/code/online-quotient-learning/environments/layer1_discrete/rocksample.py
- [X] T025 [P] [US1] Implement `GridWorldPOMDP` (small partially-observable gridworld for sanity checks) in /root/code/online-quotient-learning/environments/layer1_discrete/gridworld.py
- [X] T026 [P] [US1] Implement uniform-random exploration policy and `collect_trajectories(env, policy, n_episodes, max_steps, seed)` in /root/code/online-quotient-learning/quotient_learning/collection.py
- [X] T027 [US1] Implement `GTestEquivalenceTest` (G-statistic on histograms of length-k continuations) in /root/code/online-quotient-learning/quotient_learning/equivalence.py
- [X] T028 [US1] Implement `MMDEquivalenceTest` (kernel MMD² with RBF kernel, permutation test for p-value) in /root/code/online-quotient-learning/quotient_learning/equivalence.py
- [X] T029 [US1] Implement `QuotientEstimator(env, n_trajectories, window_length, equivalence_test)` with offline batch clustering: enumerate observed histories, pairwise equivalence-test, union-find clustering, build `CanonicalQuotient` in /root/code/online-quotient-learning/quotient_learning/estimator.py
- [X] T030 [US1] Implement transition function construction from cluster assignments (`(class_id, observation) → class_id`) inside QuotientEstimator in /root/code/online-quotient-learning/quotient_learning/estimator.py
- [X] T031 [US1] Implement `ConvergenceTracker` recording quotient size and partition refinement at logarithmic checkpoints in /root/code/online-quotient-learning/quotient_learning/convergence.py
- [X] T032 [P] [US1] Experiment script Tiger convergence (writes CSV + JSON to experiments/convergence/results/) in /root/code/online-quotient-learning/experiments/convergence/run_tiger.py
- [X] T033 [P] [US1] Experiment script RockSample(7,8) convergence in /root/code/online-quotient-learning/experiments/convergence/run_rocksample.py
- [X] T034 [P] [US1] README documenting Layer 1 experiments and how to reproduce in /root/code/online-quotient-learning/experiments/convergence/README.md
- [X] T035 [US1] Wire `quotient_learning.__init__` to export `QuotientEstimator`, `GTestEquivalenceTest`, `MMDEquivalenceTest`, `CanonicalQuotient` in /root/code/online-quotient-learning/quotient_learning/__init__.py

**Checkpoint**: User Story 1 fully functional. MVP can be demoed — Tiger and RockSample(7,8) quotients recovered with empirical convergence curves.

---

## Phase 4: User Story 2 — Apply quotient learning to an unknown environment (Priority: P2)

**Goal**: A practitioner deploys the estimator on environments without ground-truth quotient (synthetic 1000-state, T-Maze, PO-MuJoCo, sepsis, robot localization) and obtains a stable quotient with proven sample-complexity bounds; compared against PSR, bisimulation, EM, and equivalence-test ablation baselines.

**Independent Test**: `python experiments/main_table/run_all.py --envs synthetic_1000 tmaze po_halfcheetah sepsis --seeds 5` produces stabilized quotients; sample counts fall within theoretical O(n²) bound; baseline comparison table generated.

### Tests for User Story 2 ⚠️

- [X] T036 [P] [US2] Integration test synthetic 1000-state POMDP — quotient stabilizes (no partition change for ≥3 consecutive checkpoints) in /root/code/online-quotient-learning/tests/integration/test_synthetic_stability.py
- [X] T037 [P] [US2] Integration test T-Maze memory-horizon scaling (quotient size grows linearly with corridor length) in /root/code/online-quotient-learning/tests/integration/test_tmaze.py
- [X] T038 [P] [US2] Unit test sample-complexity bound predicts ≥ empirical sample requirement on Tiger/RockSample-7 (theory ≥ empirical, both ≤ FR-4 polynomial form) in /root/code/online-quotient-learning/tests/unit/test_complexity_bound.py
- [X] T039 [P] [US2] Unit test `EmpiricalBernsteinTest` confidence intervals contain truth at stated level in /root/code/online-quotient-learning/tests/unit/test_empirical_bernstein.py
- [X] T040 [P] [US2] Baseline parity test PSR spectral learning runs to completion on Tiger in /root/code/online-quotient-learning/tests/integration/test_baseline_psr.py
- [X] T041 [P] [US2] Baseline parity test bisimulation-based abstraction (oracle, model-access) runs on Tiger in /root/code/online-quotient-learning/tests/integration/test_baseline_bisimulation.py
- [X] T042 [P] [US2] Baseline parity test tabular POMDP EM runs on Tiger in /root/code/online-quotient-learning/tests/integration/test_baseline_em.py
- [X] T043 [P] [US2] Ablation test all three equivalence tests (G-test / MMD / EmpiricalBernstein) on Tiger; report Type I/II rates in /root/code/online-quotient-learning/tests/integration/test_equivalence_ablation.py

### Implementation for User Story 2

- [X] T044 [P] [US2] Implement `SyntheticPOMDPGenerator(n_states, n_obs, seed)` producing reproducible large discrete POMDPs in /root/code/online-quotient-learning/environments/layer1_discrete/synthetic.py
- [X] T045 [P] [US2] Implement `TMazePOMDP(corridor_length)` and `PassiveTMazePOMDP` in /root/code/online-quotient-learning/environments/layer2_memory/tmaze.py
- [X] T046 [P] [US2] Implement DM Lab memory-task wrapper (deepmind_lab adapter, graceful skip when dm-lab unavailable) in /root/code/online-quotient-learning/environments/layer2_memory/dmlab_wrapper.py
- [X] T047 [P] [US2] Implement PO-MuJoCo HalfCheetah wrapper (velocities masked from observation) in /root/code/online-quotient-learning/environments/layer3_continuous/po_halfcheetah.py
- [X] T048 [P] [US2] Implement PO-MuJoCo Ant wrapper (joint angles partially masked) in /root/code/online-quotient-learning/environments/layer3_continuous/po_ant.py
- [X] T049 [P] [US2] Implement Sepsis offline POMDP wrapper (Komorowski et al. dataset interface) in /root/code/online-quotient-learning/environments/layer4_realworld/sepsis.py
- [X] T050 [P] [US2] Implement robot localization / navigation POMDP in /root/code/online-quotient-learning/environments/layer4_realworld/robot_localization.py
- [X] T051 [US2] Implement `EmpiricalBernsteinTest` (empirical Bernstein concentration on MMD²) in /root/code/online-quotient-learning/quotient_learning/equivalence.py
- [X] T052 [US2] Implement partition-stability convergence criterion (no class merges/splits across W checkpoints) in /root/code/online-quotient-learning/quotient_learning/convergence.py
- [X] T053 [US2] Implement sample-complexity bound `n_trajectories_required(n_classes, window, epsilon, delta) → int` from research.md Q5 derivation in /root/code/online-quotient-learning/quotient_learning/complexity.py
- [X] T054 [US2] Implement function-approximation extension for continuous observations (kernel-binning + MMD on embedded continuations) in /root/code/online-quotient-learning/quotient_learning/estimator.py
- [X] T055 [P] [US2] Implement PSR spectral-learning baseline (Boots et al. 2011 adapted to emit a quotient via cluster-of-PSR-states) in /root/code/online-quotient-learning/experiments/baselines/psr_spectral.py
- [X] T056 [P] [US2] Implement bisimulation-distance baseline (Ferns et al., requires model access — used as oracle upper bound) in /root/code/online-quotient-learning/experiments/baselines/bisimulation.py
- [X] T057 [P] [US2] Implement tabular POMDP EM baseline in /root/code/online-quotient-learning/experiments/baselines/pomdp_em.py
- [X] T058 [P] [US2] Implement naive fixed-window histogram-clustering baseline (equivalence-test ablation) in /root/code/online-quotient-learning/experiments/baselines/histogram_naive.py
- [X] T059 [P] [US2] Scaling experiment driver across RockSample(7,8) → (15,15) → (20,20) → synthetic-1000 with sample-complexity curve plotting in /root/code/online-quotient-learning/experiments/convergence/run_scaling.py
- [X] T060 [P] [US2] Main results table driver — runs all envs × all baselines × 5 seeds, writes CSV + JSON to experiments/main_table/results/ in /root/code/online-quotient-learning/experiments/main_table/run_all.py
- [X] T061 [P] [US2] README for baselines explaining each baseline and acceptance criteria in /root/code/online-quotient-learning/experiments/baselines/README.md
- [X] T062 [P] [US2] README for main_table describing reproduction protocol in /root/code/online-quotient-learning/experiments/main_table/README.md

**Checkpoint**: User Story 2 functional. Sample-complexity scaling figure produced; baseline comparison table populated; at least one Layer 2 or Layer 3 environment shows quotient method holding where ≥1 baseline fails.

---

## Phase 5: User Story 3 — Use learned quotients to compute communication rates (Priority: P3)

**Goal**: A researcher pairs two learned quotients (different bounded-agent capacities), plugs them into the Paper 2 R_crit formula, and predicts the communication phase transition on Dec-Tiger and multi-robot warehouse benchmarks.

**Independent Test**: `python experiments/rcrit_prediction/run.py --env dec_tiger --agent_capacity 2` reports R_crit within 10% relative error of ground truth and phase-transition classification accuracy ≥ 90% across test cases.

### Tests for User Story 3 ⚠️

- [X] T063 [P] [US3] Integration test R_crit pipeline on Dec-Tiger: relative error ≤ 10% in /root/code/online-quotient-learning/tests/integration/test_rcrit_dec_tiger.py
- [X] T064 [P] [US3] Integration test phase-transition classifier accuracy ≥ 90% across ≥10 synthetic agent-pair instances in /root/code/online-quotient-learning/tests/integration/test_phase_transition.py
- [X] T065 [P] [US3] Unit test `RCritEstimate.relative_error` populated correctly when ground truth provided in /root/code/online-quotient-learning/tests/unit/test_rcrit_estimate.py

### Implementation for User Story 3

- [X] T066 [P] [US3] Implement Dec-Tiger Dec-POMDP in /root/code/online-quotient-learning/environments/layer5_marl/dec_tiger.py
- [X] T067 [P] [US3] Implement multi-robot warehouse cooperative POMDP (Semaphore AI variant) in /root/code/online-quotient-learning/environments/layer5_marl/multi_robot_warehouse.py
- [X] T068 [P] [US3] Implement Dec-POMDPx benchmark loader (parse `.dpomdp` files from openspiel/Dec-POMDPx repo) in /root/code/online-quotient-learning/environments/layer5_marl/dec_pomdpx.py
- [X] T069 [US3] Implement `compute_rcrit(quotient_a, quotient_b) → RCritEstimate` per Paper 2 (2604.09521) rate-distortion formula in /root/code/online-quotient-learning/quotient_learning/rcrit.py
- [X] T070 [US3] Implement phase-transition classifier (`feasible` vs `infeasible` from R_crit estimate + channel capacity) in /root/code/online-quotient-learning/quotient_learning/rcrit.py
- [X] T071 [P] [US3] R_crit prediction experiment (end-to-end: collect trajectories for two agents → learn quotients → compute R_crit → predict phase) in /root/code/online-quotient-learning/experiments/rcrit_prediction/run.py
- [X] T072 [P] [US3] README documenting Scenario 3 reproduction in /root/code/online-quotient-learning/experiments/rcrit_prediction/README.md

**Checkpoint**: User Story 3 functional. End-to-end pipeline from trajectories to R_crit prediction validated on Dec-Tiger; phase-transition prediction accuracy ≥ 90%.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Figures, paper scaffold, packaging cleanup, full quickstart validation.

- [X] T073 [P] Programmatic figure-generation script for convergence curves (Tiger, RockSample ladder) in /root/code/online-quotient-learning/experiments/figures/convergence_curves.py
- [X] T074 [P] Programmatic figure-generation script for sample-complexity scaling (empirical vs. theoretical O(n²)) in /root/code/online-quotient-learning/experiments/figures/sample_complexity.py
- [X] T075 [P] Programmatic figure-generation script for main results table heatmap in /root/code/online-quotient-learning/experiments/figures/main_table_figure.py
- [X] T076 [P] Programmatic figure-generation script for R_crit phase-transition diagram in /root/code/online-quotient-learning/experiments/figures/rcrit_phase.py
- [X] T077 [P] LaTeX manuscript scaffold (sections: Intro, Related Work, Algorithm, Theory, Experiments, Discussion) in /root/code/online-quotient-learning/paper/main.tex
- [X] T078 [P] Symlink experiment figure outputs to /root/code/online-quotient-learning/paper/figures/
- [X] T079 Performance pass on `QuotientEstimator` — profile and reduce O(n²) pair test loop via early-termination/locality-sensitive bucketing where safe in /root/code/online-quotient-learning/quotient_learning/estimator.py
- [X] T080 [P] Type-check and lint sweep (ruff, mypy --strict) across /root/code/online-quotient-learning/quotient_learning/ and /root/code/online-quotient-learning/environments/
- [X] T081 Run quickstart.md commands end-to-end and record outputs in /root/code/online-quotient-learning/experiments/convergence/results/quickstart_validation.log
- [X] T082 [P] Cross-cutting docstrings and module-level docs (only where non-obvious WHY) in /root/code/online-quotient-learning/quotient_learning/

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion. BLOCKS all user stories.
- **User Story 1 (Phase 3, P1)**: Depends on Foundational. MVP.
- **User Story 2 (Phase 4, P2)**: Depends on Foundational. Independent of US1 but reuses estimator + equivalence tests built in US1; if developers are sharing, US1 must land first or these tasks must avoid editing the same files concurrently.
- **User Story 3 (Phase 5, P3)**: Depends on Foundational. Independent of US1/US2 conceptually, but the `compute_rcrit` task expects `CanonicalQuotient` objects produced by the estimator — testing requires at least US1 estimator to run.
- **Polish (Phase 6)**: Depends on the user stories whose artifacts are being polished.

### Within Each User Story

- Tests written BEFORE the production code under test (and confirmed failing) per TDD.
- Data types and environments (no inter-dependencies) before the estimator.
- Estimator before convergence/complexity utilities that consume it.
- Library code before experiment scripts that drive it.

### Parallel Opportunities

- All Phase 1 tasks marked [P] (T003–T006) can run in parallel after T001+T002.
- All Phase 2 tasks marked [P] (T008–T014, T016) can run in parallel after T007. T015 depends on T008 (types). T016 depends on T008.
- All US1 tests (T017–T022) can run in parallel before implementation.
- US1 environments (T023–T025) and exploration policy (T026) can be built in parallel.
- US1 experiment scripts (T032–T034) can be authored in parallel once T029 (estimator) lands.
- T027 and T028 (G-test, MMD) both edit /root/code/online-quotient-learning/quotient_learning/equivalence.py — sequence them, do not parallelize.
- T029 and T030 both edit /root/code/online-quotient-learning/quotient_learning/estimator.py — sequence them.
- All US2 environments (T044–T050) can be built in parallel.
- All US2 baselines (T055–T058) can be built in parallel.
- T051 (EmpiricalBernsteinTest) and T054 (continuous extension) both touch shared files (equivalence.py, estimator.py) — sequence within their files.
- All US3 environments (T066–T068) can be built in parallel.
- T069 and T070 both edit /root/code/online-quotient-learning/quotient_learning/rcrit.py — sequence them.
- Polish figure scripts (T073–T076) are independent files — fully parallel.

---

## Parallel Example: User Story 1

```bash
# All US1 tests in parallel (different files, all expected to FAIL first):
Task: "T017 Contract test for GTestEquivalenceTest in tests/unit/test_gtest.py"
Task: "T018 Contract test for MMDEquivalenceTest in tests/unit/test_mmd.py"
Task: "T019 Unit test for continuation-sample extraction in tests/unit/test_sampling.py"
Task: "T020 Integration test for Tiger quotient recovery in tests/integration/test_tiger_recovery.py"
Task: "T021 Integration test for RockSample(7,8) recovery in tests/integration/test_rocksample_recovery.py"
Task: "T022 Unit test for convergence monotonicity in tests/unit/test_convergence_monotonicity.py"

# All US1 environments in parallel:
Task: "T023 Implement TigerPOMDP in environments/layer1_discrete/tiger.py"
Task: "T024 Implement RockSamplePOMDP in environments/layer1_discrete/rocksample.py"
Task: "T025 Implement GridWorldPOMDP in environments/layer1_discrete/gridworld.py"
Task: "T026 Implement exploration policy + collect_trajectories in quotient_learning/collection.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1 (Setup) → Phase 2 (Foundational) → Phase 3 (US1).
2. Stop and validate: `pytest tests/unit tests/integration/test_tiger_recovery.py tests/integration/test_rocksample_recovery.py -v` green; quickstart Tiger command prints learned quotient size = 2.
3. MVP demo-ready: Tiger + RockSample(7,8) recovered with empirical convergence curve.

### Incremental Delivery

1. Setup + Foundational → infrastructure ready.
2. US1 → Constitution gate "Finite-Case Recovery" satisfied. MVP shipped.
3. US2 → FR-3, FR-4, FR-5 satisfied; main results table + scaling figure populated; baselines in place.
4. US3 → Scenario 3 end-to-end demonstrated; paper has the R_crit prediction result.
5. Polish → figures, manuscript scaffold, quickstart validation.

### Parallel Team Strategy

- Lead engineer drives Phase 1 + Phase 2 alone (foundation is dense).
- After Phase 2 checkpoint:
  - Engineer A: US1 (P1, MVP)
  - Engineer B: US2 environments + baselines (T044–T050, T055–T058) — independent files
  - Engineer C: US3 environments (T066–T068) — independent files
- Once US1 estimator lands, Engineers B + C integrate their code with the estimator.

---

## Notes

- [P] = different files, no incomplete-task dependencies.
- [Story] label is mandatory in Phase 3+ for traceability and excluded from Setup/Foundational/Polish.
- Each user story is independently completable and testable.
- Tests are written first and confirmed failing before implementation (TDD per Theory-First gate).
- Commit after each task or logical group.
- Stop at any checkpoint to validate the story before moving on.
- Avoid: vague tasks, same-file conflicts marked [P], cross-story dependencies that break independence.
