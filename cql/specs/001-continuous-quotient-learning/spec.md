# Feature: Model-Free Quotient Learning for Continuous State/Action Spaces

**Status**: Draft
**Created**: 2026-05-25
**Author**: npow

## Problem Statement

The online-quotient-learning algorithm (arXiv companion to Nixon 2603.21399/2604.09521) recovers the Myhill-Nerode quotient from interaction trajectories in *finite* POMDPs. Its core equivalence test — a G-test on empirical continuation histograms — requires a discrete, finite continuation alphabet C = (O × A)^{k−1} × O. In continuous-observation or continuous-action environments, C is infinite and histograms degenerate.

The current workaround is manual quantile discretization before running the algorithm: e.g., 3 bins per dimension for PO-HalfCheetah. This is fragile: PO-Ant saturates the 64-code cap on every seed, an artifact of the coarse quantization rather than a meaningful quotient. The practitioner must choose a bin resolution before seeing any results, there is no principled guidance for that choice, and the resulting quotient estimates are discretization-dependent.

This direction eliminates the discretization step. The target object is the same — the fixed-policy continuation-equivalence quotient Q*_{π,k,L} — but the algorithm must recover it directly from continuous trajectory data, using kernel methods for equivalence testing and learned encoders for partition representation.

This is the **algorithmic/learning** complement to the **theoretical** extension developed in `continuous-pomdp-spaces` (which defines what the continuous quotient *is*); this work is about learning it from data.

## Goals

1. Replace the G-test histogram with a kernel two-sample test (MMD with a characteristic kernel) that operates natively on continuous continuation trajectories, with controlled Type I and II error guarantees.
2. Replace the discrete union-find partition with a learned quotient encoder f_θ: H → ℝ^d that clusters histories into quotient classes without fixing a discrete alphabet upfront.
3. Derive sample complexity bounds for continuous spaces using metric entropy (ε-covering numbers) in place of the finite alphabet size |C|.
4. Validate that the resulting algorithm matches or improves upon the discretized baseline on PO-MuJoCo benchmarks, and recovers ground-truth quotients on a continuous-observation Tiger variant.

## Non-Goals

- Proving new *theoretical* results about what the continuous quotient *is* — that is the scope of `continuous-pomdp-spaces`.
- Extending to non-compact (unbounded) observation or action spaces without structural assumptions.
- Real-time / streaming deployment (online variant remains an open question for finite spaces as well).
- Multi-agent quotient negotiation; assumes single-agent continuous quotient learning is solved first.
- Replacing the finite-space algorithm: both will coexist; continuous algorithm is additive.

## User Scenarios

### Scenario 1: Researcher applies quotient learning to a continuous-control POMDP

**As a** multi-agent RL researcher
**I want to** run the quotient learning algorithm on PO-HalfCheetah without choosing a bin resolution
**So that** I get a discretization-free quotient estimate I can trust and compare across environments

**Acceptance Criteria**:
- [ ] Algorithm produces a stable quotient (low std across seeds) on PO-HalfCheetah without any discretization hyperparameter
- [ ] Quotient size is reproducible and not sensitive to kernel bandwidth (median heuristic or cross-validated)
- [ ] Quotient size is strictly smaller than or equal to the 64-code cap that PO-Ant saturates under discretization — confirming the saturation was an artifact

### Scenario 2: Practitioner estimates R_crit for two neural agents with continuous sensors

**As a** robotics engineer deploying a large-sensor-array agent directing a bandwidth-limited controller
**I want to** estimate the minimum communication bandwidth needed for the agents to coordinate
**So that** I can design the communication link before running expensive joint training

**Acceptance Criteria**:
- [ ] End-to-end pipeline: continuous trajectories → learned quotient maps for both agents → R_crit estimate
- [ ] On continuous-observation Dec-Tiger (Gaussian observations), R_crit estimate is within ±10% of the analytical value derived from the continuous-pomdp-spaces theory
- [ ] Pipeline runs without any manual discretization step

### Scenario 3: Algorithm developer benchmarks kernel MMD vs. discretized G-test

**As a** researcher studying statistical methods for quotient recovery
**I want to** compare kernel MMD test, discretized G-test, and neural encoder approaches on a shared benchmark with known ground truth
**So that** I can understand which method is preferable under different data regimes

**Acceptance Criteria**:
- [ ] Continuous-observation Tiger variant: ground-truth quotient size is 2; all three methods recover it
- [ ] Sample complexity comparison: kernel MMD requires fewer trajectories than 3-bin G-test at matching accuracy
- [ ] Neural encoder tracks the correct quotient boundary in a 2D synthetic environment where classes are visualizable

## Functional Requirements

### FR-1: Kernel MMD Equivalence Test for Continuous Continuations

Replace the G-test histogram with a kernel Maximum Mean Discrepancy (MMD) permutation test for the equivalence test subroutine. Given two history classes h₁, h₂, draw k-step continuation trajectories c_{1,j} and c_{2,j}; compute the biased MMD² estimator using a characteristic kernel κ (RBF, Laplace, or IMQ); calibrate the test threshold via permutation to achieve exact Type I error α at finite sample size.

Kernel bandwidth: use the median heuristic on pooled continuations by default; expose a cross-validation option.

**Acceptance Criteria**:
- [ ] MMD permutation test: Type I error ≤ α for all finite m (exact permutation calibration)
- [ ] MMD permutation test: Type II error → 0 as m → ∞ for characteristic kernels (Gretton et al. 2012 Theorem 12)
- [ ] On continuous-observation Tiger: MMD test recovers n=2 classes on ≥9/10 seeds at N≤5,000 trajectories
- [ ] Median heuristic bandwidth selection: stable quotient estimates across 3× variation in bandwidth

### FR-2: Neural Quotient Encoder

Learn an encoder f_θ: H_L → ℝ^d mapping length-≤L history prefixes to quotient class embeddings. The encoder is trained with a margin-based contrastive loss: pairs labeled same-class by the MMD test are attracted, different-class pairs are repelled by at least margin γ.

Quotient partition is extracted by thresholding pairwise embedding distance: histories within ε_embed of each other are merged into one class. Class count is determined automatically (no k to choose).

**Acceptance Criteria**:
- [ ] Encoder trained on K-cycle (finite) recovers K classes with same accuracy as the union-find algorithm
- [ ] On PO-HalfCheetah, encoder quotient size is stable (std ≤ 2 classes across 5 seeds)
- [ ] Embedding space: same-class histories cluster tighter than different-class histories (Davies–Bouldin index < 1.0 on test split)
- [ ] Inference time per history: < 10ms on a single CPU core for history_max_length ≤ 10

### FR-3: Sample Complexity with Covering Numbers

Derive an upper bound on the number of trajectories N needed to recover Q* (to within ε-approximation) when the observation space O ⊆ ℝ^{d_O} and action space A ⊆ ℝ^{d_A} are compact. The |C| term in Theorem 5 of the finite case is replaced by the log ε-covering number of the continuation function class:

  log N(F_C, ε_cover, ‖·‖₂) = O(d_eff · log(1/ε_cover))

where d_eff = d_O · k + d_A · (k−1) is the effective continuation dimensionality. The resulting bound should be:

  N = O(d_eff · n² · k · τ · log(n/δ) / (T · ε²_min))

where ε_min is the minimum MMD separation between distinct quotient classes under the chosen kernel.

**Acceptance Criteria**:
- [ ] Bound stated formally with proof sketch (full proof may defer to companion theory work in continuous-pomdp-spaces)
- [ ] Bound reduces exactly to the finite-alphabet Theorem 5 when O and A are finite discrete sets
- [ ] Empirical sample complexity on PO-HalfCheetah falls within the d_eff-scaled bound on ≥4/5 seeds

### FR-4: ε-Approximate Quotient Recovery

Define an ε-approximate quotient: a partition Q̂_ε such that any two histories h₁, h₂ in the same class of Q̂_ε satisfy MMD(P_{h₁,π}^{(k)}, P_{h₂,π}^{(k)}) ≤ ε, and any two histories in different classes satisfy MMD ≥ ε_min − ε. Prove that the algorithm recovers Q̂_ε with probability ≥ 1−δ given sufficient data.

**Acceptance Criteria**:
- [ ] Definition of ε-approximate quotient is consistent with Nixon's Definition 3 in the limit ε → 0 (exact quotient recovered)
- [ ] Theorem stated: P(Q̂_ε|_{H_N} is ε-approximate) ≥ 1−δ under the standard assumptions (Assumptions 2–7 adapted to continuous spaces)
- [ ] On synthetic continuous POMDP with known ground-truth quotient size n, algorithm recovers n classes at ε = ε_min/2 on all 5 seeds

### FR-5: Benchmark Evaluation Protocol

#### Layer 1 — Continuous-observation ground-truth environments

- **Continuous-observation Tiger**: Gaussian noise N(0, σ²) added to the discrete observations; ground-truth quotient size is 2. Tests whether the algorithm is robust to observation noise.
- **Continuous-observation K-cycle**: observation is a continuous angle on [0, 2π) discretized only implicitly by the agent. Ground-truth quotient = K.

#### Layer 2 — Standard continuous-control POMDPs (no discretization)

- **PO-HalfCheetah**: compare discretized-G-test (3 bins) vs. MMD + neural encoder. Measure: quotient size, seed std, sample efficiency.
- **PO-Ant**: confirm that the saturation artifact (n=64 under discretization) disappears with the continuous algorithm; report the unsaturated quotient size.

#### Layer 3 — R_crit estimation pipeline

- **Continuous Dec-Tiger** (Gaussian observations): two agents, analytically tractable R_crit. Measure end-to-end R_crit estimation error.

**Acceptance Criteria**:
- [ ] Layer 1: both environments recover correct quotient size on ≥9/10 seeds
- [ ] Layer 2: PO-Ant quotient size < 64 (confirming artifact removal); PO-HalfCheetah quotient size ≤ baseline discretized estimate
- [ ] Layer 3: R_crit estimate within ±10% of analytical value on Dec-Tiger (continuous)
- [ ] Ablation table: MMD test vs. G-test (3-bin) vs. G-test (8-bin) on Layer 1 environments

## Success Criteria

- [ ] Continuous MMD equivalence test recovers correct quotient on all Layer 1 environments (≥9/10 seeds)
- [ ] PO-Ant quotient size is no longer saturated at the code cap: the continuous algorithm produces a stable, unsaturated estimate
- [ ] Neural quotient encoder matches union-find partition accuracy on the existing finite benchmarks (regression test)
- [ ] Sample complexity bound with covering numbers derived and stated (proof-level rigor matches Theorem 5 of online-quotient-learning)
- [ ] End-to-end R_crit pipeline validated on at least one continuous environment
- [ ] Paper contribution adds to the online-quotient-learning paper or appears as a standalone companion

## Key Entities

| Entity | Description | Relationships |
|--------|-------------|---------------|
| Continuation Kernel | Characteristic kernel κ: C × C → ℝ used to compare continuation trajectory distributions | Replaces finite continuation histogram; determines MMD test power |
| MMD Equivalence Test | Two-sample test on k-step continuation trajectories using biased MMD² estimator + permutation calibration | Replaces G-test; handles continuous C; feeds into quotient estimator |
| Neural Quotient Encoder | f_θ: H_L → ℝ^d, trained contrastively using MMD-labeled pairs | Replaces union-find discrete partition; yields a continuous embedding space |
| ε-Approximate Quotient | Partition where same-class pairs have MMD ≤ ε and different-class pairs have MMD ≥ ε_min − ε | Continuous analogue of exact quotient recovery; target of FR-4 theorem |
| Covering Number Bound | log N(F_C, ε, ‖·‖₂): metric entropy of the continuation function class | Replaces |C| in sample complexity bounds; dimension-dependent |
| Continuous-Observation Tiger | Tiger POMDP with Gaussian noise added to discrete observations | Layer 1 benchmark; tractable ground truth |

## Assumptions

- Observation and action spaces are compact metric spaces (subsets of ℝ^{d_O} and ℝ^{d_A}); boundedness is assumed.
- The exploration policy π provides sufficient coverage of the observation space, consistent with Assumption 3 of the finite case.
- A characteristic kernel κ exists for the continuation space (RBF is characteristic for ℝ^d by standard theory).
- The minimum MMD separation ε_min between distinct quotient classes is positive (continuous analogue of Assumption 6).
- PO-MuJoCo environments are accessible via the existing simulation infrastructure from online-quotient-learning.

## Dependencies

- `online-quotient-learning`: the finite-space algorithm and equivalence test; this work is an extension of it
- `continuous-pomdp-spaces`: theoretical framework defining the continuous quotient object and ε-approximate recovery
- Nixon (2603.21399): foundational Myhill-Nerode theorem whose quotient is the target object
- Nixon (2604.09521): R_crit construction to be estimated in Layer 3
- Gretton et al. (2012): MMD two-sample test theory (Type I/II error guarantees for characteristic kernels)
- PO-MuJoCo simulation environments (already present in online-quotient-learning/experiments)

## Open Questions

- Does the MMD test require more trajectories than the G-test to achieve the same Type II error, or does avoiding histogram quantization errors offset the kernel approximation overhead?
- What is the right kernel for *trajectory* (sequence) continuation data — RBF on flattened trajectories, or a structured kernel that respects temporal order (e.g., global alignment kernel)?
- Can the neural encoder be trained in a fully unsupervised way (without first running the MMD test to generate labels), e.g., via a variational autoencoder that enforces the quotient structure through its bottleneck?
- Does the ε-approximate quotient size converge monotonically as ε → 0, and is the rate quantifiable?
