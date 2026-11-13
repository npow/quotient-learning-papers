# Research: Continuous State/Action Quotient Learning

**Feature**: 001-continuous-quotient-learning
**Phase**: 0 — resolved before design

---

## R-1: Kernel Choice for Continuous Continuation Trajectories

**Decision**: RBF kernel on flattened k-step continuation vectors as the default; expose a trajectory-structured Global Alignment Kernel (GAK) as an optional alternative.

**Rationale**: A k-step continuation is a fixed-length sequence c = (o_1, a_1, …, o_k) ∈ (O × A)^{k-1} × O. When O and A are Euclidean, flattening c into ℝ^{k·d_O + (k-1)·d_A} and applying an RBF kernel is computationally O(m²) in the number of continuations and is characteristic (hence MMD-consistent). The median-heuristic bandwidth σ² = median({‖c_i − c_j‖²}) is fast, parameter-free at test time, and empirically reliable in the existing MMDEquivalenceTest.

GAK (Cuturi 2011) respects temporal alignment but costs O(m² · k²) and introduces an additional regularisation parameter. For fixed-horizon k-step windows where the sequence length is small (k ≤ 5) and temporal ordering is already implicit in the flattening order, the gain does not justify the complexity.

**Alternatives considered**:
- Laplace kernel: characteristic, slightly heavier tails; similar practical performance to RBF; RBF preferred for familiarity.
- χ² kernel κ(x,y) = 1[x=y]/p(x) on discrete spaces: reduces MMD² to a weighted χ² divergence, connecting to the G-test. Kept as a theoretical bridge but not the default; requires known p(x).
- Sliced Wasserstein (SWD): appealing geometry but not a two-sample test statistic; would require a different calibration framework.

**Finite-case reduction note** (constitution §II): With a counting kernel κ(x,y) = 1[x=y] on a discrete alphabet, biased MMD² = ‖p̂₁ − p̂₂‖²₂, the L₂ distance between empirical histograms. This is *not* the G-test (which is 2 Σ n_i log(n_i/e_i)), but both are consistent two-sample tests for the same null (P_{h₁} = P_{h₂}). Constitution compliance: the finite case is handled by the G-test (which is retained and superior for discrete), while the MMD test is the continuous replacement. These co-exist; the spec does not claim MMD *reduces* to G-test, only that both test the same null hypothesis.

---

## R-2: Neural Encoder Architecture

**Decision**: GRU history encoder (hidden size 128) → L2-normalised projection head (d=64) trained with a temperature-scaled contrastive loss (SimCLR-style InfoNCE) on MMD-labeled pairs.

**Rationale**: History sequences (o_1, a_1, …, o_t) are variable-length up to L; a GRU processes them in a single forward pass without padding ambiguity. The projection head compresses to d=64 which is large enough to separate ≥32 quotient classes empirically (K-cycle benchmark) but small enough for DBSCAN clustering to be fast. The contrastive objective is: for a batch of (anchor, positive, negatives) where positive = same MMD class, negatives = different MMD class, InfoNCE maximises agreement and minimises confusion.

Cluster extraction: DBSCAN with ε_embed set to the 10th percentile of pairwise distances (or equivalently, a fixed fraction of median pairwise distance). This auto-tunes the number of clusters without requiring k. For sanity: also report hierarchical clustering dendrogram cutoff as an alternative.

**Training curriculum**: Stage 1 — train on cheap finite benchmarks (K-cycle, Tiger) to initialise encoder. Stage 2 — fine-tune on target continuous environment using online-generated MMD labels.

**Alternatives considered**:
- Transformer encoder: better for long histories but O(L²) attention cost; not needed for L ≤ 10.
- VAE with quotient bottleneck: appealing unsupervised framing but the reconstruction objective competes with the quotient objective; the InfoNCE approach is cleaner.
- Fixed-length CNN over padded history: simpler but loses temporal ordering for variable-length histories.

---

## R-3: Sample Complexity via Covering Numbers

**Decision**: The MMD test sample complexity is O(1/ε²_MMD) independently of the ambient dimension d, because the RKHS norm converges at rate 1/√m regardless of d. The covering number of the continuation space enters only in the Bonferroni correction over possible distinct history pairs, not in the per-pair test sample requirement.

**Rationale**: In the finite case, |C| enters because there are |C| buckets to fill in each histogram, requiring m = Ω(|C|/ε²) samples for the G-test to detect separation ε in TV distance. For MMD, the RKHS mean embedding μ_P = E_P[φ(c)] (in the feature space) converges at rate 1/√m regardless of the dimensionality of the RKHS, by the RKHS central limit theorem (Gretton et al. 2012). The test threshold (from permutation or asymptotic calibration) also has the same dimension-free rate.

Therefore the main sample complexity bound is:
  N = O(n² · k · τ · log(n · K_hist / δ) / (T · ε²_MMD))

where ε_MMD = min_{[h₁]≠[h₂]} MMD(P_{h₁,π}^{(k)}, P_{h₂,π}^{(k)}) is the minimum MMD separation between distinct classes (the continuous analogue of ε_min), and K_hist = |H_N| (number of observed distinct histories) controls the Bonferroni correction. Note |C| does NOT appear; it is replaced by K_hist and the per-pair m = O(1/ε²_MMD) term.

The covering number argument is still needed for the ε-approximate quotient theorem (FR-4): to bound how well the MMD test can approximate TV separation when the continuation alphabet is continuous. This uses that TV(P, Q) ≤ C · MMD(P, Q)^{1/2} (for Lipschitz densities with bounded support), providing a bridge between the kernel-based and TV-based separation.

**Alternatives considered**:
- Direct VC-dimension argument: works but gives worse dimension-dependent constants.
- Fisher information / local minimax approach: tighter but requires smoothness on the density class; more complex.

---

## R-4: PO-MuJoCo Reuse Strategy

**Decision**: Reuse the existing `environments/` wrapper from `online-quotient-learning` via a pip editable install. The PO-HalfCheetah and PO-Ant wrappers already expose continuous observation vectors; the discretization step lives in the trajectory preprocessing, not the environment. Remove the `quantile_discretize` call and pass raw float32 observations directly to the continuous estimator.

**Rationale**: The separation is already clean in the existing code. The environments expose `obs: np.ndarray[float32]` and actions `act: np.ndarray[float32]` (MuJoCo continuous control). The current pipeline adds a `discretize_trajectory(n_bins=3)` step in the experiment scripts before calling `QuotientEstimator`. Removing that step and routing to `ContinuousQuotientEstimator` is a one-line change in the experiment runner.

---

## R-5: Continuous-Observation Tiger Benchmark Design

**Decision**: Gaussian-observation Tiger: state s ∈ {Left, Right}, observation o = s_float + N(0, σ²), action a ∈ {listen, open-left, open-right} (discrete). Ground-truth quotient under the finite algorithm is n=2 (confirmed by existing Tiger result). The continuous version with σ=0.1 is still distinguishable at k=2 steps: TV(P_Left^{(2)}, P_Right^{(2)}) ≈ 0.8 at σ=0.1 (computed analytically).

Also add a **Continuous K-cycle** variant: state is a continuous angle θ ∈ [0, 2π), observation is o = sin(θ + noise) with noise ~ N(0, 0.01). The quotient classes are still K discrete clusters on the circle; the continuous MMD test should recover them.

**Analytical R_crit for continuous Dec-Tiger**: Since both agents see the same Gaussian-perturbed observations independently, Q_A = Q_B = 2 classes; R_crit = H(Q_A | Q_B) = 0 (agents have identical quotient structure so one bit of side information suffices). Wait — for Dec-Tiger where both agents' quotient structures are symmetric, R_crit stays well-defined; we'll compute it analytically at test design time.

---

## R-6: R_crit Estimation for Continuous Agents

**Decision**: Estimate R_crit = H(Q_A | Q_B) via plug-in entropy estimation on the discrete quotient labels output by the continuous estimator. If agents A and B each produce a discrete class label (from clustering), joint entropy H(Z_A, Z_B) and marginal H(Z_B) can be estimated with a standard plugin estimator over the empirical joint distribution of (class_A, class_B) pairs collected from Dec-POMDP trajectories.

**Rationale**: The R_crit formula from Nixon (2604.09521) is defined over the quotient alphabets, which are discrete even in the continuous-space setting (the algorithm outputs a discrete partition). So R_crit estimation does not require new continuous entropy estimators; it reuses `rcrit.py` from `online-quotient-learning` unchanged.

**Alternatives considered**:
- Kernel density entropy estimation directly over continuous embeddings: more principled but loses the clean connection to the discrete R_crit formula; deferred to the `continuous-pomdp-spaces` theory work.
