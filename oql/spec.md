# Feature: Online Learning of Quotients

**Status**: Draft
**Created**: 2026-05-25
**Author**: npow

## Problem Statement

The Myhill-Nerode quotient framework for bounded agents (Nixon, 2603.21399) proves that every bounded agent in a POMDP has a unique, minimal compressed world-model (the canonical quotient). The companion communication theory (Nixon, 2604.09521) uses these quotients to derive hard phase transitions in inter-agent communication feasibility.

Both results assume the POMDP model is fully known — the transition function, observation model, and reward structure are given upfront. This is a fatal limitation for real-world deployment. Real agents interact with environments they don't have complete specifications for. Without a model-free path to quotient discovery, the entire bounded-agent communication theory remains a purely theoretical artifact.

This research direction develops algorithms that learn the minimal quotient directly from interaction trajectories, without requiring the full POMDP specification. It is the foundational enabler that makes every other extension of the quotient framework (multi-agent networks, LLM agent communication, hierarchical architectures) practically viable.

## Goals

- Develop model-free algorithms that converge to the canonical minimal quotient from finite interaction data
- Establish sample complexity bounds — how many trajectories are needed to recover the quotient to a given accuracy
- Provide convergence guarantees under standard ergodicity / mixing assumptions
- Connect the quotient learning problem to existing POMDP learning and state abstraction literature
- Produce practical algorithms that scale beyond the small tabular environments (Tiger, GridWorld, RockSample) used in the original papers

## Non-Goals

- Extending the quotient *theory* to continuous state/action spaces — PO-MuJoCo environments are included as empirical benchmarks via discretization/function approximation, not as a new theoretical contribution
- Multi-agent quotient negotiation or joint learning (assumes single-agent quotient learning is solved first)
- Adversarial or game-theoretic settings
- Real-time or streaming deployment of learned quotients in production systems

## User Scenarios

### Scenario 1: Researcher validates quotient learning on a known POMDP

**As a** multi-agent systems researcher
**I want to** run a quotient learning algorithm on a POMDP where the ground-truth quotient is known (e.g., Tiger, RockSample)
**So that** I can verify the learned quotient matches the theoretically optimal one and measure convergence speed

**Acceptance Criteria**:
- [ ] Algorithm recovers the exact canonical quotient on small POMDPs (Tiger, GridWorld) within a bounded number of episodes
- [ ] Convergence curve shows monotonic improvement toward the ground-truth quotient size
- [ ] Sample complexity scales polynomially with the number of quotient states

### Scenario 2: Practitioner applies quotient learning to an unknown environment

**As a** robotics or AI engineer
**I want to** deploy a quotient learning agent in an environment where I don't have the POMDP model
**So that** the agent automatically discovers the minimal perception it needs and I can use this to design communication protocols

**Acceptance Criteria**:
- [ ] Algorithm produces a quotient without requiring transition/observation model specification
- [ ] Learned quotient is within a provable approximation factor of the true minimal quotient
- [ ] The quotient stabilizes (stops changing) after sufficient interaction, signaling convergence

### Scenario 3: Researcher uses learned quotients to compute communication rates

**As a** researcher extending the semantic rate-distortion results
**I want to** plug learned quotients (from two agents with different capacities) into the R_crit formula from Paper 2
**So that** I can predict communication feasibility between agents without knowing the underlying POMDP

**Acceptance Criteria**:
- [ ] Learned quotients, when plugged into the rate-distortion formula, produce R_crit estimates within 10% of the true value
- [ ] The phase transition (feasible vs. infeasible communication) is correctly predicted in at least 90% of test cases
- [ ] End-to-end pipeline (learn quotients → compute R_crit → predict communication outcome) runs on benchmark environments

## Functional Requirements

### FR-1: Trajectory-Based Quotient Estimation

Given a dataset of observation-action-reward trajectories from a bounded agent interacting with a POMDP, produce an estimated quotient (equivalence relation over observation histories).

**Acceptance Criteria**: On POMDPs with known ground-truth quotients, the estimated quotient converges to the true quotient as trajectory count increases.

### FR-2: Equivalence Testing from Samples

Determine whether two observation histories belong to the same quotient class using only sampled future continuations (no model access).

**Acceptance Criteria**: Statistical test correctly classifies same-class vs. different-class history pairs with Type I and Type II error rates below specified thresholds.

### FR-3: Convergence Guarantees

Provide formal guarantees that the learned quotient converges to the canonical minimal quotient under stated assumptions (ergodicity, mixing time bounds, sufficient exploration).

**Acceptance Criteria**: Theorem statement with proof, validated empirically on benchmark POMDPs.

### FR-4: Sample Complexity Bounds

Derive upper bounds on the number of trajectories (or total interaction steps) needed to recover the quotient to within ε accuracy with probability at least 1 − δ.

**Acceptance Criteria**: Bounds are polynomial in |quotient states|, 1/ε, and log(1/δ). Empirical sample requirements fall within the theoretical bounds on benchmark environments.

### FR-5: Benchmark Evaluation Suite

Evaluate across five environment layers and four baseline methods. No single layer is optional — each answers a distinct reviewer question.

**Layer 1 — Theoretical validation** (correctness and scaling on discrete POMDPs):
- Tiger (2 states): ground-truth quotient known, used for correctness check
- RockSample(7,8): standard baseline
- RockSample(15,15): moderate scaling
- RockSample(20,20): large discrete case
- Synthetic POMDP with 1000+ states, 100+ observations: stress-tests the O(|S|²) sample complexity bound

**Layer 2 — Memory hardness** (shows quotient compression targets the right structure):
- T-Maze and Passive T-Maze: isolate memory horizon dependence
- DM Lab memory-and-planning tasks: where minimal quotient compression is visually obvious

**Layer 3 — Continuous/modern relevance** (required for NeurIPS/ICML credibility):
- PO-MuJoCo HalfCheetah with velocities hidden
- PO-MuJoCo Ant with joint angles partially masked

**Layer 4 — Real-world applicability**:
- Sepsis treatment POMDP (Komorowski et al., offline medical benchmark)
- Robot localization / navigation POMDP

**Layer 5 — MARL connection** (validates the R_crit motivation):
- Dec-Tiger and Dec-POMDPx benchmarks
- Multi-robot warehouse (cooperative partial observability)

**Baselines** (missing any of these is a rejection risk):
- Spectral/PSR methods (Boots et al., Hefny et al.)
- Bisimulation metric methods (Ferns et al., Zhang et al. 2020)
- Tabular POMDP EM (naive model-learning baseline)
- Ablation across equivalence tests: MMD, G-test, EmpiricalBernstein

**Acceptance Criteria**: Results reported with standard deviations across ≥5 seeds. Main results table covers ≥6 environments showing learned quotient size vs ground truth, R_crit prediction accuracy, and sample efficiency vs all baselines. Scaling figure shows empirical sample complexity matching theoretical O(|S|²) bound. At least one Layer 2 or Layer 3 environment where all baselines fail and the quotient method holds.

## Success Criteria

- Learned quotients match ground-truth quotients on benchmark POMDPs in 95%+ of trials given sufficient data
- Sample complexity grows at most polynomially with quotient size; empirical curve matches the O(|S|²) theoretical bound across the RockSample scaling ladder
- R_crit estimates derived from learned quotients predict the communication phase transition correctly in 90%+ of test cases
- Algorithm runs to completion on environments with at least 1,000 underlying states within 24 hours on a single GPU
- All five benchmark layers covered with ≥1 environment each; PO-MuJoCo and real-world results present
- All four baseline methods compared; at least one environment where baselines fail and the quotient method holds
- Paper accepted at NeurIPS, ICML, or ICLR (not AAAI — aim higher)

## Key Entities

| Entity | Description | Relationships |
|--------|-------------|---------------|
| Observation History | Sequence of observations seen by the agent | Grouped into Quotient Classes |
| Quotient Class | Equivalence class of histories that are indistinguishable to the bounded agent | Defines the agent's compressed world-model |
| Canonical Quotient | The unique minimal quotient (Myhill-Nerode theorem) | Target of the learning algorithm |
| Trajectory | A sampled episode (observation-action-reward sequence) | Training data for quotient estimation |
| R_crit | Critical communication rate from Paper 2 | Computed from quotients of two agents |
| Equivalence Test | Statistical test for whether two histories belong to the same class | Core subroutine of the learning algorithm |

## Assumptions

- The underlying environment is a finite POMDP (extension to continuous spaces is a separate direction)
- The agent's exploration policy provides sufficient coverage of the observation space (standard exploration assumption)
- Mixing time of the induced Markov chain is finite (standard ergodicity assumption)
- The quotient structure is identifiable from observable quantities (future observation distributions conditioned on histories)
- Computational resources equivalent to a single modern GPU are available for experiments
- The Myhill-Nerode theorem for bounded interaction (Paper 1) is correct as stated — this work builds on it, not re-proves it

## Dependencies

- Correctness of the canonical quotient existence/uniqueness theorem (Paper 1, 2603.21399)
- Rate-distortion framework and R_crit formula (Paper 2, 2604.09521)
- Standard POMDP simulation environments (POMDPs.jl, ai-gym, or equivalent)
- Existing statistical hypothesis testing frameworks for equivalence testing

## Open Questions

- What is the tightest achievable sample complexity bound? Is the polynomial dependence on quotient size improvable to near-linear?
- Can the learning algorithm be made fully online (streaming, single-pass) or does it inherently require batch processing of trajectories?
- How sensitive is convergence to the exploration policy — does the algorithm need active exploration or is passive data sufficient?
