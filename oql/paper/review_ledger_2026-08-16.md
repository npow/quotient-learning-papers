# Review Ledger — Top-Tier Peer Review of 2026-08-16

Tracks every finding from `top_tier_peer_review.md` (Strong Reject / Major
Revision, 3/10) to its resolution. Statuses: OPEN → FIXED | REBUTTED.
No OPEN rows may remain at submission (spec SC-002).

Feature: `specs/001-oql-review-revision/` · Branch: `oql/academic-review-fixes-2026-08`

## Blocking technical issues

| ID | Finding | Status | Resolution |
|---|---|---|---|
| B1 | Formal quotient conditions on full episode-prefix histories; estimator pools observation suffixes at arbitrary positions (different estimands) | FIXED | Estimand redefined as the stationary suffix quotient (main.tex §3, Defs. 1–3 rewritten: observation context / stationary continuation law / stationary suffix quotient); algorithm §5 states suffix semantics as the definition; recovery theorem re-anchored to the suffix estimand. Validated by tests/unit/test_suffix_estimand.py |
| B2 | Assumptions 2–3 unsatisfiable: ergodic chain on S×H with growing full history; P(H_t ∈ q) ≥ μ_min vacuous for t > L; Assumption 5 silently redefines H_t | FIXED | Assumption 2 → ergodic finite-context chain (S_t, C_t) on S×O^{L+} with stationary law π̄; Assumption 3 → stationary class visitation π̄(q) ≥ μ_min (no time quantifier); Assumption 5 → per-context stationary visitation π̄(h) ≥ μ_min^h, silent H_t redefinition removed; Theorem 5 re-proved with explicit δ-budget table (δ/8 coupling + δ/8 visitation + δ/4 Type-II + δ/4 Type-I = 3δ/4 ≤ δ) |
| B3 | Action-aware continuation omits a_t; Remark 1 division argument invalid; code returns observation-only futures | FIXED | Action-aware continuation redefined as (a_t, o_{t+1}, …, o_{t+k}) ∈ (A×O)^k with the immediate action first (main.tex Def. 2); Remark 1 division argument re-proved with the explicit factorisation; sampling.py extract_action_aware_continuations/enumerate emit observation-ending contexts and (action, observation) continuation pairs. Validated by tests/unit/test_action_aware_alignment.py |
| B4 | Theorem 2 analyzes unsmoothed fixed-alphabet G-test; released code uses pseudocount smoothing + data-dependent alphabet + data-dependent df; "implemented test" claim false; "Williams' correction" docstring wrong | FIXED | GTestEquivalenceTest rewritten: certified default computes the raw fixed-alphabet statistic over the environment-supplied ContinuationAlphabet with contract-derived df; pseudocount/observed-alphabet variant demoted to smoothed=True legacy path emitting NotCertifiedWarning; "Williams' correction" claim deleted; paper §4 states fixed-alphabet raw-count convention and the "implemented test" remark now cites the bit-for-bit verification suite (tests/unit/test_gtest_hand_computed.py, incl. structural-zero case) |
| B5 | Lemma 1 concludes χ²_{\|C\|−1} without full-support hypothesis; wrong df under structural zeros | FIXED | Lemma 1 restated with explicit full-support hypothesis; new Remark (structural zeros) gives the χ²_{\|supp\|−1} limit, notes conservatism of the full-alphabet critical value, and states that support declarations come from the environment contract (never data-estimated). Environments declare full_support honestly via _k1_full_support; conservative default elsewhere |

## Empirical weaknesses

| ID | Finding | Status | Resolution |
|---|---|---|---|
| E1 | Exact-recovery benchmarks mostly easy (L=1, deterministic, maximal TV); need noisy depth-≥2 benchmark with known truth and tunable ε_min | FIXED | NoisyDepth2KCycle: explicit second-order Markov symbol process, analytic suffix partition, exactly computable ε_min(η) strictly decreasing in noise, depth-2 required by construction (environments/layer2_memory/noisy_depth2_kcycle.py; tests/unit/test_noisy_depth2_env.py). Certified sweep η∈{0,0.1,0.2,0.3}: exact recovery 20/20 (experiments/separation/results_certified/noisy_depth2_sweep.csv) |
| E2 | Headline numbers use uncertified stride-1 pooling; certified (thinned + Bonferroni) configuration loses substantial power; headline must use certified procedure | FIXED | New §Certified headline results (tab:certified_main) generated from experiments/main_table/run_certified.py under thinned stride k·τ (τ=⌈(L+k)/k⌉; episodic M=1 regime for Tiger), raw fixed-alphabet G-test, Bonferroni δ=0.05, m* gate. Abstract rewritten to certified numbers; stride-1 sections demoted to labelled supplementary. Sidecars per run; scripts/check_paper_numbers.py enforces paper↔artifact identity |
| E3 | Union-find treats non-rejection as equality; low-power history can bridge separated groups (Tiger L=2 failure); need power gate / undetermined relation | FIXED | Estimator gates every pair on m*(ε_min, α, β): under-powered pairs labeled UNDETERMINED, never merged; partition reports undetermined_rate and boundary_flags (class separations resting on undetermined pairs). §5 algorithm rewritten (power-gated testing + Same-edge-only union-find). Regression tests: tests/integration/test_bridging_regression.py (constructed bridge stays out), test_tiger_l2.py (L=2 collapse corrected or flagged) |
| E4 | Baselines estimate different objects; need same-target baseline (CSSR / PST / U-Tree) with identical history set and budget | FIXED | Adapted CSSR on identical context set, samples, test, and per-pair level (experiments/baselines/cssr_baseline.py, harness raises on any divergence). Result (tab:cssr): CSSR over-splits by one class on all three ground-truth envs (Tiger 3 vs 2, K-cycle 9 vs 8, NoisyDepth2 6 vs 5) |
| E5 | PO-MuJoCo results carried over, not regenerated; regenerate in pinned env or remove from evidentiary core | FIXED | Removed from evidentiary core: Layer-3 subsection retitled and reframed as excluded carry-over record, evidence-table row deleted, discussion/provenance items updated; no claim rests on these numbers |
| E6 | Report task-level uncertainty (CIs), predefine success criteria, separate tuning from evaluation | FIXED | Certified tables carry mean ± sd and bootstrap 95% CIs (CSSR table); exact-recovery success criteria predefined by analytic target partitions (partitions_agree scoring, experiments/scoring.py); gate/level parameters fixed a priori in the runners, not tuned on outcomes |

## Citation corrections

| ID | Finding | Status | Resolution |
|---|---|---|---|
| C1 | Boots et al.: 2011 → 2010; add DOI 10.15607/RSS.2010.VI.036 | FIXED | references.bib: boots2010closing, year 2010, booktitle RSS VI, doi added; tex keys updated |
| C2 | Nixon quotient paper: return-preservation pinpoint Thm 4.4 → Thm 4.6 | FIXED | main.tex introduction citation now pinpoints Thm 4.6 |
| C3 | Roy & Gordon: normalize to NeurIPS 2002 / volume 15 | FIXED | references.bib: roy2002exponential, NeurIPS 2002 volume 15; tex keys updated |
| C4 | Shalizi & Shalizi: @article → @inproceedings | FIXED | references.bib entry converted with proper booktitle field |
| C5 | Komorowski et al.: cited work is RL/MDP clinical model, not "the Sepsis POMDP"; reframe benchmark + document transformation | FIXED | Layer-4 subsection reframed (synthetic POMDP inspired by the AI Clinician, no clinical claim); transformation documented in environments/layer4_realworld/sepsis.py docstring matching the actual generative model |
| C6 | Rivest–Schapire misused for passive-DFA hardness; split sentence, add proper citation, name the learning model | FIXED | Sentence split: minimum-consistent-DFA NP-hardness cited to Gold (1978), inapproximability to Pitt & Warmuth (1993), learning model named; Rivest–Schapire retained only for active homing sequences |
| C7 | Remove unused Hoeffding (1965) BibTeX entry | FIXED | Entry deleted from references.bib (0 citations in tex, verified) |
| C8 | Verify Lehmann–Romano Ch. 14 rate claim and Thm 15.2.1 pinpoint against cited 3rd edition | FIXED | §15.2 randomization-test finite-sample validity confirmed against the 2005 3rd edition (p. 636 argument); the O(m^{-1/2}) rate is no longer attributed to a specific TSH theorem — text now cites Ch. 14 for asymptotic null theory only and derives the rate from the standard multivariate Berry–Esseen theorem with full support stated |

## Presentation / release

| ID | Finding | Status | Resolution |
|---|---|---|---|
| P1 | Homemade single-column A4, 28 pages; conform to venue template, move proofs to appendix | REBUTTED | Venue packaging deferred by spec assumption (no target venue chosen); venue-independent content fixes only in this revision |
| P2 | Visible colored link boxes; use venue style / hidelinks | FIXED | hyperref loaded with hidelinks option |
| P3 | Anonymity: companion preprints + artifact paths defeat "Anonymous" title page | REBUTTED | Anonymization policy is venue-specific; deferred with venue packaging (P1). The revision does not add new anonymity claims |
| P4 | PDF metadata/accessibility absent | REBUTTED | Camera-ready concern; deferred with venue packaging (P1) |
| P5 | Title-page date unusual for anonymized submission | REBUTTED | Deferred with venue packaging (P1) |
| P6 | Sepsis study must not be framed as clinical validation; add explicit no-treatment-conclusion statement | FIXED | Explicit research-ethics item (vi) in the discussion + Layer-4 no-clinical-claim statement + sepsis.py docstring |
| P7 | Artifact completeness: pin versions, document commands per table, seeds, commit hash in artifact README | FIXED | Every certified run writes a JSON sidecar with python/numpy/scipy versions, seeds, exact reproduction command, and git commit (CertifiedConfig.capture_env_pins); table scripts refuse artifacts without certified sidecars; uv.lock committed |
| P8 | "Linear in n" headline needs its conditions foregrounded in abstract | FIXED | Abstract states the rate with its three conditions inline (fixed window/alphabets, K_N = O(n), μ_min^h = Ω(1/n)) and flags none as automatic; suffix-vs-canonical scope stated in the abstract |
| P9 | Uncited-claim: "exact identification of minimal DFA from passive data is hard" (see C6) | FIXED | Resolved with C6 (Gold 1978 + Pitt & Warmuth 1993) |

## Theorem-audit items subsumed by blocking fixes

Theorem 3 / Theorem 5 (not established as stated) → resolved via B1+B2 re-proofs.
Remark 1 (incorrect as written) → resolved via B3.
Theorem 2(i) Berry–Esseen regularity + MMD finite-permutation calibration → resolved via B4 text repairs (C8 for pinpoints).
δ-bookkeeping in Theorem 5 → explicit budget table (research.md R12), part of B2 resolution.
