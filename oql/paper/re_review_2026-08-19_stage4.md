# Verification Review Report (Stage 4 — roadmap-closure delta)

**Manuscript:** *Finite-Sample Recovery of Predictive History Quotients from Independent POMDP Trajectories*
**Prior decision:** Accept (roadmap scope), Stage 3' — `paper/re_review_2026-08-19.md`
**Review range:** `043fe82` (Stage-3' head) → `12cb0b3` (roadmap closed: six new experiment families, two new corollaries with proofs, full page budget used)
**Verification panel:** 3 fresh-context independent verifiers (A: theory & estimator; B: experiments & artifacts; C: text, positioning & citations). No rubber-stamping: all new mathematics re-derived from scratch, all six new experiment families independently re-run and compared byte-level against committed artifacts, all new text claims checked against line-quoted evidence, all ten new citations audited.

---

## Decision

# **ACCEPT — all panel findings fixed same-session**

Every number in the six new Implementation-checks paragraphs and the Figure 2 caption reproduces exactly under independent re-run (zero numeric mismatches across 184 compared files); both new corollaries' proofs were re-derived line-by-line and are sound (including the Hoeffding constant 18 in the amplification and the exact m\* gates 112/278/1167/6927); all ten new bibliography entries are correct and their in-text attachments accurate; the paper meets the 9-page rule with References heading page 10.

The panel surfaced **2 MAJOR and 8 MINOR findings, all fixed in this session** (see dispositions). Post-fix: paper recompiles clean (0 undefined refs, 0 overfulls, main text still ends page 9), `check_paper_numbers.py` OK, `gen_manifest.py --check` OK, 237 tests pass.

---

## Findings and dispositions

### MAJOR (both fixed)

| # | Finder | Finding | Fix |
|---|---|---|---|
| M1 | A + C independently | Budget-allocation remark (§4): "relaxed by a factor of 2K … 451→407, roughly ten percent" is internally inconsistent — factor 2K (α′=δ/K) gives m\*=363 (~20%); the quoted 407/9.8% corresponds to α′=δ/(4K), a factor-K/2 relaxation. Premise (proof spends δ/(4K)) verified TRUE; both gates sound; text paired the wrong factor with the numbers. | Reworded: "setting the pairwise budget to that actual total spend — a relaxation by a factor of K/2 — shrinks m\* by roughly ten percent (451→407)". |
| M2 | C | checklist.tex items 4/8 still said "Appendix E" for reproducibility/compute after the new Appendix D (tester/multi-endpoint proofs) shifted lettering (scope=E, reproducibility=F). Regression of the class fixed in `617a304`. | Both updated to Appendix F; item 3's theorem/appendix enumeration corrected (Theorems 1–2, Corollaries 3/5/6, Proposition 4; proofs A–D, scope E); items 1/6 updated for Figure 2 + further checks; item 7 compute claim now backed by an Appendix F timing sentence covering the new runs. |

### MINOR (all fixed)

1. **"what they certify is the boundary"** (misspec ¶) — "certify" is a formal term here; self-contradicting next to uncertified-by-design artifacts → "what they **exhibit** is the boundary" (C).
2. **Head-to-head "matching the two arms' block counts to first order"** — certified ~2,000 blocks vs exploratory ~15,600 windows; overstated parity → now states the exploratory arm has several times more raw windows, an imbalance favoring it (A + C independently).
3. **"per-run flag counts appear in every certified results row"** — false for `results_headtohead/headtohead.csv` (no flag column) → parenthetical exception added (C).
4. **Universal negative "no published implementation instantiates…"** → "we are not aware of a published implementation that instantiates…" (C).
5. **Amplification proof**: constant-error level (1/3) behind κ=⌈18 log(1/γ)⌉ now stated, Hoeffding exponent written out (A).
6. **cor:tester proof**: discard-to-min-count step and explicit per-pair union-bound accounting (≤K(K−1)/2 pairs × δ/(2K²) ≤ δ/4) added (A).
7. **README stale test count** (211 → 237); README two-sweeps clause now distinguishes pure-abstention (8-cycle) from over-splitting (merge-bearing) degradation (C).
8. **LEGACY.md self-contradiction** ("only paper figure") and undocumented Figure-2 render chain → LEGACY.md corrected; `experiments/figures/sweep_depth2.py` render step added to quickstart step 5 and main_table README (C).

### Observations (no action required)

- No unit test asserts certified loaders reject `.misspec.json` sidecars; rejection is structural (extension + `"certified": false` + narrow loader glob) and documented (B).
- `ground_truth.json` files (nondegenerate, posthoc) carry no commit field — they are protocol records, not run sidecars (B).
- Artifact `results_sweep_depth2/summary.md` truncates the gap to 0.7523 while the paper rounds to 0.7524 (exact 0.752380…); rounding-direction rationale documented in `run_certified.py` (B + C).
- `eq:klappendix` label unreferenced (pre-existing, harmless) (C).
- quickstart's "certified pipeline" scope-note phrase sweeps in the uncertified-by-design probe; qualified two paragraphs earlier (C).

---

## Verification highlights

**Verifier A (theory, 28 tool calls):** cor:tester conditioning argument re-derived (type-vector factorization; counts measurable w.r.t. types; nested exclusion forced because the longer context's sample is verbatim a subset); Hoeffding constant 18 = 1/(2(1/2−1/3)²) confirmed numerically; cor:multi gate ⌈(8/ε²)((d+1)log2 + log(2TK_t²/δ))⌉ = 112 at the experiment's configuration; all six paragraphs' gaps/gates recomputed **from transition tables, not artifacts** (nested min TV exactly 2/7, attained uniquely by the nested pair (0,)/(0,0); Tiger closed-loop 0.696 and gap 0.392 re-derived by enumeration; misspec m\*=6927). No contradiction with Theorems 1–2, Lemma 1, or scope statements.

**Verifier B (experiments, 47 tool calls):** all six runners re-run with committed seeds into scratch — every CSV and summary byte-identical, sidecars identical mod provenance fields; every summary claim recomputed from its own CSV; all 170 sidecars record clean commit `91b5c02`, repo-relative commands; misspec sidecars `"certified": false` and excluded twice over from certified loaders; manifest pins all nine families; Figure 2 regenerates byte-identically except the PDF `/CreationDate` stamp.

**Verifier C (text & citations, 46 tool calls):** every intro/limitations promise (misspec probe, per-SAME-edge margin, T=2 exercise) line-quote matched to its sec:checks delivery; zero undefined references; all ten new bib entries verified (incl. castro2020scalable correctly attributed as the single-author AAAI 2020 π-bisimulation paper); "nine experiment families" language consistent across README/LEGACY/quickstart/paper; posthoc conditional-certificate honesty judged clear enough to survive adversarial review.

## Standing

Stage-3' estimated ~7.5/10. The delta converts the two theory extensions from remarks to proved corollaries, exercises previously untested regimes (nested pairs, multi-endpoint, stochastic policy, unknown ground truth, gap-model violation), and consumes the page budget. Remaining venue risk is unchanged in kind (significance taste on a deliberately narrow estimand) — soundness/reproducibility remains exemplary and now covers the full claim set. Optional residuals from Stage 3' that remain open: Ravindran–Barto 2003 / Zhan 2023 / Yu–Künsch citations; formal two-line HMM reduction in the scope appendix; worked budget-planning example.
