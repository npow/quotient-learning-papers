# Continuous POMDP Spaces Constitution

## Core Principles

### I. Theory-First
Every component starts from the mathematical definition. Formal statements must precede proofs, which must precede experiments. No implementation work begins without a stated and justified formal claim.

### II. Finite-Case Recovery (NON-NEGOTIABLE)
All continuous extensions must reduce exactly to Nixon's finite constructions (arXiv 2603.21399, 2604.09521) when the observation/state/action spaces are finite. This is the primary correctness gate.

### III. Reproducibility
All experiments use ≥3 independent seeds. Random seeds, environment versions, and hyperparameters must be logged and committed. Results must be reproducible from a public or fully described codebase.

### IV. Approximation Honesty
Neural approximation bounds must be stated as proven (with proof) or empirically verified (with evidence). Claims about R_crit shift must distinguish between theoretical upper bounds, empirical averages, and worst-case observations.

### V. Scope Discipline
This project covers the continuous-space extension only. Model-free quotient learning, N>2 agent networks, and adversarial settings are explicitly out of scope. Any expansion must be justified and documented.

## Research Artifact Standards

- Paper draft lives in `paper/` as LaTeX
- Experiment code lives in `experiments/` as Python
- Each experiment directory has a `README.md` with: purpose, how to run, expected output
- Figures are generated programmatically (no manual editing); scripts live in `experiments/figures/`

## Quality Gates

- Phase transition claim: must have ≥1 positive result and ≥1 negative result (cliff vs. degradation conditions)
- R_crit prediction: must fall within ±15% of empirically measured threshold
- Neural approximation bound: gap between predicted and actual R_crit shift < 2× on validation benchmark
- Submission target: NeurIPS / ICML / ICLR within 18 months of project start

## Governance

All work must pass constitution gates before moving to the next phase. Violations require explicit documentation in the Complexity Tracking table. This constitution supersedes all other practices.

**Version**: 1.0.0 | **Ratified**: 2026-05-25 | **Last Amended**: 2026-05-25
