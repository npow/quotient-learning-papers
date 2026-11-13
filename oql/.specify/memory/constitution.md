# Online Quotient Learning Constitution

## Core Principles

### I. Theory-First
Every component starts from the mathematical definition. Formal statements must precede proofs, which must precede experiments. No algorithm implementation begins without a stated and justified formal claim about what it computes and why.

### II. Finite-Case Recovery (NON-NEGOTIABLE)
All algorithms must recover the exact canonical quotient (not an approximation) on the small finite POMDPs from Nixon (arXiv 2603.21399): Tiger, GridWorld-2x2, and RockSample-2. This is the primary correctness gate. Any algorithm that fails this check is incorrect, not "approximately correct."

### III. Reproducibility
All experiments use ≥3 independent random seeds. Seeds, environment versions, algorithm hyperparameters, and trajectory counts must be logged and committed. Results must be fully reproducible from the committed codebase.

### IV. Sample Complexity Honesty
Every claim about sample complexity must be labeled as: (a) proven theorem with proof, (b) empirical observation with supporting data, or (c) conjecture. Conflating categories is a constitution violation. The empirical sample requirements must fall within stated theoretical upper bounds on all benchmark environments.

### V. Scope Discipline
This project covers single-agent model-free quotient learning only. Continuous state/action spaces, multi-agent joint learning, adversarial settings, and online deployment are explicitly out of scope. Any expansion must be justified and documented before implementation.

## Research Artifact Standards

- Paper draft lives in `paper/` as LaTeX
- Experiment code lives in `experiments/` as Python
- Each experiment directory has a `README.md` with: purpose, how to run, expected output, expected runtime
- Figures are generated programmatically (no manual editing); scripts live in `experiments/figures/`
- All POMDP environments are instantiated via a unified `environments/` abstraction layer

## Quality Gates

- Finite-case correctness: Exact quotient recovery on Tiger, GridWorld-2x2, RockSample-2 across all seeds
- Convergence curve: Monotonic improvement toward true quotient size as trajectory count increases
- Sample complexity: Empirical trajectory requirements ≤ theoretical upper bound on all benchmarks
- R_crit prediction: ≥90% correct phase-transition classification when learned quotients are plugged in
- Scale: Algorithm runs to completion on a ≥1000-state environment within 24h on a single GPU
- Submission target: NeurIPS / ICML / ICLR / AAAI within 18 months of project start

## Governance

All work must pass constitution gates before moving to the next phase. Violations require explicit documentation in the Complexity Tracking table. This constitution supersedes all other practices.

**Version**: 1.0.0 | **Ratified**: 2026-05-25 | **Last Amended**: 2026-05-25
