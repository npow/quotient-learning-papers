# Implementation Plan: Online Learning of Quotients

**Branch**: `master` | **Date**: 2026-05-25 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `spec.md`

## Summary

Develop model-free algorithms that learn the canonical Myhill-Nerode quotient of a bounded agent's POMDP from interaction trajectories alone — without access to the transition function, observation model, or reward structure. The core subroutine is a statistical equivalence test on observation histories: two histories belong to the same quotient class iff their distributions over future continuations are indistinguishable. The research deliverable is a proven algorithm with sample complexity bounds, empirical validation on the environments from Nixon (2603.21399, 2604.09521), and a demonstration that learned quotients correctly predict the R_crit communication phase transition.

## Technical Context

**Language/Version**: Python 3.11 (primary); Julia 1.10 via PyJulia for POMDPs.jl environments (optional)

**Primary Dependencies**:
- `numpy`, `scipy` — core computation, statistical tests
- `pomdp-py` or `POMDPy` — Python POMDP simulation (Tiger, RockSample, Dec-Tiger)
- `gymnasium` — environment wrapper
- `mujoco`, `dm-control` — PO-MuJoCo continuous environments (Layer 3)
- `dm-lab` or `lab` — DM Lab memory tasks (Layer 2)
- `gym-pomdp` or custom wrappers — T-Maze, Passive T-Maze (Layer 2)
- `multi-robot-warehouse` (Semaphore AI) — cooperative POMDP (Layer 5)
- `openspiel` — Dec-Tiger, Dec-POMDPx (Layer 5)
- `pytest` — unit and integration testing
- `matplotlib`, `seaborn` — figure generation
- `tqdm` — experiment progress tracking
- `joblib` — parallel seed execution
- `h5py` — trajectory dataset storage

**Storage**: Files — HDF5 (trajectory datasets), JSON (experiment configs), CSV (aggregate results)

**Testing**: pytest with fixtures for canonical POMDP instances; numerical tolerance checks via `numpy.testing`

**Target Platform**: Linux server, single GPU (CUDA optional for larger environments)

**Project Type**: Research codebase — Python library (`quotient_learning/`) + experiment scripts

**Performance Goals**:
- Tiger: quotient recovery within 10k trajectories in <5 minutes
- RockSample(7,8): within 50k trajectories in <30 minutes
- RockSample(15,15): within 500k trajectories in <4 hours
- RockSample(20,20) / 1000-state synthetic: convergence within 24h on single GPU
- T-Maze, PO-MuJoCo, Sepsis, multi-robot warehouse: convergence within 48h per environment

**Constraints**:
- Algorithms must run without POMDP model access (only environment interaction)
- All randomness seeded; results reproducible across Linux machines
- Experiment reproducibility: full config logged alongside every result
- PO-MuJoCo results may use function approximation for quotient estimation — must be noted clearly in paper

**Scale/Scope**: Discrete-primary with PO-MuJoCo continuous extension; 5 environment layers (≥12 environments total); 4 baselines; 5+ seeds per configuration

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Gate | Status | Notes |
|------|--------|-------|
| Theory-First: formal algorithm definition before implementation | ✅ PASS | Spec states convergence guarantees and sample complexity bounds as requirements — no implementation starts without theorem statement |
| Finite-Case Recovery: exact quotient on Tiger/GridWorld/RockSample-2 | ✅ PASS (gated) | Primary correctness test; must pass before any scalability work |
| Reproducibility: ≥3 seeds, logged config | ✅ PASS | Spec requires standard deviations across multiple seeds |
| Sample Complexity Honesty: theoretical vs. empirical bounds | ✅ PASS | FR-4 requires empirical bounds fall within proven upper bounds |
| Scope Discipline: single-agent, discrete, model-free only | ✅ PASS | Non-goals explicitly exclude continuous spaces, multi-agent, adversarial |

**Constitution Check Result**: All gates pass. No complexity violations.

## Project Structure

### Documentation

```text
~/code/online-quotient-learning/
├── plan.md              # This file (speckit-plan output)
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── equivalence_test.md
└── tasks.md             # Phase 2 output (speckit-tasks command)
```

### Source Code

```text
~/code/online-quotient-learning/
├── quotient_learning/           # Core library
│   ├── __init__.py
│   ├── equivalence.py           # Statistical equivalence tests (FR-2)
│   ├── estimator.py             # Quotient estimator (FR-1)
│   ├── convergence.py           # Convergence tracking and stopping criteria (FR-3)
│   └── complexity.py            # Sample complexity bound computations (FR-4)
├── environments/                # POMDP wrappers (one subdir per layer)
│   ├── __init__.py
│   ├── base.py                  # Abstract environment interface
│   ├── layer1_discrete/         # Theoretical validation
│   │   ├── tiger.py
│   │   ├── rocksample.py        # Parameterized: (7,8), (15,15), (20,20)
│   │   └── synthetic.py         # Generator for 1000+ state POMDPs
│   ├── layer2_memory/           # Memory hardness
│   │   ├── tmaze.py             # T-Maze and Passive T-Maze
│   │   └── dmlab_wrapper.py     # DM Lab memory tasks
│   ├── layer3_continuous/       # Modern relevance (PO-MuJoCo)
│   │   ├── po_halfcheetah.py    # Velocities hidden
│   │   └── po_ant.py            # Joint angles partially masked
│   ├── layer4_realworld/        # Real-world applicability
│   │   ├── sepsis.py            # Komorowski et al. offline benchmark
│   │   └── robot_localization.py
│   └── layer5_marl/             # MARL connection
│       ├── dec_tiger.py
│       └── multi_robot_warehouse.py
├── experiments/
│   ├── figures/                 # Programmatic figure generation
│   ├── convergence/             # FR-3, FR-4: scaling curve + sample complexity bound
│   │   └── README.md
│   ├── rcrit_prediction/        # Scenario 3: end-to-end quotient → R_crit pipeline
│   │   └── README.md
│   ├── baselines/               # FR-5: PSR, bisimulation, EM, equivalence-test ablation
│   │   └── README.md
│   └── main_table/              # Combined results table across all layers
│       └── README.md
├── paper/                       # LaTeX manuscript
│   ├── main.tex
│   └── figures/                 # Symlinks to experiments/figures output
├── tests/
│   ├── unit/                    # Pure algorithm unit tests
│   ├── integration/             # End-to-end environment tests
│   └── conftest.py
├── pyproject.toml
└── README.md
```

**Structure Decision**: Single-project research codebase. Library in `quotient_learning/`; environments abstracted behind a unified interface so any POMDP can be plugged in. Experiments separated by research question, each self-contained with a README.

## Complexity Tracking

No constitution violations. No entries required.
