# Convergence Experiments (Layer 1) — LEGACY

> **LEGACY NOTICE:** these sweeps belong to an earlier project phase and run
> the exploratory mode (overlapping windows, G-test) against full-suffix
> ground truths — a different estimand and protocol from the paper's
> certified pipeline. Not comparable to the paper's tables; see
> `../LEGACY.md`. The description below is kept for historical provenance.

Empirical validation that the QuotientEstimator converges to the ground-truth
canonical quotient on small discrete POMDPs. Implements the Scenario-1
acceptance criteria and the FR-3 convergence claim.

## Reproduce

```bash
# Tiger — ground-truth quotient size = 2.
python experiments/convergence/run_tiger.py --n_trajectories 5000 --seeds 3

# RockSample(7,8) — smoke test, runs to completion with finite quotient.
python experiments/convergence/run_rocksample.py --grid_size 7 --n_rocks 8 --n_trajectories 2000 --seeds 3

# Scaling ladder (used by US2 / T059).
python experiments/convergence/run_rocksample.py --grid_size 15 --n_rocks 15 --n_trajectories 10000 --seeds 3
python experiments/convergence/run_rocksample.py --grid_size 20 --n_rocks 20 --n_trajectories 20000 --seeds 3
```

## Output

Each run writes to `experiments/convergence/results/`:
- `<name>.config.json` — the full ExperimentConfig (seeds, hyperparams).
- `<name>.csv` — one row per (seed, trajectory-budget) checkpoint.
- `<name>.summary.json` — wall-clock time + final quotient size per seed.

## Acceptance criteria

- **Tiger**: at least 2 of 3 seeds report `n_classes == 2` at the final budget;
  no seed differs by more than ±2.
- **RockSample(7,8)**: estimator runs to completion within 30 minutes;
  quotient is finite (≤ 50 classes).
- **Scaling ladder**: empirical trajectory requirements grow ≤ polynomially in
  the discovered quotient size (matched against `complexity.theoretical_complexity_curve`).
