# R_crit Prediction (Scenario 3) — LEGACY

> **LEGACY NOTICE:** this experiment belongs to an earlier project phase
> (Dec-POMDP communication-rate prediction). It is exploratory-mode, targets
> a different estimand, and is **not** part of the paper's certified
> pipeline; the current paper makes no R_crit or multi-agent claims. See
> `../LEGACY.md`. The description below is kept for historical provenance.

End-to-end pipeline: collect trajectories from two single-agent views of a
Dec-POMDP → learn per-agent quotients → compute R_crit → predict the
communication phase transition across a sweep of channel capacities.

## Reproduce

```bash
python experiments/rcrit_prediction/run.py --env dec_tiger --agent_capacity 2 --seeds 5
python experiments/rcrit_prediction/run.py --env multi_robot_warehouse --agent_capacity 5 --seeds 5
```

## Output

`results/rcrit.csv` — one row per (seed, channel_capacity) sweep point with
columns:
- `n_classes_a`, `n_classes_b` — learned quotient sizes
- `r_crit`, `true_r_crit`, `relative_error`
- `channel_capacity`, `predicted_feasible`, `true_feasible`, `margin`

`results/rcrit.summary.json` — aggregate phase-transition prediction accuracy
and mean relative error.

## Acceptance criteria (Scenario 3)

- Mean relative error of R_crit ≤ 10% against ground truth.
- Phase-transition classification accuracy ≥ 90% across the channel-capacity
  sweep.
- Pipeline completes on Dec-Tiger and multi-robot warehouse in under 5
  minutes.

## Surrogate vs. exact R_crit

This experiment uses the surrogate `R_crit ≈ log2(max(|Q_A|, |Q_B|))` from
`quotient_learning.rcrit.compute_rcrit`. The full Paper-2 rate-distortion
formula requires joint-distribution information — applicable when ground-truth
POMDP models are available; future work integrates the closed-form bound.
