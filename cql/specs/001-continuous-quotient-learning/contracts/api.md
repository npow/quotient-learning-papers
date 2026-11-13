# API Contract: continuous_quotient

**Type**: Python library (research artifact)
**Base package**: `continuous_quotient`
**Extends**: `quotient_learning` from `online-quotient-learning` (editable install)

---

## Public API

### ContinuousQuotientEstimator

```python
from continuous_quotient import ContinuousQuotientEstimator, ContinuousEstimatorConfig

estimator = ContinuousQuotientEstimator(config: ContinuousEstimatorConfig)

# Primary method — matches QuotientEstimator.fit() signature from online-quotient-learning
quotient: CanonicalContinuousQuotient = estimator.fit(
    trajectories: List[ContinuousTrajectory],
)

# Convenience: collect trajectories then fit
quotient = estimator.fit_from_env(
    env: gym.Env,                    # any Gymnasium environment
    policy: Callable | None = None,  # None → uniform random
)
```

**Invariants**:
- `quotient.n_classes ≥ 1`
- All histories with ≥ `config.min_history_occurrences` occurrences appear in `quotient.history_to_class`
- Bonferroni-corrected α₀ = `config.alpha / K_hist²` applied internally

---

### NeuralQuotientEstimator

```python
from continuous_quotient import NeuralQuotientEstimator, NeuralEncoderConfig

estimator = NeuralQuotientEstimator(encoder_config: NeuralEncoderConfig)

# Two-stage: generate MMD labels, then train encoder
quotient: CanonicalContinuousQuotient = estimator.fit(
    trajectories: List[ContinuousTrajectory],
    pretrain_env: str | None = None,    # e.g. "kcycle_8" for curriculum init
)

# Access the underlying encoder for embedding new histories
embedding: np.ndarray = estimator.encode(history: ContinuousTrajectory) -> np.ndarray  # (d,)
```

---

### ContinuousMMDTest

```python
from continuous_quotient.equivalence import ContinuousMMDTest

test = ContinuousMMDTest(
    kernel: str = "rbf",          # "rbf" | "laplace" | "gak"
    bandwidth: float | None = None,  # None → median heuristic
    n_permutations: int = 199,
    alpha: float = 0.01,
)

result: ContinuousEquivalenceResult = test.test(
    samples_a: np.ndarray,    # (m, d) float32
    samples_b: np.ndarray,    # (m, d) float32
)
```

**Guarantees**:
- Type I error ≤ `alpha` for all finite `m` (permutation calibration)
- `result.p_value` is the fraction of permuted statistics ≥ observed MMD²

---

### ε-Approximate Quotient Utilities

```python
from continuous_quotient.bounds import (
    mmd_separation_estimate,  # estimate ε_min from data
    trajectory_budget,        # N = O(n² kτ log(n/δ) / (T ε²_MMD))
)

eps_min: float = mmd_separation_estimate(
    quotient: CanonicalContinuousQuotient,
    trajectories: List[ContinuousTrajectory],
)

N_required: int = trajectory_budget(
    n_classes: int,
    k: int,
    tau: float,          # mixing time
    delta: float,
    T: int,              # trajectory length
    eps_mmd: float,
)
```

---

## Compatibility Contract

`ContinuousQuotientEstimator` output (`CanonicalContinuousQuotient`) is a drop-in
replacement for `CanonicalQuotient` from `online-quotient-learning` in:

- `rcrit.py`: `classify_phase_transition(quotient_a, quotient_b)` — unchanged
- `convergence.py`: `ConvergenceTracker` — unchanged
- Experiment scripts that call `.history_to_class`, `.n_classes`

**Breaking changes**: None. New types are additive; existing finite-space code is unmodified.

---

## Error Conditions

| Error | Raised when |
|-------|-------------|
| `InsufficientSamplesError` | < `min_history_occurrences` continuations for any history pair |
| `BandwidthEstimationError` | All pairwise distances are zero (degenerate embedding) |
| `EncoderNotFittedError` | `NeuralQuotientEstimator.encode()` called before `.fit()` |
| `KernelNotCharacteristicError` | Kernel is not characteristic for the given space (validation at init) |
