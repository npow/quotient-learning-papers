# Contract: EquivalenceTest

*Core subroutine contract for the quotient learning library*

## Purpose

Determine whether two observation histories `h₁` and `h₂` belong to the same quotient class, using only sampled future continuations from a trajectory dataset.

## Interface

```python
class EquivalenceTest:
    def test(
        self,
        samples_a: ContinuationSample,
        samples_b: ContinuationSample,
        alpha: float = 0.05,
    ) -> EquivalenceTestResult:
        ...
```

## Preconditions

- `samples_a.window_length == samples_b.window_length`
- `samples_a.n_samples >= 30` and `samples_b.n_samples >= 30` (minimum for CLT)
- `alpha` ∈ (0, 0.5)

## Postconditions

- Returns `same_class=True` iff the null hypothesis (same continuation distribution) is not rejected at level `alpha`
- `TVThresholdEquivalenceTest`: under iid inputs and the equality-or-TV-gap
  model, both error types are bounded non-asymptotically by
  `2**(d+1) * exp(-m * epsilon_min**2 / 8)`.
- G-test p-values use asymptotic chi-square calibration and are exploratory.

## Error Handling

- Raises `InsufficientSamplesError` if either sample has < 30 continuations
- Raises `WindowMismatchError` if continuation windows differ in length

## Implementations

| Class | Method | Use Case |
|-------|--------|----------|
| `TVThresholdEquivalenceTest` | Fixed-alphabet empirical TV threshold | Certified discrete path |
| `GTestEquivalenceTest` | χ²/G-test on histograms | Exploratory discrete analysis |
| `MMDEquivalenceTest` | Kernel MMD with RBF kernel | General case |
| `EmpiricalBernsteinTest` | Empirical Bernstein bound | When sample size is known upfront |

## Correctness Gate

On Tiger POMDP (|O|=2, n_classes=2): with 1000 continuation samples per history, Type I error ≤ 0.05 and Type II error ≤ 0.05 across 100 repeated trials.
