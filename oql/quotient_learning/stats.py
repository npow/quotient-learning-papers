"""Statistical utilities: bootstrap confidence intervals and the pairwise
power threshold m*(eps, alpha, beta) from the equivalence-test guarantees.

The implementation is intentionally self-contained (no scipy dependency) so
it can be imported anywhere in the library.
"""

from __future__ import annotations

import math
import random

from quotient_learning.types import ContinuationAlphabet


def required_samples(
    epsilon_min: float,
    alpha: float,
    beta: float,
    alphabet: ContinuationAlphabet | int,
) -> int:
    """Minimum per-history continuation samples m*(eps, alpha, beta).

    Implements the non-asymptotic histogram-TV requirement:

        m* = ceil( (8 / eps^2) * ((|C| + 1) ln 2
                                  + ln(1 / min(alpha, beta))) )

    guaranteeing Type-I error <= alpha and Type-II error <= beta for the
    fixed-alphabet empirical-TV threshold test whenever the true continuation
    laws are either equal or eps-separated in total variation. Pairs where
    either sample is below m* must be reported UNDETERMINED, not tested.
    """
    if not (0.0 < epsilon_min <= 1.0):
        raise ValueError(f"epsilon_min must be in (0, 1], got {epsilon_min}")
    if not (0.0 < alpha < 1.0) or not (0.0 < beta < 1.0):
        raise ValueError(f"alpha and beta must be in (0, 1), got {alpha}, {beta}")
    size = alphabet.size if isinstance(alphabet, ContinuationAlphabet) else int(alphabet)
    if size < 2:
        raise ValueError(f"alphabet size must be >= 2, got {size}")
    m = (8.0 / epsilon_min**2) * (
        (size + 1) * math.log(2.0) + math.log(1.0 / min(alpha, beta))
    )
    return math.ceil(m)


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute a percentile bootstrap confidence interval.

    Parameters
    ----------
    values:
        Observed scalar values.  Must contain at least one element.
    n_resamples:
        Number of bootstrap resamples to draw.  1000 is the NeurIPS standard.
    ci:
        Nominal coverage probability in (0, 1).  Defaults to 0.95.
    seed:
        Optional integer seed for reproducibility.  When *None* the current
        global random state is used.

    Returns
    -------
    (lo, hi):
        Lower and upper percentile bounds of the bootstrap distribution of the
        sample mean.

    Raises
    ------
    ValueError
        If ``values`` is empty or ``ci`` is outside (0, 1).

    Examples
    --------
    >>> import random
    >>> random.seed(42)
    >>> lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_resamples=1000)
    >>> lo < 3.0 < hi
    True
    """
    if not values:
        raise ValueError("bootstrap_ci requires at least one value.")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")

    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)

    means.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = int(alpha * n_resamples)
    hi_idx = int((1.0 - alpha) * n_resamples) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))
    return means[lo_idx], means[hi_idx]
