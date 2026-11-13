"""Finite-sample bounds for independent-trajectory quotient recovery.

The certified estimator uses half of the trajectories for context selection
and half for inference, with at most one continuation per inference
trajectory. Consequently trajectory length and mixing-time substitutions do
not appear in the guarantee.
"""

from __future__ import annotations

import math

from quotient_learning.stats import required_samples


def samples_per_pair(
    epsilon: float,
    delta: float,
    n_histories: int,
    alphabet_size: int = 2,
) -> int:
    """Per-context samples for all pairwise TV decisions.

    Uses the implementation's conservative pairwise error budget
    ``delta / (2 * n_histories**2)`` for both error types.
    """
    if not 0 < epsilon <= 1:
        raise ValueError(f"epsilon must be in (0, 1]; got {epsilon}")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1); got {delta}")
    if n_histories < 1:
        raise ValueError(f"n_histories must be >= 1; got {n_histories}")
    if alphabet_size < 2:
        raise ValueError(f"alphabet_size must be >= 2; got {alphabet_size}")
    pair_budget = delta / (2.0 * n_histories**2)
    return required_samples(epsilon, pair_budget, pair_budget, alphabet_size)


def trajectories_required(
    epsilon: float,
    delta: float,
    n_classes: int,
    window_length: int,
    episode_length: int = 50,
    n_observations: int = 2,
    mixing_time: int = 1,
    burn_in: int = 0,
    n_histories: int | None = None,
    visitation_floor: float | None = None,
) -> int:
    """Sufficient total trajectory count for the 50/50 certified split.

    ``episode_length``, ``mixing_time``, and ``burn_in`` remain in the public
    signature for backward compatibility but do not improve the iid sample
    count. Only the fixed block at the configured endpoint is analyzed.
    """
    if episode_length - burn_in < 1:
        raise ValueError("episode_length must exceed burn_in")
    if mixing_time < 1:
        raise ValueError("mixing_time must be positive")
    k_n = n_histories if n_histories is not None else max(n_classes, 2)
    mu = visitation_floor if visitation_floor is not None else 1.0 / k_n
    if not 0 < mu <= 1:
        raise ValueError("visitation_floor must be in (0, 1]")
    alphabet = n_observations**window_length
    m_star = samples_per_pair(epsilon, delta, k_n, alphabet)
    inference_half = max(
        2.0 * m_star / mu,
        8.0 * math.log(4.0 * k_n / delta) / mu,
    )
    return 2 * math.ceil(inference_half)


def theoretical_complexity_curve(
    n_classes_range: list[int],
    epsilon: float,
    delta: float,
    window_length: int = 2,
    episode_length: int = 50,
    n_observations: int = 2,
) -> list[tuple[int, int]]:
    """Return ``(class count, sufficient trajectories)`` pairs for plotting."""
    return [
        (
            n,
            trajectories_required(
                epsilon,
                delta,
                n,
                window_length,
                episode_length,
                n_observations=n_observations,
            ),
        )
        for n in n_classes_range
    ]
