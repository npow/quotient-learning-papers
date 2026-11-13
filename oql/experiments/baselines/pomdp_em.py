"""Tabular POMDP EM baseline (T057).

Naively learns a full POMDP model (transition + observation) via Baum-Welch
EM on trajectories, then reports the rank of the inferred transition matrix
as a proxy for quotient size. Intentionally strictly stronger than our
model-free learner (needs to learn the *full* model, not just the partition).

Sample complexity is far worse than the quotient learner in practice — EM
struggles with the local optima of POMDPs.

Model size is selected via BIC (Bayesian Information Criterion) over a
candidate grid n_states ∈ {2, 3, 4, 6, 8, 12}, picking the size with
lowest BIC = −2·log_likelihood + k·log(N), where k is the number of free
parameters and N is the total step count across all trajectories.
"""

from __future__ import annotations

import math

import numpy as np

from quotient_learning.dataset import TrajectoryDataset
from quotient_learning.types import CanonicalQuotient, ObservationHistory

_BIC_CANDIDATES = [2, 4, 6, 8]


def _run_em(
    sequences: list[list[int]],
    n_states: int,
    n_obs: int,
    n_iterations: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Run Baum-Welch EM; return (log_likelihood, A, B, pi)."""
    # Initial parameters: random transition + observation, uniform prior.
    A = rng.dirichlet(np.ones(n_states), size=n_states)  # P(s' | s)
    B = rng.dirichlet(np.ones(n_obs), size=n_states)     # P(o | s)
    pi = np.ones(n_states) / n_states                     # initial state
    log_like = -math.inf
    for _ in range(n_iterations):
        gamma_sum = np.zeros((n_states,))
        xi_sum = np.zeros((n_states, n_states))
        emit_sum = np.zeros((n_states, n_obs))
        initial_sum = np.zeros((n_states,))
        log_like = 0.0
        for obs_seq in sequences:
            T = len(obs_seq)
            if T == 0:
                continue
            # Forward
            alpha = np.zeros((T, n_states))
            alpha[0] = pi * B[:, obs_seq[0]]
            scale = np.zeros(T)
            scale[0] = alpha[0].sum() or 1.0
            alpha[0] /= scale[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ A) * B[:, obs_seq[t]]
                scale[t] = alpha[t].sum() or 1.0
                alpha[t] /= scale[t]
            log_like += np.log(scale).sum()
            # Backward
            beta = np.zeros((T, n_states))
            beta[T - 1] = 1.0 / scale[T - 1]
            for t in range(T - 2, -1, -1):
                beta[t] = (A @ (B[:, obs_seq[t + 1]] * beta[t + 1])) / scale[t]
            # Accumulate posteriors
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-12
            initial_sum += gamma[0]
            for t in range(T - 1):
                xi = (np.outer(alpha[t], B[:, obs_seq[t + 1]] * beta[t + 1]) * A)
                xi /= xi.sum() + 1e-12
                xi_sum += xi
                gamma_sum += gamma[t]
            for t, o in enumerate(obs_seq):
                emit_sum[:, o] += gamma[t]
        # Re-estimate
        pi = initial_sum / max(initial_sum.sum(), 1e-12)
        with np.errstate(invalid="ignore"):
            A_new = xi_sum / (gamma_sum[:, None] + 1e-12)
            A = np.where(np.isfinite(A_new), A_new, A)
        B = emit_sum / (emit_sum.sum(axis=1, keepdims=True) + 1e-12)
    return log_like, A, B, pi


def em_quotient(
    dataset: TrajectoryDataset,
    n_obs: int,
    n_iterations: int = 25,
    seed: int = 0,
    n_states: int | None = None,
) -> CanonicalQuotient:
    """Run Baum-Welch on the *observations only* (treating actions as exogenous)
    and return the BIC-selected inferred state count as the quotient size.

    If ``n_states`` is provided explicitly, skip BIC selection and use that
    fixed size (backward-compatible path used by some unit tests).
    """
    rng = np.random.default_rng(seed)
    sequences = [list(t.observations) for t in dataset.trajectories]
    # Total step count used for BIC penalty.
    N_steps = sum(len(s) for s in sequences)

    candidates = [n_states] if n_states is not None else _BIC_CANDIDATES
    best_bic = math.inf
    best_n = candidates[0]
    best_B: np.ndarray | None = None

    for k in candidates:
        # Free parameters: (n_states × n_states - n_states) for A (each row
        # sums to 1, so n_states*(n_states-1) free), (n_states*(n_obs-1)) for
        # B, and (n_states-1) for pi.
        n_free = k * (k - 1) + k * (n_obs - 1) + (k - 1)
        ll, _, B, _ = _run_em(sequences, k, n_obs, n_iterations,
                               np.random.default_rng(seed))
        # BIC = -2 * log_likelihood + n_free * log(N)
        bic = -2.0 * ll + n_free * math.log(max(N_steps, 1))
        if bic < best_bic:
            best_bic = bic
            best_n = k
            best_B = B

    assert best_B is not None
    # Effective state count: number of states with non-trivial observation profile.
    used = int((best_B.sum(axis=1) > 1e-6).sum())
    n_classes = max(used, 1)

    partition: dict[ObservationHistory, int] = {}
    for o in range(n_obs):
        partition[ObservationHistory((o,))] = int(np.argmax(best_B[:, o]))
    return CanonicalQuotient(n_classes=n_classes, partition=partition, transition={})
