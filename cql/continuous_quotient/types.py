"""Core data types for continuous-space quotient learning.

Extends the types from ``quotient_learning`` (online-quotient-learning) to
support raw float32 observation and action vectors rather than integer tokens.
The ``CanonicalContinuousQuotient`` is a drop-in replacement for
``CanonicalQuotient`` in rcrit.py and convergence.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# History key type
# ---------------------------------------------------------------------------

# A continuous history key is a tuple of (obs_vec, act_vec) pairs, where each
# vector is quantised to a bytes fingerprint for hashing.  We store the raw
# arrays separately in ContinuousTrajectoryDataset; the key is only used for
# dict look-ups.

HistoryKey = tuple  # tuple[bytes, ...] — opaque fingerprint


def _vec_to_bytes(v: np.ndarray) -> bytes:
    """Round float32 array to 3 decimal places and convert to bytes key."""
    return np.round(np.asarray(v, dtype=np.float32), 3).tobytes()


def make_history_key(obs_seq: list[np.ndarray], act_seq: list[np.ndarray | int]) -> HistoryKey:
    """Build a hashable key from parallel observation and action sequences.

    obs_seq[i] is the observation *received* at step i.
    act_seq[i] is the action *taken* at step i (may be scalar int or float array).
    len(act_seq) == len(obs_seq) - 1  (no action after last obs).
    """
    parts: list[bytes] = []
    for i, o in enumerate(obs_seq):
        parts.append(_vec_to_bytes(o))
        if i < len(act_seq):
            a = act_seq[i]
            if isinstance(a, (int, np.integer)):
                parts.append(int(a).to_bytes(4, "little"))
            else:
                parts.append(_vec_to_bytes(a))
    return tuple(parts)


# ---------------------------------------------------------------------------
# Trajectory / dataset
# ---------------------------------------------------------------------------

@dataclass
class ContinuousTrajectory:
    """Single episode with continuous observations and actions."""

    observations: list[np.ndarray]   # shape (d_O,) per step
    actions: list[Any]               # shape (d_A,) or int per step; len == len(obs)-1
    rewards: list[float]

    @property
    def length(self) -> int:
        return len(self.observations)

    def obs_array(self) -> np.ndarray:
        return np.stack(self.observations, axis=0)  # (T, d_O)

    def act_array(self) -> np.ndarray:
        acts = []
        for a in self.actions:
            if isinstance(a, (int, np.integer)):
                acts.append(np.array([float(a)], dtype=np.float32))
            else:
                acts.append(np.asarray(a, dtype=np.float32).ravel())
        return np.stack(acts, axis=0) if acts else np.zeros((0, 1), dtype=np.float32)


@dataclass
class ContinuousTrajectoryDataset:
    trajectories: list[ContinuousTrajectory]

    @property
    def n_trajectories(self) -> int:
        return len(self.trajectories)


# ---------------------------------------------------------------------------
# Continuation window (raw float matrix)
# ---------------------------------------------------------------------------

@dataclass
class ContinuousEquivalenceResult:
    """Outcome of a kernel MMD test on two continuous history classes."""

    h1_key: HistoryKey
    h2_key: HistoryKey
    mmd2: float
    p_value: float
    reject: bool          # True → different classes
    n_permutations: int
    bandwidth: float


# ---------------------------------------------------------------------------
# Quotient types
# ---------------------------------------------------------------------------

@dataclass
class ContinuousEstimatorConfig:
    n_trajectories: int = 5_000
    max_steps_per_episode: int = 50
    window_length: int = 2       # k — continuation horizon
    history_max_length: int = 1  # L — max history prefix length
    min_history_occurrences: int = 20
    alpha: float = 0.01
    kernel: str = "rbf"          # "rbf" | "laplace"
    bandwidth: float | None = None   # None → median heuristic
    n_permutations: int = 199
    # Coarse bins for history key generation.
    # Setting history_n_bins = B quantises each history observation into B
    # equal-frequency bins so samples accumulate per history class.
    # Approximation error: histories within the same bin but from different
    # true quotient classes are initially merged.  The worst-case bias is
    # bounded by the bin width w = 1/B (in normalised observation space);
    # choose B ≥ 2·n_classes so inter-class separation ≥ 2w.
    # For K-cycle environments, B = K is the natural choice.
    # For Tiger, B = 8 gives 8w = 0.125, well below the ±0.5 signal gap.
    # Special sentinel: history_n_bins = 0 → auto-select via silhouette score
    # (calls auto_select_bins() at fit time).
    history_n_bins: int = 8
    # Maximum number of continuation samples to use per MMD test pair.
    # Randomly subsampled from the available continuations.
    # Reduces runtime from O(m²) to O(max_samples²) while preserving power
    # for well-separated distributions. None = use all available samples.
    max_samples_per_test: int | None = 300


@dataclass
class ContinuousQuotientClass:
    class_id: int
    member_keys: list[HistoryKey]

    @property
    def n_members(self) -> int:
        return len(self.member_keys)


@dataclass
class CanonicalContinuousQuotient:
    """Learned partition over continuous-space history prefixes.

    Drop-in replacement for CanonicalQuotient in rcrit.py — exposes
    .n_classes and .partition (history_key → class_id).
    """

    n_classes: int
    partition: dict[HistoryKey, int]           # history key → class id
    classes: dict[int, ContinuousQuotientClass] = field(default_factory=dict)
    config: ContinuousEstimatorConfig | None = None
    env_id: str = ""
    n_trajectories: int = 0

    def class_of(self, key: HistoryKey) -> int | None:
        return self.partition.get(key)
