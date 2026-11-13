"""Abstract POMDP environment interface.

Every environment used by the quotient-learning library implements this
interface. The interface is intentionally minimal: discrete observations and
actions, finite (or terminating) episodes, and a `metadata` dict for things
like ground-truth quotient size (only known for benchmark POMDPs).
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quotient_learning.types import ContinuationAlphabet, ObservationHistory


@dataclass
class StepResult:
    observation: int
    reward: float
    terminated: bool
    truncated: bool


class POMDPEnvironment(ABC):
    """Abstract base class for discrete POMDP environments."""

    #: Set True in subclasses where EVERY observation symbol has positive
    #: stationary probability under uniform-random exploration, so the k=1
    #: observation-only alphabet is honestly full-support (drives the
    #: G-test's degrees of freedom; see ContinuationAlphabet).
    _k1_full_support: bool = False

    @property
    @abstractmethod
    def n_observations(self) -> int:
        """Size of the observation alphabet."""

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Size of the action alphabet."""

    @property
    def environment_id(self) -> str:
        """Stable identifier for this environment instance (used in dataset provenance)."""
        return type(self).__name__

    @property
    def metadata(self) -> dict[str, object]:
        """Optional metadata. Benchmark POMDPs populate `ground_truth_n_classes`."""
        return {}

    @abstractmethod
    def reset(self, seed: int | None = None) -> int:
        """Reset to a fresh episode. Returns the initial observation."""

    @abstractmethod
    def step(self, action: int) -> StepResult:
        """Apply an action; return the next observation, reward, and terminal flags."""

    def close(self) -> None:
        """Release any environment resources (default: no-op)."""
        return None

    def continuation_alphabet(self, k: int, action_aware: bool = False) -> ContinuationAlphabet:
        """Fixed continuation category set for the certified G-test.

        Default: the full product alphabet from the environment's observation
        and action counts — O^k observation-only, or (A x O)^k action-aware,
        where an action-aware continuation is the k step pairs
        ((a_t, o_{t+1}), ..., (a_{t+k-1}, o_{t+k})).

        ``full_support`` is declared honestly: True only for the k=1
        observation-only alphabet of environments that set
        ``_k1_full_support`` (every observation symbol reachable under the
        exploration policy); False otherwise, which is conservative — the
        declared support then defaults to the full category set and the
        critical value is unchanged, but no unproven full-support hypothesis
        is asserted (paper Lemma 1). Environments with known structural
        zeros SHOULD override and pass an explicit ``support`` subset.
        """
        from quotient_learning.types import ContinuationAlphabet

        obs = range(self.n_observations)
        if action_aware:
            step = list(itertools.product(range(self.n_actions), obs))
            categories = frozenset(itertools.product(step, repeat=k))
            full = False
        else:
            categories = frozenset(itertools.product(obs, repeat=k))
            full = self._k1_full_support and k == 1
        return ContinuationAlphabet(k=k, categories=categories, full_support=full)

    def target_partition(self, max_length: int) -> dict[ObservationHistory, int] | None:
        """Ground-truth suffix partition for benchmarks that know it.

        Returns a mapping from suffix word to class index, or None when no
        analytic ground truth exists (the default).
        """
        return None
