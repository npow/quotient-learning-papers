"""Core data types for the quotient-learning library.

These mirror the entities defined in data-model.md. They are immutable and
hashable where appropriate so they can be used as dictionary keys / set
elements (notably ObservationHistory).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Verdict(enum.Enum):
    """Three-valued outcome of a pairwise equivalence decision.

    UNDETERMINED means the pair had too few samples to meet the power
    requirement m*(eps_min, alpha, beta); no test was run and the pair must
    never transmit a merge.
    """

    SAME = "same"
    DIFFERENT = "different"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class ObservationHistory:
    """A finite sequence of observation tokens. Hashable, used as dict key.

    Empty history `ObservationHistory(())` represents the root node.
    """

    observations: tuple[int, ...]

    def __post_init__(self) -> None:
        for o in self.observations:
            if not isinstance(o, int) or o < 0:
                raise ValueError(f"observation token must be non-negative int, got {o!r}")

    @property
    def length(self) -> int:
        return len(self.observations)

    def extend(self, o: int) -> ObservationHistory:
        return ObservationHistory(self.observations + (int(o),))

    def __repr__(self) -> str:
        return f"H{self.observations}"


@dataclass(frozen=True)
class ActionAwareHistory:
    """A finite sequence of (observation, action) pairs. Hashable, usable as dict key.

    This is the history type from Definition 1 of the paper: each step is an
    (observation, action) pair.  Compared to ObservationHistory, this type
    distinguishes histories that share the same observations but differ in the
    actions taken, yielding a finer or equal partition (the canonical
    Myhill-Nerode quotient).

    For Tiger: H=((hear-left, listen),) is informationally different from
    H=((hear-left, open-left),) even though both have the same observation.
    The observation-only representation conflates them, collapsing n=2 to n=1.
    """

    steps: tuple  # tuple of (observation, action) pairs; each action is int or None

    def __post_init__(self) -> None:
        for o, a in self.steps:
            if not isinstance(o, int) or o < 0:
                raise ValueError(f"observation token must be non-negative int, got {o!r}")
            if a is not None and (not isinstance(a, int) or a < 0):
                raise ValueError(f"action must be non-negative int or None, got {a!r}")

    @property
    def length(self) -> int:
        return len(self.steps)

    def extend(self, o: int, a) -> ActionAwareHistory:
        """Return a new ActionAwareHistory extended by one (observation, action) pair."""
        return ActionAwareHistory(self.steps + ((int(o), None if a is None else int(a)),))

    def to_observation_history(self) -> ObservationHistory:
        """Project to the observation-only prefix (loses action context)."""
        return ObservationHistory(tuple(o for o, _ in self.steps))

    def __repr__(self) -> str:
        return f"AH{self.steps}"


@dataclass(frozen=True)
class QuotientClass:
    """An equivalence class of observation histories."""

    class_id: int
    representative: ObservationHistory


@dataclass(frozen=True)
class CanonicalQuotient:
    """A discovered partition over observation contexts with class transitions.

    Naming note: despite the legacy name, the estimator returns the
    STATIONARY SUFFIX quotient of the revised paper, not Nixon's canonical
    bounded-agent quotient (the two condition differently; see paper
    Remark "Relationship to Nixon's probe-exact quotient").

    ``undetermined_pairs`` are context pairs the power gate refused to test
    (insufficient samples for the configured (eps_min, alpha, beta)); they
    never transmit merges. ``undetermined_rate`` is their fraction of all
    tested pairs. ``boundary_flags`` are class-id pairs whose separation
    rests on at least one undetermined pair.
    """

    n_classes: int
    partition: dict[ObservationHistory, int]
    transition: dict[tuple[int, int], int] = field(default_factory=dict)
    undetermined_rate: float = 0.0
    undetermined_pairs: tuple = ()
    boundary_flags: frozenset = frozenset()

    def class_of(self, history: ObservationHistory) -> int | None:
        return self.partition.get(history)


@dataclass(frozen=True)
class ContinuationSample:
    """Future-observation samples for a fixed conditioning history.

    ``max_per_trajectory`` records the largest number of windows any single
    trajectory contributed.  It is diagnostic only: dividing a pooled count
    by this number does not turn dependent windows into iid observations.
    """

    history: ObservationHistory
    continuations: tuple[tuple[int, ...], ...]
    window_length: int
    max_per_trajectory: int = 1

    def __post_init__(self) -> None:
        for c in self.continuations:
            if len(c) != self.window_length:
                raise ValueError(
                    f"continuation length {len(c)} != window_length {self.window_length}"
                )

    @property
    def n_samples(self) -> int:
        return len(self.continuations)

    @property
    def n_effective(self) -> int:
        """Legacy conservative count; not a certification of independence."""
        if not self.continuations:
            return 0
        return max(1, len(self.continuations) // max(1, self.max_per_trajectory))


@dataclass(frozen=True)
class EquivalenceTestResult:
    """Outcome of a statistical test on two ContinuationSamples."""

    history_a: ObservationHistory
    history_b: ObservationHistory
    test_statistic: float
    p_value: float
    same_class: bool
    alpha: float

    @property
    def verdict(self) -> Verdict:
        return Verdict.SAME if self.same_class else Verdict.DIFFERENT


@dataclass(frozen=True)
class PairOutcome:
    """Three-valued pairwise decision with the power gate applied.

    When ``verdict`` is UNDETERMINED the test was never run
    (``test_statistic``/``p_value`` are None) because the pair had fewer than
    ``m_required`` continuation samples on at least one side.
    """

    history_a: ObservationHistory
    history_b: ObservationHistory
    verdict: Verdict
    alpha: float
    m_required: int
    test_statistic: float | None = None
    p_value: float | None = None

    @property
    def same_class(self) -> bool:
        """Backward-compatible view; UNDETERMINED is NOT treated as same."""
        return self.verdict is Verdict.SAME


@dataclass(frozen=True)
class ContinuationAlphabet:
    """Fixed continuation category set supplied by the environment.

    The finite-sample TV test and fixed-alphabet G-test both use this contract,
    never data-derived categories. ``support`` names categories with positive
    mass for the G-test degrees of freedom; it defaults to ``categories`` and
    must be declared independently when structural zeros exist.
    """

    k: int
    categories: frozenset
    full_support: bool
    support: frozenset | None = None

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"window length k must be >= 1, got {self.k}")
        if len(self.categories) < 2:
            raise ValueError("continuation alphabet needs >= 2 categories")
        if self.support is not None and not self.support <= self.categories:
            raise ValueError("support must be a subset of categories")
        if self.full_support and self.support is not None and self.support != self.categories:
            raise ValueError("full_support=True contradicts a strict support subset")

    @property
    def size(self) -> int:
        return len(self.categories)

    @property
    def effective_support(self) -> frozenset:
        return self.categories if self.support is None else self.support

    @property
    def df(self) -> int:
        """Degrees of freedom for the chi-square null: |support| - 1."""
        return len(self.effective_support) - 1


@dataclass(frozen=True)
class RCritEstimate:
    """Estimated critical communication rate from two learned quotients."""

    quotient_a: CanonicalQuotient
    quotient_b: CanonicalQuotient
    r_crit: float
    true_r_crit: float | None = None
    mutual_information: float | None = None
    using_independent_fallback: bool = False

    @property
    def relative_error(self) -> float | None:
        if self.true_r_crit is None or self.true_r_crit == 0:
            return None
        return abs(self.r_crit - self.true_r_crit) / abs(self.true_r_crit)
