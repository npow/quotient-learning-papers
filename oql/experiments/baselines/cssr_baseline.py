"""CSSR-style same-target baseline (review finding E4).

Adapted Causal-State Splitting Reconstruction (Shalizi & Shalizi 2004) for
the fixed-policy suffix quotient: causal states of a stationary process are
exactly the continuation-equivalence classes of suffixes, i.e. the SAME
target object as QuotientEstimator after the stationary-suffix estimand
revision. The adaptation constrains CSSR to the identical resources:

  - the identical candidate context set H_N (same enumeration thresholds),
  - the identical continuation samples (same dataset, same stride),
  - the same G-test at the same per-pair level.

Procedure (classic CSSR sufficiency loop, two-sample form): process contexts
in order of increasing length; test each context's continuation sample
against the pooled sample of every existing state; assign it to the first
state the test cannot distinguish it from, else open a new state.

``run_baseline`` enforces the identical-context-set / identical-budget
contract and raises if violated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quotient_learning.equivalence import EquivalenceTest
from quotient_learning.errors import InsufficientSamplesError
from quotient_learning.types import ContinuationSample, ObservationHistory


@dataclass
class BaselineRun:
    method: str
    partition: dict[ObservationHistory, int]
    n_states: int
    context_set: tuple[ObservationHistory, ...]
    n_total_samples: int
    skipped: tuple[ObservationHistory, ...] = field(default_factory=tuple)


def _pool(samples: list[ContinuationSample]) -> ContinuationSample:
    conts: tuple = ()
    for s in samples:
        conts = conts + s.continuations
    return ContinuationSample(
        history=samples[0].history,
        continuations=conts,
        window_length=samples[0].window_length,
    )


def cssr_partition(
    contexts: list[ObservationHistory],
    samples: dict[ObservationHistory, ContinuationSample],
    test: EquivalenceTest,
    alpha: float,
) -> BaselineRun:
    """Run the CSSR-style state construction on a fixed context set."""
    states: list[list[ObservationHistory]] = []
    skipped: list[ObservationHistory] = []
    ordered = sorted(contexts, key=lambda h: (h.length, h.observations))
    for h in ordered:
        if h not in samples:
            skipped.append(h)
            continue
        placed = False
        for members in states:
            pooled = _pool([samples[m] for m in members])
            try:
                res = test.test(samples[h], pooled, alpha=alpha)
            except InsufficientSamplesError:
                continue
            if res.same_class:
                members.append(h)
                placed = True
                break
        if not placed:
            states.append([h])
    partition = {
        h: cid for cid, members in enumerate(states) for h in members
    }
    return BaselineRun(
        method="CSSR",
        partition=partition,
        n_states=len(states),
        context_set=tuple(ordered),
        n_total_samples=sum(s.n_samples for s in samples.values()),
        skipped=tuple(skipped),
    )


def run_baseline(
    method: str,
    contexts: list[ObservationHistory],
    samples: dict[ObservationHistory, ContinuationSample],
    test: EquivalenceTest,
    alpha: float,
    reference_contexts: list[ObservationHistory] | None = None,
    reference_n_samples: int | None = None,
) -> BaselineRun:
    """Harness enforcing the identical-context-set / identical-budget contract."""
    if method != "CSSR":
        raise ValueError(f"unknown same-target baseline {method!r}")
    if reference_contexts is not None and set(reference_contexts) != set(contexts):
        raise ValueError(
            "baseline context set differs from the estimator's context set"
        )
    total = sum(s.n_samples for s in samples.values())
    if reference_n_samples is not None and total != reference_n_samples:
        raise ValueError(
            f"baseline data budget {total} != estimator budget {reference_n_samples}"
        )
    return cssr_partition(contexts, samples, test, alpha)
