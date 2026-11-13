"""Custom exceptions raised by the quotient-learning library."""

from __future__ import annotations


class QuotientLearningError(Exception):
    """Base class for all library-specific exceptions."""


class InsufficientSamplesError(QuotientLearningError):
    """Raised when an equivalence test receives fewer than the minimum required samples."""

    def __init__(self, n_samples: int, min_required: int):
        self.n_samples = n_samples
        self.min_required = min_required
        super().__init__(
            f"Equivalence test requires at least {min_required} samples; got {n_samples}."
        )


class WindowMismatchError(QuotientLearningError):
    """Raised when two ContinuationSamples have different window lengths."""

    def __init__(self, window_a: int, window_b: int):
        self.window_a = window_a
        self.window_b = window_b
        super().__init__(
            f"Continuation windows must match; got {window_a} vs {window_b}."
        )


class NonErgodicError(QuotientLearningError):
    """Raised when the collected trajectory dataset fails an ergodicity sanity check."""


class ConvergenceTimeoutError(QuotientLearningError):
    """Raised when convergence is not reached within the configured iteration budget."""
