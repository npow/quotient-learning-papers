"""Public API for the quotient-learning library."""

from quotient_learning.complexity import (
    samples_per_pair,
    theoretical_complexity_curve,
    trajectories_required,
)
from quotient_learning.convergence import (
    ConvergenceCheckpoint,
    ConvergenceTracker,
    run_convergence_ladder,
)
from quotient_learning.dataset import Trajectory, TrajectoryDataset
from quotient_learning.equivalence import (
    EmpiricalBernsteinTest,
    EquivalenceTest,
    GTestEquivalenceTest,
    MMDEquivalenceTest,
)
from quotient_learning.errors import (
    ConvergenceTimeoutError,
    InsufficientSamplesError,
    NonErgodicError,
    QuotientLearningError,
    WindowMismatchError,
)
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.rcrit import (
    PhaseTransitionResult,
    classify_phase_transition,
    compute_rcrit,
    compute_rcrit_exact,
    compute_rcrit_surrogate,
)
from quotient_learning.sampling import (
    enumerate_observed_histories,
    extract_continuation_samples,
)
from quotient_learning.types import (
    CanonicalQuotient,
    ContinuationSample,
    EquivalenceTestResult,
    ObservationHistory,
    QuotientClass,
    RCritEstimate,
)

__all__ = [
    "CanonicalQuotient",
    "ContinuationSample",
    "ConvergenceCheckpoint",
    "ConvergenceTimeoutError",
    "ConvergenceTracker",
    "EmpiricalBernsteinTest",
    "EquivalenceTest",
    "EquivalenceTestResult",
    "GTestEquivalenceTest",
    "InsufficientSamplesError",
    "MMDEquivalenceTest",
    "NonErgodicError",
    "ObservationHistory",
    "PhaseTransitionResult",
    "QuotientClass",
    "QuotientEstimator",
    "QuotientLearningError",
    "RCritEstimate",
    "Trajectory",
    "TrajectoryDataset",
    "WindowMismatchError",
    "classify_phase_transition",
    "compute_rcrit",
    "compute_rcrit_exact",
    "compute_rcrit_surrogate",
    "enumerate_observed_histories",
    "extract_continuation_samples",
    "run_convergence_ladder",
    "samples_per_pair",
    "theoretical_complexity_curve",
    "trajectories_required",
]
