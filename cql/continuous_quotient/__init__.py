"""continuous_quotient — model-free quotient learning for continuous POMDPs."""

from continuous_quotient.bounds import mmd_separation_estimate, trajectory_budget
from continuous_quotient.equivalence import ContinuousMMDTest
from continuous_quotient.estimator import (
    ContinuousQuotientEstimator,
    auto_select_bins,
    collect_continuous_trajectories,
)
from continuous_quotient.neural_encoder import NeuralQuotientEstimator, NeuralEncoderConfig
from continuous_quotient.types import (
    CanonicalContinuousQuotient,
    ContinuousEstimatorConfig,
    ContinuousTrajectory,
    ContinuousTrajectoryDataset,
)

__all__ = [
    "ContinuousMMDTest",
    "ContinuousQuotientEstimator",
    "auto_select_bins",
    "NeuralQuotientEstimator",
    "NeuralEncoderConfig",
    "CanonicalContinuousQuotient",
    "ContinuousEstimatorConfig",
    "ContinuousTrajectory",
    "ContinuousTrajectoryDataset",
    "collect_continuous_trajectories",
    "mmd_separation_estimate",
    "trajectory_budget",
]
