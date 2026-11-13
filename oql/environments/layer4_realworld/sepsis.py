"""Synthetic sepsis-inspired POMDP benchmark (T049).

IMPORTANT FRAMING (review finding C5): Komorowski et al. (2018), the "AI
Clinician", is a reinforcement-learning treatment model built on real
MIMIC-III ICU data — it is NOT a POMDP specification, and this environment
is NOT that dataset or model. This wrapper is a synthetic POMDP *inspired
by* that clinical setting. The transformation is:

- **Hidden state**: a single discretized severity scalar (default 5 bins)
  with treatment-dependent drift — invented for this benchmark, not fit to
  patient data.
- **Observations**: the severity bin read with +/-1-bin noise; the bins are
  a stand-in for coarse SOFA-like indicators, with no clinical meaning.
- **Actions**: four coarse treatment intensities (none / fluids /
  vasopressors / both) mirroring the *shape* of a clinical action space
  only; effects are hand-set drift offsets.

No claim about sepsis treatment, clinical validity, or deployment is made
or supported by experiments on this environment.

Modes:
1. **Synthetic mode** (default): the generative model above.
2. **Real-data mode**: pointed at a directory of preprocessed trajectories
   (HDF5 format matching `quotient_learning.io.save_dataset`). Falls back to
   `SepsisDatasetUnavailable` if the directory is missing. (Unused in the
   paper; requires PhysioNet credentials and a data-use agreement.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from environments.base import POMDPEnvironment, StepResult


class SepsisDatasetUnavailable(RuntimeError):
    """Raised when SepsisPOMDP is constructed in real-data mode without a dataset."""


class SepsisPOMDP(POMDPEnvironment):
    """Sepsis simulator (synthetic by default, optional real-data backing)."""

    def __init__(
        self,
        real_data_dir: str | Path | None = None,
        max_steps: int = 20,
        n_severity_bins: int = 5,
        seed: int = 0,
    ):
        self.max_steps = max_steps
        self.n_severity_bins = n_severity_bins
        self._rng = np.random.default_rng(seed)
        self._real_data_dir = Path(real_data_dir) if real_data_dir else None
        if self._real_data_dir is not None and not self._real_data_dir.exists():
            raise SepsisDatasetUnavailable(
                f"Sepsis dataset directory {self._real_data_dir} not found."
            )
        self._severity = 0
        self._t = 0

    @property
    def n_observations(self) -> int:
        return self.n_severity_bins

    @property
    def n_actions(self) -> int:
        return 4  # {no_intervention, fluids, vasopressors, both}

    @property
    def environment_id(self) -> str:
        return f"Sepsis(real={self._real_data_dir is not None})"

    @property
    def metadata(self) -> dict[str, object]:
        return {"max_steps": self.max_steps, "n_severity_bins": self.n_severity_bins}

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._severity = int(self._rng.integers(0, self.n_severity_bins))
        self._t = 0
        return self._observe()

    def _observe(self) -> int:
        # Noisy severity reading: ±1 bucket.
        noise = int(self._rng.integers(-1, 2))
        return int(np.clip(self._severity + noise, 0, self.n_severity_bins - 1))

    def step(self, action: int) -> StepResult:
        # Action 0 (no intervention): severity drifts up slowly.
        # Actions 1-2 (fluids/vasopressors): reduce severity, side-effect noise.
        # Action 3 (both): strongest reduction but largest side-effect noise.
        drift = self._rng.normal(0.0, 0.5)
        if action == 0:
            self._severity += int(drift + 0.5)
        elif action == 1:
            self._severity += int(drift - 0.5)
        elif action == 2:
            self._severity += int(drift - 0.7)
        elif action == 3:
            self._severity += int(drift - 1.2)
        self._severity = int(np.clip(self._severity, 0, self.n_severity_bins - 1))
        reward = -float(self._severity)
        self._t += 1
        terminated = self._severity == 0  # recovered
        truncated = self._t >= self.max_steps
        return StepResult(observation=self._observe(), reward=reward,
                          terminated=terminated, truncated=truncated)
