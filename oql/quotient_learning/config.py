"""Experiment configuration logging and seeding.

Every experiment writes its full config to JSON next to its result CSVs so
runs are reproducible. `seed_everything` covers Python `random`, `numpy`, and
(optionally) `torch` if installed.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ExperimentConfig:
    name: str
    environment_id: str
    n_trajectories: int
    window_length: int
    equivalence_test: str
    alpha: float
    seeds: list[int]
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        data = json.loads(Path(path).read_text())
        return cls(**data)


CERTIFIED_TESTS = ("TV_FIXED_ALPHABET",)


@dataclass
class CertifiedConfig:
    """The only configuration allowed to produce headline numbers.

    Serialized as a JSON sidecar next to every result artifact
    (contracts/certified-config.md). Table-generation scripts refuse inputs
    whose sidecar is missing or has ``certified: false``.
    """

    k: int
    tau: int
    test: str
    alpha: float
    epsilon_min: float
    beta: float
    seeds: list[int]
    level_correction: str = "BONFERRONI"
    command: str = ""
    commit: str = ""
    python: str = ""
    packages: dict[str, str] = field(default_factory=dict)
    sampling_scheme: str = "INDEPENDENT_TRAJECTORY_BLOCK"
    selection_fraction: float = 0.5
    context_end: int = 0
    policy_id: str = "uniform_random"
    n_trajectories: int = 0
    history_max_length: int = 0
    min_history_occurrences: int = 0
    max_steps_per_episode: int = 0

    def __post_init__(self) -> None:
        if self.k < 1 or self.tau < 1:
            raise ValueError(f"k and tau must be >= 1, got k={self.k}, tau={self.tau}")
        if self.test not in CERTIFIED_TESTS:
            raise ValueError(
                f"test must be one of {CERTIFIED_TESTS} (smoothed/legacy tests are "
                f"not certified), got {self.test!r}"
            )
        if self.level_correction != "BONFERRONI":
            raise ValueError(f"level_correction must be BONFERRONI, got {self.level_correction!r}")
        if not self.seeds:
            raise ValueError("seeds must be recorded (non-empty)")
        if self.sampling_scheme != "INDEPENDENT_TRAJECTORY_BLOCK":
            raise ValueError(
                "certified sidecars require INDEPENDENT_TRAJECTORY_BLOCK sampling"
            )
        if not (0.0 < self.selection_fraction < 1.0):
            raise ValueError("selection_fraction must be in (0, 1)")
        if self.context_end < 0:
            raise ValueError("context_end must be non-negative")

    @property
    def stride(self) -> int:
        """Legacy compatibility: certified sampling has no within-run stride."""
        return 0

    @staticmethod
    def _is_artifact_path(path: str) -> bool:
        """True for generated-artifact paths, which necessarily change while
        a regeneration is in flight and must not mark the run dirty.

        Segment-based so the check is independent of where the package sits
        inside the enclosing git repository."""
        parts = path.split("/")
        if parts[-1] == "ARTIFACTS_MANIFEST.json":
            return True
        for i, seg in enumerate(parts[:-1]):
            if seg == "experiments" and any(
                p.startswith("results") for p in parts[i + 1 : -1]
            ):
                return True
            if seg == "paper" and parts[i + 1] == "figures":
                return True
        return False

    def capture_env_pins(self) -> None:
        """Record Python/package versions and the current git commit.

        The dirty flag ignores changes confined to generated-artifact
        directories (``experiments/*/results*``, ``paper/figures/`` and the
        artifacts manifest), since regenerating artifacts necessarily
        modifies them mid-run; it still fires on any source change.
        """
        import contextlib
        import platform
        import subprocess

        self.python = platform.python_version()
        pins: dict[str, str] = {}
        for mod in ("numpy", "scipy"):
            with contextlib.suppress(ImportError):
                pins[mod] = __import__(mod).__version__
        self.packages = pins
        try:
            commit = (
                subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True, check=True,
                    cwd=Path(__file__).parent,
                ).stdout.strip()
            )
            porcelain = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, check=True,
                cwd=Path(__file__).parent,
            ).stdout.strip()
            dirty_paths = [
                line[3:].strip()
                for line in porcelain.splitlines()
                if line.strip()
            ]
            non_artifact = [
                p for p in dirty_paths if not self._is_artifact_path(p)
            ]
            self.commit = commit + ("-dirty" if non_artifact else "")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.commit = "unknown"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"certified": True, "stride": self.stride}
        d.update(asdict(self))
        return d

    def save_sidecar(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load_sidecar(cls, path: str | Path) -> CertifiedConfig:
        """Load and validate a sidecar; raises if it is not certified."""
        data = json.loads(Path(path).read_text())
        if not data.pop("certified", False):
            raise ValueError(f"sidecar {path} is not certified (certified != true)")
        data.pop("stride", None)
        return cls(**data)


def seed_everything(seed: int) -> None:
    """Seed Python `random`, NumPy, and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
