"""Shared pytest fixtures for the quotient-learning test suite."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_rngs() -> Iterator[None]:
    """Reset Python and NumPy RNGs to a deterministic seed per test."""
    random.seed(0)
    np.random.seed(0)
    yield


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh seeded numpy.Generator for tests that need explicit control."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def tmp_h5(tmp_path: Path) -> Path:
    """Path to a temporary HDF5 file (does not create it)."""
    return tmp_path / "trajectories.h5"


@pytest.fixture
def tiger_env():
    """A fresh Tiger POMDP — the canonical 2-class fixture."""
    from environments.layer1_discrete.tiger import TigerPOMDP
    return TigerPOMDP(seed=123)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark tests under tests/integration/ as 'integration'."""
    for item in items:
        path = str(item.fspath)
        if os.sep + "integration" + os.sep in path:
            item.add_marker(pytest.mark.integration)
        elif os.sep + "unit" + os.sep in path:
            item.add_marker(pytest.mark.unit)
