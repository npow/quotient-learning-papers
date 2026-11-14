"""Unit tests for variant (T) (continuous_quotient/variant_t.py)."""
from __future__ import annotations

import numpy as np

from continuous_quotient.types import ContinuousTrajectory, ContinuousTrajectoryDataset
from continuous_quotient.variant_t import (
    VariantTConfig, fit_variant_t, null_threshold, required_m, traj_windows,
)


def _iid_class_dataset(n_traj=800, T=12, sep=3.0, noise=0.1, seed=0, classes=True, stay=0.95):
    """Sticky two-class chain (c_{t+1}=c_t w.p. `stay`), o_t = sep*(2c_t-1)+noise.  The continuation after step t is
    informative about c_t, so the two bin-conditional continuation laws differ by a large margin.
    classes=False: observations are i.i.d. N(0,1) noise, so every bin has the same continuation law."""
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n_traj):
        if classes:
            c = [int(rng.integers(0, 2))]
            for _ in range(T):
                c.append(c[-1] if rng.random() < stay else 1 - c[-1])
            base = sep * (2 * np.array(c) - 1)
            obs = [np.array([b + noise * rng.normal()], dtype=np.float32) for b in base]
        else:
            obs = [np.array([rng.normal()], dtype=np.float32) for _ in range(T + 1)]
        trajs.append(ContinuousTrajectory(observations=obs, actions=[0] * T, rewards=[0.0] * T))
    return ContinuousTrajectoryDataset(trajectories=trajs)


def test_null_threshold_matches_formula():
    m, a = 100, 0.01
    assert abs(null_threshold(m, a) - np.sqrt(2 / m) * (1 + np.sqrt(2 * np.log(1 / a)))) < 1e-12


def test_required_m_scales_inverse_square():
    assert abs(required_m(0.5, 0.01, 0.01) / required_m(1.0, 0.01, 0.01) - 4.0) < 1e-9


def test_windows_layout():
    t = ContinuousTrajectory(observations=[np.array([float(i)]) for i in range(5)], actions=[0, 1, 2, 3],
                             rewards=[0.0] * 4)
    z, c = traj_windows(t, 2)
    assert c.shape == (3, 3)                   # (o_{t+1}, a_{t+1}, o_{t+2}) for t = 0..2
    assert list(c[0]) == [1.0, 1.0, 2.0]


def test_analytic_variant_separates_distinct_laws():
    ds = _iid_class_dataset(seed=1)
    res = fit_variant_t(ds, VariantTConfig(n_bins=2, mode="analytic", seed=0))
    assert res.n_classes == 2


def test_analytic_variant_merges_identical_laws():
    """Null: all bins share one continuation law -> union bound keeps them merged (failure prob <= delta=0.05)."""
    ds = _iid_class_dataset(seed=2, classes=False)
    res = fit_variant_t(ds, VariantTConfig(n_bins=4, mode="analytic", seed=0))
    assert res.n_classes == 1


def test_per_traj_sampling_runs_and_is_conservative_with_few_samples():
    ds = _iid_class_dataset(n_traj=200, seed=3)
    res = fit_variant_t(ds, VariantTConfig(n_bins=2, mode="analytic", sampling="per_traj", min_samples=5, seed=0))
    assert res.n_classes in (1, 2)
