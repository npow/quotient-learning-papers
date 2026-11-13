"""HDF5 trajectory dataset I/O.

Stores TrajectoryDataset objects in a compact HDF5 layout so experiments can
share trajectory snapshots between runs (e.g., compare equivalence tests on
identical data).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from quotient_learning.dataset import Trajectory, TrajectoryDataset


def save_dataset(dataset: TrajectoryDataset, path: str | Path) -> None:
    """Write a TrajectoryDataset to HDF5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["environment_id"] = dataset.environment_id
        f.attrs["policy_id"] = dataset.policy_id
        f.attrs["extra_json"] = json.dumps(dataset.extra, default=str)
        f.attrs["n_trajectories"] = dataset.n_trajectories
        grp = f.create_group("trajectories")
        for i, traj in enumerate(dataset.trajectories):
            g = grp.create_group(str(i))
            g.create_dataset("observations", data=np.asarray(traj.observations, dtype=np.int32))
            g.create_dataset("actions", data=np.asarray(traj.actions, dtype=np.int32))
            g.create_dataset("rewards", data=np.asarray(traj.rewards, dtype=np.float64))
            g.attrs["seed"] = traj.seed


def load_dataset(path: str | Path) -> TrajectoryDataset:
    """Read a TrajectoryDataset previously written with save_dataset."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        environment_id = str(f.attrs["environment_id"])
        policy_id = str(f.attrs["policy_id"])
        extra = json.loads(str(f.attrs.get("extra_json", "{}")))
        n = int(f.attrs["n_trajectories"])
        grp = f["trajectories"]
        trajectories = []
        for i in range(n):
            g = grp[str(i)]
            trajectories.append(
                Trajectory(
                    observations=tuple(int(x) for x in g["observations"][...]),
                    actions=tuple(int(x) for x in g["actions"][...]),
                    rewards=tuple(float(x) for x in g["rewards"][...]),
                    seed=int(g.attrs["seed"]),
                )
            )
    return TrajectoryDataset(
        trajectories=tuple(trajectories),
        environment_id=environment_id,
        policy_id=policy_id,
        extra=extra,
    )
