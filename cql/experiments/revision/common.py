"""Shared helpers for the revision experiments: data with latent labels, ARI scoring, env registry."""
from __future__ import annotations

import importlib.util as ilu
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from continuous_quotient.types import ContinuousTrajectory, ContinuousTrajectoryDataset  # noqa: E402


def _load(path: Path, name: str):
    spec = ilu.spec_from_file_location(name, path)
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_L1 = ROOT / "experiments" / "layer1_ground_truth" / "envs"
TigerEnv = _load(_L1 / "tiger_continuous.py", "tg_env").TigerContinuousEnv
KCycleEnv = _load(_L1 / "kcycle_continuous.py", "kc_env").KCycleContinuousEnv
GaussMixEnv = _load(_L1 / "gaussian_mixture_pomdp.py", "gm_env").GaussianMixturePOMDP


@dataclass
class EnvSpec:
    name: str
    make: callable
    n_latent: int
    N: int
    T: int
    alpha: float
    R: int


def env_spec(name: str) -> EnvSpec:
    """Configurations exactly as in the paper's Section 5 / App. C."""
    if name == "tiger":
        return EnvSpec(name, lambda: TigerEnv(sigma=0.5, episode_length=20), 2, 3000, 20, 0.01, 199)
    if name.startswith("kcycle"):
        K = int(name[6:])
        return EnvSpec(name, lambda: KCycleEnv(K=K, obs_noise=0.1, episode_length=50), K,
                       5000 if K == 16 else 3000, 50, 0.01, 199)
    if name.startswith("gauss"):
        Ns = int(name[5:])
        return EnvSpec(name, lambda: GaussMixEnv(N=Ns, sigma=0.15, episode_length=50), Ns, 2000, 50, 0.05, 99)
    raise KeyError(name)


@dataclass
class Data:
    dataset: ContinuousTrajectoryDataset
    latent: np.ndarray     # (N, T+1) latent state at each observation time
    obs: np.ndarray        # (N, T+1, d)
    act: np.ndarray        # (N, T) ints
    rew: np.ndarray        # (N, T)


def collect(env, N: int, T: int, seed: int) -> Data:
    """Random-policy rollouts (same loop as collect_continuous_trajectories) that also record the latent state."""
    rng = np.random.default_rng(seed)
    env.action_space.seed(int(seed))
    trajs, lat, obs_all, act_all, rew_all = [], [], [], [], []
    for _ in range(N):
        o, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        ol, al, rl = [np.asarray(o, dtype=np.float32)], [], []
        ll = [int(env._state)]
        for _ in range(T):
            a = env.action_space.sample()
            o, r, term, trunc, _ = env.step(a)
            al.append(a); rl.append(float(r)); ol.append(np.asarray(o, dtype=np.float32)); ll.append(int(env._state))
            if term or trunc:
                break
        trajs.append(ContinuousTrajectory(observations=ol, actions=al, rewards=rl))
        lat.append(ll); obs_all.append(np.stack(ol)); act_all.append(al); rew_all.append(rl)
    return Data(ContinuousTrajectoryDataset(trajectories=trajs), np.array(lat), np.array(obs_all),
                np.array(act_all), np.array(rew_all))


def valid_steps(T_obs: int, k: int) -> int:
    """Number of start steps t that have a full k-step continuation: t + 1 + k <= T_obs."""
    return T_obs - k


def ari_from_bin_labels(data: Data, boundaries: np.ndarray, bin_class: dict[int, int], k: int = 2):
    """ARI between latent class and estimated class over all valid steps whose bin is active.

    Returns (ari, coverage) where coverage is the fraction of valid steps in an active bin."""
    from sklearn.metrics import adjusted_rand_score
    from continuous_quotient.estimator import _obs_scalar
    N, T1 = data.latent.shape
    nv = valid_steps(T1, k)
    z = np.array([[_obs_scalar(data.obs[i, t]) for t in range(nv)] for i in range(N)])
    b = np.searchsorted(boundaries, z)
    pred = np.vectorize(lambda x: bin_class.get(int(x), -1))(b)
    true = data.latent[:, :nv]
    keep = pred >= 0
    if keep.sum() < 2:
        return float("nan"), 0.0
    return float(adjusted_rand_score(true[keep], pred[keep])), float(keep.mean())


def ari_bin_ceiling(data: Data, boundaries: np.ndarray, k: int = 2) -> float:
    """Best ARI attainable by ANY partition of the B bins: each bin is mapped to its majority latent class
    (the oracle bin->class assignment); this caps ARI because the bin is a noisy readout of the latent state."""
    from sklearn.metrics import adjusted_rand_score
    from continuous_quotient.estimator import _obs_scalar
    N, T1 = data.latent.shape
    nv = valid_steps(T1, k)
    z = np.array([[_obs_scalar(data.obs[i, t]) for t in range(nv)] for i in range(N)])
    b = np.searchsorted(boundaries, z).ravel()
    true = data.latent[:, :nv].ravel()
    maj = {}
    for bb in np.unique(b):
        vals, cnt = np.unique(true[b == bb], return_counts=True)
        maj[bb] = vals[np.argmax(cnt)]
    pred = np.array([maj[x] for x in b])
    return float(adjusted_rand_score(true, pred))
