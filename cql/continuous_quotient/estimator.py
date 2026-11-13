"""ContinuousQuotientEstimator: batch quotient discovery for continuous POMDPs.

Algorithm (mirrors QuotientEstimator from online-quotient-learning):
1. From ContinuousTrajectoryDataset, enumerate history prefixes of length ≤ L
   that occur ≥ min_history_occurrences times.
2. For each history, extract k-step continuation windows as float32 matrices.
3. Pairwise ContinuousMMDTest; Benjamini–Hochberg correction at level α over all pairs.
4. Union-find merge of same-class histories.
5. Return CanonicalContinuousQuotient.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from continuous_quotient.equivalence import ContinuousMMDTest
from continuous_quotient.types import (
    CanonicalContinuousQuotient,
    ContinuousEstimatorConfig,
    ContinuousQuotientClass,
    ContinuousTrajectory,
    ContinuousTrajectoryDataset,
    HistoryKey,
    make_history_key,
)


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, items: list) -> None:
        self._parent = {h: h for h in items}

    def find(self, h: Any) -> Any:
        root = h
        while self._parent[root] != root:
            root = self._parent[root]
        cur = h
        while self._parent[cur] != root:
            nxt = self._parent[cur]
            self._parent[cur] = root
            cur = nxt
        return root

    def union(self, a: Any, b: Any) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> dict[Any, list]:
        out: dict = defaultdict(list)
        for h in self._parent:
            out[self.find(h)].append(h)
        return dict(out)


# ---------------------------------------------------------------------------
# Trajectory collection helpers
# ---------------------------------------------------------------------------

def collect_continuous_trajectories(
    env: gym.Env,
    n_episodes: int,
    max_steps: int,
    policy: Callable | None = None,
    seed: int = 0,
) -> ContinuousTrajectoryDataset:
    """Collect episodes from a Gymnasium environment.

    policy: callable(obs, rng) → action; defaults to uniform random.
    """
    rng = np.random.default_rng(seed)
    trajs: list[ContinuousTrajectory] = []
    for ep in range(n_episodes):
        obs_list: list[np.ndarray] = []
        act_list: list[Any] = []
        rew_list: list[float] = []
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        obs_list.append(np.asarray(obs, dtype=np.float32))
        done = False
        steps = 0
        while not done and steps < max_steps:
            if policy is not None:
                action = policy(obs, rng)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            act_list.append(action)
            rew_list.append(float(reward))
            obs_list.append(np.asarray(obs, dtype=np.float32))
            steps += 1
        trajs.append(ContinuousTrajectory(
            observations=obs_list, actions=act_list, rewards=rew_list
        ))
    return ContinuousTrajectoryDataset(trajectories=trajs)


# ---------------------------------------------------------------------------
# History/continuation extraction
# ---------------------------------------------------------------------------

def _obs_scalar(o: np.ndarray) -> float:
    """Reduce a potentially multi-dim observation to a single scalar for binning.

    Uses the L2 norm of the observation vector, which is non-zero whenever any
    component is non-zero.  This avoids the single-bin failure mode that occurs
    when the first observation component is always zero (e.g., PO-HalfCheetah
    has positions zeroed out so ravel()[0] == 0 for every timestep).

    For scalar observations the L2 norm equals the absolute value, which may
    lose sign information.  When the observation is guaranteed to be 1-D and
    signed (e.g., Dec-Tiger), use ``ravel()[0]`` directly; the norm is a safe
    conservative choice for all environments.
    """
    arr = np.asarray(o, dtype=np.float32).ravel()
    if arr.size == 1:
        return float(arr[0])  # preserve sign for 1-D signed observations
    return float(np.linalg.norm(arr))


def auto_select_bins(
    dataset: ContinuousTrajectoryDataset,
    b_min: int = 2,
    b_max: int = 32,
    n_sample: int = 2000,
    seed: int = 0,
) -> int:
    """Select history_n_bins via k-means silhouette score on observation scalars.

    Reduces the bin-count selection to a standard unsupervised learning problem.
    Uses _obs_scalar to get 1D values, then finds the k with best silhouette.

    Parameters
    ----------
    dataset:  trajectory dataset to fit on
    b_min:    minimum number of bins to try (inclusive)
    b_max:    maximum number of bins to try (inclusive)
    n_sample: subsample size for speed; if dataset has fewer points, use all
    seed:     random seed for subsampling and k-means

    Returns
    -------
    Optimal bin count according to silhouette score maximisation.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Collect observation scalars
    vals = []
    for traj in dataset.trajectories:
        for o in traj.observations:
            vals.append(_obs_scalar(o))
    vals = np.array(vals).reshape(-1, 1)

    # Subsample for speed
    rng = np.random.default_rng(seed)
    if len(vals) > n_sample:
        idx = rng.choice(len(vals), n_sample, replace=False)
        vals = vals[idx]

    best_k, best_score = b_min, -1.0
    for k in range(b_min, min(b_max + 1, len(vals) // 10)):
        try:
            km = KMeans(n_clusters=k, n_init=5, random_state=seed)
            labels = km.fit_predict(vals)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(vals, labels)
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            continue
    return best_k


def _fit_obs_bins(
    dataset: ContinuousTrajectoryDataset,
    n_bins: int,
) -> np.ndarray | None:
    """Fit equal-frequency quantile boundaries for observation binning.

    Returns boundaries array of shape (n_bins-1,) or None if n_bins == 0.
    Uses _obs_scalar to reduce multi-dim observations to a single value,
    so the boundaries generalise across environments including those where
    the first observation component is always zero (e.g., PO-HalfCheetah).
    """
    if n_bins <= 0:
        return None
    all_vals = []
    for traj in dataset.trajectories:
        for o in traj.observations:
            all_vals.append(_obs_scalar(o))
    if not all_vals:
        return None
    pcts = np.linspace(0, 100, n_bins + 1)[1:-1]
    return np.percentile(all_vals, pcts)


def _bin_obs(o: np.ndarray, boundaries: np.ndarray | None) -> int:
    """Map a continuous observation to a bin index."""
    if boundaries is None:
        return 0
    v = _obs_scalar(o)
    return int(np.searchsorted(boundaries, v))


def _make_coarse_history_key(
    h_obs: list[np.ndarray],
    h_acts: list,
    boundaries: np.ndarray | None,
) -> HistoryKey:
    """Build a history key using binned observation indices (not raw floats).

    This is the crucial difference from make_history_key: by using coarse
    observation bins we accumulate many samples per history class, enabling
    the MMD test to run on the (raw float) continuation vectors.
    """
    parts: list = []
    for i, o in enumerate(h_obs):
        bin_idx = _bin_obs(o, boundaries)
        parts.append(bin_idx)
        if i < len(h_acts):
            a = h_acts[i]
            parts.append(int(a) if isinstance(a, (int, np.integer)) else 0)
    return tuple(parts)


def _extract_histories_and_continuations(
    dataset: ContinuousTrajectoryDataset,
    history_max_length: int,
    window_length: int,
    min_occurrences: int,
    history_n_bins: int = 8,
) -> dict[HistoryKey, np.ndarray]:
    """Return {history_key: continuation_matrix(shape=(m, d_cont))}.

    History keys use coarse observation bins (history_n_bins) so that
    enough continuation samples accumulate per key for the MMD test.
    Continuation vectors remain as raw float32 arrays.
    """
    # Fit quantile boundaries for history key binning.
    boundaries = _fit_obs_bins(dataset, history_n_bins)

    continuations: dict[HistoryKey, list[np.ndarray]] = defaultdict(list)

    for traj in dataset.trajectories:
        T = traj.length
        obs = traj.observations
        acts = traj.actions

        for start in range(T):
            # hlen controls how many PAST steps we include before obs[start].
            # With history_max_length=L, h_obs has hlen+1 elements (1..L+1),
            # but we want at most L elements, so hlen ≤ L-1.
            for hlen in range(0, min(history_max_length - 1, start) + 1):
                h_start = start - hlen
                h_obs = obs[h_start: start + 1]   # hlen+1 observations (length ≤ L)
                h_acts = acts[h_start: start]      # hlen actions

                cont_start = start + 1
                cont_end = cont_start + window_length
                if cont_end > T:
                    break

                # Continuation vector: raw float32 for the MMD test.
                cont_parts = []
                for ci in range(window_length):
                    o = np.asarray(obs[cont_start + ci], dtype=np.float32)
                    cont_parts.append(o)
                    if ci < window_length - 1:
                        a = acts[cont_start + ci] if cont_start + ci < len(acts) else 0
                        if isinstance(a, (int, np.integer)):
                            cont_parts.append(np.array([float(a)], dtype=np.float32))
                        else:
                            cont_parts.append(np.asarray(a, dtype=np.float32).ravel())

                cont_vec = np.concatenate(cont_parts)

                # History key: coarsely binned observation indices.
                hkey = _make_coarse_history_key(list(h_obs), list(h_acts), boundaries)
                continuations[hkey].append(cont_vec)

    result = {}
    for key, vecs in continuations.items():
        if len(vecs) >= min_occurrences:
            result[key] = np.stack(vecs, axis=0).astype(np.float64)
    return result


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

class ContinuousQuotientEstimator:
    """Offline batch quotient discovery for continuous POMDPs.

    Parameters
    ----------
    config: ContinuousEstimatorConfig with all hyperparameters
    """

    def __init__(self, config: ContinuousEstimatorConfig | None = None):
        self.config = config or ContinuousEstimatorConfig()
        self._test = ContinuousMMDTest(
            kernel=self.config.kernel,
            bandwidth=self.config.bandwidth,
            n_permutations=self.config.n_permutations,
            alpha=self.config.alpha,
        )

    def fit(
        self,
        trajectories: list[ContinuousTrajectory] | ContinuousTrajectoryDataset,
        env_id: str = "",
        verbose: bool = False,
    ) -> CanonicalContinuousQuotient:
        """Estimate the quotient partition from trajectory data."""
        if isinstance(trajectories, list):
            dataset = ContinuousTrajectoryDataset(trajectories=trajectories)
        else:
            dataset = trajectories

        cfg = self.config

        # Resolve adaptive binning sentinel: history_n_bins == 0 → auto-select.
        history_n_bins = cfg.history_n_bins
        if history_n_bins == 0:
            history_n_bins = auto_select_bins(dataset)
            if verbose:
                print(f"[ContinuousQuotientEstimator] auto_select_bins → B={history_n_bins}")

        # Extract per-history continuation matrices.
        history_continuations = _extract_histories_and_continuations(
            dataset,
            history_max_length=cfg.history_max_length,
            window_length=cfg.window_length,
            min_occurrences=cfg.min_history_occurrences,
            history_n_bins=history_n_bins,
        )

        viable = list(history_continuations.keys())
        if not viable:
            return CanonicalContinuousQuotient(
                n_classes=0, partition={}, config=cfg, env_id=env_id,
                n_trajectories=dataset.n_trajectories,
            )

        n = len(viable)
        max_m = cfg.max_samples_per_test

        def _subsample(arr: np.ndarray, pair_seed: int) -> np.ndarray:
            # Deterministic seed per pair for reproducibility (Codex D4 fix).
            if max_m is None or arr.shape[0] <= max_m:
                return arr
            rng_pair = np.random.default_rng(pair_seed)
            idx = rng_pair.choice(arr.shape[0], size=max_m, replace=False)
            return arr[idx]

        # Phase 1: collect all p-values (α is not applied per-pair yet).
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        p_values: list[float] = []
        iter_ = tqdm(pairs, desc="MMD tests", disable=not verbose)
        for i, j in iter_:
            ha, hb = viable[i], viable[j]
            pair_seed = int(abs(hash((i, j)))) % (2**31)
            result = self._test.test(
                _subsample(history_continuations[ha], pair_seed),
                _subsample(history_continuations[hb], pair_seed + 1),
                h1_key=ha,
                h2_key=hb,
                alpha=1.0,  # collect raw p-value; threshold applied via BH below
            )
            p_values.append(result.p_value)

        # Phase 2: Benjamini-Hochberg FDR correction at level cfg.alpha.
        # BH controls E[false discoveries / max(rejections,1)] ≤ alpha.
        # "Reject" = declare different class (don't merge).
        # This is strictly more powerful than Bonferroni and makes no
        # assumption about B (number of permutations), resolving the
        # infeasibility of Bonferroni at small B.
        m_pairs = len(p_values)
        sorted_idx = np.argsort(p_values)
        bh_threshold = 0.0
        for rank, idx in enumerate(sorted_idx, start=1):
            if p_values[idx] <= rank * cfg.alpha / m_pairs:
                bh_threshold = p_values[idx]
        reject_mask = [p <= bh_threshold for p in p_values]

        uf = _UnionFind(viable)
        for (i, j), reject in zip(pairs, reject_mask):
            if not reject:
                uf.union(viable[i], viable[j])

        groups = uf.groups()

        # Assign stable integer class IDs sorted by first-seen viable index.
        root_order = sorted(groups.keys(), key=lambda r: viable.index(r))
        id_of_root = {r: i for i, r in enumerate(root_order)}

        partition: dict[HistoryKey, int] = {}
        classes: dict[int, ContinuousQuotientClass] = {}
        for root, members in groups.items():
            cid = id_of_root[root]
            classes[cid] = ContinuousQuotientClass(class_id=cid, member_keys=list(members))
            for m in members:
                partition[m] = cid

        return CanonicalContinuousQuotient(
            n_classes=len(groups),
            partition=partition,
            classes=classes,
            config=cfg,
            env_id=env_id,
            n_trajectories=dataset.n_trajectories,
        )

    def fit_from_env(
        self,
        env: gym.Env,
        policy: Callable | None = None,
        seed: int = 0,
        verbose: bool = False,
    ) -> CanonicalContinuousQuotient:
        """Collect trajectories from a Gymnasium env, then fit."""
        dataset = collect_continuous_trajectories(
            env=env,
            n_episodes=self.config.n_trajectories,
            max_steps=self.config.max_steps_per_episode,
            policy=policy,
            seed=seed,
        )
        return self.fit(dataset, env_id=str(env.unwrapped.spec.id if env.unwrapped.spec else ""), verbose=verbose)
