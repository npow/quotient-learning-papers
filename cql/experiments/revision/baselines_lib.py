"""Comparison baselines for the revision (partition-level, scored by ARI against the latent state).

All methods see the same trajectories and none uses the latent index n.  Each ``*_labels`` function returns an
int array of shape (N, nv) of estimated class labels for every valid step (t + 1 + k <= T_obs) and the
number of classes found.  See the paper (Sec. 5, App. C) for what is and is not faithful to each source.

1. ``w1_*``       multivariate Wasserstein-1 between bin-conditional continuation laws (exact OT via linear
                  assignment on equal-size subsamples; Euclidean ground cost on the full continuation vector).
2. ``ferns_*``    Ferns-et-al. bisimulation metric (finite-MDP fixed point, exact LP for W1) on the empirical
                  chain over the B quantile bins.
3. ``dbc_*``      Deep Bisimulation for Control objective (Zhang et al., 2021) without the RL learner.
4. ``causal_*``   causal-state learning (Zhang et al., 2019): recurrent encoder trained to predict the next
                  observation (and reward), hidden states discretised by k-means.
5. ``rkhs_*``     RKHS epsilon-machine (Brodu & Crutchfield, 2022): regularised conditional mean embeddings of
                  the continuation given the present observation, similarity matrix G^S, diffusion-map
                  coordinates, clustering.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from common import Data, valid_steps


# ----------------------------------------------------------------------------------------------
# shared utilities
# ----------------------------------------------------------------------------------------------

def windows(data: Data, k: int = 2):
    """Continuation vectors for every valid step: (N, nv, d_c); same layout as the estimator."""
    N, T1, d = data.obs.shape
    nv = valid_steps(T1, k)
    parts = []
    for ci in range(k):
        parts.append(data.obs[:, 1 + ci: 1 + ci + nv, :])
        if ci < k - 1:
            parts.append(data.act[:, 1 + ci: 1 + ci + nv, None].astype(np.float32))
    return np.concatenate(parts, axis=2).astype(np.float64)


def scalar_obs(data: Data, nv: int):
    o = data.obs[:, :nv, :]
    return o[..., 0] if o.shape[-1] == 1 else np.linalg.norm(o, axis=-1)


def bin_index(data: Data, B: int, nv: int):
    z = scalar_obs(data, nv)
    allz = scalar_obs(data, data.obs.shape[1])
    edges = np.percentile(allz.ravel(), np.linspace(0, 100, B + 1)[1:-1])
    return np.searchsorted(edges, z), edges


def kmeans_silhouette(X: np.ndarray, kmax: int = 12, seed: int = 0, n_sub: int = 3000):
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), min(n_sub, len(X)), replace=False)]
    best, best_k = -2.0, 2
    for k in range(2, kmax + 1):
        lab = KMeans(k, n_init=5, random_state=seed).fit_predict(sub)
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(sub, lab)
        if s > best:
            best, best_k = s, k
    km = KMeans(best_k, n_init=5, random_state=seed).fit(sub)
    return km.predict(X), best_k


def cluster_distance_matrix(D: np.ndarray, kmax: int | None = None):
    """Average-linkage clustering of items with a precomputed distance matrix; k by silhouette (no n used)."""
    B = len(D)
    if B <= 2:
        return np.arange(B)
    kmax = min(B - 1, kmax or B - 1)
    best, lab_best = -2.0, np.zeros(B, int)
    for k in range(2, kmax + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(D)
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(D, lab, metric="precomputed")
        if s > best:
            best, lab_best = s, lab
    return lab_best


def threshold_merge(D: np.ndarray, eps: float):
    """Single-linkage merge of items with D < eps (same union-find semantics as the quotient estimator)."""
    B = len(D)
    parent = list(range(B))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(B):
        for j in range(i + 1, B):
            if D[i, j] < eps:
                parent[find(i)] = find(j)
    roots = {find(i) for i in range(B)}
    ids = {r: n for n, r in enumerate(sorted(roots))}
    return np.array([ids[find(i)] for i in range(B)])


def bins_to_labels(bidx: np.ndarray, bin_label: np.ndarray):
    return bin_label[bidx]


# ----------------------------------------------------------------------------------------------
# 1. multivariate Wasserstein-1
# ----------------------------------------------------------------------------------------------

def w1_distance_matrix(data: Data, B: int, k: int = 2, m: int = 300, seed: int = 0):
    W = windows(data, k)
    N, nv, d = W.shape
    bidx, _ = bin_index(data, B, nv)
    rng = np.random.default_rng(seed)
    flat, fb = W.reshape(-1, d), bidx.ravel()
    sets = []
    for b in range(B):
        idx = np.flatnonzero(fb == b)
        sets.append(flat[rng.choice(idx, min(m, len(idx)), replace=False)] if len(idx) else None)
    D = np.zeros((B, B))
    for i in range(B):
        for j in range(i + 1, B):
            if sets[i] is None or sets[j] is None:
                D[i, j] = D[j, i] = np.inf
                continue
            mm = min(len(sets[i]), len(sets[j]))
            X, Y = sets[i][:mm], sets[j][:mm]
            C = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
            r, c = linear_sum_assignment(C)
            D[i, j] = D[j, i] = C[r, c].mean()
    return D, bidx


def w1_labels(data: Data, B: int, eps: float | None):
    """eps=None -> parameter-free: average linkage on the W1 matrix, k by silhouette."""
    D, bidx = w1_distance_matrix(data, B)
    if eps is None:
        bl = cluster_distance_matrix(np.where(np.isfinite(D), D, D[np.isfinite(D)].max() * 2))
    else:
        bl = threshold_merge(D, eps)
    return bins_to_labels(bidx, bl), len(set(bl.tolist()))


# ----------------------------------------------------------------------------------------------
# 2. Ferns bisimulation metric on the binned empirical chain
# ----------------------------------------------------------------------------------------------

def _w1_finite(p: np.ndarray, q: np.ndarray, d: np.ndarray) -> float:
    B = len(p)
    if np.allclose(p, q):
        return 0.0
    # transport plan x_ij >= 0, sum_j x_ij = p_i, sum_i x_ij = q_j
    A_eq = np.zeros((2 * B, B * B))
    for i in range(B):
        A_eq[i, i * B:(i + 1) * B] = 1.0
        A_eq[B + i, i::B] = 1.0
    res = linprog(d.ravel(), A_eq=A_eq, b_eq=np.concatenate([p, q]), bounds=(0, None), method="highs")
    return float(res.fun)


def reward_signal(data: Data, env: str, nv: int):
    """Tiger: native reward (normalised to [0,1]).  Reward-free envs: next-observation emission r_t = o_{t+1}."""
    if env == "tiger":
        r = data.rew[:, :nv]
    else:
        r = data.obs[:, 1:1 + nv, 0]
    r = (r - r.min()) / max(r.max() - r.min(), 1e-9)
    return r


def ferns_metric(data: Data, env: str, B: int, c: float = 0.9, iters: int = 40):
    N, T1, _ = data.obs.shape
    nv = valid_steps(T1, 2)
    bidx_all, _ = bin_index(data, B, T1)         # bins for all T1 observations
    bidx = bidx_all[:, :nv]
    nxt = bidx_all[:, 1:1 + nv]
    act = data.act[:, :nv]
    r = reward_signal(data, env, nv)
    A = int(data.act.max()) + 1
    P = np.zeros((A, B, B)); R = np.zeros((A, B)); cnt = np.zeros((A, B))
    for a in range(A):
        for b in range(B):
            msk = (bidx == b) & (act == a)
            n_ = msk.sum()
            cnt[a, b] = n_
            if n_ == 0:
                continue
            R[a, b] = r[msk].mean()
            P[a, b] = np.bincount(nxt[msk], minlength=B) / n_
    d = np.zeros((B, B))
    for _ in range(iters):
        dn = np.zeros((B, B))
        for i in range(B):
            for j in range(i + 1, B):
                best = 0.0
                for a in range(A):
                    if cnt[a, i] == 0 or cnt[a, j] == 0:
                        continue
                    v = (1 - c) * abs(R[a, i] - R[a, j]) + c * _w1_finite(P[a, i], P[a, j], d)
                    best = max(best, v)
                dn[i, j] = dn[j, i] = best
        if np.abs(dn - d).max() < 1e-6:
            d = dn
            break
        d = dn
    return d, bidx


def ferns_labels(data: Data, env: str, B: int):
    D, bidx = ferns_metric(data, env, B)
    bl = cluster_distance_matrix(D)
    return bins_to_labels(bidx, bl), len(set(bl.tolist()))


# ----------------------------------------------------------------------------------------------
# 3. DBC (Zhang et al., 2021): bisimulation objective with a learned Gaussian latent dynamics model
# ----------------------------------------------------------------------------------------------

def dbc_labels(data: Data, env: str, seed: int = 0, steps: int = 3000, gamma: float = 0.99, zdim: int = 8):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    N, T1, d = data.obs.shape
    nv = valid_steps(T1, 2)
    A = int(data.act.max()) + 1
    o = torch.tensor(data.obs[:, :nv].reshape(-1, d), dtype=torch.float32)
    o2 = torch.tensor(data.obs[:, 1:1 + nv].reshape(-1, d), dtype=torch.float32)
    a = torch.tensor(np.eye(A)[data.act[:, :nv].reshape(-1)], dtype=torch.float32)
    r = torch.tensor(reward_signal(data, env, nv).reshape(-1), dtype=torch.float32)
    enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, zdim))
    dyn = nn.Sequential(nn.Linear(zdim + A, 64), nn.ReLU(), nn.Linear(64, 2 * zdim))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dyn.parameters()), lr=1e-3)
    n = len(o)
    for _ in range(steps):
        idx = torch.randint(0, n, (256,))
        z = enc(o[idx])
        with torch.no_grad():
            zt = enc(o2[idx])
        out = dyn(torch.cat([z.detach(), a[idx]], 1))
        mu, sig = out[:, :zdim], torch.nn.functional.softplus(out[:, zdim:]) + 1e-3
        nll = (0.5 * ((zt - mu) / sig) ** 2 + torch.log(sig)).sum(1).mean()   # transition model
        perm = torch.randperm(256)
        zi, zj = z, z[perm]
        with torch.no_grad():
            W2 = torch.sqrt(((mu - mu[perm]) ** 2).sum(1) + ((sig - sig[perm]) ** 2).sum(1) + 1e-12)
        dz = (zi - zj).abs().sum(1)
        bis = ((dz - ((r[idx] - r[idx][perm]).abs() + gamma * W2)) ** 2).mean()
        loss = bis + nll
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Z = enc(o).numpy()
    lab, kk = kmeans_silhouette(Z, kmax=12, seed=seed)
    return lab.reshape(N, nv), kk


# ----------------------------------------------------------------------------------------------
# 4. causal-state learning (Zhang et al., 2019): recurrent encoder + k-means discretisation
# ----------------------------------------------------------------------------------------------

def causal_labels(data: Data, env: str, seed: int = 0, epochs: int = 30, hidden: int = 32):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    N, T1, d = data.obs.shape
    nv = valid_steps(T1, 2)
    A = int(data.act.max()) + 1
    obs = torch.tensor(data.obs, dtype=torch.float32)                       # (N, T1, d)
    act = torch.tensor(np.eye(A)[data.act], dtype=torch.float32)            # (N, T, A)
    prev_act = torch.cat([torch.zeros(N, 1, A), act], dim=1)                # a_{t-1}, zero at t=0   (N, T1, A)
    rew = torch.tensor(reward_signal(data, env, data.act.shape[1]), dtype=torch.float32) if env == "tiger" \
        else None
    gru = nn.GRU(d + A, hidden, batch_first=True)
    head = nn.Sequential(nn.Linear(hidden + A, 64), nn.ReLU(), nn.Linear(64, d + 1))
    opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=3e-3)
    Tm = data.act.shape[1]
    for _ in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N, 128):
            ib = perm[s:s + 128]
            h, _ = gru(torch.cat([obs[ib], prev_act[ib]], 2))               # (b, T1, H)
            pred = head(torch.cat([h[:, :Tm], act[ib]], 2))                 # predict o_{t+1}, r_t from (h_t, a_t)
            loss = ((pred[..., :d] - obs[ib][:, 1:1 + Tm]) ** 2).mean()
            if rew is not None:
                loss = loss + ((pred[..., d] - rew[ib]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        h, _ = gru(torch.cat([obs, prev_act], 2))
    H = h[:, :nv].reshape(-1, hidden).numpy()
    lab, kk = kmeans_silhouette(H, kmax=12, seed=seed)
    return lab.reshape(N, nv), kk


# ----------------------------------------------------------------------------------------------
# 5. RKHS epsilon-machine (Brodu & Crutchfield, 2022), specialised to one-observation pasts
# ----------------------------------------------------------------------------------------------

def _rbf(X, Y, xi):
    sq = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None] - 2 * X @ Y.T
    return np.exp(-np.maximum(sq, 0) / xi ** 2)


def _median_dist(X, seed=0, n=500):
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    sq = (S * S).sum(1)[:, None] + (S * S).sum(1)[None] - 2 * S @ S.T
    dd = np.sqrt(np.maximum(sq[sq > 1e-12], 0))
    return float(np.median(dd)) if dd.size else 1.0


def rkhs_labels(data: Data, seed: int = 0, S: int = 1500, cluster: str = "dbscan"):
    """Past x = present observation (history length 1, as in Sec. 5); future y = k=2 continuation window.
    Omega = (G_X + eps I)^{-1} G_X, G^S = Omega^T G_Y Omega, diffusion map of G^S, then clustering.
    eps by 3-fold CV on held-out embedding error (the paper: 'eps can be set by cross-validation')."""
    N, T1, d = data.obs.shape
    nv = valid_steps(T1, 2)
    W = windows(data)
    rng = np.random.default_rng(seed)
    flat_i = rng.choice(N * nv, min(S, N * nv), replace=False)
    ti, tt = np.divmod(flat_i, nv)
    X = data.obs[ti, tt].astype(np.float64)
    Y = W[ti, tt]
    xi_x, xi_y = _median_dist(X), _median_dist(Y)
    GX, GY = _rbf(X, X, xi_x), _rbf(Y, Y, xi_y)
    n = len(X)
    best_eps, best_err = None, np.inf
    folds = np.array_split(rng.permutation(n), 3)
    for eps in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        err = 0.0
        for f in folds:
            tr = np.setdiff1d(np.arange(n), f)
            Om = np.linalg.solve(GX[np.ix_(tr, tr)] + eps * np.eye(len(tr)), GX[np.ix_(tr, f)])   # (tr, f)
            Gtr, Gcross = GY[np.ix_(tr, tr)], GY[np.ix_(tr, f)]
            # ||s_hat(x) - phi(y)||^2 = w^T G w - 2 w^T k(y_tr, y) + 1
            err += float(np.mean(np.einsum("if,ij,jf->f", Om, Gtr, Om) - 2 * np.einsum("if,if->f", Om, Gcross) + 1))
        if err < best_err:
            best_eps, best_err = eps, err
    Om = np.linalg.solve(GX + best_eps * np.eye(n), GX)
    GS = Om.T @ GY @ Om
    GS = (GS + GS.T) / 2
    # diffusion map (Coifman-Lafon alpha=1 density normalisation, then row-stochastic)
    q = np.maximum(GS.sum(1), 1e-12)
    Kt = GS / np.outer(q, q)
    Kt = np.maximum(Kt, 0)
    Mrow = Kt / Kt.sum(1, keepdims=True)
    lam, psi = np.linalg.eig(Mrow)
    order = np.argsort(-lam.real)
    lam, psi = lam.real[order], psi.real[:, order]
    top = lam[1:11]
    gaps = top[:-1] - top[1:]
    M = int(np.argmax(gaps)) + 1                     # keep coordinates before the largest spectral gap
    coords = psi[:, 1:1 + M] * lam[1:1 + M]
    coords = coords / np.maximum(np.abs(psi[:, :1]), 1e-12)   # psi_1 normalised to 1 as in the paper
    if cluster == "dbscan":
        rng_ = np.ptp(coords[:, 0]) if np.ptp(coords[:, 0]) > 0 else 1.0
        lab = DBSCAN(eps=0.1 * rng_, min_samples=max(5, n // 100)).fit_predict(coords)
        k_found = len(set(lab.tolist()) - {-1})
        lab = np.where(lab < 0, k_found, lab)        # noise -> its own extra label for scoring
    else:
        lab, k_found = kmeans_silhouette(coords, kmax=12, seed=seed, n_sub=n)
    return ti, tt, lab, k_found


def score_flat(data: Data, ti, tt, lab):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(data.latent[ti, tt], lab))


def score_grid(data: Data, lab_grid: np.ndarray):
    from sklearn.metrics import adjusted_rand_score
    nv = lab_grid.shape[1]
    return float(adjusted_rand_score(data.latent[:, :nv].ravel(), lab_grid.ravel()))
