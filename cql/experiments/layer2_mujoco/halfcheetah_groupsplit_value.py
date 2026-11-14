"""Held-out (trajectory-level) value prediction on PO-HalfCheetah: do quotient classes carry information
beyond a NONLINEAR function of instantaneous speed?

For every (policy, collection seed): 5-fold GroupKFold over TRAJECTORIES.  In each fold the quotient (auto-B
silhouette, or fixed B=8) and its quantile edges are fitted on the training trajectories only; test steps are
assigned to the class of their bin.  Targets: discounted return-to-go (gamma=0.99) within the episode.
Predictors (all fitted on train folds, scored out-of-fold, pooled):
  null            constant
  class_onehot    mean return per class (the "quotient classes" predictor)
  speed_linear    OLS on ||v||_2
  speed_gbm       gradient-boosted trees on ||v||_2 only (nonlinear, 1 feature)
  vel8_gbm        gradient-boosted trees on all 8 observed velocity coordinates (instantaneous information)
  class+speed_gbm trees on [class id one-hot, ||v||_2]
Uncertainty: 95% percentile bootstrap over TRAJECTORIES (2000 resamples) of the pooled out-of-fold R^2.
Output: results/halfcheetah_groupsplit_value.csv (one row per policy/seed/B-rule/model)
"""
from __future__ import annotations
import csv, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from halfcheetah_sinusoidal_value import make_po_halfcheetah, SinusoidalPolicy  # noqa: E402
from continuous_quotient import ContinuousEstimatorConfig, ContinuousQuotientEstimator, collect_continuous_trajectories  # noqa: E402
from continuous_quotient.estimator import _fit_obs_bins, _make_coarse_history_key, auto_select_bins  # noqa: E402
from continuous_quotient.types import ContinuousTrajectoryDataset  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

OUT = Path(__file__).parent / "results" / "halfcheetah_groupsplit_value.csv"
GAMMA = 0.99


def collect(policy_name: str, seed: int, n_ep: int = 1000, T: int = 100):
    env = make_po_halfcheetah()
    if policy_name == "sinusoidal":
        pol = SinusoidalPolicy(freq=2.0, amp=0.5, seed=seed)
    else:
        z = np.zeros(env.action_space.shape)
        pol = lambda obs, rng, _z=z: _z
    return collect_continuous_trajectories(env, n_episodes=n_ep, max_steps=T, policy=pol, seed=seed)


def returns(traj):
    G = np.zeros(traj.length); run = 0.0
    for t in range(len(traj.rewards) - 1, -1, -1):
        run = traj.rewards[t] + GAMMA * run; G[t] = run
    return G


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def run(policy_name, seed, b_rule):
    ds = collect(policy_name, seed)
    trajs = ds.trajectories
    nT = len(trajs)
    fold_of = np.zeros(nT, int)
    for f, (_, te) in enumerate(GroupKFold(5).split(np.arange(nT), groups=np.arange(nT))):
        fold_of[te] = f
    cols = {k: [] for k in ["traj", "y", "pred_null", "pred_class", "pred_lin", "pred_gbm", "pred_vel8", "pred_cls_gbm"]}
    nclass = []
    for f in range(5):
        tr = [t for i, t in enumerate(trajs) if fold_of[i] != f]
        tr_ds = ContinuousTrajectoryDataset(trajectories=tr)
        B = auto_select_bins(tr_ds) if b_rule == "auto" else 8
        cfg = ContinuousEstimatorConfig(n_trajectories=len(tr), max_steps_per_episode=100, window_length=2,
                                        history_max_length=1, min_history_occurrences=20, alpha=0.05,
                                        n_permutations=99, history_n_bins=B, max_samples_per_test=300)
        q = ContinuousQuotientEstimator(cfg).fit(tr_ds)
        nclass.append(q.n_classes)
        bnd = _fit_obs_bins(tr_ds, B)

        def feats(tlist):
            cid, sp, v8, y, tid = [], [], [], [], []
            for ti, t in tlist:
                G = returns(t)
                for s in range(t.length):
                    o = np.asarray(t.observations[s], dtype=np.float32)
                    c = q.partition.get(_make_coarse_history_key([o], [], bnd), -1)
                    cid.append(c); sp.append(float(np.linalg.norm(o))); v8.append(o[9:] if o.size == 17 else o)
                    y.append(G[s]); tid.append(ti)
            return np.array(cid), np.array(sp), np.array(v8), np.array(y), np.array(tid)
        c_tr, s_tr, v_tr, y_tr, _ = feats([(i, t) for i, t in enumerate(trajs) if fold_of[i] != f])
        c_te, s_te, v_te, y_te, id_te = feats([(i, t) for i, t in enumerate(trajs) if fold_of[i] == f])
        ok_tr, ok_te = c_tr >= 0, c_te >= 0
        c_tr, s_tr, v_tr, y_tr = c_tr[ok_tr], s_tr[ok_tr], v_tr[ok_tr], y_tr[ok_tr]
        c_te, s_te, v_te, y_te, id_te = c_te[ok_te], s_te[ok_te], v_te[ok_te], y_te[ok_te], id_te[ok_te]
        cm = {c: y_tr[c_tr == c].mean() for c in np.unique(c_tr)}
        p_class = np.array([cm.get(c, y_tr.mean()) for c in c_te])
        p_lin = LinearRegression().fit(s_tr[:, None], y_tr).predict(s_te[:, None])
        p_gbm = HistGradientBoostingRegressor(random_state=0).fit(s_tr[:, None], y_tr).predict(s_te[:, None])
        p_v8 = HistGradientBoostingRegressor(random_state=0).fit(v_tr, y_tr).predict(v_te)
        oh = lambda c: (c[:, None] == np.arange(max(cm) + 1)[None]).astype(float)
        Xtr, Xte = np.hstack([oh(c_tr), s_tr[:, None]]), np.hstack([oh(c_te), s_te[:, None]])
        p_cg = HistGradientBoostingRegressor(random_state=0).fit(Xtr, y_tr).predict(Xte)
        for k, v in [("traj", id_te), ("y", y_te), ("pred_null", np.full_like(y_te, y_tr.mean())),
                     ("pred_class", p_class), ("pred_lin", p_lin), ("pred_gbm", p_gbm), ("pred_vel8", p_v8),
                     ("pred_cls_gbm", p_cg)]:
            cols[k].append(v)
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    rng = np.random.default_rng(0)
    uniq = np.unique(cols["traj"])
    idx_by = {u: np.flatnonzero(cols["traj"] == u) for u in uniq}
    models = {"null": "pred_null", "class_onehot": "pred_class", "speed_linear": "pred_lin", "speed_gbm": "pred_gbm",
              "vel8_gbm": "pred_vel8", "class+speed_gbm": "pred_cls_gbm"}
    boots = {m: [] for m in models}
    diffs = {m: [] for m in models}
    for _ in range(2000):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx_by[u] for u in pick])
        y = cols["y"][ii]
        for m, k in models.items():
            boots[m].append(r2(y, cols[k][ii]))
            diffs[m].append(r2(y, cols[k][ii]) - r2(y, cols["pred_gbm"][ii]))
    rows = []
    for m, k in models.items():
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        dlo, dhi = np.percentile(diffs[m], [2.5, 97.5])
        rows.append(dict(policy=policy_name, seed=seed, b_rule=b_rule, n_classes_per_fold=";".join(map(str, nclass)),
                         n_traj=nT, n_steps_scored=len(cols["y"]), model=m,
                         r2_oof=round(r2(cols["y"], cols[k]), 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                         diff_vs_speed_gbm=round(r2(cols["y"], cols[k]) - r2(cols["y"], cols["pred_gbm"]), 4),
                         diff_lo=round(dlo, 4), diff_hi=round(dhi, 4)))
    return rows


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(p, s, b) for p in ["sinusoidal", "passive"] for s in range(a.seeds) for b in ["auto", "fixed8"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex, open(OUT, "w", newline="") as f:
        w = None
        for rows in ex.map(lambda j: run(*j), jobs) if False else ex.map(run, *zip(*jobs)):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            w.writerows(rows); f.flush(); print(rows[0]["policy"], rows[0]["seed"], rows[0]["b_rule"], [(r["model"], r["r2_oof"]) for r in rows], flush=True)
