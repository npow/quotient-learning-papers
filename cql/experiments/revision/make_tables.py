"""Generate the LaTeX table bodies used in the paper from the result CSVs (so no number is typed by hand)."""
import sys
from pathlib import Path
import pandas as pd
R = Path(__file__).parent / "results"
pa = pd.read_csv(R / "partition_accuracy.csv")
NAMES = {"tiger": "Tiger", "kcycle2": "K-cycle $K{=}2$", "kcycle3": "K-cycle $K{=}3$", "kcycle4": "K-cycle $K{=}4$",
         "kcycle5": "K-cycle $K{=}5$", "kcycle8": "K-cycle $K{=}8$", "gauss3": "GaussMix $N_s{=}3$",
         "gauss4": "GaussMix $N_s{=}4$", "gauss5": "GaussMix $N_s{=}5$", "gauss6": "GaussMix $N_s{=}6$"}
ORDER = list(NAMES)

def nh(x):
    v = sorted(set(int(a) for a in x))
    return str(v[0]) if len(v) == 1 else f"{v[0]}--{v[-1]}"

def cell(df, est, rule):
    d = df[(df.estimator == est) & (df.b_rule == rule)]
    if d.empty:
        return None
    return dict(B=nh(d.B), n=nh(d.n_hat), ari=d.ari.mean(), sd=d.ari.std(ddof=0), ceil=d.ari_ceiling.mean(), k=len(d))

def fmt(c):
    return "--" if c is None else f"{c['n']} & {c['ari']:.2f}"

def table_ari():
    rows = []
    for e in ORDER:
        df = pa[pa.env == e]
        o, s, f = cell(df, "P", "oracle"), cell(df, "P", "sil"), cell(df, "P", "f8")
        rows.append(f"{NAMES[e]} & {df.n_true.iloc[0]} & {o['n']} & {o['ari']:.2f} & {s['B']} & {s['n']} & {s['ari']:.2f} & {f['n']} & {f['ari']:.2f} & {f['ceil']:.2f} \\\\")
    return "\n".join(rows)

def table_T():
    rows = []
    for e in ORDER:
        df = pa[pa.env == e]
        parts = [NAMES[e], str(df.n_true.iloc[0])]
        for rule in ["oracle", "f8"]:
            for est in ["P", "T"]:
                c = cell(df, est, rule)
                parts.append(fmt(c))
        rows.append(" & ".join(parts) + " \\\\")
    return "\n".join(rows)

if __name__ == "__main__":
    print("%% ARI table\n" + table_ari()); print("\n%% T table\n" + table_T())
    print("seeds per cell:", pa.groupby(["env", "estimator", "b_rule"]).size().unique())


BENVS = ["tiger", "kcycle2", "kcycle4", "kcycle8", "gauss3", "gauss5"]
BNAMES = {"tiger": "Tiger", "kcycle2": "K2", "kcycle4": "K4", "kcycle8": "K8", "gauss3": "G3", "gauss5": "G5"}


def table_baselines():
    b = pd.read_csv(R / "partition_baselines.csv")
    methods = [
        ("MMD-quotient (P), silhouette $B$", ("pa", "P", "sil")),
        ("MMD-quotient (P), $B=8$", ("pa", "P", "f8")),
        ("RKHS $\\varepsilon$-machine, DBSCAN", ("b", "RKHS-eps-machine", "dbscan")),
        ("RKHS $\\varepsilon$-machine, $k$-means/sil.", ("b", "RKHS-eps-machine", "kmeans")),
        ("Causal-state (GRU, full history) + $k$-means", ("b", "CausalState-GRU", "kmeans-sil")),
        ("Ferns bisimulation metric, $B=8$", ("b", "Ferns-bisim-metric", "f8")),
        ("DBC objective + $k$-means", ("b", "DBC", "kmeans-sil")),
        ("$W_1$ (multivariate), silhouette, $B=8$", ("b", "W1-multivariate", "sil/f8")),
        ("$W_1$, $\\varepsilon=0.1$, $B=8$", ("b", "W1-multivariate", "eps0.1/f8")),
        ("$W_1$, $\\varepsilon=0.2$, $B=8$", ("b", "W1-multivariate", "eps0.2/f8")),
        ("$W_1$, $\\varepsilon=0.5$, $B=8$", ("b", "W1-multivariate", "eps0.5/f8")),
        ("$W_1$, $\\varepsilon=1.0$, $B=8$", ("b", "W1-multivariate", "eps1.0/f8")),
    ]
    rows = []
    for name, (src, m, v) in methods:
        cells = []
        for e in BENVS:
            if src == "pa":
                d = pa[(pa.env == e) & (pa.estimator == m) & (pa.b_rule == v)]
                ari, nn = d.ari, d.n_hat
            else:
                d = b[(b.env == e) & (b.method == m) & (b.variant == v)]
                ari, nn = d.ari, d.n_hat
            cells.append("--" if d.empty else f"{ari.mean():.2f} ({nh(nn)})")
        rows.append(name + " & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows)


def table_hc():
    h = pd.read_csv(Path(__file__).resolve().parents[1] / "layer2_mujoco" / "results" / "halfcheetah_groupsplit_value.csv")
    out = []
    names = {"class_onehot": "Quotient classes (one-hot)", "speed_linear": "Speed $\\|v\\|_2$, linear",
             "speed_gbm": "Speed $\\|v\\|_2$, gradient-boosted trees", "class+speed_gbm": "Classes + speed, trees",
             "vel8_gbm": "All 8 velocities, trees"}
    for pol in ["sinusoidal", "passive"]:
        for rule in ["auto", "fixed8"]:
            for m in ["class_onehot", "speed_linear", "speed_gbm", "class+speed_gbm", "vel8_gbm"]:
                d = h[(h.policy == pol) & (h.b_rule == rule) & (h.model == m)]
                out.append(f"{pol} & {'auto' if rule == 'auto' else '8'} & {names[m]} & {d.r2_oof.mean():.3f} & "
                           f"[{d.ci_lo.mean():.3f}, {d.ci_hi.mean():.3f}] & {d.diff_vs_speed_gbm.mean():+.3f} "
                           f"[{d.diff_lo.mean():+.3f}, {d.diff_hi.mean():+.3f}] \\\\")
    return "\n".join(out)
