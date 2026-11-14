"""Merge partial result files and print summary tables used in the paper (mean ARI, n_hat per seed)."""
import sys
from pathlib import Path
import pandas as pd
R = Path(__file__).parent / "results"
p1 = pd.read_csv(R / "partition_accuracy_part1.csv")
p1 = p1[p1.env.isin(["tiger", "kcycle2"])]   # part 1 was interrupted during kcycle3; later envs re-run in part 2
p2 = pd.read_csv(R / "partition_accuracy_part2.csv")
p3 = pd.read_csv(R / "partition_accuracy_part3.csv")   # T rows for tiger/kcycle2 (not reached in part 1)
pa = pd.concat([p1, p2, p3], ignore_index=True)
pa.to_csv(R / "partition_accuracy.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
pa["nh"] = pa.n_hat.astype(int)
g = pa.groupby(["env", "estimator", "b_rule"]).agg(n_true=("n_true", "first"), seeds=("seed", "count"),
    B=("B", lambda x: ",".join(map(str, x))), n_hat=("nh", lambda x: ",".join(map(str, x))),
    ari=("ari", "mean"), ari_sd=("ari", "std"), ceil=("ari_ceiling", "mean"))
print(g.round(3).to_string())
if len(sys.argv) > 1:
    b = pd.read_csv(R / "partition_baselines.csv")
    gb = b.groupby(["env", "method", "variant"]).agg(n_true=("n_true", "first"), seeds=("seed", "count"),
        n_hat=("n_hat", lambda x: ",".join(map(str, x))), ari=("ari", "mean"), ari_sd=("ari", "std"))
    print(gb.round(3).to_string())
