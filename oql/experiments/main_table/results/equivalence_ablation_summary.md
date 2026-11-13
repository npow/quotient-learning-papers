# Equivalence-Test Ablation Summary

All tests run on the **same** trajectory dataset per (env, seed). MMD and EmpiricalBernstein continuation samples are subsampled to max 500 per history to keep kernel matrices (O(n²)) tractable; GTest uses all samples.

| env | test | mean n_classes | std n_classes | mean wall_time_s | std wall_time_s |
|-----|------|---------------|---------------|-----------------|----------------|
| TMaze(L=4) | EmpiricalBernstein | 2.00 | 0.00 | 0.20 | 0.03 |
| TMaze(L=4) | GTest | 3.00 | 0.00 | 0.01 | 0.01 |
| TMaze(L=4) | MMD | 3.00 | 0.00 | 5.15 | 0.37 |
| Tiger | EmpiricalBernstein | 1.00 | 0.00 | 0.07 | 0.00 |
| Tiger | GTest | 2.00 | 0.00 | 0.00 | 0.00 |
| Tiger | MMD | 1.00 | 0.00 | 1.45 | 0.03 |

**Note**: GTest is the primary test used in the paper. MMD (permutation, 200 shuffles) and EmpiricalBernstein are ablations. Verified outcome (subsample cap 500, 5 seeds): GTest recovers Tiger=2.0 / T-Maze(L=4)=3.0; MMD recovers Tiger=1.0 (collapse) / T-Maze=3.0; EmpiricalBernstein recovers Tiger=1.0 / T-Maze=2.0. On Tiger the kernel tests cannot resolve the small TV gap (ε_min≈0.07) between the two listen-observation histories, so MMD/EB collapse to n=1. GTest remains the recommended default for tabular discrete alphabets.
