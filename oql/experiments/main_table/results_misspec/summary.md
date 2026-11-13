# Misspecification probe (REV-5) — NOT certified; results certify nothing

MISSPECIFICATION PROBE: eps_min=0.5 is deliberately set ABOVE the true TV (0.1) of the engineered pair ((0,), (1,)) of MisspecChain, violating the gap model (paper Eq. 2). This run exhibits the uncertified silent-merge failure mode and certifies NOTHING.

## Configuration

- Environment: MisspecChain(tau=0.1); contexts (0,), (1,), (2,) at r=0, k=1; true pairwise TVs: TV((0,),(1,)) = 0.1 (sub-gap), TV((0,),(2,)) = TV((1,),(2,)) = 0.8.
- eps_min = 0.5 (SAME threshold eps_min/2 = 0.25), delta = 0.05, beta = 0.05, Bonferroni alpha_0 = delta/(2 K^2) = 0.002778 at K = 3.
- m* = 278 per context; budget N = 2600 episodes (50/50 split; ~N/6 = 433 inference blocks per context).
- Estimator: certified machinery verbatim (disjoint split, one block per inference trajectory, fixed-alphabet empirical-TV test, m* gate); only the gap assumption is violated.
- near_threshold rule: margin < eps_min/8 = 0.0625.

## Headline results

- Silent merge on the sub-gap pair ((0,), (1,)): 10/10 seeds returned a confident SAME verdict (no abstention, no flag).
- Mean empirical TV on the sub-gap pair: 0.1090 (true 0.1); mean margin to threshold: 0.1410.
- near_threshold flags: 0/30 pair rows — the wrong merge is NOT a borderline call under the eps_min/8 rule.
- Separated pairs: 20/20 DIFFERENT verdicts (all correct).
- Reported class count: [2] in every seed (true quotient: 3) — the merge silently deletes one class.
- UNDETERMINED pairs: 0; minimum inference count 388 >= m* = 278 (the abstention gate never triggered).
- A correctly specified run (eps_min = tau = 0.1) would require m* = 6927 blocks per context, i.e. roughly N >= 41562 episodes at this layout — the honest price of resolving the pair.

## Files

- misspec_pairs.csv: one row per seed per tested pair (empirical TV, threshold, margin, verdict, near_threshold, true TV).
- misspec_chain_seed<seed>.misspec.json: per-run sidecars ("certified": false, "misspecification_probe": true).
