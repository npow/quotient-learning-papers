# Post-hoc ground truth + data-driven eps recipe

Instance: PosthocRandomHMM(n_states=3, n_obs=3, concentration=0.3, instance_seed=1000); estimand r=0, k=1, L=1, q=100, N=30000, 50/50 split, delta=0.05, beta=0.05.

Brute-force truth (ground_truth.json): 3 classes (all singleton), min pairwise TV 0.3872.

Recipe (selection half only, fixed before inference data): 0-prepended sorted pairwise empirical TV spectrum; eps_hat = midpoint of the largest gap with lower edge < 0.5; ABSTAIN-ALL fallback if the gap is < 0.1.

- seeds: 20
- abstain-all (no clean split): 0
- eps_hat: mean 0.1938, min 0.1894, max 0.1992
- validity rate (eps_hat <= true min TV): 20/20
- recovery rate (partition == brute-force truth): 20/20
- abstention rate: mean 0.0000, max 0.0000

| seed | eps_hat | valid | m* | min inf. count | n_classes | recovered | und. |
|---|---|---|---|---|---|---|---|
| 0 | 0.190323 | True | 1913 | 3594 | 3 | True | 0.0 |
| 1 | 0.19175 | True | 1884 | 3624 | 3 | True | 0.0 |
| 2 | 0.194619 | True | 1829 | 3577 | 3 | True | 0.0 |
| 3 | 0.189423 | True | 1931 | 3675 | 3 | True | 0.0 |
| 4 | 0.190201 | True | 1915 | 3677 | 3 | True | 0.0 |
| 5 | 0.194831 | True | 1825 | 3574 | 3 | True | 0.0 |
| 6 | 0.196194 | True | 1800 | 3657 | 3 | True | 0.0 |
| 7 | 0.193516 | True | 1850 | 3673 | 3 | True | 0.0 |
| 8 | 0.191133 | True | 1897 | 3583 | 3 | True | 0.0 |
| 9 | 0.196423 | True | 1796 | 3714 | 3 | True | 0.0 |
| 10 | 0.194877 | True | 1824 | 3628 | 3 | True | 0.0 |
| 11 | 0.190528 | True | 1909 | 3636 | 3 | True | 0.0 |
| 12 | 0.199157 | True | 1747 | 3673 | 3 | True | 0.0 |
| 13 | 0.197768 | True | 1772 | 3642 | 3 | True | 0.0 |
| 14 | 0.193032 | True | 1860 | 3682 | 3 | True | 0.0 |
| 15 | 0.196327 | True | 1798 | 3555 | 3 | True | 0.0 |
| 16 | 0.189558 | True | 1928 | 3611 | 3 | True | 0.0 |
| 17 | 0.198686 | True | 1755 | 3582 | 3 | True | 0.0 |
| 18 | 0.195761 | True | 1808 | 3595 | 3 | True | 0.0 |
| 19 | 0.191966 | True | 1880 | 3776 | 3 | True | 0.0 |
