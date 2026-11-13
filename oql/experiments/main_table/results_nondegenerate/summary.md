# Certified run under a nondegenerate stochastic policy

Tiger (accuracy 0.85), memoryless stochastic policy listen/open-left/open-right = (0.8, 0.1, 0.1); estimand r=1, k=1, L=1, q=10, N=8000, 50/50 split, delta=0.05, beta=0.05, eps=0.35 (brute-force closed-loop gap 0.3920; ground_truth.json).

- seeds: 20
- recovery rate (partition == brute-force truth): 20/20
- n_classes: mean 2.0 (expected 2)
- abstention rate: mean 0.0000, max 0.0000
- m* = 468, min inference count over seeds/contexts: 1556

| seed | n_classes | recovered | und. | min inf. count |
|---|---|---|---|---|
| 0 | 2 | True | 0.0 | 1588 |
| 1 | 2 | True | 0.0 | 1587 |
| 2 | 2 | True | 0.0 | 1591 |
| 3 | 2 | True | 0.0 | 1590 |
| 4 | 2 | True | 0.0 | 1589 |
| 5 | 2 | True | 0.0 | 1569 |
| 6 | 2 | True | 0.0 | 1556 |
| 7 | 2 | True | 0.0 | 1623 |
| 8 | 2 | True | 0.0 | 1590 |
| 9 | 2 | True | 0.0 | 1575 |
| 10 | 2 | True | 0.0 | 1586 |
| 11 | 2 | True | 0.0 | 1568 |
| 12 | 2 | True | 0.0 | 1577 |
| 13 | 2 | True | 0.0 | 1585 |
| 14 | 2 | True | 0.0 | 1582 |
| 15 | 2 | True | 0.0 | 1567 |
| 16 | 2 | True | 0.0 | 1573 |
| 17 | 2 | True | 0.0 | 1558 |
| 18 | 2 | True | 0.0 | 1575 |
| 19 | 2 | True | 0.0 | 1605 |
