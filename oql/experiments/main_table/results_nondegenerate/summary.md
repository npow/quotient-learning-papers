# Certified run under a nondegenerate stochastic policy

Tiger (accuracy 0.85), memoryless stochastic policy listen/open-left/open-right = (0.8, 0.1, 0.1); estimand r=1, k=1, L=1, q=10, N=8000, 50/50 split, delta=0.05, beta=0.05, eps=0.35 (brute-force closed-loop gap 0.3920; ground_truth.json).

- seeds: 20
- recovery rate (partition == brute-force truth): 20/20
- n_classes: mean 2.0 (expected 2)
- abstention rate: mean 0.0000, max 0.0000
- m* = 468, min inference count over seeds/contexts: 1548

| seed | n_classes | recovered | und. | min inf. count |
|---|---|---|---|---|
| 0 | 2 | True | 0.0 | 1572 |
| 1 | 2 | True | 0.0 | 1586 |
| 2 | 2 | True | 0.0 | 1593 |
| 3 | 2 | True | 0.0 | 1619 |
| 4 | 2 | True | 0.0 | 1564 |
| 5 | 2 | True | 0.0 | 1588 |
| 6 | 2 | True | 0.0 | 1556 |
| 7 | 2 | True | 0.0 | 1609 |
| 8 | 2 | True | 0.0 | 1587 |
| 9 | 2 | True | 0.0 | 1585 |
| 10 | 2 | True | 0.0 | 1557 |
| 11 | 2 | True | 0.0 | 1588 |
| 12 | 2 | True | 0.0 | 1590 |
| 13 | 2 | True | 0.0 | 1581 |
| 14 | 2 | True | 0.0 | 1601 |
| 15 | 2 | True | 0.0 | 1583 |
| 16 | 2 | True | 0.0 | 1584 |
| 17 | 2 | True | 0.0 | 1555 |
| 18 | 2 | True | 0.0 | 1571 |
| 19 | 2 | True | 0.0 | 1548 |
