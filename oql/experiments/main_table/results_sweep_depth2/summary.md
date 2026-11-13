# Merge-bearing abstention sweep (noisy depth-2 chain, S10)

r=0, k=L=1, 3 contexts, analytic quotient 2 classes ((1,) ~ (2,) is a genuine SAME pair); eps_min=0.5 <= analytic gap 0.7523; delta=0.05, beta=0.05, alpha_0=delta/(2K^2)=2.778e-03, d=3, m*=278; q=30; 10 seeds per budget; context masses (0,)~0.4565, (1,)=(2,)~0.2717 -> gate transition expected near N~2,050.

Degradation signature: the SAME pair abstaining OVER-SPLITS the report to 3 classes; an incorrect merge would join (0,) with (1,) or (2,).

| N | class counts {n: seeds} | mean und. | min-max und. | incorrect merges |
|---|---|---|---|---|
| 1000 | 3:10 | 1.000 | 1.000-1.000 | 0 |
| 1400 | 3:10 | 1.000 | 1.000-1.000 | 0 |
| 1800 | 3:10 | 1.000 | 1.000-1.000 | 0 |
| 2000 | 2:1, 3:9 | 0.800 | 0.000-1.000 | 0 |
| 2100 | 2:2, 3:8 | 0.533 | 0.000-0.667 | 0 |
| 2200 | 2:9, 3:1 | 0.067 | 0.000-0.667 | 0 |
| 2400 | 2:10 | 0.000 | 0.000-0.000 | 0 |
| 2800 | 2:10 | 0.000 | 0.000-0.000 | 0 |
