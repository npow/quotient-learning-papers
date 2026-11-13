# Analysed vs reported configuration

delta=0.05, tau=1, stride=window_length*tau; reported = stride 1 + per-pair alpha.

| env | reported (mean ± std) | analysed (mean ± std) | seeds |
|---|---|---|---|
| tiger | 2.0 ± 0.00 | 2.0 ± 0.00 | 10 |
| kcycle_8 | 8.0 ± 0.00 | 8.0 ± 0.00 | 5 |
| hallway | 19.3 ± 0.49 | 14.6 ± 1.13 | 7 |
| rocksample_7_8 | 5.2 ± 1.10 | 6.2 ± 1.30 | 5 |
| tmaze | 4.0 ± 0.00 | 4.0 ± 0.00 | 5 |
| sepsis | 9.8 ± 1.79 | 1.6 ± 0.55 | 5 |
| robot_localization | 26.8 ± 1.30 | 26.4 ± 0.89 | 5 |
| synthetic_200 | 10.4 ± 1.34 | 1.0 ± 0.00 | 5 |
