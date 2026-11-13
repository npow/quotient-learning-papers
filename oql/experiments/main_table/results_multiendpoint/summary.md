# Multi-endpoint certified recovery (position-flip chain, REV-2)

Family delta=0.05 split across T=2 endpoints r in [0, 1] (delta/T=0.025 each); eps_min=0.75 <= analytic TV gap 0.8 at both endpoints; beta=0.05; K=2 contexts per endpoint, d=2, m*=112; N=4000 episodes per seed, one shared trajectory pool per seed across endpoints; 50/50 split; q=10.

Analytic laws P(next=1 | context): r=0: (0,)->0.9, (1,)->0.1; r=1: (0,)->0.1, (1,)->0.9 (parity-flipped kernel). Two singleton classes at each endpoint.

| endpoint r | recovered 2 classes | zero abstention | seeds |
|---|---|---|---|
| 0 | 10/10 | 10/10 | 10 |
| 1 | 10/10 | 10/10 | 10 |
