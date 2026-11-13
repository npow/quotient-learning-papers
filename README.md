# Quotient learning: code and artifacts for two papers

| Directory | Paper |
|---|---|
| [`oql/`](oql) | Finite-Sample Recovery of Predictive History Quotients from Independent POMDP Trajectories |
| [`cql/`](cql) | Myhill–Nerode Quotient Learning for Continuous Observation–Action Spaces via Kernel Maximum Mean Discrepancy |

Each directory is self-contained (`pip install -e .` / `uv sync`, then `pytest`). LaTeX sources are in `*/paper/`.
Result artifacts and SHA-256 manifests: `oql/experiments/ARTIFACTS_MANIFEST.json`, `cql/experiments/*/results/`.
