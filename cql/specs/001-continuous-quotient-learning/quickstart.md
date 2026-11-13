# Quickstart: continuous_quotient

## Setup

```bash
# From repo root
pip install -e /root/code/online-quotient-learning   # upstream dependency (editable)
pip install -e .                                      # this package

# MuJoCo environments
pip install "gymnasium[mujoco]"
```

## Run the canonical correctness check

```bash
# Continuous-observation Tiger: should recover n=2 on all seeds
python experiments/layer1_ground_truth/run_tiger_continuous.py --seeds 10

# K-cycle (continuous angle observations): recover K classes
python experiments/layer1_ground_truth/run_kcycle_continuous.py --K 8 --seeds 5
```

## Run PO-MuJoCo without discretization

```bash
# PO-HalfCheetah: compare continuous vs. discretized baseline
python experiments/layer2_mujoco/run_halfcheetah.py --mode continuous
python experiments/layer2_mujoco/run_halfcheetah.py --mode discretized  # baseline

# PO-Ant: confirm saturation artifact removed
python experiments/layer2_mujoco/run_ant.py --mode continuous
```

## Estimate R_crit on continuous Dec-Tiger

```bash
python experiments/layer3_rcrit/run_dec_tiger_continuous.py --seeds 10
```

## Run tests

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v --slow       # requires MuJoCo
```

## Key config knobs

| Flag | Default | Effect |
|------|---------|--------|
| `--kernel rbf` | rbf | Kernel for MMD test |
| `--bandwidth auto` | auto (median) | Kernel bandwidth |
| `--n-permutations 199` | 199 | Permutation calibration count |
| `--window-length 2` | 2 | k-step continuation horizon |
| `--history-max-length 1` | 1 | Max history prefix length |
| `--alpha 0.01` | 0.01 | Per-test significance level |
