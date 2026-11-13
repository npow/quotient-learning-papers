# Data Model: Continuous Quotient Learning

**Feature**: 001-continuous-quotient-learning

---

## Core Types

### ContinuousTrajectory
Extends `Trajectory` from `online-quotient-learning` to carry raw float observations and actions.

```
ContinuousTrajectory:
  observations:  List[np.ndarray[float32, shape=(d_O,)]]   # per-step obs vectors
  actions:       List[np.ndarray[float32, shape=(d_A,)]]   # per-step act vectors (or int for discrete)
  rewards:       List[float]
  length:        int
```

**Relationships**: Used by `ContinuousTrajectoryDataset`; source of `ContinuationWindow`.

---

### ContinuationWindow
A single k-step continuation from a fixed history prefix — the primitive sample for equivalence testing.

```
ContinuationWindow:
  history_key:   tuple[int|float, ...]    # (o_1, a_1, ..., o_L) as hashable key
  window:        np.ndarray[float32,
                   shape=(k, d_O + d_A)]  # flattened (obs, act) per step; last step obs only
  k:             int                      # continuation horizon
```

**Constraints**:
- `window` shape is `(k, d_O + d_A)` for steps 1..k-1; last entry is `(d_O,)` padded to full width.
- For discrete action spaces (int), action is stored as a one-hot float vector.

---

### ContinuousEquivalenceResult
Output of a single MMD permutation test between two history prefixes.

```
ContinuousEquivalenceResult:
  h1_key:        tuple
  h2_key:        tuple
  mmd2:          float          # biased MMD² statistic
  p_value:       float          # permutation p-value (or asymptotic)
  reject:        bool           # True → different classes
  n_permutations: int
  bandwidth:     float          # kernel bandwidth used (median heuristic or CV)
```

**State transitions**:
- `reject=False` (same class) → h1 and h2 are merged in union-find
- `reject=True` (different class) → h1 and h2 remain in separate classes

---

### QuotientClass
One equivalence class in the learned partition.

```
QuotientClass:
  class_id:      int
  member_keys:   List[tuple]     # history keys assigned to this class
  centroid:      np.ndarray | None   # mean embedding vector (neural encoder only)
  n_members:     int
```

---

### CanonicalContinuousQuotient
The full output of `ContinuousQuotientEstimator.fit()`.

```
CanonicalContinuousQuotient:
  classes:          Dict[int, QuotientClass]
  history_to_class: Dict[tuple, int]       # lookup map
  n_classes:        int
  estimator_config: ContinuousEstimatorConfig
  env_id:           str
  n_trajectories:   int
```

**Relationships**:
- Consumed by `rcrit.py` (plug-in entropy estimation over class labels) — unchanged from finite case.
- Compared against ground-truth quotient via `partition_accuracy()` (Rand index or class-count match).

---

### NeuralQuotientEncoder (stateful model)

```
NeuralQuotientEncoder:
  encoder:       GRU(input_dim=(d_O+d_A), hidden=128, layers=2)
  projection:    Linear(128 → 64) + L2-normalize
  embedding_dim: 64
  cluster_eps:   float           # DBSCAN threshold (10th pctile of pairwise dist by default)
  training_meta: TrainingMeta
```

```
TrainingMeta:
  n_epochs:      int
  loss_history:  List[float]
  val_rand_index: float          # Rand index vs. MMD-labeled partition on held-out histories
  pretrain_env:  str | None      # e.g. "kcycle_K8" for curriculum init
```

**Persistence**: model checkpoint at `experiments/<run>/encoder_<epoch>.pt`; config at `experiments/<run>/config.json`.

---

### ContinuousEstimatorConfig

```
ContinuousEstimatorConfig:
  n_trajectories:         int   = 5_000
  max_steps_per_episode:  int   = 50
  window_length:          int   = 2        # k
  history_max_length:     int   = 1        # L
  min_history_occurrences: int  = 20
  alpha:                  float = 0.01
  kernel:                 str   = "rbf"    # "rbf" | "laplace" | "gak"
  bandwidth:              float | None = None  # None → median heuristic
  n_permutations:         int   = 199
```

---

## State Transitions

### QuotientEstimation Lifecycle

```
trajectories collected
        │
        ▼
histories enumerated (min_occurrences filter)
        │
        ▼
continuation windows extracted per history
        │
        ▼
pairwise MMD tests (Bonferroni-corrected α)
        │
        ├─ reject=False ─→ union-find merge
        └─ reject=True  ─→ classes remain separate
        │
        ▼
CanonicalContinuousQuotient (stable IDs)
        │
        └─→ rcrit.py (if multi-agent)
```

### NeuralEncoder Training Lifecycle

```
MMD-labeled pairs generated (from above pipeline)
        │
        ▼
Stage 1: pretrain on finite benchmark (K-cycle, Tiger)
        │
        ▼
Stage 2: fine-tune on target continuous environment
        │
        ▼
DBSCAN clustering over embeddings
        │
        ▼
CanonicalContinuousQuotient (from embedding clusters)
```

---

## Validation Rules

| Entity | Field | Rule |
|--------|-------|------|
| ContinuationWindow | window | shape must be (k, d_O + d_A); no NaN |
| ContinuousEquivalenceResult | p_value | ∈ [0, 1] |
| CanonicalContinuousQuotient | n_classes | ≥ 1; ≤ n_trajectories |
| NeuralQuotientEncoder | cluster_eps | > 0; validated against median pairwise distance |
| ContinuousEstimatorConfig | alpha | ∈ (0, 0.5); Bonferroni correction applied automatically |
