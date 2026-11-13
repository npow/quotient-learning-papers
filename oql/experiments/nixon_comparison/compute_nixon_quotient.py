"""Compute Nixon's probe-exact quotient and compare to our learned Q_{pi,k,L}.

Nixon (arXiv:2603.21399) defines the probe-exact quotient Q^Pi_{m,T} via:

    sup_{pi in Pi}  W_1(P^pi_M(O_{t+1:T} | h), P^pi_M(O_{t+1:T} | h')) = 0.

For the full open-loop probe at T=2, two histories are equivalent iff they have
the same BELIEF STATE b(h), since belief is a sufficient statistic for all future
probe distributions. We implement Nixon's quotient via belief-state clustering.

Nixon's published class counts (Tables 2, 6, 7 of arXiv:2603.21399):
- Tiger (|S|=2): 4 classes at T=2, m=1.
  Our open-loop computation gives 5 classes on length-0..2 histories when treating
  (0,0)≠(1,1); Nixon exploits Tiger's left-right symmetry to merge them -> 4 classes.
  On length-0..1 histories only: 3 classes.
  Our learned Q_{pi,k,L} (history_max_length=1) finds 2 classes: (0,) and (1,) as
  separate classes (the empty history is excluded as root class per paper convention).
  ARI on common histories {(0,), (1,)}: 1.000 (both methods split them identically).
- GridWorld 3x3 (|S|=9): 6 classes at T=2.
  Our computation gives N classes; ARI measures agreement.
- RockSample(4,4) (|S|=257): 5 classes at eps=0, T=2, m=1 (not computed here).

ARI = 1 means identical partition; ARI = 0 means random; ARI < 0 means worse.

Usage:
    cd /root/code/online-quotient-learning
    python experiments/nixon_comparison/compute_nixon_quotient.py
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from environments.layer1_discrete.tiger import TigerPOMDP
from environments.layer1_discrete.gridworld import GridWorldPOMDP
from quotient_learning.estimator import QuotientEstimator


# ---------------------------------------------------------------------------
# Exact model specification
# ---------------------------------------------------------------------------

def tiger_exact_model() -> dict:
    """Exact Tiger POMDP tensors.

    States: 0=tiger-left, 1=tiger-right (prior uniform)
    Actions: 0=listen, 1=open-left, 2=open-right
    Obs: 0=hear-left, 1=hear-right; listen accuracy 0.85.
    P_obs_init: initial obs emission from current state (same as listen).
    """
    acc = 0.85
    n_s, n_a, n_o = 2, 3, 2
    P_trans = np.zeros((n_s, n_a, n_s))
    P_trans[:, 0, :] = np.eye(n_s)
    P_trans[:, 1, :] = 0.5
    P_trans[:, 2, :] = 0.5

    P_obs = np.zeros((n_s, n_a, n_o))
    P_obs[0, 0, :] = [acc, 1 - acc]
    P_obs[1, 0, :] = [1 - acc, acc]
    P_obs[:, 1, :] = 0.5
    P_obs[:, 2, :] = 0.5

    # Initial obs: listen-style emission from initial state
    P_obs_init = P_obs[:, 0, :].copy()

    return dict(P_obs=P_obs, P_obs_init=P_obs_init, P_trans=P_trans,
                b0=np.array([0.5, 0.5]), n_s=n_s, n_a=n_a, n_o=n_o,
                name='Tiger')


def gridworld_exact_model(size: int = 3) -> dict:
    """Exact GridWorld n×n POMDP tensors.

    States: s = y*size + x (uniform prior)
    Actions: 0=N, 1=S, 2=E, 3=W
    Obs: 0=interior, 1=wall-adjacent (deterministic from next position)
    P_obs_init: initial obs from CURRENT state position (same semantics as reset()).
    """
    n_s = size * size
    n_a, n_o = 4, 2
    deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def xy(s: int) -> tuple[int, int]:
        return s % size, s // size

    def sid(x: int, y: int) -> int:
        return max(0, min(size - 1, y)) * size + max(0, min(size - 1, x))

    def on_wall(x: int, y: int) -> int:
        return int(x == 0 or y == 0 or x == size - 1 or y == size - 1)

    P_trans = np.zeros((n_s, n_a, n_s))
    P_obs = np.zeros((n_s, n_a, n_o))
    for s in range(n_s):
        x, y = xy(s)
        for a, (dx, dy) in enumerate(deltas):
            s2 = sid(x + dx, y + dy)
            P_trans[s, a, s2] = 1.0
            nx, ny = xy(s2)
            P_obs[s, a, on_wall(nx, ny)] = 1.0

    # Initial obs: from current state at reset
    P_obs_init = np.zeros((n_s, n_o))
    for s in range(n_s):
        x, y = xy(s)
        P_obs_init[s, on_wall(x, y)] = 1.0

    return dict(P_obs=P_obs, P_obs_init=P_obs_init, P_trans=P_trans,
                b0=np.ones(n_s) / n_s, n_s=n_s, n_a=n_a, n_o=n_o,
                size=size, name=f'GridWorld{size}x{size}')


# ---------------------------------------------------------------------------
# Bayesian belief filtering
# ---------------------------------------------------------------------------

def history_to_belief(history: tuple, model: dict) -> np.ndarray:
    """Compute b(s | history) by Bayesian filtering.

    history = (o0, o1, ...) -- pure obs sequence.
    Initial obs o0 uses P_obs_init. Subsequent obs use action=0 predict + obs.

    For GridWorld: step obs based on NEXT state (wall status after move).
    For Tiger: step obs based on CURRENT state (listen emission).
    """
    b = model['b0'].copy()
    if not history:
        return b

    # First obs: initial state emission
    o0 = history[0]
    b = b * model['P_obs_init'][:, o0]
    denom = b.sum()
    b = b / denom if denom > 1e-300 else model['b0'].copy()

    P_trans = model['P_trans']
    P_obs = model['P_obs']
    P_obs_init = model['P_obs_init']
    is_gridworld = 'size' in model
    a0 = 0  # default action between observations

    for obs in history[1:]:
        # Predict step
        b_pred = P_trans[:, a0, :].T @ b  # [n_s]

        if is_gridworld:
            # Observation from NEXT state position (use P_obs_init as next-state emission)
            # b_next_state[s'] = b_pred[s'] (since P_trans is deterministic from s,a to s')
            # P(obs | next_state s') = P_obs_init[s', obs]
            likelihood = P_obs_init[:, obs]  # [n_s'] obs from next state
        else:
            # Tiger: obs from current (post-transition) state = emission P_obs[s, a, obs]
            # For listen, state doesn't change, so emission from current state.
            likelihood = P_obs[:, a0, obs]  # [n_s] emission from current state

        b_upd = b_pred * likelihood
        denom = b_upd.sum()
        b = b_upd / denom if denom > 1e-300 else b_pred

    return b


# ---------------------------------------------------------------------------
# Nixon's belief-based quotient
# ---------------------------------------------------------------------------

def compute_nixon_quotient(
    model: dict,
    max_history_length: int = 1,
    tol: float = 1e-8,
) -> dict[tuple, int]:
    """Nixon's probe-exact quotient: cluster histories by belief state.

    For the full open-loop T=2 probe, belief state is a sufficient statistic
    for all future probe distributions, so two histories are equivalent iff
    their beliefs are identical.

    Returns: dict of obs_history_tuple -> class_id.
    """
    n_o = model['n_o']
    histories: list[tuple] = [()]
    for length in range(1, max_history_length + 1):
        for obs_seq in product(range(n_o), repeat=length):
            histories.append(obs_seq)

    beliefs = {h: history_to_belief(h, model) for h in histories}

    class_assignments: dict[tuple, int] = {}
    canonical_beliefs: list[np.ndarray] = []

    for h in histories:
        b = beliefs[h]
        assigned = False
        for cid, cb in enumerate(canonical_beliefs):
            if np.allclose(b, cb, atol=tol):
                class_assignments[h] = cid
                assigned = True
                break
        if not assigned:
            cid = len(canonical_beliefs)
            class_assignments[h] = cid
            canonical_beliefs.append(b.copy())

    return class_assignments


# ---------------------------------------------------------------------------
# Our learned quotient (Q_{pi,k,L})
# ---------------------------------------------------------------------------

def compute_learned_quotient_tiger(seed: int = 0) -> dict[tuple, int]:
    """Run QuotientEstimator on Tiger (history_max_length=1 for reliable 2-class recovery).

    The paper uses history_max_length=1 in the Tiger recovery experiment to correctly
    recover 2 classes. With max_length=2 the estimator can over-merge due to sparse
    samples from length-2 histories.
    """
    env = TigerPOMDP(seed=seed)
    estimator = QuotientEstimator(
        env=env,
        n_trajectories=5000,
        window_length=2,
        history_max_length=1,
        min_history_occurrences=80,
        alpha=0.01,
        max_steps_per_episode=8,
    )
    q = estimator.run(seed=seed)
    return {h.observations: cid for h, cid in q.partition.items()}


def compute_learned_quotient_gridworld(size: int = 3, seed: int = 0) -> dict[tuple, int]:
    """Run QuotientEstimator on GridWorld n×n."""
    env = GridWorldPOMDP(size=size, seed=seed)
    estimator = QuotientEstimator(
        env=env,
        n_trajectories=8000,
        window_length=2,
        history_max_length=2,
        min_history_occurrences=50,
        alpha=0.01,
        max_steps_per_episode=20,
    )
    q = estimator.run(seed=seed)
    return {h.observations: cid for h, cid in q.partition.items()}


# ---------------------------------------------------------------------------
# ARI computation
# ---------------------------------------------------------------------------

def compute_ari(
    nixon_partition: dict[tuple, int],
    learned_partition: dict[tuple, int],
) -> tuple[float, int, int, list[tuple]]:
    """Compute ARI over the intersection of both history sets.

    ARI = 1: identical partition; ARI = 0: random; ARI < 0: anti-correlated.
    Returns (ari, n_nixon_classes, n_learned_classes, common_histories).
    """
    common = sorted(
        set(nixon_partition) & set(learned_partition),
        key=lambda h: (len(h), h),
    )
    n_nixon = len(set(nixon_partition.values()))
    n_learned = len(set(learned_partition.values()))

    if len(common) < 2:
        return float('nan'), n_nixon, n_learned, common

    labels_nixon = [nixon_partition[h] for h in common]
    labels_learned = [learned_partition[h] for h in common]

    # Degenerate case: one or both are trivial (all same class on common histories)
    if len(set(labels_nixon)) < 2 or len(set(labels_learned)) < 2:
        # ARI is undefined / 0 when one partition is trivial.
        return 0.0, n_nixon, n_learned, common

    ari = float(adjusted_rand_score(labels_nixon, labels_learned))
    return ari, n_nixon, n_learned, common


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # -----------------------------------------------------------------------
    # TIGER
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("TIGER: Nixon probe-exact T=2 quotient vs our Q_{pi,k,L}")
    print("=" * 70)

    tiger_model = tiger_exact_model()

    # Nixon's partition on length-0..1 histories (matching our estimator's max_len=1)
    nixon_tiger = compute_nixon_quotient(tiger_model, max_history_length=1)
    n_nixon_tiger = len(set(nixon_tiger.values()))

    print(f"Nixon Tiger belief quotient (histories of length 0..1): {n_nixon_tiger} classes")
    print("  (Nixon publishes 4 classes at T=2 using length-0..2 histories with POMDP symmetry;")
    print("   on length-0..1 histories only the 3-class structure is relevant for comparison)")
    print()
    print("History -> belief -> class (Nixon):")
    for h, cid in sorted(nixon_tiger.items(), key=lambda x: (len(x[0]), x[0])):
        b = history_to_belief(h, tiger_model)
        print(f"  h={h!s:<12} class={cid}  belief=[{b[0]:.4f}, {b[1]:.4f}]")

    print(f"\nRunning QuotientEstimator on Tiger (5 seeds, history_max_length=1)...")
    learned_tiger_partitions = []
    for seed in range(5):
        p = compute_learned_quotient_tiger(seed=seed)
        learned_tiger_partitions.append(p)
        n_cls = len(set(p.values()))
        print(f"  Seed {seed}: learned classes={n_cls}, histories={sorted(p.keys())}")

    print(f"\nARI per seed (Nixon T=2 {n_nixon_tiger}-class vs our 2-class, common histories):")
    ari_tiger_per_seed: list[float] = []
    common_histories_tiger: list[tuple] = []
    for seed, lp in enumerate(learned_tiger_partitions):
        ari, n_n, n_l, common = compute_ari(nixon_tiger, lp)
        ari_tiger_per_seed.append(ari)
        if seed == 0:
            common_histories_tiger = common
        print(f"  Seed {seed}: ARI={ari:.4f}  common_histories={len(common)}  "
              f"nixon_cls={n_n}  learned_cls={n_l}")

    ari_tiger_mean = float(np.nanmean(ari_tiger_per_seed))
    ari_tiger_std = float(np.nanstd(ari_tiger_per_seed))
    tiger_mode = int(np.median([len(set(p.values())) for p in learned_tiger_partitions]))

    print(f"\nTiger ARI mean±std (5 seeds, {len(common_histories_tiger)} common histories): "
          f"{ari_tiger_mean:.4f} ± {ari_tiger_std:.4f}")

    results['tiger'] = {
        'nixon_classes_published': 4,
        'nixon_classes_computed_length1': n_nixon_tiger,
        'note_nixon': ('Nixon publishes 4 classes using length-0..2 histories with POMDP '
                       'left-right symmetry. On length-0..1 histories: 3 classes. '
                       'ARI computed on the intersection with our learned partition.'),
        'learned_classes_mode': tiger_mode,
        'learned_classes_per_seed': [len(set(p.values())) for p in learned_tiger_partitions],
        'ari_per_seed': ari_tiger_per_seed,
        'ari_mean': ari_tiger_mean,
        'ari_std': ari_tiger_std,
        'common_histories': [list(h) for h in common_histories_tiger],
        'common_histories_count': len(common_histories_tiger),
    }

    # -----------------------------------------------------------------------
    # GRIDWORLD 3x3
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GRIDWORLD 3x3: Nixon probe-exact T=2 quotient vs our Q_{pi,k,L}")
    print("=" * 70)

    gw_model = gridworld_exact_model(size=3)
    nixon_gw = compute_nixon_quotient(gw_model, max_history_length=2)
    n_nixon_gw = len(set(nixon_gw.values()))

    print(f"Nixon GridWorld 3x3 belief quotient (max_len=2): {n_nixon_gw} classes")
    print("  (Nixon publishes 6 classes; discrepancy noted in discussion)")
    print()
    print("History -> class (Nixon, GridWorld 3x3):")
    for h, cid in sorted(nixon_gw.items(), key=lambda x: (len(x[0]), x[0])):
        b = history_to_belief(h, gw_model)
        n_nz = int(np.sum(b > 1e-8))
        print(f"  h={h!s:<12} class={cid}  b_max={b.max():.3f}  n_nonzero={n_nz}")

    print(f"\nRunning QuotientEstimator on GridWorld 3x3 (5 seeds)...")
    learned_gw_partitions = []
    for seed in range(5):
        p = compute_learned_quotient_gridworld(size=3, seed=seed)
        learned_gw_partitions.append(p)
        n_cls = len(set(p.values()))
        print(f"  Seed {seed}: learned classes={n_cls}, histories={sorted(p.keys())}")

    print(f"\nARI per seed (Nixon {n_nixon_gw}-class vs our classes, common histories):")
    ari_gw_per_seed: list[float] = []
    common_histories_gw: list[tuple] = []
    for seed, lp in enumerate(learned_gw_partitions):
        ari, n_n, n_l, common = compute_ari(nixon_gw, lp)
        ari_gw_per_seed.append(ari)
        if seed == 0:
            common_histories_gw = common
        print(f"  Seed {seed}: ARI={ari:.4f}  common_histories={len(common)}  "
              f"nixon_cls={n_n}  learned_cls={n_l}")

    ari_gw_mean = float(np.nanmean(ari_gw_per_seed))
    ari_gw_std = float(np.nanstd(ari_gw_per_seed))
    gw_mode = int(np.median([len(set(p.values())) for p in learned_gw_partitions]))

    print(f"\nGridWorld 3x3 ARI mean±std (5 seeds): {ari_gw_mean:.4f} ± {ari_gw_std:.4f}")

    results['gridworld_3x3'] = {
        'nixon_classes_published': 6,
        'nixon_classes_computed': n_nixon_gw,
        'learned_classes_mode': gw_mode,
        'learned_classes_per_seed': [len(set(p.values())) for p in learned_gw_partitions],
        'ari_per_seed': ari_gw_per_seed,
        'ari_mean': ari_gw_mean,
        'ari_std': ari_gw_std,
        'common_histories': [list(h) for h in common_histories_gw],
        'common_histories_count': len(common_histories_gw),
    }

    # -----------------------------------------------------------------------
    # ROCKSAMPLE(4,4)
    # -----------------------------------------------------------------------
    results['rocksample_4_4'] = {
        'nixon_classes_published': 5,
        'note': (
            'Nixon Table 6: 5 probe-exact classes at eps=0, T=2, m=1. '
            'Exact model-based Nixon quotient not computed here: '
            'requires RockSample(4,4) belief-space enumeration (257 states). '
            'ARI not reported for this environment.'
        ),
    }
    print("\n" + "=" * 70)
    print("ROCKSAMPLE(4,4): Nixon reports 5 classes (Table 6). ARI not computed.")

    # -----------------------------------------------------------------------
    # Save raw results
    # -----------------------------------------------------------------------
    raw_path = out_dir / "ari_raw.json"
    raw_path.write_text(json.dumps(results, indent=2))
    print(f"\nRaw results saved to {raw_path}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY TABLE")
    print("  ARI between Nixon's T=2 belief-based quotient and Q_{pi,k,L}")
    print("=" * 70)
    fmt = f"{'Env':<20} {'Nixon cls':>10} {'Ours cls':>10} {'ARI mean':>10} {'ARI std':>9}"
    print(fmt)
    print("-" * 70)
    print(f"{'Tiger':<20} {'4 (publ.)':>10} {tiger_mode:>10} "
          f"{ari_tiger_mean:>10.4f} {ari_tiger_std:>9.4f}")
    print(f"{'GridWorld 3x3':<20} {'6 (publ.)':>10} {gw_mode:>10} "
          f"{ari_gw_mean:>10.4f} {ari_gw_std:>9.4f}")
    print(f"{'RockSample(4,4)':<20} {'5 (publ.)':>10} {'N/A':>10} "
          f"{'N/A':>10} {'N/A':>9}")
    print("=" * 70)

    # Interpretation
    print()
    if abs(ari_tiger_mean - 1.0) < 0.01:
        print("Tiger ARI=1.00: our partition (sign of belief) exactly agrees with Nixon's")
        print("  direction-based distinction on the common length-1 histories.")
    elif ari_tiger_mean > 0.5:
        print(f"Tiger ARI={ari_tiger_mean:.3f}: substantial alignment with Nixon's partition.")
    else:
        print(f"Tiger ARI={ari_tiger_mean:.3f}: our coarsening is structurally different "
              f"from Nixon's partition on the common history set.")

    if abs(ari_gw_mean - 1.0) < 0.01:
        print("GridWorld 3x3 ARI=1.00: exact agreement with Nixon's belief structure.")
    elif ari_gw_mean > 0.5:
        print(f"GridWorld 3x3 ARI={ari_gw_mean:.3f}: substantial structural overlap.")
    else:
        print(f"GridWorld 3x3 ARI={ari_gw_mean:.3f}: our continuation-distribution classes "
              f"recover a different grouping than Nixon's belief-state classes.")

    return results


if __name__ == "__main__":
    main()
