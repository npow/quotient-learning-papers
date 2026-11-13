"""R_crit computation and phase-transition classification (T069, T070).

Implements the critical communication rate from Paper 2 (Nixon, 2604.09521):
given two bounded agents A and B with learned canonical quotients Q_A and
Q_B, R_crit is the minimum rate (bits per timestep) at which their internal
state representations can be communicated faithfully across a noiseless
channel.

Three estimation modes are provided:

**Surrogate** (``compute_rcrit_surrogate``):
    R_crit ≈ log2(max(|Q_A|, |Q_B|))
    Monotone in quotient size — coarse but always valid as an upper bound.

**Exact / rate-distortion** (``compute_rcrit_exact``):
    Given a shared ``TrajectoryDataset``, estimates R_crit = H(A | B), the
    conditional entropy of agent A's quotient class given agent B's, computed
    from the empirical joint distribution over (class_A, class_B) pairs.
    This is the Slepian-Wolf / source-coding-with-side-information bound:
    the minimum bits per timestep at which A must transmit so that B can
    perfectly reconstruct A's class.

**Paired exact** (``compute_rcrit_exact_v2``):
    Given *two separate* ``TrajectoryDataset`` objects (one per agent),
    estimates R_crit = H(A | B) from the empirical joint distribution of
    (Q_A.class_of(h_a), Q_B.class_of(h_b)) where h_a and h_b are
    observation histories from *concurrent* (same episode index) trajectories
    of agent A and agent B respectively.  This is the physically correct
    formulation when the two agents observe the same underlying environment
    through independent, noisy channels and are not co-located.  Unlike the
    single-dataset variant, it avoids trivially inflating MI when both
    quotients are trained on the same observation stream.

The public entry point ``compute_rcrit`` dispatches to the exact version
when a ``dataset`` argument is supplied, and to the surrogate otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quotient_learning.types import CanonicalQuotient, RCritEstimate


@dataclass
class PhaseTransitionResult:
    """A classified communication-feasibility outcome."""
    r_crit: float
    channel_capacity: float
    feasible: bool
    margin: float


# ---------------------------------------------------------------------------
# Surrogate estimator (original implementation, now renamed)
# ---------------------------------------------------------------------------

def compute_rcrit_surrogate(
    quotient_a: CanonicalQuotient,
    quotient_b: CanonicalQuotient,
    true_r_crit: float | None = None,
) -> RCritEstimate:
    """Estimate R_crit using log2 of the larger quotient size as the surrogate.

    This is the original implementation. It is monotone in quotient size and
    serves as a valid upper bound on the exact rate.  If both quotients are
    empty (no classes), returns r_crit = 0.
    """
    n_max = max(quotient_a.n_classes, quotient_b.n_classes)
    if n_max <= 1:
        r_crit = 0.0
    else:
        r_crit = math.log2(n_max)
    return RCritEstimate(
        quotient_a=quotient_a,
        quotient_b=quotient_b,
        r_crit=float(r_crit),
        true_r_crit=true_r_crit,
        using_independent_fallback=False,
    )


# ---------------------------------------------------------------------------
# Exact estimator: H(A | B) from empirical joint distribution
# ---------------------------------------------------------------------------

def compute_rcrit_exact(
    quotient_a: CanonicalQuotient,
    quotient_b: CanonicalQuotient,
    dataset,  # TrajectoryDataset — avoid circular import; type-checked at runtime
    true_r_crit: float | None = None,
) -> RCritEstimate:
    """Estimate R_crit = H(A | B) via the empirical joint distribution.

    Algorithm
    ---------
    1. Map every observation history *h* in *dataset* to
       (Q_A.class_of(h), Q_B.class_of(h)).  Only histories that appear in
       *both* partitions contribute; the rest are skipped.
    2. Compute the joint distribution π_{A,B}(c_A, c_B) and marginals
       π_A(c_A), π_B(c_B).
    3. Compute mutual information I(A; B) =
       Σ π_{A,B} log2(π_{A,B} / (π_A · π_B)).
    4. Compute H(A) = -Σ π_A log2 π_A.
    5. R_crit = H(A | B) = H(A) - I(A; B).

    Fallback
    --------
    If no observation history appears in *both* quotient partitions (i.e. the
    two datasets are completely disjoint), the function falls back to the
    surrogate R_crit = log2(|Q_A|) and sets
    ``using_independent_fallback=True`` on the returned ``RCritEstimate``.
    """
    from quotient_learning.dataset import TrajectoryDataset  # local to avoid circularity
    from quotient_learning.types import ObservationHistory  # local to avoid circularity

    if not isinstance(dataset, TrajectoryDataset):
        raise TypeError(f"dataset must be TrajectoryDataset, got {type(dataset)}")

    # Walk every sub-history up to the maximum partition depth of *either*
    # quotient.  For each history, we ask both quotients for their class
    # assignment.  If Q_A doesn't know it, we skip.  If Q_B doesn't know it,
    # we treat B as having a unique "unknown" sentinel class for that history
    # (B provides zero side information for those A classes).  Using a sentinel
    # rather than skipping preserves H(A)'s full entropy when Q_B is coarser
    # than Q_A.
    _SENTINEL_B = -1  # a class id Q_B will never produce (Q_B ids are ≥ 0)

    joint_counts: dict[tuple[int, int], int] = {}

    max_len_a = max((h.length for h in quotient_a.partition), default=0)
    max_len_b = max((h.length for h in quotient_b.partition), default=0)
    max_len = max(max_len_a, max_len_b, 0)

    for traj in dataset.trajectories:
        obs = traj.observations
        n = len(obs)
        for start in range(n):
            for length in range(0, min(max_len, n - start) + 1):
                h = ObservationHistory(obs[start: start + length])
                ca = quotient_a.class_of(h)
                if ca is None:
                    continue  # Q_A doesn't know this history; skip entirely.
                cb = quotient_b.class_of(h)
                if cb is None:
                    cb = _SENTINEL_B  # B has no info; treat as a unique class.
                key = (ca, cb)
                joint_counts[key] = joint_counts.get(key, 0) + 1

    total = sum(joint_counts.values())
    all_sentinel = total > 0 and all(cb == _SENTINEL_B for (_, cb) in joint_counts)
    if total == 0 or all_sentinel:
        # Independent-fallback: Q_A has no recognisable histories in dataset,
        # or Q_B does not recognise any of the histories Q_A encounters.
        # Use the product-distribution upper bound: R_crit = log2(|Q_A|).
        n_a = quotient_a.n_classes
        r_crit = math.log2(n_a) if n_a > 1 else 0.0
        return RCritEstimate(
            quotient_a=quotient_a,
            quotient_b=quotient_b,
            r_crit=float(r_crit),
            true_r_crit=true_r_crit,
            mutual_information=None,
            using_independent_fallback=True,
        )

    # Build joint probability table (unnormalised counts → probs).
    joint: dict[tuple[int, int], float] = {
        k: v / total for k, v in joint_counts.items()
    }

    # Full marginal of A (over all histories that Q_A recognises in dataset,
    # including those where Q_B uses the sentinel class).
    marginal_a: dict[int, float] = {}
    for (ca, _cb), p in joint.items():
        marginal_a[ca] = marginal_a.get(ca, 0.0) + p

    # H(A) = -Σ π_A log2 π_A
    h_a = -sum(p * math.log2(p) for p in marginal_a.values() if p > 0.0)

    # For the mutual-information calculation we restrict to histories where
    # Q_B actually provides side information (cb != _SENTINEL_B).  We
    # renormalise the non-sentinel sub-distribution to form a proper joint.
    non_sentinel_total = sum(
        v for (_, cb), v in joint_counts.items() if cb != _SENTINEL_B
    )

    if non_sentinel_total == 0:
        # Q_B provides zero side information for any history Q_A sees.
        mi = 0.0
    else:
        # Compute I(A; B) on the renormalised sub-joint.
        ns_weight = non_sentinel_total / total  # fraction of data with real B class
        sub_joint: dict[tuple[int, int], float] = {
            (ca, cb): v / non_sentinel_total
            for (ca, cb), v in joint_counts.items()
            if cb != _SENTINEL_B
        }
        sub_marginal_a: dict[int, float] = {}
        sub_marginal_b: dict[int, float] = {}
        for (ca, cb), p in sub_joint.items():
            sub_marginal_a[ca] = sub_marginal_a.get(ca, 0.0) + p
            sub_marginal_b[cb] = sub_marginal_b.get(cb, 0.0) + p

        sub_mi = 0.0
        for (ca, cb), p_ab in sub_joint.items():
            p_a = sub_marginal_a.get(ca, 0.0)
            p_b = sub_marginal_b.get(cb, 0.0)
            if p_ab > 0.0 and p_a > 0.0 and p_b > 0.0:
                sub_mi += p_ab * math.log2(p_ab / (p_a * p_b))

        # Weight MI by the fraction of data where B's info is available.
        mi = ns_weight * max(0.0, sub_mi)

    # Clamp to [0, H(A)] to guard against floating-point noise.
    mi = max(0.0, min(mi, h_a))
    r_crit = max(0.0, h_a - mi)

    return RCritEstimate(
        quotient_a=quotient_a,
        quotient_b=quotient_b,
        r_crit=float(r_crit),
        true_r_crit=true_r_crit,
        mutual_information=float(mi),
        using_independent_fallback=False,
    )


# ---------------------------------------------------------------------------
# Paired exact estimator: H(A | B) from two independent datasets
# ---------------------------------------------------------------------------

def compute_rcrit_exact_v2(
    quotient_a: CanonicalQuotient,
    quotient_b: CanonicalQuotient,
    dataset_a,  # TrajectoryDataset for agent A
    dataset_b,  # TrajectoryDataset for agent B
    true_r_crit: float | None = None,
) -> RCritEstimate:
    """Estimate R_crit = H(A | B) from two independent agent datasets.

    This is the physically correct estimator for the Dec-POMDP setting where
    agent A and agent B observe the same underlying world state through
    independent noisy channels.  Unlike ``compute_rcrit_exact`` (which applies
    both quotient maps to a single dataset and therefore obtains H(A|B) = 0
    whenever both quotients are trained on the same observation alphabet), this
    function pairs trajectories by episode index across the two datasets,
    giving a joint distribution that captures the statistical dependence
    induced by the shared hidden state.

    Algorithm
    ---------
    1. Pair trajectory i from dataset_a with trajectory i from dataset_b.
    2. At each timestep t, form h_a = observation t from traj_a (as a
       length-1 history) and h_b = observation t from traj_b.
    3. Look up Q_A.class_of(h_a) and Q_B.class_of(h_b).  Skip if either is
       unknown.
    4. Accumulate the empirical joint count over (class_A, class_B) pairs.
    5. Compute H(A | B) = H(A, B) - H(B) from the empirical joint.

    If no paired observations are found (e.g. both partitions are empty),
    falls back to the surrogate R_crit = log2(|Q_A|) as in
    ``compute_rcrit_exact``.

    Honesty disclaimer on ``true_r_crit``
    --------------------------------------
    When ``true_r_crit`` is ``None``, the caller should supply the
    *analytically computed* H(Q_A | Q_B) under the environment's true joint
    observation distribution — **not** simply log2(agent_capacity).  The
    quantity log2(agent_capacity) is the worst-case channel capacity needed
    when the two agents' quotients are maximally independent; it is a valid
    upper bound but may overestimate the typical-case R_crit by > 10% when
    both agents share correlated observations (as in Dec-Tiger, where
    analytical H(A|B) ≈ 0.819 bits with listen_accuracy=0.85, not 1.0 bit).
    """
    from quotient_learning.dataset import TrajectoryDataset  # local to avoid circularity
    from quotient_learning.types import ObservationHistory   # local to avoid circularity

    if not isinstance(dataset_a, TrajectoryDataset):
        raise TypeError(f"dataset_a must be TrajectoryDataset, got {type(dataset_a)}")
    if not isinstance(dataset_b, TrajectoryDataset):
        raise TypeError(f"dataset_b must be TrajectoryDataset, got {type(dataset_b)}")

    max_len_a = max((h.length for h in quotient_a.partition), default=0)
    max_len_b = max((h.length for h in quotient_b.partition), default=0)
    # Only use maximal-length sub-histories to avoid weighting H() by O(n).
    use_len_a = max_len_a
    use_len_b = max_len_b

    joint_counts: dict[tuple[int, int], int] = {}
    trajs_a = list(dataset_a.trajectories)
    trajs_b = list(dataset_b.trajectories)
    n_pairs = min(len(trajs_a), len(trajs_b))

    for i in range(n_pairs):
        obs_a = trajs_a[i].observations
        obs_b = trajs_b[i].observations
        n_a = len(obs_a)
        n_b = len(obs_b)
        # Walk length-use_len_a windows in traj_a and length-use_len_b in traj_b,
        # pairing them by position within the shorter trajectory.
        if use_len_a > 0 and use_len_b > 0:
            n_steps = min(n_a - use_len_a, n_b - use_len_b)
            for t in range(max(n_steps, 0) + 1):
                if t + use_len_a > n_a or t + use_len_b > n_b:
                    break
                h_a = ObservationHistory(obs_a[t: t + use_len_a])
                h_b = ObservationHistory(obs_b[t: t + use_len_b])
                ca = quotient_a.class_of(h_a)
                cb = quotient_b.class_of(h_b)
                if ca is None or cb is None:
                    continue
                key = (ca, cb)
                joint_counts[key] = joint_counts.get(key, 0) + 1
        else:
            # If either partition only has H() (length 0), fall through to fallback.
            pass

    total = sum(joint_counts.values())
    if total == 0:
        # No paired observations found; use surrogate fallback.
        n_a_classes = quotient_a.n_classes
        r_crit = math.log2(n_a_classes) if n_a_classes > 1 else 0.0
        return RCritEstimate(
            quotient_a=quotient_a,
            quotient_b=quotient_b,
            r_crit=float(r_crit),
            true_r_crit=true_r_crit,
            mutual_information=None,
            using_independent_fallback=True,
        )

    # Build joint probability table.
    joint: dict[tuple[int, int], float] = {k: v / total for k, v in joint_counts.items()}

    marginal_a: dict[int, float] = {}
    marginal_b: dict[int, float] = {}
    for (ca, cb), p in joint.items():
        marginal_a[ca] = marginal_a.get(ca, 0.0) + p
        marginal_b[cb] = marginal_b.get(cb, 0.0) + p

    h_a = -sum(p * math.log2(p) for p in marginal_a.values() if p > 0.0)
    h_b = -sum(p * math.log2(p) for p in marginal_b.values() if p > 0.0)
    h_joint = -sum(p * math.log2(p) for p in joint.values() if p > 0.0)

    h_a_given_b = max(0.0, h_joint - h_b)
    mi = max(0.0, min(h_a - h_a_given_b, h_a))
    r_crit = h_a_given_b

    return RCritEstimate(
        quotient_a=quotient_a,
        quotient_b=quotient_b,
        r_crit=float(r_crit),
        true_r_crit=true_r_crit,
        mutual_information=float(mi),
        using_independent_fallback=False,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_rcrit(
    quotient_a: CanonicalQuotient,
    quotient_b: CanonicalQuotient,
    dataset=None,
    true_r_crit: float | None = None,
) -> RCritEstimate:
    """Estimate R_crit from two learned quotients.

    Parameters
    ----------
    quotient_a, quotient_b:
        Learned canonical quotients for the two agents.
    dataset:
        Optional ``TrajectoryDataset``.  When provided, the exact
        rate-distortion estimator H(A|B) is used; otherwise the surrogate
        log2(max(|Q_A|, |Q_B|)) is returned.
    true_r_crit:
        Ground-truth value for relative-error tracking (optional).
    """
    if dataset is not None:
        return compute_rcrit_exact(quotient_a, quotient_b, dataset, true_r_crit)
    return compute_rcrit_surrogate(quotient_a, quotient_b, true_r_crit)


# ---------------------------------------------------------------------------
# Phase-transition classifier (unchanged)
# ---------------------------------------------------------------------------

def classify_phase_transition(
    estimate: RCritEstimate,
    channel_capacity: float,
) -> PhaseTransitionResult:
    """Predict whether communication is feasible across a channel of given capacity.

    Returns `feasible=True` iff r_crit ≤ channel_capacity. The `margin`
    captures how far we are from the phase-transition boundary.
    """
    margin = channel_capacity - estimate.r_crit
    return PhaseTransitionResult(
        r_crit=estimate.r_crit,
        channel_capacity=channel_capacity,
        feasible=margin >= 0.0,
        margin=margin,
    )
