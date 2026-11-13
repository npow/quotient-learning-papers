"""Extract continuation samples from trajectory datasets."""

from __future__ import annotations

from collections import defaultdict

from quotient_learning.dataset import TrajectoryDataset
from quotient_learning.types import ActionAwareHistory, ContinuationSample, ObservationHistory


def extract_continuation_samples(
    dataset: TrajectoryDataset,
    history: ObservationHistory,
    window_length: int,
    stride: int = 1,
) -> ContinuationSample:
    """Find every occurrence of `history` as a contiguous observation block in
    `dataset` and return the next `window_length` observations as continuations.

    With ``stride=1`` (default) a single
    trajectory of length T contributes up to T - len(history) - window_length + 1
    *overlapping* continuation windows, one per position where the history
    matches; consecutive windows are statistically dependent.  With
    ``stride=s > 1`` only positions ``0, s, 2s, ...`` of each trajectory are
    scanned.  A stride only controls overlap; it does not by itself certify
    independence.  If a history has length ``L``, the free gap between two
    sampled ``(history, continuation)`` blocks is ``stride - (L+k)``.
    """
    if window_length < 1:
        raise ValueError(f"window_length must be >= 1, got {window_length}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    hlen = history.length
    target = history.observations
    continuations: list[tuple[int, ...]] = []
    max_per_traj = 0
    for traj in dataset.trajectories:
        obs = traj.observations
        n = len(obs)
        max_start = n - hlen - window_length
        count_this = 0
        for start in range(0, max_start + 1, stride):
            if obs[start : start + hlen] == target:
                cont = obs[start + hlen : start + hlen + window_length]
                continuations.append(cont)
                count_this += 1
        max_per_traj = max(max_per_traj, count_this)
    return ContinuationSample(
        history=history,
        continuations=tuple(continuations),
        window_length=window_length,
        max_per_trajectory=max(1, max_per_traj),
    )


def enumerate_observed_histories(
    dataset: TrajectoryDataset,
    max_length: int,
    min_occurrences: int = 1,
) -> list[ObservationHistory]:
    """Return every distinct observation prefix (up to max_length) appearing in
    at least `min_occurrences` positions across the dataset.

    Length 0 (empty history) is always included if any data is present.
    """
    counts: dict[tuple[int, ...], int] = defaultdict(int)
    for traj in dataset.trajectories:
        obs = traj.observations
        n = len(obs)
        for start in range(n):
            for length in range(0, min(max_length, n - start) + 1):
                counts[obs[start : start + length]] += 1
    return [
        ObservationHistory(h)
        for h, c in counts.items()
        if c >= min_occurrences
    ]


def enumerate_independent_histories(
    dataset: TrajectoryDataset,
    max_length: int,
    context_end: int,
    min_occurrences: int = 1,
    include_root: bool = False,
) -> list[ObservationHistory]:
    """Enumerate contexts only at the prespecified endpoint of each trajectory."""
    if max_length < 0 or context_end < 0:
        raise ValueError("max_length and context_end must be non-negative")
    counts: dict[tuple[int, ...], int] = defaultdict(int)
    for traj in dataset.trajectories:
        if context_end >= len(traj.observations):
            continue
        if include_root:
            counts[()] += 1
        for length in range(1, min(max_length, context_end + 1) + 1):
            start = context_end - length + 1
            counts[tuple(traj.observations[start : context_end + 1])] += 1
    return [ObservationHistory(h) for h, count in counts.items()
            if count >= min_occurrences]


def enumerate_observed_action_aware_histories(
    dataset: TrajectoryDataset,
    max_length: int,
    min_occurrences: int = 1,
) -> list:
    """Return every distinct action-aware context (up to max_length steps)
    appearing in at least `min_occurrences` positions across the dataset.

    A context of length l is (observation, action) step pairs whose FINAL
    step has action None: the context ends at its last observation o_t and
    conditions on nothing after it (paper Definition 1, revised).  The
    immediate action a_t belongs to the continuation, not the context.
    Length 0 (empty context) is always included if any data is present.
    """
    counts: dict = defaultdict(int)
    for traj in dataset.trajectories:
        obs = traj.observations
        acts = traj.actions
        n_obs = len(obs)
        for end in range(n_obs):  # `end` = index of the context's final observation
            for length in range(0, min(max_length, end + 1) + 1):
                if length == 0:
                    counts[()] += 1
                    continue
                start = end - length + 1
                step_slice = tuple(
                    (int(obs[start + i]), int(acts[start + i]))
                    for i in range(length - 1)
                ) + ((int(obs[end]), None),)
                counts[step_slice] += 1
    return [
        ActionAwareHistory(h)
        for h, c in counts.items()
        if c >= min_occurrences
    ]


def enumerate_independent_action_aware_histories(
    dataset: TrajectoryDataset,
    max_length: int,
    context_end: int,
    min_occurrences: int = 1,
    include_root: bool = False,
) -> list[ActionAwareHistory]:
    """Enumerate action-aware contexts at one endpoint per trajectory."""
    if max_length < 0 or context_end < 0:
        raise ValueError("max_length and context_end must be non-negative")
    counts: dict[tuple, int] = defaultdict(int)
    for traj in dataset.trajectories:
        obs, acts = traj.observations, traj.actions
        if context_end >= len(obs):
            continue
        if include_root:
            counts[()] += 1
        for length in range(1, min(max_length, context_end + 1) + 1):
            start = context_end - length + 1
            steps = tuple(
                (int(obs[start + i]), int(acts[start + i]))
                for i in range(length - 1)
            ) + ((int(obs[context_end]), None),)
            counts[steps] += 1
    return [ActionAwareHistory(h) for h, count in counts.items()
            if count >= min_occurrences]


def extract_action_aware_continuations(
    dataset: TrajectoryDataset,
    history: ActionAwareHistory,
    window_length: int,
    stride: int = 1,
) -> ContinuationSample:
    """Find every occurrence of an action-aware context in `dataset` and
    return the next `window_length` (action, observation) pairs.

    The context ends at an observation o_t (its final step's action is
    None).  The continuation is the k step pairs

        (a_t, o_{t+1}), (a_{t+1}, o_{t+2}), ..., (a_{t+k-1}, o_{t+k})

    whose first symbol is the immediate action a_t generating o_{t+1}
    (paper Definition 2, revised).  For the empty context every position
    contributes one continuation window.
    """
    if window_length < 1:
        raise ValueError(f"window_length must be >= 1, got {window_length}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    hlen = history.length
    target_steps = history.steps
    continuations: list = []
    max_per_traj = 0
    for traj in dataset.trajectories:
        obs = traj.observations
        acts = traj.actions
        n_acts = len(acts)
        # `end` = index of the context's final observation; the continuation
        # uses actions end .. end+k-1 and observations end+1 .. end+k.
        count_this = 0
        first_end = max(hlen - 1, 0)
        for end in range(first_end, n_acts - window_length + 1, stride):
            if hlen > 0:
                start = end - hlen + 1
                candidate = tuple(
                    (int(obs[start + i]), int(acts[start + i]))
                    for i in range(hlen - 1)
                ) + ((int(obs[end]), None),)
                if candidate != target_steps:
                    continue
            cont = tuple(
                (int(acts[end + j]), int(obs[end + 1 + j]))
                for j in range(window_length)
            )
            continuations.append(cont)
            count_this += 1
        max_per_traj = max(max_per_traj, count_this)
    obs_history = history.to_observation_history()
    return ContinuationSample(
        history=obs_history,
        continuations=tuple(continuations),
        window_length=window_length,
        max_per_trajectory=max(1, max_per_traj),
    )


def extract_independent_continuation_samples(
    dataset: TrajectoryDataset,
    history: ObservationHistory,
    window_length: int,
    context_end: int = 0,
) -> ContinuationSample:
    """Take at most one prespecified block from each trajectory.

    ``context_end`` is an observation index fixed before seeing trajectory
    contents.  A trajectory contributes iff the history ends at that index
    and enough subsequent observations exist.  Consequently independent
    trajectories yield independent continuation samples; there is no random
    within-trajectory denominator or overlapping-window approximation.
    """
    if window_length < 1:
        raise ValueError(f"window_length must be >= 1, got {window_length}")
    if context_end < 0:
        raise ValueError(f"context_end must be >= 0, got {context_end}")
    hlen = history.length
    target = history.observations
    continuations: list[tuple[int, ...]] = []
    for traj in dataset.trajectories:
        obs = traj.observations
        start = context_end - hlen + 1
        stop = context_end + 1 + window_length
        if start < 0 or stop > len(obs):
            continue
        if tuple(obs[start : context_end + 1]) != target:
            continue
        continuations.append(tuple(obs[context_end + 1 : stop]))
    return ContinuationSample(
        history=history,
        continuations=tuple(continuations),
        window_length=window_length,
        max_per_trajectory=1,
    )


def extract_independent_action_aware_continuations(
    dataset: TrajectoryDataset,
    history: ActionAwareHistory,
    window_length: int,
    context_end: int = 0,
) -> ContinuationSample:
    """Action-aware counterpart of :func:`extract_independent_continuation_samples`."""
    if window_length < 1:
        raise ValueError(f"window_length must be >= 1, got {window_length}")
    if context_end < 0:
        raise ValueError(f"context_end must be >= 0, got {context_end}")
    hlen = history.length
    continuations: list[tuple[tuple[int, int], ...]] = []
    for traj in dataset.trajectories:
        obs, acts = traj.observations, traj.actions
        start = context_end - hlen + 1
        if start < 0 or context_end + window_length > len(acts):
            continue
        if hlen:
            candidate = tuple(
                (int(obs[start + i]), int(acts[start + i]))
                for i in range(hlen - 1)
            ) + ((int(obs[context_end]), None),)
            if candidate != history.steps:
                continue
        continuations.append(tuple(
            (int(acts[context_end + j]), int(obs[context_end + 1 + j]))
            for j in range(window_length)
        ))
    return ContinuationSample(
        history=history.to_observation_history(),
        continuations=tuple(continuations),
        window_length=window_length,
        max_per_trajectory=1,
    )
