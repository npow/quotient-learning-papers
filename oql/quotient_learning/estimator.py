"""QuotientEstimator: offline batch quotient discovery from trajectories.

Algorithm:
1. Collect `n_trajectories` episodes via uniform-random exploration.
2. Enumerate all observed observation histories up to `history_max_length`
   that occur >= `min_occurrences` times in the dataset.
3. For each history, extract `ContinuationSample` of `window_length`.
4. Pairwise run the configured `EquivalenceTest`. Union-find clusters histories
   whose continuation distributions are not statistically distinguished.
5. Build a `CanonicalQuotient` with deterministic class IDs and an empirical
   transition function `(class_id, observation) -> class_id`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from environments.base import POMDPEnvironment
from quotient_learning.collection import (
    Policy,
    collect_trajectories,
    count_based_exploration_policy,
    uniform_random_policy,
)
from quotient_learning.dataset import TrajectoryDataset
from quotient_learning.equivalence import (
    EquivalenceTest,
    GTestEquivalenceTest,
    TVThresholdEquivalenceTest,
)
from quotient_learning.errors import InsufficientSamplesError
from quotient_learning.sampling import (
    enumerate_independent_action_aware_histories,
    enumerate_independent_histories,
    enumerate_observed_action_aware_histories,
    enumerate_observed_histories,
    extract_action_aware_continuations,
    extract_continuation_samples,
    extract_independent_action_aware_continuations,
    extract_independent_continuation_samples,
)
from quotient_learning.stats import required_samples
from quotient_learning.types import (
    ActionAwareHistory,
    CanonicalQuotient,
    ContinuationAlphabet,
    ObservationHistory,
)


@dataclass
class EstimatorConfig:
    n_trajectories: int = 5000
    max_steps_per_episode: int = 50
    window_length: int = 2
    history_max_length: int = 1
    min_history_occurrences: int = 30
    alpha: float = 0.01
    window_stride: int = 1


class _UnionFind:
    def __init__(self, items) -> None:
        self._parent = {h: h for h in items}

    def find(self, h):
        root = h
        while self._parent[root] != root:
            root = self._parent[root]
        cur = h
        while self._parent[cur] != root:
            nxt = self._parent[cur]
            self._parent[cur] = root
            cur = nxt
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> dict:
        out: dict = defaultdict(list)
        for h in self._parent:
            out[self.find(h)].append(h)
        return dict(out)


class QuotientEstimator:
    """Offline batch quotient discovery.

    Parameters
    ----------
    env:
        The POMDP environment to collect data from.
    n_trajectories:
        Number of episodes to collect.
    window_length:
        Length of the continuation window used for distribution comparison.
    history_max_length:
        Maximum length of observation histories to consider.
    min_history_occurrences:
        Minimum times a history must appear in the dataset to be included.
    equivalence_test:
        Statistical test used to decide if two histories are equivalent.
        Defaults to ``GTestEquivalenceTest()``.
    alpha:
        Fixed significance level for pairwise equivalence tests. Used when
        ``delta`` is ``None``. Defaults to 0.01.
    delta:
        Optional overall failure probability. When provided, the per-pair
        significance level is set to the Bonferroni-corrected value
        alpha_0 = delta / (2 * |H_N|^2) as prescribed by Theorem 2 of the
        paper, with |H_N| the selection-half candidate count in certified
        mode. Supersedes ``alpha`` when set.
    max_steps_per_episode:
        Maximum number of steps per episode.
    policy:
        Optional explicit policy callable. If provided, ``policy_kind`` is
        ignored.
    policy_id:
        String identifier for the policy (used in dataset metadata).
    policy_kind:
        Which built-in policy to use when ``policy`` is not provided.
        ``"uniform_random"`` (default) satisfies the iid assumption of
        Theorem 1; ``"count_based"`` is non-stationary and does not.
    drop_terminal_obs:
        When True, the last observation of each trajectory (the post-terminate
        observation appended after the episode ends) is excluded before
        extracting continuation windows.
    window_stride:
        Spacing between scanned continuation-window start positions within a
        trajectory in exploratory mode.  A stride changes overlap but does not
        certify independence.  The theorem-facing path instead uses
        ``certified_independent=True`` and one fixed block per trajectory.
    action_aware:
        When True, use action-aware histories (Definition 1 of the paper) as
        the clustering keys instead of observation-only histories.  This yields
        a finer or equal partition because two histories that share the same
        observations but differ in the actions taken are kept separate.
        For Tiger, this is load-bearing: H=(hear-left) after listen is
        informationally different from H=(hear-left) after open-left (episode
        boundary), and conflating them collapses the quotient to n=1.
        The returned CanonicalQuotient.partition maps ObservationHistory keys
        (observation-only projection) for compatibility with downstream code.
    certified_independent:
        Enable the finite-sample path: split whole trajectories for context
        selection/inference and take at most one prespecified inference block
        per trajectory. Requires ``TVThresholdEquivalenceTest``,
        ``epsilon_min``, and ``power_beta``.
    """

    def __init__(
        self,
        env: POMDPEnvironment,
        n_trajectories: int = 5000,
        window_length: int = 2,
        history_max_length: int = 1,
        min_history_occurrences: int = 30,
        equivalence_test: EquivalenceTest | None = None,
        alpha: float = 0.01,
        delta: float | None = None,
        max_steps_per_episode: int = 50,
        policy: Policy | None = None,
        policy_id: str = "uniform_random",
        policy_kind: Literal["uniform_random", "count_based"] = "uniform_random",
        drop_terminal_obs: bool = False,
        action_aware: bool = False,
        window_stride: int = 1,
        epsilon_min: float | None = None,
        power_beta: float | None = None,
        episodic_blocks: bool = False,
        certified_independent: bool = False,
        selection_fraction: float = 0.5,
        independent_context_end: int = 0,
        include_root: bool = True,
    ):
        self.env = env
        self.drop_terminal_obs = drop_terminal_obs
        self.action_aware = action_aware
        self.delta = delta
        # Power gate (certified path): with both set, a pair is tested only
        # when both contexts carry >= m*(epsilon_min, alpha, power_beta)
        # samples; below that the pair is UNDETERMINED and never merges.
        self.epsilon_min = epsilon_min
        self.power_beta = power_beta
        # Legacy diagnostic only. Pooled within-episode windows are not iid;
        # certification requires ``certified_independent=True`` instead.
        self.episodic_blocks = episodic_blocks
        self.certified_independent = certified_independent
        self.selection_fraction = selection_fraction
        self.independent_context_end = independent_context_end
        self.include_root = include_root
        if certified_independent and episodic_blocks:
            raise ValueError(
                "certified_independent and legacy episodic_blocks are mutually exclusive"
            )
        if certified_independent and not (0.0 < selection_fraction < 1.0):
            raise ValueError("selection_fraction must be in (0, 1)")
        self.config = EstimatorConfig(
            n_trajectories=n_trajectories,
            max_steps_per_episode=max_steps_per_episode,
            window_length=window_length,
            history_max_length=history_max_length,
            min_history_occurrences=min_history_occurrences,
            alpha=alpha,
            window_stride=window_stride,
        )
        self.equivalence_test = equivalence_test or GTestEquivalenceTest()
        if certified_independent:
            if not isinstance(self.equivalence_test, TVThresholdEquivalenceTest):
                raise ValueError(
                    "certified_independent requires TVThresholdEquivalenceTest"
                )
            if epsilon_min is None or power_beta is None:
                raise ValueError(
                    "certified_independent requires epsilon_min and power_beta"
                )
            if abs(self.equivalence_test.epsilon_min - epsilon_min) > 1e-12:
                raise ValueError(
                    "estimator epsilon_min must match the TV test epsilon_min"
                )
        if policy is not None:
            self.policy = policy
            self.policy_id = policy_id
        elif policy_kind == "count_based":
            self.policy = count_based_exploration_policy(
                n_actions=env.n_actions,
                history_max_length=history_max_length,
            )
            self.policy_id = "count_based"
        else:
            self.policy = uniform_random_policy(env.n_actions)
            self.policy_id = policy_id

    def collect(self, seed: int) -> TrajectoryDataset:
        return collect_trajectories(
            env=self.env,
            policy=self.policy,
            n_episodes=self.config.n_trajectories,
            max_steps=self.config.max_steps_per_episode,
            seed=seed,
            policy_id=self.policy_id,
        )

    def estimate(self, dataset: TrajectoryDataset) -> CanonicalQuotient:
        """Run the clustering algorithm on a fixed dataset."""
        if self.action_aware:
            return self._estimate_action_aware(dataset)
        return self._estimate_obs_only(dataset)

    # ------------------------------------------------------------------
    # Observation-only path (original algorithm)
    # ------------------------------------------------------------------

    def _estimate_obs_only(self, dataset: TrajectoryDataset) -> CanonicalQuotient:
        """Cluster using observation-only histories."""
        if self.certified_independent:
            selection_data, inference_data = dataset.split(self.selection_fraction)
        else:
            selection_data = inference_data = dataset
        if self.certified_independent:
            histories = enumerate_independent_histories(
                selection_data,
                max_length=self.config.history_max_length,
                context_end=self.independent_context_end,
                min_occurrences=self.config.min_history_occurrences,
                include_root=self.include_root,
            )
        else:
            histories = enumerate_observed_histories(
                selection_data,
                max_length=self.config.history_max_length,
                min_occurrences=self.config.min_history_occurrences,
            )
        if not self.include_root:
            histories = [h for h in histories if h.length > 0]
        if not histories:
            return CanonicalQuotient(n_classes=0, partition={}, transition={})

        samples = {}
        viable = []
        for h in histories:
            if self.certified_independent:
                # Certified contract: the candidate set is a function of the
                # selection half only.  Every selected context is kept; a
                # context with too few inference samples abstains pairwise
                # (m* gate / InsufficientSamplesError) rather than being
                # dropped, and the Bonferroni K below therefore equals the
                # selection-half count as Theorem 2 requires.
                cs = extract_independent_continuation_samples(
                    inference_data, h, self.config.window_length,
                    self.independent_context_end,
                )
                samples[h] = cs
                viable.append(h)
                continue
            if self.config.window_stride == 1:
                cs = extract_continuation_samples(inference_data, h, self.config.window_length)
            else:
                cs = extract_continuation_samples(
                    inference_data, h, self.config.window_length, self.config.window_stride
                )
            if cs.n_samples >= max(30, self.config.min_history_occurrences // 2):
                samples[h] = cs
                viable.append(h)
        if not viable:
            return CanonicalQuotient(n_classes=0, partition={}, transition={})

        alpha = self._effective_alpha(len(viable))
        uf, undetermined, n_pairs = self._pairwise_cluster(viable, samples, alpha)
        groups = uf.groups()

        # Root-singleton exclusion: the empty-history class is excluded from
        # n_classes and the partition when it forms a singleton group.
        root_history = ObservationHistory(())
        if any(root_history in members for members in groups.values()):
            root_key = next(
                k for k, members in groups.items() if root_history in members
            )
            if len(groups[root_key]) == 1:
                del groups[root_key]

        sorted_roots = sorted(
            groups.keys(),
            key=lambda h: (h.length, h.observations),
        )
        class_id_of_root = {root: i for i, root in enumerate(sorted_roots)}
        partition = {}
        for root, members in groups.items():
            cid = class_id_of_root[root]
            for m in members:
                partition[m] = cid

        # The empirical transition table scans overlapping windows and is not
        # covered by the certified analysis; certified mode omits it.
        transition = (
            {} if self.certified_independent
            else self._build_transition(partition, inference_data)
        )
        return CanonicalQuotient(
            n_classes=len(groups), partition=partition, transition=transition,
            undetermined_rate=(len(undetermined) / n_pairs if n_pairs else 0.0),
            undetermined_pairs=tuple(undetermined),
            boundary_flags=self._boundary_flags(undetermined, partition),
        )

    # ------------------------------------------------------------------
    # Action-aware path (Definition 1 of the paper)
    # ------------------------------------------------------------------

    def _estimate_action_aware(self, dataset: TrajectoryDataset) -> CanonicalQuotient:
        """Cluster using action-aware (observation, action) histories."""
        if self.certified_independent:
            selection_data, inference_data = dataset.split(self.selection_fraction)
        else:
            selection_data = inference_data = dataset
        if self.certified_independent:
            aa_histories = enumerate_independent_action_aware_histories(
                selection_data,
                max_length=self.config.history_max_length,
                context_end=self.independent_context_end,
                min_occurrences=self.config.min_history_occurrences,
                include_root=self.include_root,
            )
        else:
            aa_histories = enumerate_observed_action_aware_histories(
                selection_data,
                max_length=self.config.history_max_length,
                min_occurrences=self.config.min_history_occurrences,
            )
        if not self.include_root:
            aa_histories = [h for h in aa_histories if h.length > 0]
        if not aa_histories:
            return CanonicalQuotient(n_classes=0, partition={}, transition={})

        samples = {}
        viable = []
        for h in aa_histories:
            if self.certified_independent:
                # Same certified contract as the observation-only path:
                # selection-half contexts are never dropped on inference-half
                # evidence; low-count contexts abstain pairwise instead.
                cs = extract_independent_action_aware_continuations(
                    inference_data, h, self.config.window_length,
                    self.independent_context_end,
                )
                samples[h] = cs
                viable.append(h)
                continue
            cs = extract_action_aware_continuations(
                inference_data, h, self.config.window_length,
                self.config.window_stride,
            )
            if cs.n_samples >= max(30, self.config.min_history_occurrences // 2):
                samples[h] = cs
                viable.append(h)
        if not viable:
            return CanonicalQuotient(n_classes=0, partition={}, transition={})

        alpha = self._effective_alpha(len(viable))
        uf, undetermined, n_pairs = self._pairwise_cluster(viable, samples, alpha)
        groups = uf.groups()

        # Root-singleton exclusion for the action-aware empty history.
        root_aa = ActionAwareHistory(())
        if any(root_aa in members for members in groups.values()):
            root_key_aa = next(
                k for k, members in groups.items() if root_aa in members
            )
            if len(groups[root_key_aa]) == 1:
                del groups[root_key_aa]

        sorted_roots_aa = sorted(
            groups.keys(),
            key=lambda h: (h.length, h.steps),
        )
        class_id_of_root_aa = {root: i for i, root in enumerate(sorted_roots_aa)}

        # Build obs-only partition for CanonicalQuotient compatibility.
        obs_partition = {}
        aa_cid_of = {}
        for root, members in groups.items():
            cid = class_id_of_root_aa[root]
            for m in members:
                aa_cid_of[m] = cid
                obs_key = m.to_observation_history()
                if obs_key not in obs_partition or obs_partition[obs_key] > cid:
                    obs_partition[obs_key] = cid

        transition = (
            {} if self.certified_independent
            else self._build_transition(obs_partition, inference_data)
        )
        return CanonicalQuotient(
            n_classes=len(groups),
            partition=obs_partition,
            transition=transition,
            undetermined_rate=(len(undetermined) / n_pairs if n_pairs else 0.0),
            undetermined_pairs=tuple(undetermined),
            boundary_flags=self._boundary_flags(undetermined, aa_cid_of),
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _alphabet_size(self) -> int:
        """Continuation-alphabet size for the power gate, preferring the
        environment's contract and falling back to the full product count."""
        if isinstance(self.equivalence_test, TVThresholdEquivalenceTest):
            return self.equivalence_test.alphabet.size
        try:
            alph = self.env.continuation_alphabet(
                self.config.window_length, self.action_aware
            )
            if isinstance(alph, ContinuationAlphabet):
                return alph.size
        except Exception:
            pass
        base = self.env.n_observations * (self.env.n_actions if self.action_aware else 1)
        return max(2, int(base) ** self.config.window_length)

    def _gate_threshold(self, alpha: float) -> int:
        """m* for the configured power gate; 0 when the gate is disabled."""
        if self.epsilon_min is None or self.power_beta is None:
            return 0
        return required_samples(
            self.epsilon_min, alpha, self.power_beta, self._alphabet_size()
        )

    def _pairwise_cluster(self, viable, samples, alpha):
        """Gated pairwise testing: returns (union_find, undetermined, n_pairs).

        Pairs below the power threshold — and pairs whose test refuses for
        insufficient samples — are recorded as undetermined and NEVER unioned
        (undetermined is not evidence of equality)."""
        m_req = self._gate_threshold(alpha)
        uf = _UnionFind(viable)
        undetermined: list = []
        n_pairs = 0
        for i, ha in enumerate(viable):
            for hb in viable[i + 1:]:
                n_pairs += 1
                n_a, n_b = samples[ha].n_samples, samples[hb].n_samples
                if m_req > 0 and (n_a < m_req or n_b < m_req):
                    undetermined.append((ha, hb))
                    continue
                try:
                    res = self.equivalence_test.test(samples[ha], samples[hb], alpha=alpha)
                except InsufficientSamplesError:
                    undetermined.append((ha, hb))
                    continue
                if res.same_class:
                    uf.union(ha, hb)
        return uf, undetermined, n_pairs

    @staticmethod
    def _boundary_flags(undetermined, cid_of) -> frozenset:
        """Class-id pairs whose separation rests on an undetermined pair."""
        flags = set()
        for ha, hb in undetermined:
            ca, cb = cid_of.get(ha), cid_of.get(hb)
            if ca is not None and cb is not None and ca != cb:
                flags.add((min(ca, cb), max(ca, cb)))
        return frozenset(flags)

    def _effective_alpha(self, n_viable: int) -> float:
        """Return per-pair alpha, Bonferroni-corrected when delta is set."""
        if self.delta is not None and n_viable >= 2:
            return self.delta / (2.0 * n_viable ** 2)
        return self.config.alpha

    def _build_transition(
        self,
        partition,
        dataset: TrajectoryDataset,
    ) -> dict:
        """Empirically estimate (class_id, observation) -> class_id.

        Exploratory-only: scans all start positions with overlapping
        windows, which the certified analysis does not cover.  Certified
        mode returns an empty table instead of calling this."""
        votes: dict = defaultdict(Counter)
        max_h = max((h.length for h in partition), default=0)
        for traj in dataset.trajectories:
            obs = traj.observations
            n = len(obs)
            for start in range(n):
                for length in range(0, min(max_h, n - start - 1) + 1):
                    h = ObservationHistory(obs[start : start + length])
                    if h not in partition:
                        continue
                    nxt_obs = obs[start + length]
                    h_next = ObservationHistory(obs[start : start + length + 1])
                    cid = partition[h]
                    if h_next in partition:
                        cid_next = partition[h_next]
                        votes[(cid, nxt_obs)][cid_next] += 1
        transition = {}
        for key, ctr in votes.items():
            transition[key] = ctr.most_common(1)[0][0]
        return transition

    def run(self, seed: int = 0) -> CanonicalQuotient:
        """Collect data + estimate the quotient in one call."""
        dataset = self.collect(seed=seed)
        return self.estimate(dataset)
