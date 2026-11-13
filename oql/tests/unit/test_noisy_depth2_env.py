"""Noisy depth-2 benchmark: analytic ground truth verified by brute force.

The environment is an explicit second-order Markov symbol process: contexts
ending in a non-zero symbol share one exact law at EVERY noise level, the
depth-2 contexts (x, 0) have distinct laws, and eps_min(eta) is exactly
computable and strictly decreasing in eta. This test re-derives the laws and
the partition independently (long-run empirical frequencies + direct mixture
computation) for small instances.
"""

import itertools

import numpy as np
import pytest

from environments.layer2_memory.noisy_depth2_kcycle import NoisyDepth2KCycle
from quotient_learning.types import ObservationHistory


def _bruteforce_law(env, word: tuple[int, ...], steps: int = 400_000, seed: int = 0):
    """Empirical next-symbol law after `word`, from one long rollout."""
    rng = np.random.default_rng(seed)
    n = env.n_observations
    prev, cur = 1, 0
    seq = [prev, cur]
    for _ in range(steps):
        row = env.transition_row(seq[-2], seq[-1])
        seq.append(int(rng.choice(n, p=row)))
    counts = np.zeros(n)
    L = len(word)
    for i in range(1, len(seq) - L):
        if tuple(seq[i : i + L]) == word:
            counts[seq[i + L]] += 1
    assert counts.sum() > 500, f"word {word} too rare in rollout"
    return counts / counts.sum()


def _tv(p, q) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


class TestNoisyDepth2KCycle:
    def test_depth1_zero_context_is_aliased_but_depth2_resolves(self):
        env = NoisyDepth2KCycle(m=2, eta=0.1)
        part = env.target_partition(max_length=2)
        h10, h20, h0 = (ObservationHistory(w) for w in ((1, 0), (2, 0), (0,)))
        assert part[h10] != part[h20]
        assert part[h0] not in (part[h10], part[h20])

    def test_nonzero_contexts_share_one_exact_class_at_every_noise_level(self):
        # The design guarantee that noise creates no spurious near-classes.
        for eta in (0.0, 0.1, 0.3):
            env = NoisyDepth2KCycle(m=3, eta=eta)
            part = env.target_partition(max_length=2)
            cids = {
                part[h]
                for h in part
                if h.observations[-1] != 0
            }
            assert len(cids) == 1, f"eta={eta}: non-zero-ending contexts split: {cids}"

    def test_exact_laws_match_bruteforce_rollout(self):
        env = NoisyDepth2KCycle(m=2, eta=0.2)
        for word in [(0,), (1,), (1, 0), (2, 0), (0, 0)]:
            exact = env.exact_continuation_law(word)
            empirical = _bruteforce_law(env, word)
            assert _tv(exact, empirical) < 0.02, (
                f"word={word}: exact={exact}, empirical={empirical.tolist()}"
            )

    def test_depth2_rows_are_exact_table_rows(self):
        # A length-2 context pins the pair state, so its law IS the table row.
        env = NoisyDepth2KCycle(m=3, eta=0.15)
        for x in (1, 2, 3):
            law = env.exact_continuation_law((x, 0))
            row = env.transition_row(x, 0)
            assert law == pytest.approx(list(row), abs=1e-12)

    def test_epsilon_min_decreases_with_noise(self):
        eps = [
            NoisyDepth2KCycle(m=2, eta=e).epsilon_min(max_length=2)
            for e in (0.0, 0.2, 0.4, 0.6)
        ]
        assert all(a > b for a, b in zip(eps, eps[1:])), f"eps_min not decreasing: {eps}"
        assert eps[0] > 0.4
        assert eps[-1] > 0.0

    def test_partition_matches_bruteforce_grouping(self):
        env = NoisyDepth2KCycle(m=2, eta=0.25)
        part = env.target_partition(max_length=2)
        words = list(part.keys())
        laws = {w: env.exact_continuation_law(w.observations) for w in words}
        for wa, wb in itertools.combinations(words, 2):
            same_law = _tv(laws[wa], laws[wb]) < 1e-9
            assert same_law == (part[wa] == part[wb]), f"{wa} vs {wb}"

    def test_rollout_alphabet_and_full_support(self):
        env = NoisyDepth2KCycle(m=2, eta=0.1, seed=3)
        assert env.continuation_alphabet(k=1).full_support
        obs = [env.reset(seed=3)]
        for _ in range(200):
            obs.append(env.step(0).observation)
        assert set(obs) <= set(range(env.n_observations))
        assert len(set(obs)) == env.n_observations
