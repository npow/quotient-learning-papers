"""Tests for the post-hoc ground-truth experiments (REV-3a / REV-3c).

Covers, per the revision item:
- the brute-force HMM continuation-law helper on tiny hand-computable
  instances (deterministic, soft-transition, and mixed-emission cases);
- the data-driven eps recipe, including the documented ABSTAIN-ALL fallback
  on a spectrum with no clean split;
- the Tiger stochastic-policy closed-loop law helper (laws sum to 1,
  termination-before-r+k conditioning handled exactly, hand-computed
  values, degenerate-policy reduction to the analytic always-listen gap).
"""

import numpy as np
import pytest

from environments.layer1_discrete.posthoc_random import PosthocRandomHMM
from experiments.main_table.run_nondegenerate import (
    tiger_closed_loop_laws,
    true_partition_and_gap,
)
from experiments.main_table.run_posthoc import (
    eps_hat_from_spectrum,
    selection_tv_spectrum,
)
from quotient_learning.dataset import Trajectory, TrajectoryDataset
from quotient_learning.types import ObservationHistory


# ----------------------------------------------------------------------
# Brute-force HMM law helper on hand-computable instances
# ----------------------------------------------------------------------

def _hmm(pi0, A, B):
    pi0 = np.asarray(pi0, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return PosthocRandomHMM(
        n_states=A.shape[0], n_obs=B.shape[1], pi0=pi0, A=A, B=B
    )


def test_brute_force_law_deterministic_chain() -> None:
    # State 0 -> 1 -> 0 -> ..., identity emission, start surely in state 0.
    env = _hmm([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])
    assert env.exact_context_mass((0,), r=0) == pytest.approx(1.0)
    law = env.exact_continuation_law((0,), r=0, k=1)
    assert law == pytest.approx({(0,): 0.0, (1,): 1.0})
    law2 = env.exact_continuation_law((0,), r=0, k=2)
    assert law2[(1, 0)] == pytest.approx(1.0)
    assert sum(law2.values()) == pytest.approx(1.0)
    # At r=1 the surely-observed context is (1,), continuing back to 0.
    assert env.exact_context_mass((1,), r=1) == pytest.approx(1.0)
    assert env.exact_continuation_law((1,), r=1, k=1)[(0,)] == pytest.approx(1.0)
    # Zero-mass contexts are excluded from eligibility and rejected for laws.
    assert env.eligible_contexts(r=0) == [(0,)]
    with pytest.raises(ValueError):
        env.exact_continuation_law((1,), r=0, k=1)


def test_brute_force_law_soft_transition_hand_computed() -> None:
    # Identity emission, sticky chain: P(next obs | obs 0) = (0.9, 0.1).
    env = _hmm([0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]], [[1.0, 0.0], [0.0, 1.0]])
    assert env.exact_context_mass((0,), r=0) == pytest.approx(0.5)
    law = env.exact_continuation_law((0,), r=0, k=1)
    assert law == pytest.approx({(0,): 0.9, (1,): 0.1})
    # Two-step law by hand: 0.81 / 0.09 / 0.01 / 0.09.
    law2 = env.exact_continuation_law((0,), r=0, k=2)
    assert law2 == pytest.approx(
        {(0, 0): 0.81, (0, 1): 0.09, (1, 0): 0.01, (1, 1): 0.09}
    )
    # TV matrix by hand: |0.9 - 0.1| = 0.8.
    assert env.min_pairwise_tv(r=0, k=1) == pytest.approx(0.8)
    assert env.true_partition(r=0, k=1) == {(0,): 0, (1,): 1}


def test_brute_force_law_mixed_emission_posterior() -> None:
    # Static states, noisy emission: the posterior after observing 0 is
    # (0.75, 0.25), so the k=1 law is 0.75*(0.75, 0.25) + 0.25*(0.25, 0.75).
    env = _hmm([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], [[0.75, 0.25], [0.25, 0.75]])
    assert env.exact_context_mass((0,), r=0) == pytest.approx(0.5)
    law = env.exact_continuation_law((0,), r=0, k=1)
    assert law == pytest.approx({(0,): 0.625, (1,): 0.375})
    assert env.min_pairwise_tv(r=0, k=1) == pytest.approx(0.25)


def test_shipped_instance_matches_documented_scan_result() -> None:
    # The documented protocol result: instance seed 1000 of the (3, 3, 0.3)
    # family has min pairwise TV 0.3872 >= 0.15 and an all-singleton quotient.
    env = PosthocRandomHMM()
    assert env.instance_seed == 1000
    assert env.min_pairwise_tv(r=0, k=1) == pytest.approx(0.3872, abs=1e-4)
    assert env.true_partition(r=0, k=1) == {(0,): 0, (1,): 1, (2,): 2}


# ----------------------------------------------------------------------
# Data-driven eps recipe
# ----------------------------------------------------------------------

def test_eps_recipe_clean_split_midpoint() -> None:
    # Spectrum [0, .02, .03, .40, .45]: largest gap .03 -> .40, midpoint .215.
    assert eps_hat_from_spectrum([0.02, 0.03, 0.40, 0.45]) == pytest.approx(0.215)


def test_eps_recipe_all_singleton_uses_zero_anchor() -> None:
    # No near-zero pairs: the gap is 0 -> smallest TV; eps_hat = tv_min / 2.
    assert eps_hat_from_spectrum([0.38, 0.59, 0.64]) == pytest.approx(0.19)


def test_eps_recipe_no_split_returns_fallback() -> None:
    # Evenly spread spectrum, all gaps 0.08 < GAMMA=0.10: documented fallback.
    spectrum = [0.08, 0.16, 0.24, 0.32, 0.40]
    assert eps_hat_from_spectrum(spectrum) is None
    # Empty spectrum also abstains.
    assert eps_hat_from_spectrum([]) is None


def test_eps_recipe_ignores_gaps_starting_at_or_above_ceiling() -> None:
    # Below the ceiling every gap is < 0.10; the only large gap (0.55 ->
    # 0.95) starts at 0.55 >= 0.5 and must not be used: fallback.
    spectrum = [0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.55, 0.95]
    assert eps_hat_from_spectrum(spectrum) is None


def test_eps_recipe_on_selection_half_dataset() -> None:
    # Hand-built selection half at r=0, k=1: context (0,) always continues
    # with 0, context (1,) always continues with 1 -> empirical TV 1.0 in a
    # 2-symbol alphabet; spectrum [1.0], gap 0 -> 1.0, eps_hat = 0.5.
    trajs = tuple(
        Trajectory(observations=(o, o), actions=(0,), rewards=(0.0,), seed=i)
        for i, o in enumerate([0, 1] * 10)
    )
    data = TrajectoryDataset(trajs, "toy", "uniform_random")
    env = _hmm([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
    alphabet = env.continuation_alphabet(1)
    contexts = [ObservationHistory((0,)), ObservationHistory((1,))]
    spectrum = selection_tv_spectrum(data, contexts, 1, 0, alphabet)
    assert spectrum == pytest.approx([1.0])
    assert eps_hat_from_spectrum(spectrum) == pytest.approx(0.5)


def test_eps_recipe_dataset_abstains_when_context_has_no_block() -> None:
    # Context (1,) occurs at the endpoint but its trajectory is too short to
    # supply a continuation block: the recipe must return None (abstain-all).
    trajs = (
        Trajectory(observations=(0, 0), actions=(0,), rewards=(0.0,), seed=0),
        Trajectory(observations=(1,), actions=(), rewards=(), seed=1),
    )
    data = TrajectoryDataset(trajs, "toy", "uniform_random")
    env = _hmm([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
    alphabet = env.continuation_alphabet(1)
    contexts = [ObservationHistory((0,)), ObservationHistory((1,))]
    assert selection_tv_spectrum(data, contexts, 1, 0, alphabet) is None


# ----------------------------------------------------------------------
# Tiger stochastic-policy closed-loop laws
# ----------------------------------------------------------------------

def test_tiger_closed_loop_laws_sum_to_one_and_masses() -> None:
    laws, masses = tiger_closed_loop_laws(
        listen_accuracy=0.85, action_probs=(0.8, 0.1, 0.1), r=1, k=1
    )
    assert set(laws) == {(0,), (1,)}
    for law in laws.values():
        assert sum(law.values()) == pytest.approx(1.0)
    # mu_h = p_listen / 2 per context; total mass = P(reach r+k) = p_listen.
    assert masses[(0,)] == pytest.approx(0.4)
    assert masses[(1,)] == pytest.approx(0.4)
    assert sum(masses.values()) == pytest.approx(0.8)


def test_tiger_closed_loop_law_hand_computed() -> None:
    # P(O_2=hl | O_1=hl, reach) = 0.8*(0.85^2 + 0.15^2) + 0.2*0.5 = 0.696.
    laws, _ = tiger_closed_loop_laws(
        listen_accuracy=0.85, action_probs=(0.8, 0.1, 0.1), r=1, k=1
    )
    assert laws[(0,)][(0,)] == pytest.approx(0.696)
    assert laws[(0,)][(1,)] == pytest.approx(0.304)
    # Symmetry between the two contexts.
    assert laws[(1,)][(1,)] == pytest.approx(laws[(0,)][(0,)])
    partition, gap = true_partition_and_gap(laws)
    assert partition == {(0,): 0, (1,): 1}
    # Gap = p_listen * (2a-1)^2 = 0.8 * 0.49 = 0.392.
    assert gap == pytest.approx(0.392)


def test_tiger_conditioning_reduces_to_always_listen() -> None:
    # Degenerate policy check: with p_listen=1 no episode terminates early,
    # and the closed-loop laws must equal the analytic always-listen laws
    # with gap (2a-1)^2 = 0.49 and full context mass 1/2 each.
    laws, masses = tiger_closed_loop_laws(
        listen_accuracy=0.85, action_probs=(1.0, 0.0, 0.0), r=1, k=1
    )
    assert laws[(0,)][(0,)] == pytest.approx(0.85**2 + 0.15**2)
    assert masses[(0,)] == pytest.approx(0.5)
    _, gap = true_partition_and_gap(laws)
    assert gap == pytest.approx(0.49)


def test_tiger_conditioning_excludes_early_door_opens() -> None:
    # With r=1, k=2 an open at step 1 ends the episode BEFORE index 3, so
    # only all-listen prefixes survive: mass = p_listen^2 / 2 per context,
    # and a door opened at the LAST step (index 2) still contributes.
    p_listen = 0.8
    laws, masses = tiger_closed_loop_laws(
        listen_accuracy=0.85, action_probs=(p_listen, 0.1, 0.1), r=1, k=2
    )
    assert masses[(0,)] == pytest.approx(p_listen**2 / 2)
    for law in laws.values():
        assert sum(law.values()) == pytest.approx(1.0)
    # Final-step mixture: P(last obs = hl | s=left, all listens so far)
    # = p_listen * 0.85 + (1 - p_listen) * 0.5 marginally within the law;
    # verify via the k=2 law's marginal on its SECOND symbol given first=hl.
    law = laws[(0,)]
    p_first_hl = law[(0, 0)] + law[(0, 1)]
    p_second_hl_given_first_hl = law[(0, 0)] / p_first_hl
    # Posterior after two hear-lefts: (0.85^2)/(0.85^2 + 0.15^2) for left.
    w_left = 0.85**2 / (0.85**2 + 0.15**2)
    expected = p_listen * (w_left * 0.85 + (1 - w_left) * 0.15) \
        + (1 - p_listen) * 0.5
    assert p_second_hl_given_first_hl == pytest.approx(expected)
