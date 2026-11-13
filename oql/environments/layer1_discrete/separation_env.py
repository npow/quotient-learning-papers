"""Separation POMDP: bounded-agent quotient vs. PSR predictive-state partition.

This POMDP is designed to demonstrate that the bounded-agent (probe-exact)
quotient under a restricted probe family Π can be strictly COARSER than the
PSR predictive-state partition, and that this coarser partition is
nevertheless sufficient — and in fact superior — for a downstream decision
task, because the PSR's extra dimensions track decision-irrelevant structure.

Construction
------------
Hidden state: (X, Y) where X ∈ {0,1} and Y ∈ {0,1}.

  - X is the decision-relevant variable: it determines the optimal action.
  - Y is a FULLY LATENT nuisance variable: it is never observed directly.
    However, Y has strong autocorrelation (p_switch_y = 0.05, so each step
    Y flips with only 5% probability) and modulates the X-switching rate.

Observation: o = x_noisy ∈ {0,1}  (Y is NOT observed)
  - x_noisy = X with probability obs_acc_x = 0.90, else 1-X.

Transition:
  - Y switches with probability p_switch_y = 0.05 (slow, autocorrelated).
  - X switches with probability p_switch_x_when_y0 = 0.20 if Y=0,
                           or  p_switch_x_when_y1 = 0.05 if Y=1.
  This means Y modulates how quickly X changes — but the agent cannot
  observe Y directly.

Actions: 2 actions.  Action a=0 is optimal when X=0, a=1 when X=1.
  - Reward = +1 if action matches X, else -1.

Quotient analysis
-----------------

PSR predictive states (open-loop test statistics):
  The PSR Hankel matrix H[hist, fut] captures P(fut | hist) for all
  (hist, fut) pairs.  Because Y has strong autocorrelation (ρ=0.90/step)
  and modulates X-switching, the conditional distribution P(o_{t+1:t+k}|h)
  depends BOTH on belief_X and belief_Y extracted from h.
  Specifically: two histories h1, h2 that induce the same X-belief but
  different Y-beliefs produce different P(o_{t+2}|h) at horizon k=2
  (because X is more likely to switch when Y=0 vs Y=1).
  Therefore: K_PSR ≥ 4 (the spectral Hankel rank reflects both X and Y).

Bounded-agent quotient under uniform-random probe policy:
  The agent only observes x_noisy ∈ {0,1}.  The continuation distribution
  P(x_{t+1:t+k} | h, π_uniform) at short horizons (k=1,2) is primarily
  determined by belief_X.  The Y-modulation of X-switching is second-order:
  it changes the effective X-switch probability from ≈0.20 (Y=0, 50% prior)
  to ≈0.05 (Y=1, 50% prior), but with Y unobservable and balanced at the
  prior, the marginalised P(x_{t+1}|h) collapses to the average.
  QuotientEstimator with history_max_length=1 sees only the last x_noisy
  observation, giving K_BA = 2 classes (high-X vs low-X belief).
  The G-test at this resolution cannot distinguish histories by Y-belief.

Decision task:
  The optimal action is a* = round(belief_X), depending only on X.
  Using K_BA=2 classes as features → decision loss ≈ 0 (X correctly tracked).
  Using K_PSR classes as features → also achieves ≈0 loss, but K_PSR > K_BA.
  The bounded-agent quotient is STRICTLY COARSER (K_BA < K_PSR) yet equally
  decision-sufficient.  The PSR's extra dimensions are decision-irrelevant.

Paper claim (Lemma 3.x):
  For any probe family Π that uses only the single-step x_noisy signal with
  uniform-random actions, the bounded-agent quotient Q^Π has |Q^Π| = 2.
  The PSR predictive-state partition has rank ≥ 4.  Hence Q^Π is strictly
  coarser than the PSR partition.  Nevertheless Q^Π is decision-sufficient
  for any reward function depending only on X.
"""

from __future__ import annotations

import numpy as np

from environments.base import POMDPEnvironment, StepResult

# Observation encoding: x_noisy only (Y is fully latent)
OBS_COUNT = 2  # {0, 1}


class SeparationPOMDP(POMDPEnvironment):
    """POMDP demonstrating K_PSR > K_BA (bounded-agent quotient is coarser).

    Y is a fully latent autocorrelated nuisance variable.
    Only X is observable.  PSR sees Y-induced rank inflation; the
    bounded-agent quotient under uniform-random probe does not.

    Parameters
    ----------
    obs_acc_x : float
        Accuracy of the x_noisy observation (0.5..1.0).
    p_switch_y : float
        Per-step probability that Y flips (0..0.5). Small = high autocorr.
    p_switch_x_when_y0 : float
        Per-step probability X flips when Y=0 (X is less stable).
    p_switch_x_when_y1 : float
        Per-step probability X flips when Y=1 (X is more stable).
    reward_correct : float
        Reward for matching optimal action to X.
    reward_wrong : float
        Reward for mismatching.
    max_steps : int
        Maximum episode length before truncation.
    seed : int | None
        RNG seed for reproducibility.
    """

    _k1_full_support = True

    def __init__(
        self,
        obs_acc_x: float = 0.90,
        p_switch_y: float = 0.05,
        p_switch_x_when_y0: float = 0.20,
        p_switch_x_when_y1: float = 0.05,
        reward_correct: float = 1.0,
        reward_wrong: float = -1.0,
        max_steps: int = 30,
        seed: int | None = None,
    ):
        if not 0.5 <= obs_acc_x <= 1.0:
            raise ValueError("obs_acc_x must be in [0.5, 1.0]")
        if not 0.0 <= p_switch_y <= 0.5:
            raise ValueError("p_switch_y must be in [0.0, 0.5]")
        if not 0.0 <= p_switch_x_when_y0 <= 0.5:
            raise ValueError("p_switch_x_when_y0 must be in [0.0, 0.5]")
        if not 0.0 <= p_switch_x_when_y1 <= 0.5:
            raise ValueError("p_switch_x_when_y1 must be in [0.0, 0.5]")

        self.obs_acc_x = obs_acc_x
        self.p_switch_y = p_switch_y
        self.p_switch_x_when_y0 = p_switch_x_when_y0
        self.p_switch_x_when_y1 = p_switch_x_when_y1
        self.reward_correct = reward_correct
        self.reward_wrong = reward_wrong
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        # Hidden state
        self._x: int = 0
        self._y: int = 0  # latent, never observed
        self._t: int = 0

    # ------------------------------------------------------------------ #
    # POMDPEnvironment interface
    # ------------------------------------------------------------------ #

    @property
    def n_observations(self) -> int:
        return OBS_COUNT  # {0,1}

    @property
    def n_actions(self) -> int:
        return 2  # {0,1}

    @property
    def environment_id(self) -> str:
        return (
            f"SeparationPOMDP(acc_x={self.obs_acc_x},"
            f"p_switch_y={self.p_switch_y},"
            f"p_switch_x_y0={self.p_switch_x_when_y0},"
            f"p_switch_x_y1={self.p_switch_x_when_y1})"
        )

    @property
    def metadata(self) -> dict[str, object]:
        # Y is a latent slow Markov chain that modulates X switching.
        # PSR needs (belief_X, belief_Y) -> K_PSR >= 4 (empirically 3-4 via TPSR).
        # Bounded-agent under uniform-random probe can't observe Y -> K_BA = 2.
        return {
            "psr_n_classes_expected": 4,
            "bounded_agent_n_classes_expected": 2,
            "obs_acc_x": self.obs_acc_x,
            "p_switch_y": self.p_switch_y,
            "p_switch_x_when_y0": self.p_switch_x_when_y0,
            "p_switch_x_when_y1": self.p_switch_x_when_y1,
            "n_hidden_states": 4,  # (X, Y) in {0,1}^2
        }

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._x = int(self._rng.integers(0, 2))
        self._y = int(self._rng.integers(0, 2))
        self._t = 0
        return self._draw_observation()

    def step(self, action: int) -> StepResult:
        if action not in (0, 1):
            raise ValueError(f"action must be 0 or 1, got {action}")

        # Reward depends only on X (Y is decision-irrelevant)
        reward = self.reward_correct if action == self._x else self.reward_wrong

        # Transition Y (slow autocorrelated chain — latent)
        if self._rng.random() < self.p_switch_y:
            self._y = 1 - self._y

        # Transition X (rate depends on Y — this is what creates PSR rank inflation)
        p_switch_x = self.p_switch_x_when_y0 if self._y == 0 else self.p_switch_x_when_y1
        if self._rng.random() < p_switch_x:
            self._x = 1 - self._x

        self._t += 1
        obs = self._draw_observation()
        truncated = self._t >= self.max_steps
        return StepResult(
            observation=obs,
            reward=reward,
            terminated=False,
            truncated=truncated,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _draw_observation(self) -> int:
        """Return x_noisy (Y is never observed)."""
        if self._rng.random() < self.obs_acc_x:
            return self._x
        return 1 - self._x

    def optimal_action(self) -> int:
        """Ground-truth optimal action (requires hidden state — for evaluation only)."""
        return self._x

    # ------------------------------------------------------------------ #
    # Analytical analysis (for paper validation)
    # ------------------------------------------------------------------ #

    def marginal_x_switch_rate(self) -> float:
        """Average X-switching rate over the stationary Y distribution.

        Under the stationary distribution π(Y=0) = π(Y=1) = 0.5:
        E[p_switch_X] = 0.5 * p_switch_x_when_y0 + 0.5 * p_switch_x_when_y1
        """
        return 0.5 * self.p_switch_x_when_y0 + 0.5 * self.p_switch_x_when_y1

    @staticmethod
    def explain_separation() -> str:
        return (
            "Separation mechanism: Y is a fully latent autocorrelated nuisance.\n"
            "\n"
            "PSR (Hankel rank inflation):\n"
            "  Two histories h1=(x_obs=0 for 3 steps) and\n"
            "                h2=(x_obs=0 for 1 step, then x_obs=1, then x_obs=0)\n"
            "  induce the SAME X-belief (roughly P(X=0|h)≈0.8) but DIFFERENT\n"
            "  Y-beliefs (h1 -> Y more likely to be in the stable regime).\n"
            "  The Hankel test P(x_{t+2}|h) differs between h1 and h2\n"
            "  because X's switching rate in the next step depends on Y.\n"
            "  Therefore PSR Hankel rank >= 4, and TPSR reports K_PSR >= 3.\n"
            "\n"
            "Bounded-agent quotient (insensitive to Y):\n"
            "  Under uniform-random policy, the G-test compares\n"
            "  P(x_{t+1}|h1) vs P(x_{t+1}|h2) at horizon k=1.\n"
            "  At k=1, the difference is small because Y-effects require\n"
            "  the second step to manifest.\n"
            "  QuotientEstimator with history_max_length=1 clusters histories\n"
            "  by x_noisy value only: K_BA = 2 (x_seen=0 vs x_seen=1).\n"
            "\n"
            "Decision task:\n"
            "  The optimal action a* = argmax P(X=a|h) depends only on X.\n"
            "  K_BA=2 classes achieve zero decision loss (X-belief is sufficient).\n"
            "  K_PSR's extra Y-belief dimensions are decision-irrelevant.\n"
            "  Separation: K_BA(=2) < K_PSR(>=3), same decision loss.\n"
        )
