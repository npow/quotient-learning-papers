"""Lower-bound simulation for quotient learning sample complexity.

Verifies the minimax lower bound  N = Omega(n / (T eps^2))  via two approaches:

1. Le Cam two-point method (direct hypothesis test):
   - P1: n-class i.i.d. POMDP with emission parameter 1/2+eps at each state
   - P2: (n-1)-class version where states 0 and 1 share the same emission
   - Measures log-likelihood ratio test power as a function of N
   - Shows phase transition at the Le Cam threshold N* = Omega(n/(T*eps^2))

2. QuotientEstimator empirical validation:
   - Builds K-cycle POMDPs with varying eps
   - Shows the estimator's recovery threshold matches theory

Construction (Le Cam pair — i.i.d. emission POMDP)
---------------------------------------------------
P1: n independent binary channels, one per state.
    State s emits: obs=0 w.p. 1/2+eps  (if s is even)
                   obs=1 w.p. 1/2+eps  (if s is odd)
    Episodes: start state drawn uniformly; state persists for T steps (self-loop).
    Canonical quotient: n classes (all emissions distinct).

P2: Same as P1, except state 1 is changed to emit obs=0 w.p. 1/2+eps
    (same as state 0).  States 0 and 1 now have identical continuation
    distributions.  Canonical quotient: n-1 classes.

Per-step KL(P1||P2) at state 1 = 2*eps*ln((1+2eps)/(1-2eps)) ~ 8*eps^2.
Per-trajectory contribution (state 1 visited T/n steps on average):
  KL_traj ~ (T/n) * 8*eps^2.
Le Cam lower bound: N >= Omega(n / (T * eps^2)).

Usage
-----
python experiments/theory/lower_bound_simulation.py
"""

from __future__ import annotations

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environments.base import POMDPEnvironment, StepResult
from environments.layer1_discrete.kcycle import KCyclePOMDP
from quotient_learning.estimator import QuotientEstimator


# ---------------------------------------------------------------------------
# Clean Le Cam pair: i.i.d. binary channel POMDP
# ---------------------------------------------------------------------------

class BinaryChannelPOMDP(POMDPEnvironment):
    """n-class i.i.d. POMDP for the Le Cam lower bound construction.

    Hidden state s in {0, ..., n-1}.  Self-loop dynamics: state persists
    for the entire episode.  Start state drawn uniformly from {0, ..., n-1}.

    Observation alphabet: {0, 1} (binary, same for all states).
    Emission model:
      State s (even):  obs=0 w.p. 1/2+eps,  obs=1 w.p. 1/2-eps
      State s (odd):   obs=0 w.p. 1/2-eps,  obs=1 w.p. 1/2+eps

    For even/odd pairs: TV(emission_even, emission_odd) = 2*eps.
    All states have distinct emissions (alternating direction).
    Canonical quotient: n classes.
    """

    def __init__(
        self,
        n: int,
        eps: float = 0.10,
        seed: int | None = None,
        randomise_start: bool = True,
    ):
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        self._n = n
        self._eps = eps
        self._rng = np.random.default_rng(seed)
        self._state: int = 0
        self._randomise_start = randomise_start
        # Probability of obs=0 for each state
        self._p0 = [0.5 + eps if s % 2 == 0 else 0.5 - eps for s in range(n)]

    @property
    def n_observations(self) -> int:
        return 2

    @property
    def n_actions(self) -> int:
        return 1

    @property
    def environment_id(self) -> str:
        return f"BinaryChannelPOMDP(n={self._n},eps={self._eps:.3f})"

    @property
    def metadata(self) -> dict[str, object]:
        return {"ground_truth_n_classes": self._n, "eps": self._eps}

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._randomise_start:
            self._state = int(self._rng.integers(0, self._n))
        else:
            self._state = 0
        return self._emit()

    def step(self, action: int) -> StepResult:  # noqa: ARG002
        obs = self._emit()
        return StepResult(observation=obs, reward=0.0, terminated=False, truncated=False)

    def _emit(self) -> int:
        return 0 if self._rng.random() < self._p0[self._state] else 1


class BinaryChannelMergedPOMDP(BinaryChannelPOMDP):
    """(n-1)-class version: states 0 (even) and 1 (odd) share state 0's emission.

    State 1's emission is changed from [1/2-eps, 1/2+eps] to [1/2+eps, 1/2-eps]
    (same as state 0).  States 0 and 1 now have identical continuation
    distributions (self-loop → emission determines future completely).
    Canonical quotient: n-1 classes.

    TV(P1_state1, P2_state1) = 2*eps exactly.
    KL(P1_state1 || P2_state1) = h_b(1/2-eps) - h_b(1/2+eps)... more precisely:
    KL = (1/2+eps)*ln((1/2+eps)/(1/2-eps)) + (1/2-eps)*ln((1/2-eps)/(1/2+eps))
       = 2*eps * ln((1+2eps)/(1-2eps))  ~  8*eps^2 for small eps.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override state 1 to use state 0's emission
        self._p0[1] = self._p0[0]  # state 1 now matches state 0


# ---------------------------------------------------------------------------
# KL / TV analysis
# ---------------------------------------------------------------------------

def binary_kl(p: float, q: float) -> float:
    """KL(Bernoulli(p) || Bernoulli(q))."""
    if p <= 0 or p >= 1 or q <= 0 or q >= 1:
        return float("inf")
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def per_step_kl_p1_to_p2(n: int, eps: float) -> float:
    """KL per step at state 1: P1_state1 (Bernoulli(1/2-eps)) || P2_state1 (Bernoulli(1/2+eps))."""
    p_obs0_p1 = 0.5 - eps   # state 1 in P1: minority on obs=0
    p_obs0_p2 = 0.5 + eps   # state 1 in P2: majority on obs=0 (merged with state 0)
    return binary_kl(p_obs0_p1, p_obs0_p2)


def tv_between_emissions(n: int, eps: float) -> float:
    """TV between P1_state1 and P2_state1 emissions (binary)."""
    return 2 * eps  # exactly 2*eps for binary flip


# ---------------------------------------------------------------------------
# Le Cam power measurement (direct likelihood ratio test)
# ---------------------------------------------------------------------------

def lecam_power_experiment(
    n_states: int,
    n_traj_list: list[int],
    T: int,
    eps: float,
    n_seeds: int,
    rng_base: int = 42,
) -> dict[str, object]:
    """Measure log-likelihood ratio test power as a function of N.

    For each N in n_traj_list, we:
    1. Draw N trajectories from P1 and N from P2.
    2. Compute the LLR test statistic.
    3. Record whether the test correctly identifies the source (power).

    The Le Cam lower bound says: for N <= Omega(n/(T*eps^2)), the power
    cannot exceed 3/4 (i.e., error >= 1/4).
    """
    p1_power = {n: [] for n in n_traj_list}
    p2_power = {n: [] for n in n_traj_list}

    kl_step = per_step_kl_p1_to_p2(n_states, eps)
    kl_traj = kl_step * T / n_states  # state 1 visited T/n steps per trajectory
    lecam_n_star = 0.5 / kl_traj if kl_traj > 0 else float("inf")

    rng = np.random.default_rng(rng_base)

    # Precompute emission probabilities for vectorized generation
    # P1 state 1: P(obs=0) = 1/2-eps
    # P2 state 1: P(obs=0) = 1/2+eps
    p_obs0_p1_s1 = 0.5 - eps
    p_obs0_p2_s1 = 0.5 + eps

    for seed_idx in range(n_seeds):
        seed = rng_base + seed_idx * 1000

        for n in n_traj_list:
            # Generate N trajectories from P1 (state 1 emits [1/2-eps, 1/2+eps])
            # and from P2 (state 1 emits [1/2+eps, 1/2-eps]).
            # For simplicity: generate T obs from state 1 only (worst case pair).
            # In each trajectory, state 1 is visited on average T/n_states steps.
            # For maximum power, we use the T/n_states steps from state 1.
            local_rng = np.random.default_rng(seed)

            n_state1_steps = max(1, int(round(n * T / n_states)))
            # P1 samples from state 1: Bernoulli(1/2-eps)
            obs_p1 = (local_rng.random(n_state1_steps) > p_obs0_p1_s1).astype(int)
            # P2 samples from state 1: Bernoulli(1/2+eps)
            obs_p2 = (local_rng.random(n_state1_steps) > p_obs0_p2_s1).astype(int)

            # LLR: given obs, decide P1 or P2 by sign of sum of log-odds
            # log-odds per step = log(P1(obs)) - log(P2(obs))
            def log_odds(obs_arr: np.ndarray) -> float:
                """Compute log P1/P2 ratio for the obs sequence."""
                # P1(obs=0) = 1/2-eps, P1(obs=1) = 1/2+eps
                # P2(obs=0) = 1/2+eps, P2(obs=1) = 1/2-eps
                logr = 0.0
                for o in obs_arr:
                    if o == 0:
                        logr += np.log((0.5 - eps) / (0.5 + eps))
                    else:
                        logr += np.log((0.5 + eps) / (0.5 - eps))
                return logr

            # Decision: positive LLR -> P1, negative -> P2
            llr_on_p1_data = log_odds(obs_p1)
            llr_on_p2_data = log_odds(obs_p2)

            # Power: P(correct decision)
            # On P1 data: correct if LLR > 0 (decide P1)
            p1_power[n].append(int(llr_on_p1_data > 0))
            # On P2 data: correct if LLR < 0 (decide P2)
            p2_power[n].append(int(llr_on_p2_data < 0))

    return {
        "n_states": n_states,
        "T": T,
        "eps": eps,
        "kl_step": kl_step,
        "kl_traj": kl_traj,
        "lecam_n_star": lecam_n_star,
        "fano_n_star": n_states / (T * (2 * eps) ** 2),
        "n_traj_list": n_traj_list,
        "p1_power": {n: float(np.mean(v)) for n, v in p1_power.items()},
        "p2_power": {n: float(np.mean(v)) for n, v in p2_power.items()},
    }


# ---------------------------------------------------------------------------
# QuotientEstimator validation on K-cycle family
# ---------------------------------------------------------------------------

def run_kcycle_recovery(
    k: int,
    n_traj_list: list[int],
    T: int,
    obs_accuracy: float,
    n_seeds: int,
    rng_base: int = 42,
) -> dict[str, object]:
    """Run QuotientEstimator on K-cycle with varying N.  Shows phase transition."""
    accuracy = {n: [] for n in n_traj_list}

    for seed_idx in range(n_seeds):
        seed = rng_base + seed_idx * 1000
        for n in n_traj_list:
            env = KCyclePOMDP(k=k, obs_accuracy=obs_accuracy, seed=seed + 1)
            est = QuotientEstimator(
                env=env,
                n_trajectories=n,
                window_length=2,
                history_max_length=1,
                min_history_occurrences=max(5, n // (20 * k)),
                alpha=0.05,
                max_steps_per_episode=T,
            )
            ds = est.collect(seed=seed + 1)
            q = est.estimate(ds)
            # K-cycle: estimator returns k+1 raw (k canonical + root class)
            accuracy[n].append(int(q.n_classes == k + 1))

    # Fano threshold for K-cycle recovery: K / (T * eps_min^2)
    # where eps_min is the min TV between adjacent cycle continuation distributions
    # Approximate: eps_min ~ (obs_accuracy - 1/k) * (k-1)/k (cylinder set TV)
    eps_min_approx = obs_accuracy - 1.0 / k
    fano_n_star = k / (T * eps_min_approx**2) if eps_min_approx > 0 else float("inf")

    return {
        "k": k,
        "T": T,
        "obs_accuracy": obs_accuracy,
        "eps_min_approx": eps_min_approx,
        "fano_n_star": fano_n_star,
        "n_traj_list": n_traj_list,
        "accuracy": {n: float(np.mean(v)) for n, v in accuracy.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Lower-bound simulation: N = Omega(n / (T * eps^2))")
    print("=" * 70)

    # ---- Part 1: Le Cam power analysis ----
    # Use small eps so the Le Cam threshold is visible in N_LIST.
    # With eps=0.01: KL_step ~ 8*(0.01)^2 = 0.0008
    #                KL_traj ~ (T/n) * 0.0008 = (20/4)*0.0008 = 0.004
    #                Le Cam N* = 1/(2*0.004) = 125
    # => phase transition near N=125.
    EPS = 0.01          # TV = 2*eps = 0.02
    N_STATES = 4        # n = 4 canonical classes (P1), n-1=3 (P2)
    T = 20              # episode length
    N_LIST_LECAM = [25, 50, 100, 200, 500, 1000, 2000]
    N_SEEDS = 200       # many seeds for stable power estimates

    print(f"\n--- Part 1: Le Cam power (binary channel, n={N_STATES}, T={T}, eps={EPS}) ---")
    lecam = lecam_power_experiment(
        n_states=N_STATES, n_traj_list=N_LIST_LECAM, T=T, eps=EPS, n_seeds=N_SEEDS
    )

    kl_step = lecam["kl_step"]
    kl_traj = lecam["kl_traj"]
    n_star_lecam = lecam["lecam_n_star"]
    n_star_fano = lecam["fano_n_star"]

    print(f"TV between P1 and P2 at state 1:        {tv_between_emissions(N_STATES, EPS):.4f}")
    print(f"KL per step (state 1):                  {kl_step:.6f}")
    print(f"KL per trajectory (T/n * KL_step):       {kl_traj:.6f}")
    print(f"Le Cam threshold N* = 1/(2*KL_traj):    {n_star_lecam:.1f}")
    print(f"Fano-form threshold n/(T*(2eps)^2):      {n_star_fano:.1f}")
    print()
    print(f"{'N':>6}  {'P1 power':>10}  {'P2 power':>10}  {'minimax error':>14}")
    print("-" * 48)
    for n in N_LIST_LECAM:
        p1 = lecam["p1_power"][n]
        p2 = lecam["p2_power"][n]
        # Minimax error = 1 - max_power (worst case over P1/P2)
        minimax_err = 1.0 - max(p1, p2)
        tag = " <- error >= 1/4" if minimax_err >= 0.25 else ""
        print(f"{n:>6}  {p1:>10.3f}  {p2:>10.3f}  {minimax_err:>14.3f}{tag}")

    # ---- Part 2: K-cycle QuotientEstimator (moderate obs_accuracy for visibility) ----
    K_CYCLE_ACC = 0.60  # eps_min approx 0.60 - 1/4 = 0.35; Fano N* ~ 4/(20*0.35^2) ~ 1.6
    # Use smaller accuracy to push the threshold to a visible N range
    K_CYCLE_ACC2 = 0.52  # near-uniform; eps_min approx 0.52 - 0.25 = 0.27; harder
    print(f"\n--- Part 2: K-cycle QuotientEstimator (K=4, T={T}) ---")
    N_LIST_KCYCLE = [50, 100, 200, 500, 1000]
    kcycle = run_kcycle_recovery(
        k=4, n_traj_list=N_LIST_KCYCLE, T=T, obs_accuracy=0.85, n_seeds=20
    )
    print(f"Fano threshold approx: {kcycle['fano_n_star']:.1f}")
    print()
    print(f"{'N':>6}  {'accuracy':>10}")
    print("-" * 20)
    for n in N_LIST_KCYCLE:
        acc = kcycle["accuracy"][n]
        print(f"{n:>6}  {acc:>10.2f}")

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Plot 1: Le Cam power vs N
    p1_powers = [lecam["p1_power"][n] for n in N_LIST_LECAM]
    p2_powers = [lecam["p2_power"][n] for n in N_LIST_LECAM]
    minimax_errs = [1.0 - max(lecam["p1_power"][n], lecam["p2_power"][n]) for n in N_LIST_LECAM]

    ax = axes[0]
    ax.plot(N_LIST_LECAM, p1_powers, "o-b", label="LLR power on P1")
    ax.plot(N_LIST_LECAM, p2_powers, "s--r", label="LLR power on P2")
    ax.plot(N_LIST_LECAM, minimax_errs, "^:k", label="Minimax error")
    ax.axhline(0.25, color="red", linestyle="--", linewidth=1.5, label="Le Cam bound (error ≥ 1/4)")
    ax.axvline(n_star_lecam, color="orange", linestyle="-.", label=f"N*≈{n_star_lecam:.0f} (Le Cam)")
    ax.axvline(n_star_fano, color="gray", linestyle=":", label=f"N*≈{n_star_fano:.0f} (Fano form)")
    ax.set_xlabel("N (number of trajectories)")
    ax.set_ylabel("Power / error probability")
    ax.set_title(f"Le Cam two-point: n={N_STATES}, T={T}, ε={EPS}")
    ax.legend(fontsize=7)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)

    # Plot 2: K-cycle recovery threshold
    ax = axes[1]
    accs = [kcycle["accuracy"][n] for n in N_LIST_KCYCLE]
    ax.plot(N_LIST_KCYCLE, accs, "o-b", label="Recovery accuracy")
    ax.axvline(kcycle["fano_n_star"], color="gray", linestyle=":", label=f"Fano N*≈{kcycle['fano_n_star']:.0f}")
    ax.set_xlabel("N (number of trajectories)")
    ax.set_ylabel("Recovery accuracy")
    ax.set_title(f"QuotientEstimator: K-cycle (K=4, acc=0.85, T={T})")
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    out_dir = os.path.join(ROOT, "experiments", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lower_bound_phase_transition.pdf")
    fig.savefig(out_path)
    print(f"\nFigure saved to {out_path}")

    # ---- Verification ----
    print("\n--- Verification ---")
    # Find smallest N where minimax error drops below 0.25
    n_star_empirical = next(
        (n for n in N_LIST_LECAM if (1.0 - max(lecam["p1_power"][n], lecam["p2_power"][n])) < 0.25),
        "not reached"
    )
    print(f"Le Cam N* (theory):   {n_star_lecam:.1f}")
    print(f"Fano N* (theory):     {n_star_fano:.1f}")
    print(f"Empirical N* (LLR error < 0.25 first at): {n_star_empirical}")
    print(
        f"Theorem 4 conclusion: any algorithm requires N >= Omega(n/(T*eps^2)) ="
        f" Omega({N_STATES}/({T}*{2*EPS:.2f}^2)) = {N_STATES/(T*(2*EPS)**2):.1f} trajectories."
    )
    lecam_kcycle_threshold = next(
        (n for n in N_LIST_KCYCLE if kcycle["accuracy"][n] >= 0.8), "not reached"
    )
    print(f"K-cycle recovery: 80% accuracy first reached at N = {lecam_kcycle_threshold}")
    print(f"Fano threshold for K-cycle: {kcycle['fano_n_star']:.1f}")


if __name__ == "__main__":
    main()
