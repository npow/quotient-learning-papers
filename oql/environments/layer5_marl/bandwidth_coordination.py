"""BandwidthCoordination Dec-POMDP — parameterized coordination task (§5.6).

Two agents share a Dec-POMDP in which the optimal joint action depends on a
private signal observed only by agent A.  Agent B must take a conditioned
action based on a *bandwidth-limited* message from A.

Environment semantics
---------------------
Hidden state  s ∈ {0, …, K-1}: the "signal" drawn uniformly at episode start.
Agent A observes s directly (noiseless) and can send a quantized message m to B.
Agent B observes only the quantized message (no direct signal access).

The message channel has capacity C bits per timestep, meaning B can receive at
most 2^C distinct symbols per step.  We model the channel as a deterministic
quantizer Q_C that maps A's quotient class to one of ⌊2^C⌋ channel symbols.
When C ≥ log2(K) the quantizer is lossless; when C < log2(K) it is lossy and
coordination suffers.

Reward
------
B must pick an action a_B ∈ {0, …, K-1}.  Reward = +1 if a_B == s, else 0.
The optimal policy for A: encode s faithfully; for B: decode and act.

This environment is used to measure whether R_crit predicts the bandwidth
threshold at which empirical coordination success transitions from ~1/K (random
guessing) to ~1 (perfect coordination).

Quotient structure
------------------
- Agent A: the canonical quotient has K classes (one per signal value), since
  each signal value induces a unique continuation distribution (always correct
  action is always the same within a class, distinct across classes).
- Agent B: same structure — its "observation" is the channel symbol, so if C
  bits are available and A encodes faithfully, B's quotient has min(2^C, K)
  classes.

Analytical R_crit
-----------------
With uniform signal distribution and no observation noise:
  H(Q_A) = log2(K)   (uniform over K classes)
  H(Q_B) = log2(min(2^C, K))
  H(Q_A | Q_B) = log2(K) - log2(min(2^C, K)) = max(0, log2(K) - C)

So R_crit = max(0, log2(K) - C).  The coordination phase transition occurs
at C = log2(K), i.e. when the channel is wide enough to losslessly transmit A's
signal.  For C ≥ log2(K), R_crit = 0 and success = 1.
For C < log2(K), the residual uncertainty log2(K) - C leads to errors.

Usage
-----
This module exposes:
  - ``BandwidthCoordinationSampler``: generates paired (A, B) trajectories for a
    given agent capacity K and channel capacity C.
  - ``analytical_r_crit(K, C)``: the closed-form R_crit for this environment.
  - ``analytical_success_rate(K, C)``: the expected success rate under the
    optimal quantization policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quotient_learning.dataset import Trajectory, TrajectoryDataset


@dataclass
class CoordinationConfig:
    """Configuration for the bandwidth coordination Dec-POMDP."""
    K: int          # Number of distinct signals / quotient classes for agent A
    C: float        # Channel capacity in bits per timestep
    n_steps: int    # Episode length (number of coordination steps)
    seed: int


def analytical_r_crit(K: int, C: float) -> float:
    """Analytical R_crit = max(0, log2(K) - C) for the coordination Dec-POMDP.

    Derivation: H(Q_A) = log2(K) (uniform signal), H(Q_B | Q_A) = 0 (A encodes
    deterministically), so H(Q_A | Q_B) = H(Q_A) - MI(Q_A; Q_B).
    With optimal quantization at C bits: MI = min(C, log2(K)), giving
    H(Q_A | Q_B) = max(0, log2(K) - C).
    """
    if K <= 1:
        return 0.0
    return max(0.0, math.log2(K) - C)


def analytical_success_rate(K: int, C: float) -> float:
    """Expected coordination success rate under optimal quantization policy.

    With C bits, agent A can distinguish min(2^C, K) symbols.  When 2^C >= K,
    the signal is transmitted losslessly and success = 1.  Otherwise, the K
    signals are partitioned into 2^C buckets of size K / 2^C; within each
    bucket B guesses uniformly, giving success = 1/bucket_size = 2^C / K.
    """
    if K <= 1:
        return 1.0
    n_symbols = min(2 ** C, float(K))
    if n_symbols >= K:
        return 1.0
    return n_symbols / K


class BandwidthCoordinationSampler:
    """Generate paired trajectories for the bandwidth coordination Dec-POMDP.

    Agent A observes the hidden signal directly.  Agent A's message to B is
    quantized through a C-bit channel using uniform quantization: signal values
    {0, …, K-1} are grouped into ceil(2^C) bins; A sends the bin index; B
    decodes the bin center as its signal estimate.

    Parameters
    ----------
    K : int
        Number of distinct signal values (agent A's quotient size).
    C : float
        Channel capacity in bits per timestep.
    n_steps : int
        Trajectory length (number of episodes steps per trajectory).
    seed : int
        Base seed for reproducibility.
    """

    def __init__(self, K: int, C: float, n_steps: int = 1, seed: int = 0) -> None:
        if K < 2:
            raise ValueError(f"K must be >= 2, got {K}")
        if C < 0:
            raise ValueError(f"C must be >= 0, got {C}")
        self.K = K
        self.C = C
        self.n_steps = n_steps
        self.seed = seed
        # Number of channel symbols available (capped at K to avoid over-engineering).
        self.n_symbols: int = min(int(2 ** C), K)
        # Quantization: assign each signal to a channel symbol uniformly.
        # Signals 0..K-1 → symbols 0..n_symbols-1 via floor(signal * n_symbols / K).
        self._signal_to_symbol: list[int] = [
            min(int(s * self.n_symbols / K), self.n_symbols - 1)
            for s in range(K)
        ]
        # B's decoding: given symbol m, pick the signal in that bucket.
        # Bucket for symbol m contains signals in [m*K/n_symbols, (m+1)*K/n_symbols).
        # B's optimal action is the bucket-floor signal (or any representative).
        self._symbol_to_action: list[int] = [
            int(m * K / self.n_symbols)
            for m in range(self.n_symbols)
        ]

    def _make_obs_a(self, signal: int) -> int:
        """Agent A's observation = the signal directly."""
        return signal

    def _make_obs_b(self, signal: int) -> int:
        """Agent B's observation = the channel symbol (quantized signal)."""
        return self._signal_to_symbol[signal]

    def _b_action(self, obs_b: int) -> int:
        """B's optimal action given its observation (channel symbol)."""
        return self._symbol_to_action[obs_b]

    def sample_episode(self, episode_seed: int) -> tuple[Trajectory, Trajectory, bool]:
        """Sample one paired episode.

        Returns
        -------
        (traj_a, traj_b, success)
            success = True iff B's action equals the hidden signal.
        """
        rng = np.random.default_rng(episode_seed)
        signal = int(rng.integers(0, self.K))

        obs_a_seq: list[int] = []
        obs_b_seq: list[int] = []
        acts_a: list[int] = []
        acts_b: list[int] = []
        rews_a: list[float] = []
        rews_b: list[float] = []

        for _ in range(self.n_steps):
            obs_a = self._make_obs_a(signal)
            obs_b = self._make_obs_b(signal)
            obs_a_seq.append(obs_a)
            obs_b_seq.append(obs_b)
            # A always sends the correct channel symbol (action = channel symbol index).
            act_a = self._signal_to_symbol[signal]
            # B decodes and acts.
            act_b = self._b_action(obs_b)
            acts_a.append(act_a)
            acts_b.append(act_b)
            reward = 1.0 if act_b == signal else 0.0
            rews_a.append(reward)
            rews_b.append(reward)

        # Trajectories need one more observation than actions.
        obs_a_seq.append(self._make_obs_a(signal))
        obs_b_seq.append(self._make_obs_b(signal))

        traj_a = Trajectory(
            observations=tuple(obs_a_seq),
            actions=tuple(acts_a),
            rewards=tuple(rews_a),
            seed=episode_seed,
        )
        traj_b = Trajectory(
            observations=tuple(obs_b_seq),
            actions=tuple(acts_b),
            rewards=tuple(rews_b),
            seed=episode_seed,
        )
        success = acts_b[-1] == signal if acts_b else False
        return traj_a, traj_b, success

    def sample_datasets(
        self,
        n_trajectories: int,
        seed: int = 0,
    ) -> tuple[TrajectoryDataset, TrajectoryDataset, float]:
        """Sample n_trajectories paired episodes.

        Returns
        -------
        (dataset_a, dataset_b, empirical_success_rate)
        """
        trajs_a: list[Trajectory] = []
        trajs_b: list[Trajectory] = []
        n_success = 0
        for ep in range(n_trajectories):
            ep_seed = int(seed) * 100003 + ep
            traj_a, traj_b, success = self.sample_episode(ep_seed)
            trajs_a.append(traj_a)
            trajs_b.append(traj_b)
            if success:
                n_success += 1

        env_id = f"BandwidthCoord(K={self.K},C={self.C:.3f})"
        dataset_a = TrajectoryDataset(
            trajectories=tuple(trajs_a),
            environment_id=env_id,
            policy_id="optimal_encode",
            extra={
                "agent": "A",
                "K": self.K,
                "C": self.C,
                "n_symbols": self.n_symbols,
                "seed": seed,
            },
        )
        dataset_b = TrajectoryDataset(
            trajectories=tuple(trajs_b),
            environment_id=env_id,
            policy_id="optimal_decode",
            extra={
                "agent": "B",
                "K": self.K,
                "C": self.C,
                "n_symbols": self.n_symbols,
                "seed": seed,
            },
        )
        empirical_success_rate = n_success / max(n_trajectories, 1)
        return dataset_a, dataset_b, empirical_success_rate
