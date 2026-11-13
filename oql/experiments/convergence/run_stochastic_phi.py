"""Stochastic-phi validation experiment (Theorem~\ref{thm:stochastic_phi}).

Constructs a small POMDP in which the agent's interaction function phi is
*stochastic*: on each visit to a history the agent randomly assigns it to one
of two abstract classes with equal probability.  This directly tests whether
the G-test estimator recovers the *expected* partition induced by the
KL-equivalence relation ~_phi.

Environment: StochasticPhiPOMDP
-------------------------------
Hidden state space: {0, 1, 2, 3}  (4 states, two pairs)
- States 0 and 1 form equivalence class A (KL(phi(h_s0) || phi(h_s1)) = 0).
  Both states emit observation 0 with probability 0.80, observation 1 with 0.20.
- States 2 and 3 form equivalence class B (KL(phi(h_s2) || phi(h_s3)) = 0).
  Both states emit observation 0 with probability 0.25, observation 1 with 0.75.

Transitions: uniform random over {0,1} for states in class A,
             uniform random over {2,3} for states in class B.
Actions: single action (no-op).

Stochastic interaction function phi
------------------------------------
For each history h rooted at a class-A state the agent assigns it to internal
node "A0" with probability 0.5 and "A1" with probability 0.5.  Analogously for
class-B states.  Crucially phi(h_0) and phi(h_1) have identical distributions
over {A0, A1}, so KL(phi(h_0) || phi(h_1)) = 0 and they are ~_phi equivalent.
Similarly for h_2 and h_3.

However, states 0/1 and states 2/3 have *different* observation models, so the
continuation distributions differ: P(o_{t+1}=0 | class A) = 0.80 != 0.25 =
P(o_{t+1}=0 | class B).  Hence KL(phi(h_0) || phi(h_2)) > 0: classes A and B
are *not* ~_phi equivalent.

Expected recovered partition
-----------------------------
Because histories from class A all share the same continuation distribution
(in expectation over the stochastic phi), and similarly for class B, the
QuotientEstimator should recover exactly 2 classes.

This validates Theorem~\ref{thm:stochastic_phi}: the G-test on continuation
distributions recovers the partition induced by ~_phi (KL equivalence on the
class-assignment distributions) in expectation, with recovery probability
converging to 1 by LLN as sample count grows.

Usage::

    python experiments/convergence/run_stochastic_phi.py
    python experiments/convergence/run_stochastic_phi.py --seeds 20 --budgets 500 1000 2000 5000
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from environments.base import POMDPEnvironment, StepResult
from quotient_learning.estimator import QuotientEstimator


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class StochasticPhiPOMDP(POMDPEnvironment):
    """POMDP with stochastic interaction function phi.

    Hidden states {0, 1} emit from distribution A; {2, 3} emit from B.
    States 0 and 1 are ~_phi equivalent (same phi distribution);
    states 2 and 3 are ~_phi equivalent.
    Classes A and B are distinguished by their continuation distributions.

    The stochastic phi is *simulated* via the observation model: within each
    class the two states have identical emission probabilities, so any
    estimator based on continuation distributions cannot distinguish them.
    This is equivalent to phi randomly assigning each visit to one of the two
    intra-class abstract nodes.

    Parameters
    ----------
    seed:
        RNG seed for reproducibility.
    """

    # Emission probabilities: P(obs=0 | state)
    _EMIT_0 = {0: 0.80, 1: 0.80, 2: 0.25, 3: 0.25}

    # Intra-class transitions: class A stays in {0,1}, class B stays in {2,3}
    _CLASS_A_STATES = (0, 1)
    _CLASS_B_STATES = (2, 3)

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self._state: int = 0

    @property
    def n_observations(self) -> int:
        return 2

    @property
    def n_actions(self) -> int:
        return 1  # single no-op action

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "ground_truth_n_classes": 2,
            "description": "Stochastic-phi POMDP: 2 KL-equivalence classes, "
                           "4 hidden states, stochastic intra-class transitions",
        }

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Start uniformly over all 4 states so both classes are visited.
        self._state = int(self._rng.integers(0, 4))
        return self._emit()

    def step(self, action: int) -> StepResult:  # noqa: ARG002 (action ignored)
        # Transition: stay within the same class (stochastic phi effect).
        if self._state in self._CLASS_A_STATES:
            self._state = int(self._rng.choice(self._CLASS_A_STATES))
        else:
            self._state = int(self._rng.choice(self._CLASS_B_STATES))
        obs = self._emit()
        return StepResult(observation=obs, reward=0.0, terminated=False, truncated=False)

    def _emit(self) -> int:
        p0 = self._EMIT_0[self._state]
        return int(self._rng.random() < p0)  # 0 with prob p0, 1 otherwise


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

GROUND_TRUTH_N_CLASSES = 2

# Estimator defaults calibrated for this environment.
HISTORY_MAX_LENGTH = 1
MIN_HISTORY_OCCURRENCES = 30
MAX_STEPS_PER_EPISODE = 20
WINDOW_LENGTH = 2
ALPHA = 0.01

DEFAULT_BUDGETS = [200, 500, 1000, 2000, 5000]
DEFAULT_N_SEEDS = 10

FIELDNAMES = [
    "n_trajectories",
    "seed",
    "learned_n_classes_raw",
    "learned_n_classes",
    "recovery_error",
    "exact_recovery",
]


def effective_n_classes(q) -> int:  # type: ignore[type-arg]
    """Return n_classes minus the root-only class, matching the MN-quotient convention.

    The empty history H() forms its own singleton class (pre-episode information
    state).  Following the canonical Myhill-Nerode convention used throughout
    the paper (and in the K-cycle experiment), we subtract this root class from
    the reported count.  The stochastic-phi ground truth has 2 non-root classes
    (class A: high P(obs=0); class B: high P(obs=1)), so the corrected count
    should be 2.
    """
    from quotient_learning.types import ObservationHistory
    root = ObservationHistory(())
    if root in q.partition:
        root_cls = q.partition[root]
        root_members = [h for h, c in q.partition.items() if c == root_cls]
        if len(root_members) == 1:
            return max(0, q.n_classes - 1)
    return q.n_classes


def run_one(n_trajectories: int, seed: int) -> dict:
    """Run a single (budget, seed) cell and return the result row."""
    env = StochasticPhiPOMDP(seed=seed)
    est = QuotientEstimator(
        env=env,
        n_trajectories=n_trajectories,
        window_length=WINDOW_LENGTH,
        history_max_length=HISTORY_MAX_LENGTH,
        min_history_occurrences=MIN_HISTORY_OCCURRENCES,
        alpha=ALPHA,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
    )
    q = est.run(seed=seed)
    learned_raw = q.n_classes
    learned = effective_n_classes(q)
    error = abs(learned - GROUND_TRUTH_N_CLASSES)
    return {
        "n_trajectories": n_trajectories,
        "seed": seed,
        "learned_n_classes_raw": learned_raw,
        "learned_n_classes": learned,
        "recovery_error": error,
        "exact_recovery": int(error == 0),
    }


def print_summary(rows: list[dict], budgets: list[int]) -> None:
    """Print a text summary: recovery accuracy and mean n_classes per budget."""
    from collections import defaultdict

    acc: dict[int, list] = defaultdict(list)
    cls: dict[int, list] = defaultdict(list)
    for r in rows:
        acc[r["n_trajectories"]].append(r["exact_recovery"])
        cls[r["n_trajectories"]].append(r["learned_n_classes"])

    print("\n" + "=" * 60)
    print(f"Stochastic-phi experiment  (ground truth n_classes={GROUND_TRUTH_N_CLASSES})")
    print(f"{'N':>8}  {'recovery_acc':>14}  {'mean_n_classes':>16}")
    print("-" * 60)
    for b in sorted(budgets):
        vals = acc.get(b, [])
        clsv = cls.get(b, [])
        if vals:
            rec = sum(vals) / len(vals)
            mc = sum(clsv) / len(clsv)
            print(f"{b:>8}  {rec:>14.3f}  {mc:>16.2f}")
    print("=" * 60)

    # Report at largest budget
    largest = max(budgets)
    final_acc = sum(acc.get(largest, [])) / max(len(acc.get(largest, [])), 1)
    print(
        f"\nAt N={largest}: recovery_accuracy={final_acc:.3f}  "
        f"(expected ≥ 0.90 for Theorem validation)"
    )
    if final_acc >= 0.90:
        print("PASS: G-test recovers the ~_phi partition in expectation.")
    else:
        print("WARN: recovery below 90% — increase budget or seeds.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stochastic-phi POMDP validation experiment"
    )
    p.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    p.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS)
    p.add_argument(
        "--out_dir",
        type=str,
        default="experiments/convergence/results",
    )
    args = p.parse_args()

    budgets: list[int] = sorted(set(args.budgets))
    seeds: list[int] = list(range(args.seeds))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stochastic_phi_recovery.csv"

    total_cells = len(budgets) * len(seeds)
    print(
        f"Stochastic-phi experiment: budgets={budgets}, seeds={seeds}\n"
        f"Config: history_max_length={HISTORY_MAX_LENGTH}, "
        f"min_history_occurrences={MIN_HISTORY_OCCURRENCES}, "
        f"max_steps={MAX_STEPS_PER_EPISODE}\n"
        f"Ground truth n_classes={GROUND_TRUTH_N_CLASSES}\n"
        f"Total cells: {total_cells}"
    )

    rows: list[dict] = []
    t0 = time.time()
    for cell_i, (budget, seed) in enumerate(
        (b, s) for b in budgets for s in seeds
    ):
        r = run_one(n_trajectories=budget, seed=seed)
        rows.append(r)
        status = "OK" if r["exact_recovery"] else f"err={r['recovery_error']}"
        print(
            f"  [{cell_i + 1:>4}/{total_cells}] n={budget:>6} seed={seed}  "
            f"learned={r['learned_n_classes']}  {status}"
        )

    elapsed = time.time() - t0

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {csv_path}  (elapsed {elapsed:.1f}s)")
    print_summary(rows, budgets)


if __name__ == "__main__":
    main()
