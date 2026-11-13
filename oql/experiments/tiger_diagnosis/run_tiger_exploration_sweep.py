"""Tiger exploration policy sweep — recover n=2 quotient classes.

Tests five families of exploration policies against the Tiger POMDP to
diagnose why uniform-random collapses to n=1, and to find configurations
that reliably recover the expected n=2.

Policies tested (all implemented inline; library unchanged):
  1. count_based_exploration_policy  (from library, varied params)
  2. listen_biased          – stationary p(listen) ∈ {0.7,0.8,0.9,0.95}
  3. listen_k_then_open     – K listens then force-open; repeat
  4. ucb1_exploration       – maximise count[a]^{-0.5}, break ties randomly
  5. thompson_sampling      – posterior over tiger-left, sample & act greedy

Protocol: 5 seeds × N=5000 trajectories × max_steps_per_episode=15.
Output:
  results/exploration_sweep.csv       – one row per (config, seed)
  results/exploration_sweep_summary.md – best / worst configs

Usage:
    cd /root/code/online-quotient-learning
    python experiments/tiger_diagnosis/run_tiger_exploration_sweep.py
"""

from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from environments.layer1_discrete.tiger import TigerPOMDP
from quotient_learning.collection import (
    collect_trajectories,
    count_based_exploration_policy,
)
from quotient_learning.equivalence import GTestEquivalenceTest
from quotient_learning.estimator import QuotientEstimator

# ── constants ────────────────────────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4]
N_TRAJECTORIES = 5000
MAX_STEPS = 15
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

Policy = Callable[[int, np.random.Generator], int]


# ── inline policy factories (never added to the library) ────────────────────

def listen_biased_policy(p_listen: float) -> Policy:
    """Stationary: listen with prob p_listen, open-left/right equally."""
    p_open = (1.0 - p_listen) / 2.0

    def _policy(_obs: int, rng: np.random.Generator) -> int:
        r = rng.random()
        if r < p_listen:
            return 0   # listen
        if r < p_listen + p_open:
            return 1   # open-left
        return 2       # open-right

    return _policy


def listen_k_then_open_policy(k: int) -> Policy:
    """Listen K times then force an open action; repeat each episode.

    State is reset at episode boundary by the caller (collect_trajectories
    resets per episode via ep_seed / env.reset; the counter here tracks
    steps within the current episode via the closure).

    Because collect_trajectories calls policy once per step and the
    episode boundary resets the env but NOT the policy closure, we track
    wall-clock step count mod (k+1) to approximate episode-local counting.
    The +1 accounts for the forced-open step itself.
    """
    _step: list[int] = [0]

    def _policy(_obs: int, rng: np.random.Generator) -> int:
        s = _step[0]
        _step[0] += 1
        # Every (k+1)-th step (0-indexed: step k) force an open action.
        if (s % (k + 1)) == k:
            return int(rng.integers(1, 3))  # 1=open-left, 2=open-right
        return 0  # listen

    # Expose reset hook so collect_trajectories can clear per-episode.
    _policy._step = _step  # type: ignore[attr-defined]
    return _policy


def ucb1_exploration_policy(n_actions: int) -> Policy:
    """UCB1-style: pick action a maximising count[a]^{-0.5}; uniform on ties."""
    counts: list[int] = [0] * n_actions

    def _policy(_obs: int, rng: np.random.Generator) -> int:
        # Unvisited actions have infinite bonus — prefer those.
        unvisited = [a for a in range(n_actions) if counts[a] == 0]
        if unvisited:
            action = int(rng.choice(unvisited))
        else:
            scores = [counts[a] ** -0.5 for a in range(n_actions)]
            max_s = max(scores)
            candidates = [a for a, s in enumerate(scores) if s == max_s]
            action = int(rng.choice(candidates))
        counts[action] += 1
        return action

    return _policy


def thompson_sampling_policy(listen_accuracy: float = 0.85) -> Policy:
    """Thompson sampling over tiger-left belief; act greedily on sample.

    Maintains a Beta(alpha_L, alpha_R) posterior over P(tiger-left).
    On 'listen' we update with the observation (noisy signal).
    After an open action terminates the episode, the belief resets to
    Beta(1,1) at the next reset.  The policy object exposes a _reset()
    hook that collect_trajectories calls at episode start via the
    _obs_window sentinel pattern — here we piggyback on the same interface.
    """
    # Beta distribution parameters: (a, b) = (#times evidence for left, #right)
    alpha_l: list[float] = [1.0]
    alpha_r: list[float] = [1.0]
    _prev_terminated: list[bool] = [False]

    def _policy(obs: int, rng: np.random.Generator) -> int:
        # If last action terminated episode, reset belief.
        if _prev_terminated[0]:
            alpha_l[0] = 1.0
            alpha_r[0] = 1.0
            _prev_terminated[0] = False

        # Sample tiger-left probability from posterior.
        p_left = rng.beta(alpha_l[0], alpha_r[0])

        if p_left > 0.5:
            # Tiger likely left: open right (safe) if confident, else listen.
            confidence = abs(p_left - 0.5) * 2  # 0..1
            if confidence > 0.7:
                _prev_terminated[0] = True
                return 2  # open-right (away from tiger-left)
            else:
                # Listen and update belief.
                if obs == 0:  # hear-left -> evidence for tiger-left
                    alpha_l[0] += listen_accuracy
                    alpha_r[0] += (1 - listen_accuracy)
                else:         # hear-right -> evidence for tiger-right
                    alpha_l[0] += (1 - listen_accuracy)
                    alpha_r[0] += listen_accuracy
                return 0  # listen
        else:
            confidence = abs(p_left - 0.5) * 2
            if confidence > 0.7:
                _prev_terminated[0] = True
                return 1  # open-left (away from tiger-right)
            else:
                if obs == 0:
                    alpha_l[0] += listen_accuracy
                    alpha_r[0] += (1 - listen_accuracy)
                else:
                    alpha_l[0] += (1 - listen_accuracy)
                    alpha_r[0] += listen_accuracy
                return 0

    # Expose _obs_window attribute so collect_trajectories resets per episode.
    # We repurpose the hook: clearing _obs_window triggers belief reset above.
    _obs_window: list[int] = []

    def _reset_wrapper(obs: int, rng: np.random.Generator) -> int:
        # When collect_trajectories clears _obs_window at episode start,
        # the list is empty; use that as signal to reset belief.
        if len(_obs_window) == 0:
            alpha_l[0] = 1.0
            alpha_r[0] = 1.0
            _prev_terminated[0] = False
        _obs_window.append(obs)
        return _policy(obs, rng)

    _reset_wrapper._obs_window = _obs_window  # type: ignore[attr-defined]
    return _reset_wrapper


# ── main sweep runner ────────────────────────────────────────────────────────

def run_config(
    config_name: str,
    policy_factory: Callable[[], Policy],
    history_max_length: int = 2,
    min_history_occurrences: int = 30,
    alpha: float = 0.01,
    window_length: int = 2,
) -> list[dict]:
    """Run 5 seeds and return one result-row dict per seed."""
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        env = TigerPOMDP()
        policy = policy_factory()
        estimator = QuotientEstimator(
            env=env,
            n_trajectories=N_TRAJECTORIES,
            window_length=window_length,
            history_max_length=history_max_length,
            min_history_occurrences=min_history_occurrences,
            equivalence_test=GTestEquivalenceTest(),
            alpha=alpha,
            max_steps_per_episode=MAX_STEPS,
            policy=policy,
            policy_id=config_name,
        )
        quotient = estimator.run(seed=seed)
        elapsed = time.time() - t0
        rows.append({
            "config_name": config_name,
            "seed": seed,
            "n_classes": quotient.n_classes,
            "wall_time_s": round(elapsed, 3),
        })
    return rows


def build_configs() -> list[tuple[str, Callable[[], Policy], dict]]:
    """Return list of (config_name, policy_factory_no_args, estimator_kwargs)."""
    configs: list[tuple[str, Callable[[], Policy], dict]] = []

    # ── 1. count_based_exploration_policy ─────────────────────────────────
    for hml in [1, 2, 3]:
        for bonus in [0.5, 1.0, 2.0, 5.0]:
            name = f"count_based_hml{hml}_bonus{bonus}"
            # Capture loop variables correctly.
            def _cbf(h=hml, b=bonus):
                return count_based_exploration_policy(
                    n_actions=3, history_max_length=h, exploration_bonus=b
                )
            configs.append((name, _cbf, {"history_max_length": hml}))

    # ── 2. Listen-biased stationary ───────────────────────────────────────
    for p in [0.7, 0.8, 0.9, 0.95]:
        name = f"listen_biased_p{int(p*100)}"
        def _lbf(p=p):
            return listen_biased_policy(p)
        configs.append((name, _lbf, {}))

    # ── 3. Listen-K-then-open ─────────────────────────────────────────────
    for k in [3, 5, 10]:
        name = f"listen_{k}_then_open"
        def _lkf(k=k):
            return listen_k_then_open_policy(k)
        configs.append((name, _lkf, {}))

    # ── 4. UCB1-style ─────────────────────────────────────────────────────
    name = "ucb1_exploration"
    def _ucb1f():
        return ucb1_exploration_policy(n_actions=3)
    configs.append((name, _ucb1f, {}))

    # ── 5. Thompson sampling on belief ────────────────────────────────────
    name = "thompson_sampling"
    def _tsf():
        return thompson_sampling_policy(listen_accuracy=0.85)
    configs.append((name, _tsf, {}))

    return configs


def summarise(all_rows: list[dict]) -> None:
    """Compute mean ± std per config and write markdown summary."""
    from collections import defaultdict

    by_config: dict[str, list[int]] = defaultdict(list)
    times: dict[str, list[float]] = defaultdict(list)
    for r in all_rows:
        by_config[r["config_name"]].append(r["n_classes"])
        times[r["config_name"]].append(r["wall_time_s"])

    stats = []
    for cname, vals in by_config.items():
        arr = np.array(vals, dtype=float)
        stats.append({
            "config_name": cname,
            "mean_n": float(arr.mean()),
            "std_n": float(arr.std()),
            "seeds_n2": int((arr == 2).sum()),
            "mean_time_s": float(np.mean(times[cname])),
        })

    stats.sort(key=lambda x: (-x["seeds_n2"], x["mean_n"]))

    winners = [s for s in stats if s["mean_n"] >= 1.8 and s["std_n"] <= 0.5]
    losers = [s for s in stats if s["mean_n"] <= 1.2]

    lines = [
        "# Tiger Exploration Sweep — Summary",
        "",
        f"Protocol: {len(SEEDS)} seeds × N={N_TRAJECTORIES} traj × max_steps={MAX_STEPS}",
        "Estimator: GTest, alpha=0.01, history_max_length=2 (default), window_length=2",
        "",
        "## All configs (sorted by seeds_n2 desc, mean_n asc)",
        "",
        f"| {'Config':<45} | mean n | std  | seeds n=2 | mean time(s) |",
        f"|{'-'*47}|--------|------|-----------|-------------|",
    ]
    for s in stats:
        lines.append(
            f"| {s['config_name']:<45} | {s['mean_n']:6.2f} | {s['std_n']:4.2f} "
            f"| {s['seeds_n2']:>9}/5 | {s['mean_time_s']:>11.1f} |"
        )

    lines += [
        "",
        "## Best configs (mean_n >= 1.8, std <= 0.5)",
        "",
    ]
    if winners:
        for s in winners:
            lines.append(
                f"- **{s['config_name']}**: mean={s['mean_n']:.2f} ± {s['std_n']:.2f}, "
                f"{s['seeds_n2']}/5 seeds hit n=2"
            )
    else:
        lines.append("*No config reliably recovered n=2 under these conditions.*")

    lines += [
        "",
        "## Worst configs (mean_n <= 1.2)",
        "",
    ]
    if losers:
        for s in losers:
            lines.append(
                f"- {s['config_name']}: mean={s['mean_n']:.2f} ± {s['std_n']:.2f}"
            )
    else:
        lines.append("*No config collapsed below mean=1.2.*")

    md_path = RESULTS_DIR / "exploration_sweep_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary written to {md_path}")

    # Print condensed table to stdout.
    print("\n" + "="*72)
    print(f"{'Config':<45}  mean_n   std  n=2/5")
    print("-"*72)
    for s in stats:
        marker = " ***" if s["seeds_n2"] >= 4 else ""
        print(f"{s['config_name']:<45}  {s['mean_n']:5.2f}  {s['std_n']:4.2f}  "
              f"{s['seeds_n2']}/5{marker}")

    if winners:
        print(f"\nWINNERS: {[s['config_name'] for s in winners]}")
    else:
        print("\nNo winner found: no config reliably recovers n=2.")


def main() -> None:
    configs = build_configs()
    print(f"Tiger exploration sweep: {len(configs)} configs × {len(SEEDS)} seeds")
    print(f"  N={N_TRAJECTORIES} trajectories, max_steps={MAX_STEPS}\n")

    all_rows: list[dict] = []
    csv_path = RESULTS_DIR / "exploration_sweep.csv"

    # Write header immediately so partial results are always readable.
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["config_name", "seed", "n_classes", "wall_time_s"]
        )
        writer.writeheader()

    for idx, (cname, pfactory, ekwargs) in enumerate(configs):
        hml = ekwargs.get("history_max_length", 2)
        print(f"[{idx+1}/{len(configs)}] {cname}  (hml={hml})", flush=True)
        rows = run_config(
            config_name=cname,
            policy_factory=pfactory,
            history_max_length=hml,
        )
        all_rows.extend(rows)
        nc = [r["n_classes"] for r in rows]
        print(f"  seeds n_classes: {nc}  mean={np.mean(nc):.2f}", flush=True)

        # Append rows to CSV.
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["config_name", "seed", "n_classes", "wall_time_s"]
            )
            writer.writerows(rows)

    print(f"\nCSV written to {csv_path} ({len(all_rows)} rows)")
    summarise(all_rows)


if __name__ == "__main__":
    main()
