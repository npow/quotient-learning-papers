"""Certified run under a genuinely stochastic, nondegenerate policy (REV-3c).

Every published run uses a single-action or constant policy (kcycle's one
action, Tiger's always-listen), so the paper's fixed-policy POMDP framing is
empirically unexercised beyond degenerate cases.  This experiment runs the
UNCHANGED certified procedure on Tiger under a memoryless stochastic policy

    listen with prob 0.8, open-left 0.1, open-right 0.1,

at r=1, k=1, observation-only continuations.  No environment or estimator
code changes: the estimator already accepts an explicit policy callable
(run_certified.py passes one for always_listen); this runner supplies a
stochastic one.

GROUND TRUTH (post hoc, brute force).  The closed-loop continuation laws are
computed EXACTLY by enumerating the latent chain under the policy
(``tiger_closed_loop_laws``): the tiger state is static within an episode;
opening a door emits a uniform terminal observation and ends the episode.
Termination-before-r+k conditioning is handled exactly as the estimand
defines it: a context contributes if and only if the trajectory reaches
observation index r+k, so a path that opens a door at any step before the
last one never enters the law (its episode ends before index r+k), while a
door opened at the LAST step does (its terminal observation IS O_{r+k}).
For r=1, k=1, accuracy a=0.85, p_listen=0.8 this yields, per context
mass mu = p_listen/2 = 0.4:

    P(O_2 = hear-left | O_1 = hear-left, reach 2)
        = p_listen * (a^2 + (1-a)^2) + (1-p_listen) * 1/2 = 0.696,

two singleton classes with true TV gap 0.8 * (2a-1)^2 = 0.392 (the
always-listen gap 0.49 shrunk by the probability the second action is a
listen).  eps is chosen post hoc BELOW that brute-force gap, mirroring the
analytic-gap protocol of run_certified.py: eps = 0.35 <= 0.392, checked at
runtime against the brute-force computation.

BUDGET.  d = 2, K = 2 contexts, alpha_0 = delta/(2K^2) = 0.00625, so
m*(0.35, alpha_0, 0.05, 2) = 468.  Context masses are 0.4 each (from the
brute-force computation), so the inference half N/2 = 4,000 yields ~1,600
blocks per context, ~3.4x the gate.

Usage:
    uv run python experiments/main_table/run_nondegenerate.py [--seeds 0-19]
        [--out experiments/main_table/results_nondegenerate]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np

from environments.layer1_discrete.tiger import ACT_LISTEN, TigerPOMDP
from quotient_learning.config import CertifiedConfig, seed_everything
from quotient_learning.equivalence import TVThresholdEquivalenceTest
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.sampling import extract_independent_continuation_samples
from quotient_learning.stats import required_samples

DELTA = 0.05
BETA = 0.05
EPS_MIN = 0.35              # post-hoc protocol: must be <= brute-force gap
ACTION_PROBS = (0.8, 0.1, 0.1)   # (listen, open-left, open-right)
POLICY_ID = "stochastic_listen80"
LISTEN_ACCURACY = 0.85

R_ENDPOINT = 1
K_WINDOW = 1
L_MAX = 1
Q_MIN_OCC = 10
N_TRAJECTORIES = 8_000
MAX_STEPS = 15
DEFAULT_SEEDS = 20

ENV_NAME = "tiger_stochastic"

_LAW_TOL = 1e-9


# ----------------------------------------------------------------------
# Brute-force closed-loop ground truth
# ----------------------------------------------------------------------

def stochastic_tiger_policy(action_probs=ACTION_PROBS):
    """Memoryless stochastic policy (obs, rng) -> action over Tiger's 3 actions."""
    cum = np.cumsum(np.asarray(action_probs, dtype=float))
    if abs(cum[-1] - 1.0) > 1e-12:
        raise ValueError(f"action_probs must sum to 1, got {action_probs}")

    def _policy(_obs: int, rng: np.random.Generator) -> int:
        return int(np.searchsorted(cum, rng.random(), side="right"))

    return _policy


def tiger_closed_loop_laws(
    listen_accuracy: float = LISTEN_ACCURACY,
    action_probs: tuple[float, float, float] = ACTION_PROBS,
    r: int = R_ENDPOINT,
    k: int = K_WINDOW,
):
    """Exact closed-loop continuation laws for length-one Tiger contexts.

    Brute force over the latent chain under the memoryless policy: enumerate
    the latent state (uniform over tiger-left/right; static within an
    episode), every action sequence, and every observation sequence up to
    index r + k.  Opening a door emits a uniform terminal observation and
    ends the episode, so paths that open before step r+k-1 never reach
    observation index r+k and are excluded from mass and law -- exactly the
    estimand's ``conditioned on reaching r+k`` event.  The initial
    observation O_0 is uninformative and marginalizes out (contexts here are
    the length-one (O_r) with L=1).

    Returns ``(laws, masses)``: for each context ``(o_r,)``,
    ``laws[(o_r,)]`` is a dict over length-k observation tuples summing to 1
    and ``masses[(o_r,)]`` is mu_h = P(O_r = o_r, Z reaches r+k) per episode.
    """
    if r < 1 or k < 1:
        raise ValueError(f"r and k must be >= 1, got r={r}, k={k}")
    p_listen, p_ol, p_or = (float(p) for p in action_probs)
    acc = float(listen_accuracy)
    n_steps = r + k  # actions 0..r+k-1 emit observations 1..r+k

    # joint[(o_r, continuation)] = P(context, continuation, reach r+k)
    joint: dict[tuple[int, tuple[int, ...]], float] = {}

    for state in (0, 1):  # tiger-left, tiger-right; uniform prior
        p_state = 0.5

        def obs_prob(obs: int, action: int) -> float:
            if action == ACT_LISTEN:
                correct = obs == state  # hear-left iff tiger-left, etc.
                return acc if correct else 1.0 - acc
            return 0.5  # door open: uniform terminal draw

        # Depth-first enumeration of (action, observation) paths.
        # Element: (step j, observations o_1..o_j, path probability).
        stack: list[tuple[int, tuple[int, ...], float]] = [(0, (), p_state)]
        while stack:
            j, obs_path, prob = stack.pop()
            if j == n_steps:
                context_obs = obs_path[r - 1]  # O_r (obs_path[i] is O_{i+1})
                cont = obs_path[r:]
                key = ((context_obs,), cont)
                joint[key] = joint.get(key, 0.0) + prob
                continue
            for action, p_act in ((0, p_listen), (1, p_ol), (2, p_or)):
                if p_act == 0.0:
                    continue
                terminates = action != ACT_LISTEN
                if terminates and j < n_steps - 1:
                    continue  # episode ends before reaching index r+k
                for obs in (0, 1):
                    p_obs = obs_prob(obs, action)
                    if p_obs == 0.0:
                        continue
                    stack.append((j + 1, obs_path + (obs,), prob * p_act * p_obs))

    contexts = sorted({ctx for ctx, _ in joint})
    words = list(itertools.product((0, 1), repeat=k))
    masses = {
        ctx: sum(joint.get((ctx, w), 0.0) for w in words) for ctx in contexts
    }
    laws = {}
    for ctx in contexts:
        if masses[ctx] <= 0.0:
            continue
        laws[ctx] = {w: joint.get((ctx, w), 0.0) / masses[ctx] for w in words}
    return laws, masses


def true_partition_and_gap(laws: dict) -> tuple[dict[tuple[int, ...], int], float]:
    """Group contexts by exact law equality; return (partition, min pairwise TV)."""
    contexts = sorted(laws)
    words = sorted(next(iter(laws.values())))
    partition: dict[tuple[int, ...], int] = {}
    reps: list[tuple[int, ...]] = []

    def tv(a, b) -> float:
        return 0.5 * sum(abs(laws[a][w] - laws[b][w]) for w in words)

    for c in contexts:
        for cid, rep in enumerate(reps):
            if tv(c, rep) < _LAW_TOL:
                partition[c] = cid
                break
        else:
            reps.append(c)
            partition[c] = len(reps) - 1
    gap = min(
        tv(a, b) for a, b in itertools.combinations(contexts, 2)
    ) if len(contexts) >= 2 else float("nan")
    return partition, gap


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

def _partition_blocks(partition: dict) -> set[frozenset]:
    blocks: dict[int, set] = {}
    for key, cid in partition.items():
        obs = key.observations if hasattr(key, "observations") else tuple(key)
        blocks.setdefault(cid, set()).add(obs)
    return {frozenset(b) for b in blocks.values()}


def run_one(seed: int, truth_blocks: set[frozenset], out_dir: Path) -> dict:
    seed_everything(seed)
    env = TigerPOMDP(seed=seed)
    alphabet = env.continuation_alphabet(K_WINDOW)
    est = QuotientEstimator(
        env=env,
        equivalence_test=TVThresholdEquivalenceTest(alphabet, EPS_MIN),
        delta=DELTA,
        epsilon_min=EPS_MIN,
        power_beta=BETA,
        certified_independent=True,
        selection_fraction=0.5,
        independent_context_end=R_ENDPOINT,
        include_root=False,
        policy=stochastic_tiger_policy(),
        policy_id=POLICY_ID,
        n_trajectories=N_TRAJECTORIES,
        window_length=K_WINDOW,
        history_max_length=L_MAX,
        min_history_occurrences=Q_MIN_OCC,
        max_steps_per_episode=MAX_STEPS,
    )
    dataset = est.collect(seed=seed)
    q = est.estimate(dataset)

    _, inference_data = dataset.split(0.5)
    contexts = sorted(q.partition.keys(), key=lambda h: h.observations)
    inference_counts = [
        extract_independent_continuation_samples(
            inference_data, h, K_WINDOW, R_ENDPOINT
        ).n_samples
        for h in contexts
    ] or [0]
    n_contexts = max(len(contexts), 2)
    alpha0 = DELTA / (2.0 * n_contexts**2)
    m_star = required_samples(EPS_MIN, alpha0, BETA, alphabet.size)
    recovered = _partition_blocks(q.partition) == truth_blocks

    row = {
        "env": ENV_NAME,
        "seed": seed,
        "n_classes": q.n_classes,
        "expected": len(truth_blocks),
        "eps_min": EPS_MIN,
        "m_star": m_star,
        "min_inference_count": min(inference_counts),
        "undetermined_rate": round(q.undetermined_rate, 4),
        "n_boundary_flags": len(q.boundary_flags),
        "recovered": recovered,
    }
    try:
        recorded_out = out_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        recorded_out = out_dir.resolve()
    cfg = CertifiedConfig(
        k=K_WINDOW, tau=1, test="TV_FIXED_ALPHABET", alpha=DELTA,
        epsilon_min=EPS_MIN, beta=BETA, seeds=[seed],
        selection_fraction=0.5, context_end=R_ENDPOINT,
        policy_id=POLICY_ID,
        n_trajectories=N_TRAJECTORIES,
        history_max_length=L_MAX,
        min_history_occurrences=Q_MIN_OCC,
        max_steps_per_episode=MAX_STEPS,
        command=(
            "python experiments/main_table/run_nondegenerate.py "
            f"--seeds {seed} --out {recorded_out}"
        ),
    )
    cfg.capture_env_pins()
    cfg.save_sidecar(out_dir / f"{ENV_NAME}_seed{seed}.certified.json")
    return row


def write_ground_truth(out_dir: Path) -> set[frozenset]:
    laws, masses = tiger_closed_loop_laws()
    partition, gap = true_partition_and_gap(laws)
    if EPS_MIN > gap:
        raise ValueError(
            f"eps_min={EPS_MIN} exceeds the brute-force closed-loop TV gap "
            f"{gap:.6f}; the certificate's gap model would be violated"
        )
    n_contexts = len(partition)
    alpha0 = DELTA / (2.0 * max(n_contexts, 2) ** 2)
    m_star = required_samples(EPS_MIN, alpha0, BETA, 2**K_WINDOW)
    doc = {
        "protocol": {
            "policy": {
                "id": POLICY_ID,
                "type": "memoryless stochastic",
                "action_probs": {
                    "listen": ACTION_PROBS[0],
                    "open_left": ACTION_PROBS[1],
                    "open_right": ACTION_PROBS[2],
                },
            },
            "eps_rule": (
                "post hoc, mirroring run_certified.py's analytic-gap "
                f"protocol: eps = {EPS_MIN} chosen <= the brute-force "
                f"closed-loop min TV gap {gap:.6f}; checked at runtime"
            ),
            "ground_truth": (
                "exact brute-force enumeration of the latent chain under "
                "the stochastic policy (tiger_closed_loop_laws); episodes "
                "that open a door before observation index r+k terminate "
                "early and are excluded exactly as the estimand's "
                "reach-conditioning prescribes"
            ),
        },
        "estimand": {
            "r": R_ENDPOINT, "k": K_WINDOW, "L": L_MAX,
            "listen_accuracy": LISTEN_ACCURACY,
        },
        "budget": {
            "n_trajectories": N_TRAJECTORIES,
            "m_star": m_star,
            "expected_inference_count_per_context":
                0.5 * N_TRAJECTORIES * min(masses.values()),
        },
        "context_masses": {str(c): m for c, m in masses.items()},
        "closed_loop_laws": {
            str(c): {str(w): p for w, p in laws[c].items()} for c in laws
        },
        "true_partition": {str(c): cid for c, cid in partition.items()},
        "true_n_classes": len(set(partition.values())),
        "true_min_tv": gap,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ground_truth.json").write_text(json.dumps(doc, indent=2))
    blocks: dict[int, set] = {}
    for c, cid in partition.items():
        blocks.setdefault(cid, set()).add(c)
    return {frozenset(b) for b in blocks.values()}


def parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default=f"0-{DEFAULT_SEEDS - 1}")
    ap.add_argument(
        "--out", default=str(Path(__file__).parent / "results_nondegenerate")
    )
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    truth_blocks = write_ground_truth(out_dir)
    laws, masses = tiger_closed_loop_laws()
    _, gap = true_partition_and_gap(laws)
    print(f"brute-force truth: {len(truth_blocks)} classes, "
          f"min TV gap {gap:.4f}, masses "
          f"{ {str(c): round(m, 3) for c, m in masses.items()} }; "
          f"eps={EPS_MIN}")

    rows = []
    for seed in parse_seeds(args.seeds):
        row = run_one(seed, truth_blocks, out_dir)
        rows.append(row)
        print(f"  seed={seed} n={row['n_classes']} "
              f"recovered={row['recovered']} und={row['undetermined_rate']} "
              f"min_count={row['min_inference_count']}", flush=True)

    with (out_dir / "nondegenerate_main.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)

    n_rec = sum(1 for r in rows if r["recovered"])
    und = [float(r["undetermined_rate"]) for r in rows]
    ns = [float(r["n_classes"]) for r in rows]
    lines = [
        "# Certified run under a nondegenerate stochastic policy",
        "",
        f"Tiger (accuracy {LISTEN_ACCURACY}), memoryless stochastic policy "
        f"listen/open-left/open-right = {ACTION_PROBS}; estimand r={R_ENDPOINT}, "
        f"k={K_WINDOW}, L={L_MAX}, q={Q_MIN_OCC}, N={N_TRAJECTORIES}, 50/50 "
        f"split, delta={DELTA}, beta={BETA}, eps={EPS_MIN} (brute-force "
        f"closed-loop gap {gap:.4f}; ground_truth.json).",
        "",
        f"- seeds: {len(rows)}",
        f"- recovery rate (partition == brute-force truth): "
        f"{n_rec}/{len(rows)}",
        f"- n_classes: mean {np.mean(ns):.1f} "
        f"(expected {len(truth_blocks)})",
        f"- abstention rate: mean {np.mean(und):.4f}, max {max(und):.4f}",
        f"- m* = {rows[0]['m_star']}, min inference count over seeds/contexts: "
        f"{min(r['min_inference_count'] for r in rows)}",
        "",
        "| seed | n_classes | recovered | und. | min inf. count |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['n_classes']} | {r['recovered']} | "
            f"{r['undetermined_rate']} | {r['min_inference_count']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_dir / 'nondegenerate_main.csv'}; "
          f"elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
