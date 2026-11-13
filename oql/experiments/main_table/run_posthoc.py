"""Post-hoc ground truth + data-driven eps recipe (review item REV-3a).

Every published certified run sets eps below an ANALYTICALLY KNOWN gap on an
environment DESIGNED to have a known quotient.  This experiment leaves that
regime: the environment is a randomly parameterized HMM
(environments/layer1_discrete/posthoc_random.py) whose quotient at r=0, k=1
is a numerical accident of a documented Dirichlet draw, ground truth is
computed only POST HOC by brute force from the drawn matrices, and eps is
NOT supplied analytically -- it is chosen per seed by a data-driven recipe
that sees only the selection half.

DATA-DRIVEN EPS RECIPE (fixed before seeing any inference-half data)
--------------------------------------------------------------------
1. On the SELECTION half only, retain the candidate contexts exactly as the
   certified estimator does (endpoint r, lengths 1..L, >= q occurrences) and
   form each context's empirical continuation law from the same
   one-block-per-trajectory extraction the inference half will use.
2. Sort the pairwise empirical total-variation distances into a spectrum and
   prepend 0 (the anchor of the SAME cluster).
3. Find the largest gap between consecutive spectrum values whose lower edge
   lies below 0.5.  If that gap is smaller than GAMMA = 0.10 -- or any
   selected context has no selection-half continuation block -- the spectrum
   has no clean split and the procedure ABSTAINS ON ALL PAIRS (fallback:
   no certificate is attempted, reported as such).
4. Otherwise set eps_hat to the midpoint of the split gap and run the
   UNCHANGED certified procedure on the inference half: TV threshold
   eps_hat/2, Bonferroni delta/(2K^2), m*(eps_hat, alpha_0, beta, d) gate.

Because both the candidate set and eps_hat are functions of the selection
half alone, the inference half remains untouched until the certified tests
run, preserving the split logic of Theorem 2 (with eps selection-measurable
rather than fixed a priori).  The certificate is VALID only if
eps_hat <= the true minimum TV between distinct laws; this run validates
that condition post hoc against the brute-force truth and reports the
validity rate honestly -- an eps_hat above the true gap is a finding
(the uncertified regime), not a bug.

INSTANCE-SELECTION PROTOCOL (fixed before any estimator output; evaluated
on brute-force ground truth only): see posthoc_random.py.  Family
(n_states=3, n_obs=3, Dirichlet 0.3), scan from instance seed 1000, first
seed with exact min pairwise TV >= 0.15.  Result: seed 1000 (min TV 0.3872).

Usage:
    uv run python experiments/main_table/run_posthoc.py [--seeds 0-19]
        [--out experiments/main_table/results_posthoc]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from environments.layer1_discrete.posthoc_random import PosthocRandomHMM
from quotient_learning.collection import collect_trajectories, uniform_random_policy
from quotient_learning.config import CertifiedConfig, seed_everything
from quotient_learning.equivalence import TVThresholdEquivalenceTest
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.sampling import (
    enumerate_independent_histories,
    extract_independent_continuation_samples,
)
from quotient_learning.stats import required_samples

DELTA = 0.05        # family-wise failure budget
BETA = 0.05         # per-pair Type-II budget for the power gate
GAMMA = 0.10        # minimum spectrum gap accepted as a clean split
CEILING = 0.5       # split gaps must start below this TV value

# Instance-generation family and scan rule (see posthoc_random.py docstring).
N_STATES = 3
N_OBS = 3
CONCENTRATION = 0.3
SCAN_START = 1000
SCAN_LIMIT = 20
SCAN_MIN_TV = 0.15

# Estimand and budget.  r=0, k=1, L=1: length-one contexts at the episode
# start; d = 3.  Budget sizing (from the brute-force ground truth, computed
# BEFORE running the estimator): the recipe's typical eps_hat is about half
# the true min TV 0.3872, i.e. ~0.19, giving m* ~ 1920 at
# alpha_0 = delta/(2*3^2); the minimum context mass is 0.2426, so the
# inference half (N/2 = 15,000) yields >= ~3,600 blocks per context,
# ~1.9x the gate.
R_ENDPOINT = 0
K_WINDOW = 1
L_MAX = 1
Q_MIN_OCC = 100
N_TRAJECTORIES = 30_000
MAX_STEPS = 2
DEFAULT_SEEDS = 20

ENV_NAME = "posthoc_hmm"


# ----------------------------------------------------------------------
# Instance selection (documented scan; ground-truth quantities only)
# ----------------------------------------------------------------------

def select_instance_seed() -> int:
    """First instance seed in the documented scan whose exact min pairwise
    TV at r=0, k=1 is >= SCAN_MIN_TV.  Uses brute-force ground truth only."""
    for inst in range(SCAN_START, SCAN_START + SCAN_LIMIT):
        env = PosthocRandomHMM(N_STATES, N_OBS, instance_seed=inst,
                               concentration=CONCENTRATION)
        if env.min_pairwise_tv(R_ENDPOINT, K_WINDOW, L_MAX) >= SCAN_MIN_TV:
            return inst
    raise RuntimeError(
        f"no instance with min TV >= {SCAN_MIN_TV} in "
        f"[{SCAN_START}, {SCAN_START + SCAN_LIMIT})"
    )


# ----------------------------------------------------------------------
# The eps recipe (selection half only)
# ----------------------------------------------------------------------

def eps_hat_from_spectrum(
    sorted_tvs: list[float], gamma: float = GAMMA, ceiling: float = CEILING
) -> float | None:
    """Midpoint of the largest clean gap in the 0-prepended TV spectrum.

    Returns None (the documented ABSTAIN-ALL fallback) when no consecutive
    gap with lower edge below ``ceiling`` reaches ``gamma``.  Ties are broken
    toward the lowest spectrum position (first maximal gap).
    """
    if not sorted_tvs:
        return None
    spectrum = [0.0] + sorted(sorted_tvs)
    best_gap, best_i = -1.0, -1
    for i in range(len(spectrum) - 1):
        if spectrum[i] >= ceiling:
            break
        gap = spectrum[i + 1] - spectrum[i]
        if gap > best_gap + 1e-15:
            best_gap, best_i = gap, i
    if best_i < 0 or best_gap < gamma:
        return None
    return (spectrum[best_i] + spectrum[best_i + 1]) / 2.0


def selection_tv_spectrum(
    selection_data,
    contexts,
    window_length: int,
    context_end: int,
    alphabet,
) -> list[float] | None:
    """Sorted pairwise empirical TVs between selection-half context laws.

    Laws use the same one-block-per-trajectory extraction as inference,
    applied to the SELECTION half.  Returns None (abstain-all) if any
    selected context has no selection-half continuation block.
    """
    categories = sorted(alphabet.categories)
    laws = {}
    for h in contexts:
        cs = extract_independent_continuation_samples(
            selection_data, h, window_length, context_end
        )
        if cs.n_samples == 0:
            return None
        counts = Counter(cs.continuations)
        laws[h] = np.array([counts[c] / cs.n_samples for c in categories])
    tvs = [
        float(0.5 * np.abs(laws[a] - laws[b]).sum())
        for a, b in itertools.combinations(contexts, 2)
    ]
    return sorted(tvs)


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

def _partition_blocks(partition: dict) -> set[frozenset]:
    """Set-of-blocks view; keys may be ObservationHistory or plain tuples."""
    blocks: dict[int, set] = {}
    for key, cid in partition.items():
        obs = key.observations if hasattr(key, "observations") else tuple(key)
        blocks.setdefault(cid, set()).add(obs)
    return {frozenset(b) for b in blocks.values()}


def run_one(seed: int, instance_seed: int, truth: dict, out_dir: Path) -> dict:
    seed_everything(seed)
    env = PosthocRandomHMM(
        N_STATES, N_OBS, instance_seed=instance_seed,
        concentration=CONCENTRATION, seed=seed,
    )
    policy = uniform_random_policy(env.n_actions)
    dataset = collect_trajectories(
        env=env, policy=policy, n_episodes=N_TRAJECTORIES,
        max_steps=MAX_STEPS, seed=seed, policy_id="uniform_random",
    )
    # The estimator will re-split the same dataset identically (collection
    # order, same fraction): the recipe sees exactly its selection half.
    selection_data, inference_data = dataset.split(0.5)
    contexts = enumerate_independent_histories(
        selection_data, max_length=L_MAX, context_end=R_ENDPOINT,
        min_occurrences=Q_MIN_OCC, include_root=False,
    )
    contexts = sorted(
        (h for h in contexts if h.length > 0),
        key=lambda h: (h.length, h.observations),
    )
    alphabet = env.continuation_alphabet(K_WINDOW)
    spectrum = (
        selection_tv_spectrum(
            selection_data, contexts, K_WINDOW, R_ENDPOINT, alphabet
        )
        if len(contexts) >= 2 else None
    )
    eps_hat = eps_hat_from_spectrum(spectrum) if spectrum is not None else None

    true_min_tv = truth["min_pairwise_tv"]
    true_blocks = _partition_blocks(truth["partition"])
    row = {
        "env": ENV_NAME,
        "seed": seed,
        "expected": len(true_blocks),
        "true_min_tv": round(true_min_tv, 6),
    }
    if eps_hat is None:
        # Documented fallback: no clean split -> abstain on all pairs.
        row.update({
            "abstain_all": True, "eps_hat": "", "eps_valid": "",
            "m_star": "", "min_inference_count": "",
            "n_classes": 0, "recovered": False,
            "undetermined_rate": 1.0, "n_boundary_flags": 0,
        })
        return row

    est = QuotientEstimator(
        env=env,
        equivalence_test=TVThresholdEquivalenceTest(alphabet, eps_hat),
        delta=DELTA,
        epsilon_min=eps_hat,
        power_beta=BETA,
        certified_independent=True,
        selection_fraction=0.5,
        independent_context_end=R_ENDPOINT,
        include_root=False,
        n_trajectories=N_TRAJECTORIES,
        window_length=K_WINDOW,
        history_max_length=L_MAX,
        min_history_occurrences=Q_MIN_OCC,
        max_steps_per_episode=MAX_STEPS,
    )
    q = est.estimate(dataset)

    n_contexts = len(contexts)
    alpha0 = DELTA / (2.0 * n_contexts**2) if n_contexts >= 2 else DELTA
    m_star = required_samples(eps_hat, alpha0, BETA, alphabet.size)
    inference_counts = [
        extract_independent_continuation_samples(
            inference_data, h, K_WINDOW, R_ENDPOINT
        ).n_samples
        for h in contexts
    ]
    recovered = _partition_blocks(q.partition) == true_blocks
    row.update({
        "abstain_all": False,
        "eps_hat": round(eps_hat, 6),
        "eps_valid": bool(eps_hat <= true_min_tv),
        "m_star": m_star,
        "min_inference_count": min(inference_counts),
        "n_classes": q.n_classes,
        "recovered": recovered,
        "undetermined_rate": round(q.undetermined_rate, 4),
        "n_boundary_flags": len(q.boundary_flags),
    })

    try:
        recorded_out = out_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        recorded_out = out_dir.resolve()
    cfg = CertifiedConfig(
        k=K_WINDOW, tau=1, test="TV_FIXED_ALPHABET", alpha=DELTA,
        epsilon_min=eps_hat, beta=BETA, seeds=[seed],
        selection_fraction=0.5, context_end=R_ENDPOINT,
        policy_id="uniform_random",
        n_trajectories=N_TRAJECTORIES,
        history_max_length=L_MAX,
        min_history_occurrences=Q_MIN_OCC,
        max_steps_per_episode=MAX_STEPS,
        command=(
            "python experiments/main_table/run_posthoc.py "
            f"--seeds {seed} --out {recorded_out}"
        ),
    )
    cfg.capture_env_pins()
    cfg.save_sidecar(out_dir / f"{ENV_NAME}_seed{seed}.certified.json")
    return row


def write_ground_truth(env: PosthocRandomHMM, out_dir: Path) -> dict:
    contexts, tv, laws, masses = env.exact_tv_matrix(R_ENDPOINT, K_WINDOW, L_MAX)
    partition = env.true_partition(R_ENDPOINT, K_WINDOW, L_MAX)
    iu = np.triu_indices(len(contexts), 1)
    truth = {
        "min_pairwise_tv": float(tv[iu].min()),
        "partition": partition,
        "contexts": contexts,
    }
    doc = {
        "protocol": {
            "instance_family": {
                "n_states": N_STATES, "n_obs": N_OBS,
                "dirichlet_concentration": CONCENTRATION,
            },
            "scan_rule": (
                f"first instance seed in [{SCAN_START}, "
                f"{SCAN_START + SCAN_LIMIT}) whose exact min pairwise TV at "
                f"r={R_ENDPOINT}, k={K_WINDOW} is >= {SCAN_MIN_TV}; evaluated "
                "on brute-force ground truth only, fixed before any "
                "estimator ran (scan history in posthoc_random.py)"
            ),
            "instance_seed": env.instance_seed,
            "eps_recipe": (
                "selection half only: 0-prepended sorted pairwise empirical "
                "TV spectrum from one-block-per-trajectory laws; eps_hat = "
                "midpoint of the largest consecutive gap with lower edge "
                f"below {CEILING}; ABSTAIN-ALL if that gap < {GAMMA} or any "
                "context lacks a selection-half block"
            ),
        },
        "estimand": {
            "r": R_ENDPOINT, "k": K_WINDOW, "L": L_MAX,
            "policy_id": "uniform_random",
        },
        "matrices": {
            "pi0": env.pi0.tolist(),
            "A": env.A.tolist(),
            "B": env.B.tolist(),
        },
        "contexts": [list(c) for c in contexts],
        "context_masses": {str(c): masses[c] for c in contexts},
        "continuation_laws": {
            str(c): {str(w): p for w, p in laws[c].items()} for c in contexts
        },
        "tv_matrix": tv.tolist(),
        "min_pairwise_tv": truth["min_pairwise_tv"],
        "true_partition": {str(c): cid for c, cid in partition.items()},
        "true_n_classes": len(set(partition.values())),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ground_truth.json").write_text(json.dumps(doc, indent=2))
    return truth


def parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default=f"0-{DEFAULT_SEEDS - 1}")
    ap.add_argument(
        "--out", default=str(Path(__file__).parent / "results_posthoc")
    )
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    instance_seed = select_instance_seed()
    env0 = PosthocRandomHMM(
        N_STATES, N_OBS, instance_seed=instance_seed,
        concentration=CONCENTRATION,
    )
    truth = write_ground_truth(env0, out_dir)
    print(f"instance seed {instance_seed}: "
          f"true min TV {truth['min_pairwise_tv']:.4f}, "
          f"{len(set(truth['partition'].values()))} true classes")

    rows = []
    for seed in parse_seeds(args.seeds):
        row = run_one(seed, instance_seed, truth, out_dir)
        rows.append(row)
        print(f"  seed={seed} eps_hat={row['eps_hat']} "
              f"valid={row['eps_valid']} n={row['n_classes']} "
              f"recovered={row['recovered']} und={row['undetermined_rate']}",
              flush=True)

    fieldnames = list(rows[0].keys())
    with (out_dir / "posthoc_main.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    active = [r for r in rows if not r["abstain_all"]]
    eps_vals = [float(r["eps_hat"]) for r in active]
    n_valid = sum(1 for r in active if r["eps_valid"])
    n_rec = sum(1 for r in rows if r["recovered"])
    und = [float(r["undetermined_rate"]) for r in rows]
    lines = [
        "# Post-hoc ground truth + data-driven eps recipe",
        "",
        f"Instance: PosthocRandomHMM(n_states={N_STATES}, n_obs={N_OBS}, "
        f"concentration={CONCENTRATION}, instance_seed={instance_seed}); "
        f"estimand r={R_ENDPOINT}, k={K_WINDOW}, L={L_MAX}, "
        f"q={Q_MIN_OCC}, N={N_TRAJECTORIES}, 50/50 split, "
        f"delta={DELTA}, beta={BETA}.",
        "",
        f"Brute-force truth (ground_truth.json): "
        f"{len(set(truth['partition'].values()))} classes (all singleton), "
        f"min pairwise TV {truth['min_pairwise_tv']:.4f}.",
        "",
        "Recipe (selection half only, fixed before inference data): "
        "0-prepended sorted pairwise empirical TV spectrum; eps_hat = "
        f"midpoint of the largest gap with lower edge < {CEILING}; "
        f"ABSTAIN-ALL fallback if the gap is < {GAMMA}.",
        "",
        f"- seeds: {len(rows)}",
        f"- abstain-all (no clean split): {len(rows) - len(active)}",
        f"- eps_hat: mean {np.mean(eps_vals):.4f}, "
        f"min {min(eps_vals):.4f}, max {max(eps_vals):.4f}"
        if eps_vals else "- eps_hat: (none)",
        f"- validity rate (eps_hat <= true min TV): {n_valid}/{len(active)}"
        if active else "- validity rate: n/a",
        f"- recovery rate (partition == brute-force truth): "
        f"{n_rec}/{len(rows)}",
        f"- abstention rate: mean {np.mean(und):.4f}, max {max(und):.4f}",
        "",
        "| seed | eps_hat | valid | m* | min inf. count | n_classes | "
        "recovered | und. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['eps_hat']} | {r['eps_valid']} | "
            f"{r['m_star']} | {r['min_inference_count']} | {r['n_classes']} "
            f"| {r['recovered']} | {r['undetermined_rate']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_dir / 'posthoc_main.csv'}; "
          f"elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
