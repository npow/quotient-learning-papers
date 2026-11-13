"""Same-target CSSR comparison (review finding E4).

Runs QuotientEstimator (certified configuration) and the adapted CSSR
baseline on the IDENTICAL context set, continuation samples, test, and
per-pair level, then reports classes recovered and exact recovery against
ground truth where it is known.

Usage:
    python experiments/baselines/run_cssr_comparison.py \
        --envs tiger kcycle_8 noisy_depth2 --seeds 0-4
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from environments.layer1_discrete.kcycle import KCyclePOMDP
from environments.layer1_discrete.tiger import TigerPOMDP
from environments.layer2_memory.noisy_depth2_kcycle import NoisyDepth2KCycle
from experiments.baselines.cssr_baseline import run_baseline
from experiments.scoring import merge_result_rows, partitions_agree
from quotient_learning.config import CertifiedConfig, seed_everything
from quotient_learning.equivalence import TVThresholdEquivalenceTest
from quotient_learning.estimator import QuotientEstimator
from quotient_learning.sampling import (
    enumerate_independent_histories,
    extract_independent_continuation_samples,
)
from quotient_learning.stats import bootstrap_ci

DELTA = 0.05
BETA = 0.05
EPS_MIN = 0.5


def certified_tau(k: int, max_len: int) -> int:
    import math
    return max(1, math.ceil((max_len + k) / k))


CONFIGS: dict = {
    "tiger": (
        lambda seed: TigerPOMDP(seed=seed),
        dict(n_trajectories=20000, window_length=1, history_max_length=1,
             min_history_occurrences=10, max_steps_per_episode=15,
             independent_context_end=1, policy_id="always_listen")),
    "kcycle_8": (
        lambda seed: KCyclePOMDP(8, seed=seed),
        dict(n_trajectories=8000, window_length=1, history_max_length=1,
             min_history_occurrences=100, max_steps_per_episode=20)),
    "noisy_depth2": (
        lambda seed: NoisyDepth2KCycle(m=2, eta=0.1, seed=seed),
        dict(n_trajectories=16000, window_length=1, history_max_length=2,
             min_history_occurrences=30, max_steps_per_episode=60)),
}


def run_one(name: str, seed: int, out_dir: Path, calibration: str = "bonferroni") -> dict:
    ctor, kwargs = CONFIGS[name]
    kwargs = dict(kwargs)
    seed_everything(seed)
    env = ctor(seed)
    k = kwargs["window_length"]
    tau = certified_tau(k, kwargs["history_max_length"])
    context_end = kwargs.pop("independent_context_end", 0)
    policy_id = kwargs.pop("policy_id", "uniform_random")
    policy = (lambda _obs, _rng: 0) if policy_id == "always_listen" else None
    alphabet = env.continuation_alphabet(k)
    test = TVThresholdEquivalenceTest(alphabet, EPS_MIN)

    est = QuotientEstimator(
        env=env, equivalence_test=test, delta=DELTA,
        epsilon_min=EPS_MIN, power_beta=BETA,
        certified_independent=True, selection_fraction=0.5,
        independent_context_end=context_end, include_root=False, **kwargs,
        policy=policy, policy_id=policy_id,
    )
    dataset = est.collect(seed=seed)
    q = est.estimate(dataset)

    # Identical context set + samples for the baseline (same enumeration
    # thresholds, same stride, same dataset).
    selection_data, inference_data = dataset.split(0.5)
    contexts = enumerate_independent_histories(
        selection_data, max_length=kwargs["history_max_length"],
        context_end=context_end,
        min_occurrences=kwargs["min_history_occurrences"],
        include_root=False,
    )
    contexts = [h for h in contexts if h.length > 0]
    samples = {}
    for h in contexts:
        cs = extract_independent_continuation_samples(
            inference_data, h, k, context_end
        )
        if cs.n_samples >= max(30, kwargs["min_history_occurrences"] // 2):
            samples[h] = cs
    if calibration == "native":
        # Sensitivity variant: CSSR at its conventional per-test level,
        # no Bonferroni correction (Shalizi & Shalizi use fixed per-test
        # significance).
        alpha_pair = 0.05
    else:
        alpha_pair = DELTA / (2.0 * max(len(samples), 2) ** 2)
    base = run_baseline(
        "CSSR", list(samples.keys()), samples, test, alpha_pair,
        reference_contexts=list(samples.keys()),
        reference_n_samples=sum(s.n_samples for s in samples.values()),
    )

    truth = None
    if hasattr(env, "target_partition"):
        truth = env.target_partition(kwargs["history_max_length"])
    row = {
        "env": name,
        "seed": seed,
        "estimator_n": q.n_classes,
        "cssr_n": base.n_states,
        "calibration": calibration,
        "expected": env.metadata.get("ground_truth_n_classes", ""),
        "estimator_exact": int(partitions_agree(q.partition, truth)) if truth else "",
        "cssr_exact": int(partitions_agree(base.partition, truth)) if truth else "",
        "undetermined_rate": round(q.undetermined_rate, 4),
    }
    cfg = CertifiedConfig(
        k=k, tau=tau, test="TV_FIXED_ALPHABET", alpha=DELTA,
        epsilon_min=EPS_MIN, beta=BETA, seeds=[seed],
        selection_fraction=0.5, context_end=context_end,
        policy_id=policy_id,
        n_trajectories=kwargs["n_trajectories"],
        history_max_length=kwargs["history_max_length"],
        min_history_occurrences=kwargs["min_history_occurrences"],
        max_steps_per_episode=kwargs["max_steps_per_episode"],
        command=(
            f"python experiments/baselines/run_cssr_comparison.py "
            f"--envs {name} --seeds {seed}"
        ),
    )
    cfg.capture_env_pins()
    cfg.save_sidecar(out_dir / f"cssr_{name}_seed{seed}.certified.json")
    return row


def parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", nargs="*", default=list(CONFIGS.keys()))
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results_certified"))
    ap.add_argument("--calibration", choices=["bonferroni", "native"],
                    default="bonferroni")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for name in args.envs:
        for seed in parse_seeds(args.seeds):
            row = run_one(name, seed, out_dir, calibration=args.calibration)
            rows.append(row)
            print(f"  {name} seed={seed} est={row['estimator_n']} "
                  f"cssr={row['cssr_n']} expected={row['expected']}", flush=True)

    csv_name = ("cssr_comparison.csv" if args.calibration == "bonferroni"
                else "cssr_comparison_native.csv")
    csv_path = out_dir / csv_name
    all_rows = merge_result_rows(csv_path, rows, ('env', 'seed'))
    fieldnames: list[str] = []
    for r in all_rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(all_rows)

    lines = [
        "# Same-target CSSR comparison (identical contexts, samples, test, level)",
        "",
        "| env | estimator n (mean, 95% CI) | CSSR n (mean, 95% CI) | expected |",
        "|---|---|---|---|",
    ]
    summary_envs = [e for e in CONFIGS if any(r["env"] == e for r in all_rows)]
    for name in summary_envs:
        sub = [r for r in all_rows if r["env"] == name]
        en = [float(r["estimator_n"]) for r in sub]
        cn = [float(r["cssr_n"]) for r in sub]
        elo, ehi = bootstrap_ci(en, seed=0)
        clo, chi = bootstrap_ci(cn, seed=0)
        lines.append(
            f"| {name} | {np.mean(en):.1f} [{elo:.1f}, {ehi:.1f}] "
            f"| {np.mean(cn):.1f} [{clo:.1f}, {chi:.1f}] | {sub[0]['expected']} |")
    summary_name = ("cssr_summary.md" if args.calibration == "bonferroni"
                    else "cssr_summary_native.md")
    (out_dir / summary_name).write_text("\n".join(lines) + "\n")
    print(f"\nWrote {csv_path}; elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
