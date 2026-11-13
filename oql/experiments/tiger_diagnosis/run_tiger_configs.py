"""Tiger POMDP diagnosis: systematically test configurations to recover n=2.

Investigates:
1. Count-based exploration policy
2. Listen-biased stationary policy (listen=0.7, open-left=0.15, open-right=0.15)
3. Longer history (max_length=3,4) with sufficient min_occurrences
4. MMD equivalence test
5. Larger budgets (N=20000, N=50000)
6. Different alpha (0.05 vs 0.01)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environments.layer1_discrete.tiger import TigerPOMDP
from quotient_learning.collection import (
    collect_trajectories,
    count_based_exploration_policy,
    uniform_random_policy,
)
from quotient_learning.equivalence import GTestEquivalenceTest, MMDEquivalenceTest
from quotient_learning.estimator import QuotientEstimator


SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ConfigResult:
    config_name: str
    n_trajectories: int
    history_max_length: int
    min_history_occurrences: int
    alpha: float
    policy_name: str
    test_name: str
    seeds: list[int]
    n_classes_per_seed: list[int]
    mean_n_classes: float
    std_n_classes: float
    seeds_recovering_2: int
    time_seconds: float
    partition_details: list[dict]


def listen_biased_policy(listen_prob: float = 0.7):
    """Stationary policy: listen with listen_prob, open-left/right with equal remaining."""
    open_prob = (1.0 - listen_prob) / 2.0

    def _policy(_obs: int, rng: np.random.Generator) -> int:
        r = rng.random()
        if r < listen_prob:
            return 0  # listen
        elif r < listen_prob + open_prob:
            return 1  # open-left
        else:
            return 2  # open-right

    return _policy


def run_config(
    config_name: str,
    n_trajectories: int,
    history_max_length: int,
    min_history_occurrences: int,
    alpha: float,
    policy_name: str,
    test_name: str,
    window_length: int = 2,
    verbose: bool = True,
) -> ConfigResult:
    """Run 5 seeds of a given configuration and return results."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"  N={n_trajectories}, history_max_length={history_max_length}")
        print(f"  min_history_occurrences={min_history_occurrences}, alpha={alpha}")
        print(f"  policy={policy_name}, test={test_name}")

    t0 = time.time()
    n_classes_per_seed = []
    partition_details = []

    for seed in SEEDS:
        env = TigerPOMDP()

        # Select policy
        if policy_name == "uniform_random":
            policy = uniform_random_policy(env.n_actions)
        elif policy_name == "count_based":
            policy = count_based_exploration_policy(
                n_actions=env.n_actions,
                history_max_length=history_max_length,
            )
        elif policy_name == "listen_biased_70":
            policy = listen_biased_policy(0.70)
        elif policy_name == "listen_biased_80":
            policy = listen_biased_policy(0.80)
        elif policy_name == "listen_biased_90":
            policy = listen_biased_policy(0.90)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        # Select equivalence test
        if test_name == "gtest":
            equiv_test = GTestEquivalenceTest()
        elif test_name == "mmd":
            # Use n_permutations=20 for diagnosis speed; the full 200-permutation
            # run is prohibitively slow (>20 min) on Tiger with N=5000 trajectories.
            equiv_test = MMDEquivalenceTest(n_observations=env.n_observations, n_permutations=20, seed=seed)
        else:
            raise ValueError(f"Unknown test: {test_name}")

        estimator = QuotientEstimator(
            env=env,
            n_trajectories=n_trajectories,
            window_length=window_length,
            history_max_length=history_max_length,
            min_history_occurrences=min_history_occurrences,
            equivalence_test=equiv_test,
            alpha=alpha,
            max_steps_per_episode=50,
            policy=policy,
            policy_id=policy_name,
        )

        quotient = estimator.run(seed=seed)
        n_classes_per_seed.append(quotient.n_classes)

        # Collect partition details
        partition_info = {
            "seed": seed,
            "n_classes": quotient.n_classes,
            "partition_keys": [str(h) for h in quotient.partition.keys()],
            "class_assignments": {str(h): c for h, c in quotient.partition.items()},
        }
        partition_details.append(partition_info)

        if verbose:
            print(f"  seed={seed}: n_classes={quotient.n_classes}, "
                  f"partition={partition_info['class_assignments']}")

    elapsed = time.time() - t0
    n_arr = np.array(n_classes_per_seed)
    seeds_recovering_2 = int(np.sum(n_arr == 2))

    result = ConfigResult(
        config_name=config_name,
        n_trajectories=n_trajectories,
        history_max_length=history_max_length,
        min_history_occurrences=min_history_occurrences,
        alpha=alpha,
        policy_name=policy_name,
        test_name=test_name,
        seeds=SEEDS,
        n_classes_per_seed=n_classes_per_seed,
        mean_n_classes=float(n_arr.mean()),
        std_n_classes=float(n_arr.std()),
        seeds_recovering_2=seeds_recovering_2,
        time_seconds=elapsed,
        partition_details=partition_details,
    )

    if verbose:
        print(f"  -> mean n_classes={result.mean_n_classes:.2f} ± {result.std_n_classes:.2f}, "
              f"seeds with n=2: {seeds_recovering_2}/5, time: {elapsed:.1f}s")

    return result


def main():
    all_results = []

    # -----------------------------------------------------------------------
    # Baseline: verify the reported failure mode with exact same config
    # -----------------------------------------------------------------------
    r = run_config(
        config_name="baseline_uniform_hml2_N5000",
        n_trajectories=5000,
        history_max_length=2,
        min_history_occurrences=80,
        alpha=0.01,
        policy_name="uniform_random",
        test_name="gtest",
    )
    all_results.append(r)

    # -----------------------------------------------------------------------
    # 1. Count-based exploration policy
    # -----------------------------------------------------------------------
    r = run_config(
        config_name="count_based_hml2_N5000",
        n_trajectories=5000,
        history_max_length=2,
        min_history_occurrences=80,
        alpha=0.01,
        policy_name="count_based",
        test_name="gtest",
    )
    all_results.append(r)

    # -----------------------------------------------------------------------
    # 2. Listen-biased stationary policy (0.70, 0.80, 0.90)
    # -----------------------------------------------------------------------
    for listen_p, policy_name in [
        (0.70, "listen_biased_70"),
        (0.80, "listen_biased_80"),
        (0.90, "listen_biased_90"),
    ]:
        r = run_config(
            config_name=f"listen_biased_{int(listen_p*100)}_hml2_N5000",
            n_trajectories=5000,
            history_max_length=2,
            min_history_occurrences=80,
            alpha=0.01,
            policy_name=policy_name,
            test_name="gtest",
        )
        all_results.append(r)

    # -----------------------------------------------------------------------
    # 3. Longer history: max_length=3 and 4 with appropriate min_occurrences
    # -----------------------------------------------------------------------
    for hml, min_occ in [(3, 50), (4, 30)]:
        r = run_config(
            config_name=f"uniform_hml{hml}_N5000_minocc{min_occ}",
            n_trajectories=5000,
            history_max_length=hml,
            min_history_occurrences=min_occ,
            alpha=0.01,
            policy_name="uniform_random",
            test_name="gtest",
        )
        all_results.append(r)

    # listen-biased + longer history (hml=3 only; hml=4 is computationally
    # prohibitive: generates O(n^2) history pairs × 5 seeds per config)
    for listen_p, policy_name in [
        (0.80, "listen_biased_80"),
        (0.90, "listen_biased_90"),
    ]:
        for hml, min_occ in [(3, 50)]:
            r = run_config(
                config_name=f"listen_{int(listen_p*100)}_hml{hml}_N5000",
                n_trajectories=5000,
                history_max_length=hml,
                min_history_occurrences=min_occ,
                alpha=0.01,
                policy_name=policy_name,
                test_name="gtest",
            )
            all_results.append(r)

    # -----------------------------------------------------------------------
    # 4. MMD equivalence test
    # -----------------------------------------------------------------------
    for policy_name in ["uniform_random", "listen_biased_80"]:
        r = run_config(
            config_name=f"mmd_{policy_name}_hml2_N5000",
            n_trajectories=5000,
            history_max_length=2,
            min_history_occurrences=80,
            alpha=0.05,  # MMD permutation test needs looser alpha
            policy_name=policy_name,
            test_name="mmd",
        )
        all_results.append(r)

    # -----------------------------------------------------------------------
    # 5. Larger budgets (N=20000 only; N=50000 is omitted here — at N=20000
    # the uniform-random result is already n=1 on all 5 seeds, confirming
    # the failure mode is not a budget issue)
    # -----------------------------------------------------------------------
    for N, policy_name in [
        (20000, "uniform_random"),
        (20000, "listen_biased_80"),
    ]:
        r = run_config(
            config_name=f"N{N}_{policy_name}_hml2",
            n_trajectories=N,
            history_max_length=2,
            min_history_occurrences=80,
            alpha=0.01,
            policy_name=policy_name,
            test_name="gtest",
        )
        all_results.append(r)

    # -----------------------------------------------------------------------
    # 6. Different alpha (more permissive)
    # -----------------------------------------------------------------------
    for alpha, policy_name in [
        (0.05, "uniform_random"),
        (0.05, "listen_biased_80"),
        (0.10, "uniform_random"),
    ]:
        r = run_config(
            config_name=f"alpha{int(alpha*100)}_{policy_name}_hml2_N5000",
            n_trajectories=5000,
            history_max_length=2,
            min_history_occurrences=80,
            alpha=alpha,
            policy_name=policy_name,
            test_name="gtest",
        )
        all_results.append(r)

    # -----------------------------------------------------------------------
    # Combined: listen-biased + large N + permissive alpha
    # -----------------------------------------------------------------------
    r = run_config(
        config_name="listen_biased_90_hml3_N20000_alpha05",
        n_trajectories=20000,
        history_max_length=3,
        min_history_occurrences=50,
        alpha=0.05,
        policy_name="listen_biased_90",
        test_name="gtest",
    )
    all_results.append(r)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY: Tiger POMDP Diagnosis Results")
    print("="*70)
    print(f"{'Config':<50} {'Mean n_cls':>10} {'Std':>6} {'n=2 seeds':>10} {'Time(s)':>8}")
    print("-"*70)

    any_success = False
    successful_configs = []
    for r in all_results:
        success = r.seeds_recovering_2 >= 4  # >=4/5 seeds
        marker = " ***" if success else ""
        print(f"{r.config_name:<50} {r.mean_n_classes:>10.2f} {r.std_n_classes:>6.2f} "
              f"{r.seeds_recovering_2:>10}/5 {r.time_seconds:>8.1f}{marker}")
        if success:
            any_success = True
            successful_configs.append(r)

    print("\n" + "="*70)
    if any_success:
        print("DECISION: At least one config recovers n=2 on >=4/5 seeds.")
        print("Successful configs:")
        for r in successful_configs:
            print(f"  - {r.config_name}: mean={r.mean_n_classes:.2f}, {r.seeds_recovering_2}/5 seeds")
    else:
        print("DECISION: No config recovers n=2 on >=4/5 seeds. Must retract Tiger n=2 claim.")

    # Save results to JSON
    results_data = {
        "configs": [asdict(r) for r in all_results],
        "any_success": any_success,
        "successful_configs": [r.config_name for r in successful_configs] if any_success else [],
    }
    outfile = RESULTS_DIR / "tiger_diagnosis.json"
    with open(outfile, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {outfile}")

    return any_success, successful_configs


if __name__ == "__main__":
    any_success, successful_configs = main()
    sys.exit(0 if any_success else 1)
