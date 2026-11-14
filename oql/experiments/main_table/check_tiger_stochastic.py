"""Derivation check for the stochastic-policy Tiger gap (survival-conditioned).

Timeline at r=1, k=1: reset emits O_0 (uniform, uninformative); action a_0
emits O_1 (the context); action a_1 emits O_2 (the continuation).  A door
opening ends the episode but its (uniform) terminal draw IS recorded as the
observation of that step.  Hence

  * the episode reaches index r+k = 2 iff a_0 = listen (prob p_listen);
    if a_0 opens a door, O_1 is recorded but there is no O_2, so no block;
  * a_1 is unrestricted: listen (p_listen) gives the noisy informative draw,
    a door opening gives a uniform recorded O_2.

Closed form, with a the listen accuracy:
  P(O_2=L | O_1=L, reach) = p_listen (a^2+(1-a)^2) + (1-p_listen)/2
  P(O_2=L | O_1=R, reach) = p_listen 2a(1-a)       + (1-p_listen)/2
  TV = p_listen (2a-1)^2        (a_0's listen factor cancels in conditioning:
                                 it multiplies numerator and mass alike).
This script verifies the closed form against (i) the brute-force enumeration
in run_nondegenerate.py and (ii) a Monte-Carlo run through the real
collection code and one-block extraction.  Output: results_nondegenerate/
tiger_gap_check.json.

Usage: uv run python experiments/main_table/check_tiger_stochastic.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from environments.layer1_discrete.tiger import TigerPOMDP  # noqa: E402
from quotient_learning.collection import collect_trajectories  # noqa: E402
from quotient_learning.sampling import extract_independent_continuation_samples  # noqa: E402
from quotient_learning.types import ObservationHistory  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_nd", HERE / "run_nondegenerate.py")
nd = importlib.util.module_from_spec(spec)
sys.modules["_nd"] = nd
spec.loader.exec_module(nd)

N_MC = 400_000


def main() -> int:
    a, p = nd.LISTEN_ACCURACY, nd.ACTION_PROBS[0]
    pLL = p * (a**2 + (1 - a) ** 2) + (1 - p) / 2
    pLR = p * 2 * a * (1 - a) + (1 - p) / 2
    closed_tv = abs(pLL - pLR)
    laws, masses = nd.tiger_closed_loop_laws()
    brute_tv = nd.true_partition_and_gap(laws)[1]

    env = TigerPOMDP(seed=0)
    ds = collect_trajectories(
        env=env, policy=nd.stochastic_tiger_policy(), n_episodes=N_MC,
        max_steps=nd.MAX_STEPS, seed=123, policy_id=nd.POLICY_ID,
    )
    reach = sum(len(t.observations) >= 3 for t in ds.trajectories) / N_MC
    mc = {}
    for o in (0, 1):
        cs = extract_independent_continuation_samples(
            ds, ObservationHistory((o,)), 1, 1)
        n = cs.n_samples
        mc[o] = {
            "n": n,
            "mass": n / N_MC,
            "p_next_left": sum(c == (0,) for c in cs.continuations) / n,
        }
    mc_tv = abs(mc[0]["p_next_left"] - mc[1]["p_next_left"])
    out = {
        "p_listen": p, "listen_accuracy": a,
        "closed_form": {"P(O2=L|O1=L)": pLL, "P(O2=L|O1=R)": pLR,
                        "tv": closed_tv, "reach_prob": p, "context_mass": p / 2},
        "brute_force_enumeration": {"tv": brute_tv,
                                    "masses": {str(k): v for k, v in masses.items()}},
        "monte_carlo": {"n_episodes": N_MC, "reach_fraction": reach,
                        "contexts": {str(k): v for k, v in mc.items()},
                        "tv": mc_tv},
    }
    print(json.dumps(out, indent=1))
    assert abs(closed_tv - 0.392) < 1e-12 and abs(brute_tv - closed_tv) < 1e-12
    assert abs(reach - p) < 0.005 and abs(mc_tv - closed_tv) < 0.01
    dest = HERE / "results_nondegenerate" / "tiger_gap_check.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
