"""Dec-POMDP (.dpomdp) file parser and single-agent environment view (T068).

Parses the standard .dpomdp format used by the Dec-POMDPx benchmarks
(Oliehoek et al., Cassandra-style POMDP format extended to Dec-POMDPs).

Supports:
- Named and indexed states, actions, observations
- Uniform / explicit start distributions
- T: transitions (inline tuple and matrix forms)
- O: observations (inline tuple and matrix forms)
- R: rewards (with wildcard '*' for state/obs dimensions)
- Multi-agent joint actions/observations as space-separated tuples
- Comment lines starting with '#'

Sufficient for: Dec-Tiger, BoxPushing, GridSmall, Mars-Recon-2.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from environments.base import POMDPEnvironment, StepResult


class DecPOMDPxFileMissing(RuntimeError):
    """Raised when no .dpomdp file is found at the configured path."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DecPOMDPxModel:
    """Parsed Dec-POMDP model.

    Attributes
    ----------
    states:
        List of state names (length S).
    actions_per_agent:
        List of lists, one per agent; each contains that agent's action names.
    obs_per_agent:
        List of lists, one per agent; each contains that agent's observation names.
    T:
        Transition tensor, shape (joint_actions, S, S).
        T[ja, s, s'] = P(s' | s, ja).
    O:
        Observation tensor, shape (joint_actions, S, joint_obs).
        O[ja, s', jo] = P(jo | s', ja).
    R:
        Reward tensor, shape (joint_actions, S, S, joint_obs).
        R[ja, s, s', jo] = immediate reward.
    gamma:
        Discount factor.
    start:
        Initial state distribution, shape (S,).
    n_agents:
        Number of agents.
    """

    states: list[str]
    actions_per_agent: list[list[str]]
    obs_per_agent: list[list[str]]
    T: np.ndarray          # (joint_actions, S, S)
    O: np.ndarray          # (joint_actions, S, joint_obs)
    R: np.ndarray          # (joint_actions, S, S, joint_obs)
    gamma: float
    start: np.ndarray      # (S,)

    @property
    def n_agents(self) -> int:
        return len(self.actions_per_agent)

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_joint_actions(self) -> int:
        return self.T.shape[0]

    @property
    def n_joint_obs(self) -> int:
        return self.O.shape[2]

    def joint_action_index(self, per_agent_actions: tuple[int, ...]) -> int:
        """Convert per-agent action tuple to joint action index (row-major)."""
        idx = 0
        for i, a in enumerate(per_agent_actions):
            stride = 1
            for j in range(i + 1, self.n_agents):
                stride *= len(self.actions_per_agent[j])
            idx += a * stride
        return idx

    def joint_obs_index(self, per_agent_obs: tuple[int, ...]) -> int:
        """Convert per-agent observation tuple to joint obs index (row-major)."""
        idx = 0
        for i, o in enumerate(per_agent_obs):
            stride = 1
            for j in range(i + 1, self.n_agents):
                stride *= len(self.obs_per_agent[j])
            idx += o * stride
        return idx

    def joint_action_to_per_agent(self, ja: int) -> tuple[int, ...]:
        """Decompose joint action index into per-agent actions."""
        result = []
        for i in range(self.n_agents - 1, -1, -1):
            n = len(self.actions_per_agent[i])
            result.append(ja % n)
            ja //= n
        return tuple(reversed(result))

    def joint_obs_to_per_agent(self, jo: int) -> tuple[int, ...]:
        """Decompose joint observation index into per-agent observations."""
        result = []
        for i in range(self.n_agents - 1, -1, -1):
            n = len(self.obs_per_agent[i])
            result.append(jo % n)
            jo //= n
        return tuple(reversed(result))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _clean_lines(text: str) -> list[str]:
    """Strip comments and blank lines; join continuation lines (ending with \\)."""
    lines = []
    for raw in text.splitlines():
        # Strip inline comments
        if "#" in raw:
            raw = raw[:raw.index("#")]
        raw = raw.strip()
        if raw:
            lines.append(raw)
    return lines


def _parse_names_or_count(token: str, rest_tokens: list[str]) -> list[str]:
    """Parse either a plain integer count or a list of names.

    If `token` is a digit string, return ['0', '1', ..., n-1].
    Otherwise, combine [token] + rest_tokens as name list.
    """
    if token.isdigit():
        n = int(token)
        return [str(i) for i in range(n)]
    names = [token] + rest_tokens
    return names


def parse_dpomdp(path: str | Path) -> DecPOMDPxModel:
    """Parse a .dpomdp file and return a DecPOMDPxModel.

    Parameters
    ----------
    path:
        Path to the .dpomdp file.

    Returns
    -------
    DecPOMDPxModel
        Fully populated model with T, O, R tensors.

    Raises
    ------
    DecPOMDPxFileMissing
        If the file does not exist.
    ValueError
        If the file format is invalid or required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        raise DecPOMDPxFileMissing(f".dpomdp file not found at {path}")

    text = path.read_text()
    lines = _clean_lines(text)

    # ---- Phase 1: parse header fields ----
    n_agents: int | None = None
    gamma: float = 0.95
    values_type: str = "reward"
    states: list[str] = []
    start_spec: str | list[float] = "uniform"
    actions_per_agent: list[list[str]] = []
    obs_per_agent: list[list[str]] = []

    i = 0
    # We'll collect T/O/R lines after the header
    dynamics_lines: list[str] = []

    while i < len(lines):
        line = lines[i]
        llow = line.lower()

        if llow.startswith("agents:"):
            rest = line.split(":", 1)[1].strip()
            parts = rest.split()
            if len(parts) == 1 and parts[0].isdigit():
                n_agents = int(parts[0])
            else:
                n_agents = len(parts)
            i += 1

        elif llow.startswith("discount:"):
            gamma = float(line.split(":", 1)[1].strip())
            i += 1

        elif llow.startswith("values:"):
            values_type = line.split(":", 1)[1].strip().lower()
            i += 1

        elif llow.startswith("states:"):
            rest = line.split(":", 1)[1].strip()
            parts = rest.split()
            if parts:
                states = _parse_names_or_count(parts[0], parts[1:])
            i += 1

        elif llow.startswith("start:"):
            rest = line.split(":", 1)[1].strip()
            if rest.lower() == "uniform":
                start_spec = "uniform"
                i += 1
            else:
                # Could be inline probabilities or continue on next lines
                probs_str = rest.split()
                # Collect until we have len(states) numbers or hit a keyword
                j = i + 1
                while len(probs_str) < len(states) and j < len(lines):
                    nxt = lines[j]
                    if ":" in nxt and any(
                        nxt.lower().startswith(k)
                        for k in ("t:", "o:", "r:", "actions:", "observations:", "states:")
                    ):
                        break
                    probs_str += nxt.split()
                    j += 1
                start_spec = [float(p) for p in probs_str]
                i = j

        elif llow.startswith("actions:"):
            # Actions for each agent appear on successive lines (one per agent)
            # or all on one line if only 1 agent.
            rest = line.split(":", 1)[1].strip()
            if n_agents is None:
                raise ValueError("'agents:' must come before 'actions:' in .dpomdp file")
            if rest:
                # Names/count provided on same line (agent 0)
                parts = rest.split()
                agent_actions = _parse_names_or_count(parts[0], parts[1:])
                actions_per_agent.append(agent_actions)
                i += 1
                # Read remaining agents from following lines
                while len(actions_per_agent) < n_agents and i < len(lines):
                    nxt = lines[i]
                    # Stop if we hit another section
                    if ":" in nxt and any(
                        nxt.lower().startswith(k)
                        for k in ("observations:", "states:", "start:", "discount:", "t:", "o:", "r:")
                    ):
                        break
                    parts = nxt.split()
                    if parts:
                        agent_actions = _parse_names_or_count(parts[0], parts[1:])
                        actions_per_agent.append(agent_actions)
                    i += 1
            else:
                # All agents on following lines
                i += 1
                while len(actions_per_agent) < n_agents and i < len(lines):
                    nxt = lines[i]
                    if ":" in nxt and any(
                        nxt.lower().startswith(k)
                        for k in ("observations:", "states:", "start:", "discount:", "t:", "o:", "r:")
                    ):
                        break
                    parts = nxt.split()
                    if parts:
                        agent_actions = _parse_names_or_count(parts[0], parts[1:])
                        actions_per_agent.append(agent_actions)
                    i += 1

        elif llow.startswith("observations:"):
            rest = line.split(":", 1)[1].strip()
            if n_agents is None:
                raise ValueError("'agents:' must come before 'observations:' in .dpomdp file")
            if rest:
                parts = rest.split()
                agent_obs = _parse_names_or_count(parts[0], parts[1:])
                obs_per_agent.append(agent_obs)
                i += 1
                while len(obs_per_agent) < n_agents and i < len(lines):
                    nxt = lines[i]
                    if ":" in nxt and any(
                        nxt.lower().startswith(k)
                        for k in ("t:", "o:", "r:", "states:", "start:", "actions:")
                    ):
                        break
                    parts = nxt.split()
                    if parts:
                        agent_obs = _parse_names_or_count(parts[0], parts[1:])
                        obs_per_agent.append(agent_obs)
                    i += 1
            else:
                i += 1
                while len(obs_per_agent) < n_agents and i < len(lines):
                    nxt = lines[i]
                    if ":" in nxt and any(
                        nxt.lower().startswith(k)
                        for k in ("t:", "o:", "r:", "states:", "start:", "actions:")
                    ):
                        break
                    parts = nxt.split()
                    if parts:
                        agent_obs = _parse_names_or_count(parts[0], parts[1:])
                        obs_per_agent.append(agent_obs)
                    i += 1

        elif (llow.startswith("t:") or llow.startswith("o:") or llow.startswith("r:")):
            # Rest of file is dynamics; collect remaining lines
            dynamics_lines = lines[i:]
            break

        else:
            i += 1

    # Validate header
    if n_agents is None:
        raise ValueError("Missing 'agents:' field in .dpomdp file")
    if not states:
        raise ValueError("Missing 'states:' field in .dpomdp file")
    if len(actions_per_agent) != n_agents:
        raise ValueError(
            f"Expected {n_agents} action lists, got {len(actions_per_agent)}"
        )
    if len(obs_per_agent) != n_agents:
        raise ValueError(
            f"Expected {n_agents} observation lists, got {len(obs_per_agent)}"
        )

    S = len(states)
    state_idx = {name: idx for idx, name in enumerate(states)}

    # Build joint action / obs name->index maps
    joint_action_names = [
        " ".join(combo)
        for combo in itertools.product(*actions_per_agent)
    ]
    n_ja = len(joint_action_names)
    ja_idx = {name: idx for idx, name in enumerate(joint_action_names)}

    joint_obs_names = [
        " ".join(combo)
        for combo in itertools.product(*obs_per_agent)
    ]
    n_jo = len(joint_obs_names)
    jo_idx = {name: idx for idx, name in enumerate(joint_obs_names)}

    # ---- Phase 2: initialise tensors ----
    # T: uniform default (will be overwritten)
    T = np.zeros((n_ja, S, S), dtype=np.float64)
    # O: uniform default
    O = np.full((n_ja, S, n_jo), 1.0 / n_jo, dtype=np.float64)
    # R: zero default
    R = np.zeros((n_ja, S, S, n_jo), dtype=np.float64)

    # Track which (ja, s) rows have been assigned in T and O
    _t_assigned: set[tuple[int, int]] = set()
    _o_assigned: set[tuple[int, int]] = set()

    # ---- Helper: resolve wildcard or name ----
    def _resolve_states(token: str) -> list[int]:
        if token == "*":
            return list(range(S))
        return [state_idx[token]]

    def _resolve_ja(token: str) -> list[int]:
        if token == "*":
            return list(range(n_ja))
        return [ja_idx[token]]

    def _resolve_jo(token: str) -> list[int]:
        if token == "*":
            return list(range(n_jo))
        return [jo_idx[token]]

    # ---- Phase 3: parse dynamics lines ----
    di = 0
    while di < len(dynamics_lines):
        line = dynamics_lines[di]
        llow = line.lower()

        # ---- T: entry ----
        if llow.startswith("t:"):
            parts = [p.strip() for p in line.split(":")]
            # T: <joint_action> : <start_state> : <end_state> : <prob>  (inline)
            # T: <joint_action>                                          (matrix follows)
            # T: <joint_action> : <start_state>                         (row vector follows)
            if len(parts) == 5:
                # Inline: T: ja : s : s' : prob
                ja_token = parts[1].strip()
                s_token = parts[2].strip()
                sp_token = parts[3].strip()
                prob = float(parts[4].strip())
                for ja in _resolve_ja(ja_token):
                    for s in _resolve_states(s_token):
                        for sp in _resolve_states(sp_token):
                            T[ja, s, sp] = prob
                            _t_assigned.add((ja, s))
                di += 1

            elif len(parts) == 4:
                # Row vector: T: ja : s : followed by S probabilities on next line
                ja_token = parts[1].strip()
                s_token = parts[2].strip()
                # Check if the 4th part has content (might be the keyword "uniform"/"identity")
                kw = parts[3].strip().lower()
                di += 1
                if kw in ("uniform", "identity", ""):
                    if kw == "uniform":
                        for ja in _resolve_ja(ja_token):
                            for s in _resolve_states(s_token):
                                T[ja, s, :] = 1.0 / S
                                _t_assigned.add((ja, s))
                    elif kw == "identity":
                        for ja in _resolve_ja(ja_token):
                            for s in _resolve_states(s_token):
                                T[ja, s, :] = 0.0
                                T[ja, s, s] = 1.0
                                _t_assigned.add((ja, s))
                    else:
                        # Empty 4th segment -- read probabilities from next line
                        probs = list(map(float, dynamics_lines[di].split()))
                        di += 1
                        for ja in _resolve_ja(ja_token):
                            for s in _resolve_states(s_token):
                                T[ja, s, :] = probs
                                _t_assigned.add((ja, s))
                else:
                    # 4th part is the first probability row
                    probs = list(map(float, kw.split()))
                    # may need more tokens
                    while len(probs) < S and di < len(dynamics_lines):
                        probs += list(map(float, dynamics_lines[di].split()))
                        di += 1
                    for ja in _resolve_ja(ja_token):
                        for s in _resolve_states(s_token):
                            T[ja, s, :] = probs[:S]
                            _t_assigned.add((ja, s))

            elif len(parts) == 3:
                # Matrix form: T: ja : followed by S rows, each S numbers
                ja_token = parts[1].strip()
                kw = parts[2].strip().lower()
                di += 1
                if kw == "uniform":
                    for ja in _resolve_ja(ja_token):
                        T[ja, :, :] = 1.0 / S
                        for s in range(S):
                            _t_assigned.add((ja, s))
                elif kw == "identity":
                    for ja in _resolve_ja(ja_token):
                        T[ja, :, :] = np.eye(S)
                        for s in range(S):
                            _t_assigned.add((ja, s))
                else:
                    # kw might be the start of the first row of numbers
                    row_data: list[float] = []
                    if kw:
                        row_data += list(map(float, kw.split()))
                    while len(row_data) < S * S and di < len(dynamics_lines):
                        nxt = dynamics_lines[di]
                        if any(nxt.lower().startswith(k) for k in ("t:", "o:", "r:")):
                            break
                        row_data += list(map(float, nxt.split()))
                        di += 1
                    mat = np.array(row_data[:S * S]).reshape(S, S)
                    for ja in _resolve_ja(ja_token):
                        T[ja, :, :] = mat
                        for s in range(S):
                            _t_assigned.add((ja, s))

            elif len(parts) == 2:
                # T: <joint_action> followed by S*S numbers (matrix)
                ja_token = parts[1].strip()
                di += 1
                row_data = []
                while len(row_data) < S * S and di < len(dynamics_lines):
                    nxt = dynamics_lines[di]
                    if any(nxt.lower().startswith(k) for k in ("t:", "o:", "r:")):
                        break
                    row_data += list(map(float, nxt.split()))
                    di += 1
                mat = np.array(row_data[:S * S]).reshape(S, S)
                for ja in _resolve_ja(ja_token):
                    T[ja, :, :] = mat
                    for s in range(S):
                        _t_assigned.add((ja, s))
            else:
                di += 1

        # ---- O: entry ----
        elif llow.startswith("o:"):
            parts = [p.strip() for p in line.split(":")]
            if len(parts) == 5:
                # O: ja : s' : jo : prob
                ja_token = parts[1].strip()
                sp_token = parts[2].strip()
                jo_token = parts[3].strip()
                prob = float(parts[4].strip())
                for ja in _resolve_ja(ja_token):
                    for sp in _resolve_states(sp_token):
                        for jo in _resolve_jo(jo_token):
                            O[ja, sp, jo] = prob
                            _o_assigned.add((ja, sp))
                di += 1

            elif len(parts) == 4:
                # Row vector: O: ja : s' : <kw or probs>
                ja_token = parts[1].strip()
                sp_token = parts[2].strip()
                kw = parts[3].strip().lower()
                di += 1
                if kw == "uniform":
                    for ja in _resolve_ja(ja_token):
                        for sp in _resolve_states(sp_token):
                            O[ja, sp, :] = 1.0 / n_jo
                            _o_assigned.add((ja, sp))
                else:
                    probs = list(map(float, kw.split()))
                    while len(probs) < n_jo and di < len(dynamics_lines):
                        nxt = dynamics_lines[di]
                        if any(nxt.lower().startswith(k) for k in ("t:", "o:", "r:")):
                            break
                        probs += list(map(float, nxt.split()))
                        di += 1
                    for ja in _resolve_ja(ja_token):
                        for sp in _resolve_states(sp_token):
                            O[ja, sp, :] = probs[:n_jo]
                            _o_assigned.add((ja, sp))

            elif len(parts) == 3:
                # Matrix form: O: ja : <kw or numbers>
                ja_token = parts[1].strip()
                kw = parts[2].strip().lower()
                di += 1
                if kw == "uniform":
                    for ja in _resolve_ja(ja_token):
                        O[ja, :, :] = 1.0 / n_jo
                        for sp in range(S):
                            _o_assigned.add((ja, sp))
                else:
                    row_data = []
                    if kw:
                        row_data += list(map(float, kw.split()))
                    while len(row_data) < S * n_jo and di < len(dynamics_lines):
                        nxt = dynamics_lines[di]
                        if any(nxt.lower().startswith(k) for k in ("t:", "o:", "r:")):
                            break
                        row_data += list(map(float, nxt.split()))
                        di += 1
                    mat = np.array(row_data[:S * n_jo]).reshape(S, n_jo)
                    for ja in _resolve_ja(ja_token):
                        O[ja, :, :] = mat
                        for sp in range(S):
                            _o_assigned.add((ja, sp))

            elif len(parts) == 2:
                ja_token = parts[1].strip()
                di += 1
                row_data = []
                while len(row_data) < S * n_jo and di < len(dynamics_lines):
                    nxt = dynamics_lines[di]
                    if any(nxt.lower().startswith(k) for k in ("t:", "o:", "r:")):
                        break
                    row_data += list(map(float, nxt.split()))
                    di += 1
                mat = np.array(row_data[:S * n_jo]).reshape(S, n_jo)
                for ja in _resolve_ja(ja_token):
                    O[ja, :, :] = mat
                    for sp in range(S):
                        _o_assigned.add((ja, sp))
            else:
                di += 1

        # ---- R: entry ----
        elif llow.startswith("r:"):
            # R: ja : s : s' : jo : reward
            # Wildcards allowed.
            parts = [p.strip() for p in line.split(":")]
            if len(parts) == 6:
                ja_token = parts[1].strip()
                s_token = parts[2].strip()
                sp_token = parts[3].strip()
                jo_token = parts[4].strip()
                reward = float(parts[5].strip())
                for ja in _resolve_ja(ja_token):
                    for s in _resolve_states(s_token):
                        for sp in _resolve_states(sp_token):
                            for jo in _resolve_jo(jo_token):
                                R[ja, s, sp, jo] = reward
            elif len(parts) == 5:
                # R: ja : s : s' : reward  (no jo dimension — fill all jo)
                ja_token = parts[1].strip()
                s_token = parts[2].strip()
                sp_token = parts[3].strip()
                reward = float(parts[4].strip())
                for ja in _resolve_ja(ja_token):
                    for s in _resolve_states(s_token):
                        for sp in _resolve_states(sp_token):
                            R[ja, s, sp, :] = reward
            di += 1

        else:
            di += 1

    # ---- Phase 4: build start distribution ----
    if start_spec == "uniform":
        start = np.full(S, 1.0 / S)
    else:
        start = np.array(start_spec, dtype=np.float64)
        if start.sum() > 0:
            start /= start.sum()

    return DecPOMDPxModel(
        states=states,
        actions_per_agent=actions_per_agent,
        obs_per_agent=obs_per_agent,
        T=T,
        O=O,
        R=R,
        gamma=gamma,
        start=start,
    )


# ---------------------------------------------------------------------------
# Single-agent environment view
# ---------------------------------------------------------------------------

class DecPOMDPxSingleAgentView(POMDPEnvironment):
    """Exposes one agent's view of a Dec-POMDP as a single-agent POMDP.

    The 'other agents' act under a fixed policy (default: uniform random).
    Each call to ``step(action)`` samples the other agents' actions from their
    policies, computes the joint action, samples next state from T, samples
    joint observation from O, and returns the controlled agent's marginal
    observation plus a per-agent reward (joint reward / n_agents).

    Parameters
    ----------
    model:
        A parsed DecPOMDPxModel.
    agent_idx:
        Index of the controlled agent (0-based).
    other_policy:
        Either ``'uniform_random'`` or a list of per-agent action probability
        arrays (one array per *other* agent, length = that agent's n_actions).
    max_steps:
        Maximum steps per episode before truncation.
    seed:
        Random seed.
    """

    def __init__(
        self,
        model: DecPOMDPxModel,
        agent_idx: int = 0,
        other_policy: str | list[np.ndarray] = "uniform_random",
        max_steps: int = 50,
        seed: int = 0,
    ):
        self.model = model
        self.agent_idx = agent_idx
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._state: int = 0
        self._t: int = 0

        # Build per-other-agent action distributions
        other_indices = [i for i in range(model.n_agents) if i != agent_idx]
        if other_policy == "uniform_random":
            self._other_probs: list[np.ndarray] = [
                np.ones(len(model.actions_per_agent[i])) / len(model.actions_per_agent[i])
                for i in other_indices
            ]
        else:
            self._other_probs = list(other_policy)
        self._other_agent_indices = other_indices

    @property
    def n_observations(self) -> int:
        return len(self.model.obs_per_agent[self.agent_idx])

    @property
    def n_actions(self) -> int:
        return len(self.model.actions_per_agent[self.agent_idx])

    @property
    def environment_id(self) -> str:
        return f"DecPOMDPx(agent={self.agent_idx})"

    @property
    def metadata(self) -> dict[str, object]:
        return {"n_agents": self.model.n_agents, "n_states": self.model.n_states}

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = int(self._rng.choice(self.model.n_states, p=self.model.start))
        self._t = 0
        # Sample initial observation marginal (uniform or from first T/O step)
        # Conventionally return uniform initial obs
        return int(self._rng.integers(0, self.n_observations))

    def step(self, action: int) -> StepResult:
        # Sample other agents' actions
        per_agent_actions = {}
        per_agent_actions[self.agent_idx] = action
        for i, other_idx in enumerate(self._other_agent_indices):
            probs = self._other_probs[i]
            per_agent_actions[other_idx] = int(
                self._rng.choice(len(probs), p=probs)
            )

        # Build ordered tuple
        joint_tuple = tuple(per_agent_actions[i] for i in range(self.model.n_agents))
        ja = self.model.joint_action_index(joint_tuple)

        # Sample next state
        trans_probs = self.model.T[ja, self._state]
        if trans_probs.sum() == 0:
            trans_probs = np.ones(self.model.n_states) / self.model.n_states
        else:
            trans_probs = trans_probs / trans_probs.sum()
        next_state = int(self._rng.choice(self.model.n_states, p=trans_probs))

        # Sample joint observation
        obs_probs = self.model.O[ja, next_state]
        if obs_probs.sum() == 0:
            obs_probs = np.ones(self.model.n_joint_obs) / self.model.n_joint_obs
        else:
            obs_probs = obs_probs / obs_probs.sum()
        joint_obs = int(self._rng.choice(self.model.n_joint_obs, p=obs_probs))

        # Per-agent reward = joint reward / n_agents
        joint_reward = self.model.R[ja, self._state, next_state, joint_obs]
        reward = float(joint_reward) / self.model.n_agents

        # Extract this agent's marginal observation
        per_agent_obs = self.model.joint_obs_to_per_agent(joint_obs)
        my_obs = int(per_agent_obs[self.agent_idx])

        self._state = next_state
        self._t += 1
        truncated = self._t >= self.max_steps

        return StepResult(
            observation=my_obs,
            reward=reward,
            terminated=False,
            truncated=truncated,
        )
