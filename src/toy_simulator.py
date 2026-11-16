"""Exact finite testbed for the acquisition-and-position problem (Workstream A).

Everything here is exact: a belief is the conditional of a finite joint pmf over
(r, y0, A, B), so posterior weights, predictive branches and expectations are
finite sums with no quadrature error. The module implements the ``Belief``
protocol of ``src.interfaces`` so that the same planner code can be validated
on a case where every quantity can be checked by enumeration.

Contents
--------
* ``DiscreteJointModel``       exact joint pmf over (r, y0, A, B).
* ``DiscreteBelief``           exact posterior given observed signal values.
* ``myopic_voi``               exact net one-step VOI (own code path).
* ``exact_two_step_value``     independent Bellman recursion (does NOT import
                               ``src.planner``); returns (value, action, action_values).
* ``enumerate_policy_trees`` / ``brute_force_optimal`` / ``evaluate_policy_tree``
                               enumerate every deterministic adaptive policy tree and
                               evaluate each by enumerating the hidden world. No
                               Bellman recursion is used.
* ``scenarios()``              named deterministic instances with documented properties.
* ``random_instance(seed)``    seeded random small instances for cross-checks.

Conventions (see docs/notes_A.md): money in fractions of capital; query costs are
subtracted from the realized utility; the position utility includes the transaction
cost relative to q_prev; each source is bought at most once; a source is affordable
when cost <= budget_left + BUDGET_ATOL; STOP wins ties; between sources, "A" wins ties.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from src.interfaces import BUDGET_ATOL, SOURCES, STOP, AcquisitionSetup
from src.utility import POSITIONS, UtilityParams, choose_position, expected_utility, realized_utility

AXES: tuple[str, ...] = ("r", "y0", "A", "B")
SIGNALS: tuple[str, ...] = ("y0", "A", "B")

# Value-level tolerance for "query strictly better than the current best". Must match the
# tolerance used by the planner under test (src.planner.PLAN_TOL) for action comparisons.
QUERY_TOL = 1e-12
# Tolerance for two brute-force trees to count as equally good (keeps the first in
# enumeration order, i.e. STOP first, then A before B).
TREE_TIE_TOL = 1e-13
# Relative tolerance for matching an observed numeric value to a signal alphabet entry.
VALUE_MATCH_RTOL = 1e-12


# --------------------------------------------------------------------------- model
class DiscreteJointModel:
    """Exact joint pmf over (r, y0, A, B) on finite alphabets.

    r_values:      1-D finite array, the return values indexed by the first pmf axis.
    pmf:           ndarray of shape (len(r_values), n_y0, n_A, n_B), nonnegative, sums to 1
                   within 1e-12.
    signal_values: mapping with keys "y0", "A", "B"; each a 1-D finite array of distinct
                   numeric values; entry i is the value that index i denotes.
    """

    def __init__(self, r_values, pmf, signal_values: Mapping[str, Sequence[float]]):
        r = np.array(r_values, dtype=float)
        if r.ndim != 1 or r.size == 0 or not np.all(np.isfinite(r)):
            raise ValueError("r_values must be a nonempty 1-D finite array")
        if set(signal_values) != set(SIGNALS):
            raise ValueError(f"signal_values must have keys {SIGNALS}, got {sorted(signal_values)}")
        sig: dict[str, np.ndarray] = {}
        for name in SIGNALS:
            v = np.array(signal_values[name], dtype=float)
            if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)):
                raise ValueError(f"signal_values[{name!r}] must be a nonempty 1-D finite array")
            scale = max(1.0, float(np.max(np.abs(v))))
            if v.size > 1 and np.min(np.diff(np.sort(v))) <= 2 * VALUE_MATCH_RTOL * scale:
                raise ValueError(f"signal_values[{name!r}] must contain distinct values")
            sig[name] = v
        p = np.array(pmf, dtype=float)
        shape = (r.size, sig["y0"].size, sig["A"].size, sig["B"].size)
        if p.shape != shape:
            raise ValueError(f"pmf shape {p.shape} does not match alphabets {shape}")
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("pmf must be finite and nonnegative")
        if abs(float(p.sum()) - 1.0) > 1e-12:
            raise ValueError(f"pmf must sum to 1 within 1e-12, sums to {float(p.sum())!r}")
        self.r_values = r
        self.signal_values = sig
        self.pmf = p
        for arr in (self.r_values, self.pmf, *self.signal_values.values()):
            arr.setflags(write=False)

    def signal_index(self, name: str, value: float) -> int:
        """Index of the alphabet entry equal to ``value``. Raises ValueError if none."""
        if name not in SIGNALS:
            raise KeyError(f"unknown signal {name!r}")
        alpha = self.signal_values[name]
        v = float(value)
        hits = np.flatnonzero(np.abs(alpha - v) <= VALUE_MATCH_RTOL * max(1.0, abs(v)))
        if hits.size != 1:
            raise ValueError(f"value {v!r} is out of the {name} alphabet {alpha.tolist()}")
        return int(hits[0])


# --------------------------------------------------------------------------- belief
class DiscreteBelief:
    """Exact posterior over r and the unobserved signals, given observed signal values.

    Implements ``src.interfaces.Belief``. The posterior is the model pmf restricted to
    the observed values and renormalized; unobserved signal values enter only through
    this marginalization. ``predictive_branches`` ignores ``n_nodes`` (the outcome set
    is finite and enumerated exactly).
    """

    def __init__(self, model: DiscreteJointModel, observations: Mapping[str, float] | None = None):
        if not isinstance(model, DiscreteJointModel):
            raise TypeError("model must be a DiscreteJointModel")
        obs = {} if observations is None else dict(observations)
        idx: dict[str, int] = {}
        vals: dict[str, float] = {}
        for name, value in obs.items():
            if name not in SIGNALS:
                raise KeyError(f"cannot observe {name!r}")
            i = model.signal_index(name, value)
            idx[name] = i
            vals[name] = float(model.signal_values[name][i])
        index = []
        free: list[str] = []
        for name in AXES:
            if name in idx:
                index.append(idx[name])
            else:
                index.append(slice(None))
                free.append(name)
        sub = model.pmf[tuple(index)]
        mass = float(sub.sum())
        if not mass > 0.0:
            raise ValueError(f"observations {vals} have zero probability")
        self.model = model
        self._obs = vals
        self._free = tuple(free)  # always starts with "r"
        self._post = sub / mass

    # ------------------------------------------------------------ protocol
    @property
    def observed(self) -> frozenset[str]:
        return frozenset(self._obs)

    @property
    def observations(self) -> dict[str, float]:
        return dict(self._obs)

    def _marginal(self, name: str) -> np.ndarray:
        if name not in self._free:
            raise KeyError(f"{name!r} is not an unobserved coordinate")
        axis = self._free.index(name)
        other = tuple(i for i in range(len(self._free)) if i != axis)
        return self._post.sum(axis=other) if other else self._post.copy()

    def return_moments(self) -> tuple[float, float]:
        p = self._marginal("r")
        r = self.model.r_values
        return float(p @ r), float(p @ (r * r))

    def r_marginal(self) -> np.ndarray:
        """P(r = r_values[i] | b) as an array aligned with ``model.r_values``."""
        return self._marginal("r")

    def signal_marginal(self, source: str) -> np.ndarray:
        """P(Y_source = alphabet[k] | b) for an unobserved signal."""
        return self._marginal(source)

    def condition(self, source: str, value: float) -> "DiscreteBelief":
        if source not in SIGNALS:
            raise KeyError(f"cannot condition on {source!r}")
        if source in self._obs:
            raise KeyError(f"{source!r} already observed")
        i = self.model.signal_index(source, value)  # raises ValueError if out of alphabet
        if not self._marginal(source)[i] > 0.0:
            raise ValueError(f"{source}={value!r} has zero posterior probability")
        new = dict(self._obs)
        new[source] = float(self.model.signal_values[source][i])
        return DiscreteBelief(self.model, new)

    def predictive_branches(self, source: str, n_nodes: int | None = None) -> list[tuple[float, "DiscreteBelief"]]:
        """Exact (P(Y_source = y | b), b^{source, y}) over positive-probability outcomes."""
        if source not in SIGNALS:
            raise KeyError(f"unknown source {source!r}")
        if source in self._obs:
            raise KeyError(f"{source!r} already observed")
        p = self._marginal(source)
        values = self.model.signal_values[source]
        out: list[tuple[float, DiscreteBelief]] = []
        for k in np.flatnonzero(p > 0.0):
            new = dict(self._obs)
            new[source] = float(values[k])
            out.append((float(p[k]), DiscreteBelief(self.model, new)))
        total = sum(w for w, _ in out)
        if abs(total - 1.0) > 1e-12:
            raise FloatingPointError(f"predictive weights sum to {total!r}")
        return out

    def expected_stop_value_after(self, source: str, q_prev: float, params: UtilityParams) -> float:
        """Exact E_{Y_source|b}[G(b^{source, Y})] (finite sum over branches)."""
        return float(sum(w * stop_value(b, q_prev, params) for w, b in self.predictive_branches(source)))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DiscreteBelief(obs={self._obs})"


# --------------------------------------------------------------------------- values
def stop_value(belief, q_prev: float, params: UtilityParams) -> float:
    """G(b) = max_q E_b[u(q, r)] from the first two moments (src.utility formula)."""
    m, s2 = belief.return_moments()
    return max(float(expected_utility(q, m, s2, q_prev, params)) for q in POSITIONS)


def stop_choice(belief, q_prev: float, params: UtilityParams) -> tuple[float, int]:
    """(G(b), terminal position) under the CONTRACTS tie rule (src.utility.choose_position)."""
    m, s2 = belief.return_moments()
    return choose_position(m, s2, q_prev, params)


def myopic_voi(belief, source: str, cost: float, q_prev: float, params: UtilityParams) -> float:
    """Exact net one-step VOI_j(b) = E[G(b^{j,Y})] - G(b) - c_j, computed branch by branch."""
    if source in belief.observed:
        raise KeyError(f"{source!r} already observed")
    gross = sum(w * stop_value(b, q_prev, params) for w, b in belief.predictive_branches(source))
    return float(gross - stop_value(belief, q_prev, params) - cost)


def gross_myopic_voi(belief, source: str, q_prev: float, params: UtilityParams) -> float:
    """Exact gross one-step VOI E[G(b^{j,Y})] - G(b) (>= 0 by Jensen)."""
    return myopic_voi(belief, source, 0.0, q_prev, params)


def _check_setup(setup: AcquisitionSetup) -> None:
    extra = set(setup.costs) - set(SOURCES)
    if extra:
        raise ValueError(f"costs given for non-purchasable sources {sorted(extra)}")
    for name, c in setup.costs.items():
        if not (np.isfinite(c) and c >= 0):
            raise ValueError(f"cost of {name} must be finite and >= 0, got {c}")
    if not (np.isfinite(setup.budget) and setup.budget >= 0):
        raise ValueError(f"budget must be finite and >= 0, got {setup.budget}")


def _remaining_sources(belief, setup: AcquisitionSetup) -> tuple[str, ...]:
    """Purchasable sources not yet observed, in the tie-breaking order ("A", "B")."""
    return tuple(s for s in SOURCES if s in setup.costs and s not in belief.observed)


def exact_two_step_value(belief, setup: AcquisitionSetup, q_prev: float):
    """Exact Bellman recursion V(b, B, U) with the CONTRACTS tie rules.

    Returns (value, best_first_action, action_values). ``action_values`` contains
    STOP and every affordable remaining source; a source's value is
    -c_j + E[V(b^{j,Y}, B - c_j, U minus j)]. STOP wins ties; a source is preferred only
    if it beats the current best by more than QUERY_TOL; "A" is considered before "B".

    Independent of ``src.planner`` by construction.
    """
    _check_setup(setup)
    params = setup.params
    remaining = _remaining_sources(belief, setup)

    def value(b, budget_left: float, unqueried: tuple[str, ...]) -> float:
        best = stop_value(b, q_prev, params)
        for j in unqueried:
            if not setup.affordable(j, budget_left):
                continue
            cand = query_value(b, j, budget_left, unqueried)
            if cand > best + QUERY_TOL:
                best = cand
        return best

    def query_value(b, j: str, budget_left: float, unqueried: tuple[str, ...]) -> float:
        c = float(setup.costs[j])
        rest = tuple(s for s in unqueried if s != j)
        cont = sum(w * value(child, budget_left - c, rest) for w, child in b.predictive_branches(j))
        return -c + cont

    g_now = stop_value(belief, q_prev, params)
    action_values: dict[str, float] = {STOP: g_now}
    best_val, best_action = g_now, STOP
    for j in remaining:
        if not setup.affordable(j, setup.budget):
            continue
        cand = query_value(belief, j, setup.budget, remaining)
        action_values[j] = cand
        if cand > best_val + QUERY_TOL:
            best_val, best_action = cand, j
    return best_val, best_action, action_values


# --------------------------------------------------------------------------- brute force
PolicyTree = dict  # node key -> ("stop", q) | ("query", source); node key = ((src, value), ...)


def _hidden_worlds(belief: DiscreteBelief):
    """Positive-probability hidden worlds (weight, r, {unobserved signal: value}).

    Built directly from the model pmf restricted to the observed values; the weights are
    conditional on the observations. Independent of DiscreteBelief's posterior arrays.
    """
    if not isinstance(belief, DiscreteBelief):
        raise TypeError("brute force needs a DiscreteBelief")
    model = belief.model
    idx = {name: model.signal_index(name, v) for name, v in belief.observations.items()}
    index = tuple(idx.get(name, slice(None)) for name in AXES)
    sub = model.pmf[index]
    free = [name for name in AXES if name not in idx]
    total = float(sub.sum())
    worlds = []
    for pos in np.ndindex(sub.shape):
        w = float(sub[pos])
        if w <= 0.0:
            continue
        coords = dict(zip(free, pos))
        r = float(model.r_values[coords["r"]])
        hidden = {name: float(model.signal_values[name][coords[name]]) for name in free if name != "r"}
        worlds.append((w / total, r, hidden))
    return worlds


def _subtrees(history, budget_left, unqueried, worlds, setup) -> list[PolicyTree]:
    """All deterministic adaptive subtrees rooted at ``history``.

    Options at a node: stop with any position in POSITIONS, or query any affordable
    unqueried source. For every positive-probability outcome of a query there is a child
    node, and each child gets its own independent subtree. Zero-probability outcomes are
    not enumerated: their actions cannot affect any expected value.
    """
    out: list[PolicyTree] = [{history: ("stop", q)} for q in POSITIONS]
    for j in unqueried:
        if not setup.affordable(j, budget_left):
            continue
        consistent = [hidden for _, _, hidden in worlds
                      if all(hidden[src] == v for src, v in history)]
        outcomes = sorted({hidden[j] for hidden in consistent})
        children = [
            _subtrees(history + ((j, v),), budget_left - setup.costs[j],
                      tuple(s for s in unqueried if s != j), worlds, setup)
            for v in outcomes
        ]
        for combo in itertools.product(*children):
            tree: PolicyTree = {history: ("query", j)}
            for sub in combo:
                tree.update(sub)
            out.append(tree)
    return out


def enumerate_policy_trees(belief: DiscreteBelief, setup: AcquisitionSetup) -> list[PolicyTree]:
    """Every deterministic adaptive acquisition-and-position policy tree.

    The list is ordered so that STOP options come before queries and "A" before "B",
    which makes the first of several equally good trees the one that stops or buys in
    the contract tie order.
    """
    _check_setup(setup)
    worlds = _hidden_worlds(belief)
    return _subtrees((), setup.budget, _remaining_sources(belief, setup), worlds, setup)


def _evaluate(tree: PolicyTree, worlds, setup: AcquisitionSetup, q_prev: float) -> float:
    params = setup.params
    total = 0.0
    for w, r, hidden in worlds:
        history: tuple = ()
        spent = 0.0
        queried: list[str] = []
        budget_left = setup.budget
        while True:
            if history not in tree:
                raise KeyError(f"policy tree has no action at node {history}")
            kind, arg = tree[history]
            if kind == "stop":
                if arg not in POSITIONS:
                    raise ValueError(f"illegal position {arg}")
                payoff = float(realized_utility(arg, r, q_prev, params))
                break
            if kind != "query":
                raise ValueError(f"unknown action {kind!r}")
            if arg not in SOURCES or arg in queried or arg not in setup.costs:
                raise ValueError(f"illegal query {arg!r} at node {history}")
            if not setup.affordable(arg, budget_left):
                raise ValueError(f"unaffordable query {arg!r} at node {history}")
            spent += float(setup.costs[arg])
            budget_left -= float(setup.costs[arg])
            queried.append(arg)
            history = history + ((arg, hidden[arg]),)
        total += w * (payoff - spent)
    return total


def evaluate_policy_tree(tree: PolicyTree, belief: DiscreteBelief, setup: AcquisitionSetup, q_prev: float) -> float:
    """Expected net utility of a policy tree by enumerating the hidden world."""
    _check_setup(setup)
    return _evaluate(tree, _hidden_worlds(belief), setup, q_prev)


def brute_force_optimal(belief: DiscreteBelief, setup: AcquisitionSetup, q_prev: float) -> tuple[float, PolicyTree]:
    """Best value over all policy trees and one optimal tree (first in enumeration order)."""
    _check_setup(setup)
    worlds = _hidden_worlds(belief)
    best_v, best_t = -np.inf, None
    for tree in enumerate_policy_trees(belief, setup):
        v = _evaluate(tree, worlds, setup, q_prev)
        if v > best_v + TREE_TIE_TOL:
            best_v, best_t = v, tree
    return float(best_v), best_t


def reachable_depth_actions(tree: PolicyTree) -> dict[int, set]:
    """Map node depth (number of observed values) to the set of actions taken there."""
    out: dict[int, set] = {}
    for key, act in tree.items():
        out.setdefault(len(key), set()).add(act)
    return out


# --------------------------------------------------------------------------- scenarios
@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    belief: DiscreteBelief
    setup: AcquisitionSetup
    q_prev: float
    expect: Mapping[str, object] = field(default_factory=dict)


def _factor(p_r, p_y0, p_a_given_r, p_b_given_r) -> np.ndarray:
    return np.einsum("r,y,ra,rb->ryab", np.asarray(p_r, float), np.asarray(p_y0, float),
                     np.asarray(p_a_given_r, float), np.asarray(p_b_given_r, float))


def _model(r_vals, pmf, a_vals=(-1.0, 1.0), b_vals=(-1.0, 1.0), y0_vals=(0.0,)) -> DiscreteJointModel:
    return DiscreteJointModel(np.asarray(r_vals, float), pmf,
                              {"y0": np.asarray(y0_vals, float), "A": np.asarray(a_vals, float),
                               "B": np.asarray(b_vals, float)})


R_SIGNED = (-0.01, 0.01)
LAM, TAU = 1.0, 1e-4
DELTA = 0.01  # |r| in the XOR instance


def _informative_pmf(p_up: float = 0.5, hit: float = 0.75):
    """A and B conditionally independent given r; P(sign(signal) = sign(r)) = hit."""
    p_r = [1 - p_up, p_up]
    cond = np.array([[hit, 1 - hit], [1 - hit, hit]])  # rows: r=-, r=+; cols: signal -1, +1
    return _factor(p_r, [1.0], cond, cond)


def _scen_zero_cost_information() -> Scenario:
    model = _model(R_SIGNED, _informative_pmf(0.5, 0.7))
    setup = AcquisitionSetup(costs={"A": 0.0, "B": 0.0}, budget=0.0, params=UtilityParams(LAM, TAU))
    return Scenario(
        "zero_cost_information",
        "Both sources free and informative. V equals the full-information value E[G(b^{AB})] "
        "and is at least G; the optimal first action is a source, never STOP.",
        DiscreteBelief(model, {"y0": 0.0}), setup, 0.0,
        {"G": 0.0, "action_in": ("A", "B"), "full_info": True},
    )


def _scen_uninformative() -> Scenario:
    p_a = np.array([0.5, 0.5])
    pmf = _factor([0.4, 0.6], [1.0], np.tile(p_a, (2, 1)), np.tile(p_a, (2, 1)))
    model = _model(R_SIGNED, pmf)
    setup = AcquisitionSetup(costs={"A": 1e-4, "B": 1e-4}, budget=1e-3, params=UtilityParams(LAM, TAU))
    return Scenario(
        "uninformative",
        "Signals independent of r and of each other. Gross VOI is 0, net VOI equals -c exactly, "
        "and the optimal policy stops immediately.",
        DiscreteBelief(model, {"y0": 0.0}), setup, 0.0,
        {"net_voi": {"A": -1e-4, "B": -1e-4}, "action": STOP},
    )


def _scen_uninformative_zero_cost() -> Scenario:
    p_a = np.array([0.5, 0.5])
    pmf = _factor([0.4, 0.6], [1.0], np.tile(p_a, (2, 1)), np.tile(p_a, (2, 1)))
    model = _model(R_SIGNED, pmf)
    setup = AcquisitionSetup(costs={"A": 0.0, "B": 0.0}, budget=0.0, params=UtilityParams(LAM, TAU))
    return Scenario(
        "uninformative_zero_cost",
        "Free but worthless signals: net VOI is exactly 0 for both sources, so the STOP-wins-ties "
        "rule makes the optimal action STOP.",
        DiscreteBelief(model, {"y0": 0.0}), setup, 0.0,
        {"net_voi": {"A": 0.0, "B": 0.0}, "action": STOP},
    )


def _scen_duplicate() -> Scenario:
    p_a_given_r = np.array([[0.2, 0.8], [0.8, 0.2]])  # rows r=-, r=+ ; cols A=-1, A=+1
    pmf = np.einsum("r,y,ra,ab->ryab", [0.3, 0.7], [1.0], p_a_given_r, np.eye(2))  # B == A
    model = _model(R_SIGNED, pmf)
    setup = AcquisitionSetup(costs={"A": 2e-4, "B": 2e-4}, budget=1e-3, params=UtilityParams(LAM, TAU))
    return Scenario(
        "duplicate",
        "B is an exact copy of A (B == A in every world). After A, VOI_B = -c_B exactly; the "
        "optimal policy buys A (tie with B resolved to A) and never buys both.",
        DiscreteBelief(model, {"y0": 0.0}), setup, 0.0,
        {"action": "A", "after_A_voi_B": -2e-4, "never_buy_both": True},
    )


def _scen_independent_informative() -> Scenario:
    model = _model(R_SIGNED, _informative_pmf(0.5, 0.75))
    setup = AcquisitionSetup(costs={"A": 1e-4, "B": 1e-4}, budget=1e-3, params=UtilityParams(LAM, TAU))
    return Scenario(
        "independent_informative",
        "A and B conditionally independent given r, each informative. Both one-step VOIs are "
        "positive; the optimal policy acquires and V >= G + max_j VOI_j.",
        DiscreteBelief(model, {"y0": 0.0}), setup, 0.0,
        {"voi_positive": ("A", "B"), "action_in": ("A", "B")},
    )


def _scen_xor() -> Scenario:
    # r = +d if A xor B = 1 else -d, with A, B independent fair bits.
    # Each signal alone is uninformative about the sign, the pair reveals it.
    pmf = np.zeros((2, 1, 2, 2))
    for a in range(2):
        for b in range(2):
            pmf[a ^ b, 0, a, b] = 0.25
    model = _model((-DELTA, DELTA), pmf, a_vals=(0.0, 1.0), b_vals=(0.0, 1.0))
    setup = AcquisitionSetup(costs={"A": 1e-4, "B": 1e-4}, budget=2e-4, params=UtilityParams(LAM, TAU))
    # G(prior) = 0 (E r = 0, q = 0). After one signal E r = 0 so G stays 0 (VOI = -c).
    # After both, E r = +-d, E r^2 = d^2, G = d - lam d^2 - tau = 0.0098 in each world.
    # Two-step value: -2c + E G(full) = 0.0098 - 2e-4 = 0.0096.
    return Scenario(
        "xor",
        "Each one-step VOI is -c <= 0 (myopic stops) but buying both reveals the sign of r. "
        "Two-step planning buys A then B with V = 0.0096 > G = 0.",
        DiscreteBelief(model, {"y0": 0.0}), setup, 0.0,
        {"G": 0.0, "net_voi": {"A": -1e-4, "B": -1e-4}, "value": 0.0096, "action": "A"},
    )


def _scen_insufficient_budget() -> Scenario:
    model = _model(R_SIGNED, _informative_pmf(0.5, 0.75))
    setup = AcquisitionSetup(costs={"A": 1e-4, "B": 1e-4}, budget=5e-5, params=UtilityParams(LAM, TAU))
    return Scenario(
        "insufficient_budget",
        "Budget below the cheapest source: no source is affordable, so V = G and action is STOP.",
        DiscreteBelief(model, {"y0": 0.0}), setup, 0.0,
        {"action": STOP, "value_equals_G": True, "action_values_keys": (STOP,)},
    )


def _scen_budget_one_source() -> Scenario:
    model = _model(R_SIGNED, _informative_pmf(0.5, 0.75))
    setup = AcquisitionSetup(costs={"A": 1e-4, "B": 1e-4}, budget=1.5e-4, params=UtilityParams(LAM, TAU))
    return Scenario(
        "budget_one_source",
        "Budget covers exactly one purchase: after the first purchase the remaining budget is "
        "below every cost, so no second purchase. V = G + max(0, VOI_A, VOI_B).",
        DiscreteBelief(model, {"y0": 0.0}), setup, 0.0,
        {"no_second_purchase": True, "action": "A"},
    )


def _scen_budget_equals_cost() -> Scenario:
    model = _model(R_SIGNED, _informative_pmf(0.5, 0.75))
    setup = AcquisitionSetup(costs={"A": 1e-4, "B": 2e-4}, budget=1e-4, params=UtilityParams(LAM, TAU))
    return Scenario(
        "budget_equals_cost",
        "Budget equals c_A exactly, so A is affordable (budget comparison uses BUDGET_ATOL); "
        "B (cost 2e-4) is not. Action values contain only STOP and A.",
        DiscreteBelief(model, {"y0": 0.0}), setup, 0.0,
        {"action_values_keys": (STOP, "A"), "action": "A"},
    )


def _scen_no_remaining_sources() -> Scenario:
    model = _model(R_SIGNED, _informative_pmf(0.5, 0.75))
    setup = AcquisitionSetup(costs={"A": 1e-4, "B": 1e-4}, budget=1e-3, params=UtilityParams(LAM, TAU))
    return Scenario(
        "no_remaining_sources",
        "Both sources already observed: nothing to acquire, V = G and the only action is STOP.",
        DiscreteBelief(model, {"y0": 0.0, "A": -1.0, "B": 1.0}), setup, 0.0,
        {"action": STOP, "value_equals_G": True, "action_values_keys": (STOP,)},
    )


def _scen_terminal_tie(q_prev: float) -> Scenario:
    # r in {0, 1}, P(r = 1) = 0.3, tau = 0: E r = E r^2 = 0.3, so u(1) = u(0) = 0 exactly.
    p_a = np.array([0.5, 0.5])
    pmf = _factor([0.7, 0.3], [1.0], np.tile(p_a, (2, 1)), np.tile(p_a, (2, 1)))
    model = _model((0.0, 1.0), pmf)
    setup = AcquisitionSetup(costs={"A": 1e-3, "B": 1e-3}, budget=1e-3, params=UtilityParams(1.0, 0.0))
    expected_q = 0 if q_prev == 0.0 else 1
    return Scenario(
        f"terminal_tie_qprev{int(q_prev)}",
        "q=0 and q=1 tie exactly for the stop decision; the CONTRACTS rule picks the smallest "
        "|q - q_prev|.",
        DiscreteBelief(model, {"y0": 0.0}), setup, q_prev,
        {"G": 0.0, "stop_position": expected_q, "action": STOP},
    )


def scenarios() -> list[Scenario]:
    """All named deterministic scenarios, in a fixed order."""
    return [
        _scen_zero_cost_information(),
        _scen_uninformative(),
        _scen_uninformative_zero_cost(),
        _scen_duplicate(),
        _scen_independent_informative(),
        _scen_xor(),
        _scen_insufficient_budget(),
        _scen_budget_one_source(),
        _scen_budget_equals_cost(),
        _scen_no_remaining_sources(),
        _scen_terminal_tie(0.0),
        _scen_terminal_tie(1.0),
    ]


def scenario(name: str) -> Scenario:
    for s in scenarios():
        if s.name == name:
            return s
    raise KeyError(name)


# --------------------------------------------------------------------------- random instances
def random_instance(seed: int, n_a: int = 2, n_b: int = 2, n_r: int | None = None):
    """Seeded random small instance: (belief, setup, q_prev).

    The belief observes y0 with probability 0.6 and each purchasable source with
    probability 0.15 (at a value that has positive probability, drawn from a random
    world), so the belief is always well defined.
    """
    rng = np.random.default_rng(seed)
    nr = int(rng.integers(2, 4)) if n_r is None else n_r
    ny = int(rng.integers(1, 3))
    r_vals = np.sort(rng.uniform(-0.02, 0.02, nr))
    y_vals = _random_alphabet(rng, ny)
    a_vals = _random_alphabet(rng, n_a)
    b_vals = _random_alphabet(rng, n_b)
    shape = (nr, ny, n_a, n_b)
    w = rng.gamma(0.5, size=int(np.prod(shape)))
    w[rng.random(w.size) < 0.25] = 0.0
    if w.sum() <= 0:
        w[0] = 1.0
    pmf = (w / w.sum()).reshape(shape)
    model = DiscreteJointModel(r_vals, pmf, {"y0": y_vals, "A": a_vals, "B": b_vals})

    world = np.unravel_index(rng.choice(pmf.size, p=pmf.ravel()), shape)
    obs: dict[str, float] = {}
    if rng.random() < 0.6:
        obs["y0"] = float(y_vals[world[1]])
    if rng.random() < 0.15:
        obs["A"] = float(a_vals[world[2]])
    if rng.random() < 0.15:
        obs["B"] = float(b_vals[world[3]])
    belief = DiscreteBelief(model, obs)

    costs = {}
    for name in ("A", "B"):
        costs[name] = 0.0 if rng.random() < 0.15 else float(rng.uniform(1e-4, 3e-3))
    cA, cB = costs["A"], costs["B"]
    budget_options = [0.0, cA, cB, cA + cB, 1.5 * max(cA, cB), float(rng.uniform(0.0, 4e-3)),
                      (cA + cB) * (1.0 - 1e-9)]
    budget = float(budget_options[int(rng.integers(len(budget_options)))])
    params = UtilityParams(lam=float(rng.choice([0.0, 0.5, 1.0, 3.0])),
                           tau=float(rng.choice([0.0, 1e-4, 1e-3])))
    q_prev = float(rng.choice([-1.0, 0.0, 1.0]))
    return belief, AcquisitionSetup(costs=costs, budget=budget, params=params), q_prev


def _random_alphabet(rng: np.random.Generator, n: int) -> np.ndarray:
    base = np.linspace(-1.0, 1.0, n) if n > 1 else np.array([0.0])
    return base + rng.uniform(-0.05, 0.05, n)
