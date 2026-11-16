"""Acquisition policies, the hidden-value forecast vault, and the episode runner.

Every policy maps the current belief and the remaining budget to one action: STOP or the
name of one source to buy next. A policy never sees a forecast value; values are revealed
only by ``ForecastVault.query``, which charges the cost once and refuses repeats and
unaffordable requests. After the last purchase, the terminal position is chosen by
``utility.choose_position`` on the final belief for every policy.

Policies
--------
* FreeOnly                  never buys anything.
* AlwaysBuy                 buys every affordable remaining source in order A, B.
* FixedOrder(order)         optimal stopping along a fixed order (``planner.solve_fixed_order``).
* RandomAcquisition(p)      each step: with probability p buy a uniformly random affordable
                            remaining source, else stop. Randomness comes only from the rng
                            passed to ``run_episode``.
* MyopicVOI                 buy the argmax net one-step VOI if it exceeds ``PLAN_TOL``.
* OptimalVOI                act per ``planner.solve`` and re-solve after each observation.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src import planner, voi
from src.interfaces import BUDGET_ATOL, SOURCES, STOP, AcquisitionSetup
from src.utility import choose_position

Action = str  # STOP or a source name


class Policy:
    """Interface: ``next_action`` returns (action, diagnostics). Deterministic unless
    ``deterministic`` is False. ``rng`` is only used by stochastic policies."""

    name: str = "policy"
    deterministic: bool = True

    def next_action(
        self,
        belief: Any,
        remaining: Sequence[str],
        budget_left: float,
        setup: AcquisitionSetup,
        q_prev: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[Action, dict[str, Any]]:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.name})"


def _available(remaining: Iterable[str], budget_left: float, setup: AcquisitionSetup) -> list[str]:
    """Remaining sources that are affordable now, in the SOURCES order (A before B)."""
    rem = set(remaining)
    return [s for s in SOURCES if s in rem and setup.affordable(s, budget_left)]


class FreeOnly(Policy):
    name = "free_only"

    def next_action(self, belief, remaining, budget_left, setup, q_prev, rng=None):
        return STOP, {"rule": "free_only"}


class AlwaysBuy(Policy):
    name = "always_buy"

    def next_action(self, belief, remaining, budget_left, setup, q_prev, rng=None):
        cands = _available(remaining, budget_left, setup)
        if not cands:
            return STOP, {"rule": "always_buy", "affordable": []}
        return cands[0], {"rule": "always_buy", "affordable": cands}


class FixedOrder(Policy):
    def __init__(self, order: Sequence[str]):
        self.order = tuple(order)
        if len(set(self.order)) != len(self.order) or not set(self.order) <= set(SOURCES):
            raise ValueError(f"invalid fixed order {self.order}")
        self.name = "fixed_" + "".join(self.order) if self.order else "fixed_none"

    def next_action(self, belief, remaining, budget_left, setup, q_prev, rng=None):
        order = tuple(s for s in self.order if s in set(remaining))
        res = planner.solve_fixed_order(belief, order, budget_left, setup, q_prev)
        return res.action, {
            "rule": "fixed_order",
            "order": order,
            "value": res.value,
            "stop_value": res.stop_value,
            "action_values": dict(res.action_values),
        }


class RandomAcquisition(Policy):
    name = "random"
    deterministic = False

    def __init__(self, p: float):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)

    def next_action(self, belief, remaining, budget_left, setup, q_prev, rng=None):
        if rng is None:
            raise ValueError("RandomAcquisition needs the episode rng")
        cands = _available(remaining, budget_left, setup)
        u = float(rng.random())
        if not cands or u >= self.p:
            return STOP, {"rule": "random", "u": u, "candidates": cands}
        pick = cands[int(rng.integers(len(cands)))]
        return pick, {"rule": "random", "u": u, "candidates": cands}


class MyopicVOI(Policy):
    name = "myopic_voi"

    def __init__(self, n_nodes: int | None = None):
        self.n_nodes = n_nodes

    def next_action(self, belief, remaining, budget_left, setup, q_prev, rng=None):
        cands = _available(remaining, budget_left, setup)
        vois = {
            j: voi.one_step_voi(belief, j, setup.costs[j], q_prev, setup.params, self.n_nodes)
            for j in cands
        }
        diag: dict[str, Any] = {"rule": "myopic_voi", "voi": dict(vois)}
        if not cands:
            return STOP, diag
        best = cands[0]
        for j in cands[1:]:
            if vois[j] > vois[best] + planner.PLAN_TOL:
                best = j
        if vois[best] > planner.PLAN_TOL:
            return best, diag
        return STOP, diag


class OptimalVOI(Policy):
    name = "optimal_voi"

    def __init__(self, n_nodes: int | None = None):
        self.n_nodes = n_nodes

    def next_action(self, belief, remaining, budget_left, setup, q_prev, rng=None):
        res = planner.solve(belief, remaining, budget_left, setup, q_prev, self.n_nodes)
        return res.action, {
            "rule": "optimal_voi",
            "value": res.value,
            "stop_value": res.stop_value,
            "action_values": dict(res.action_values),
        }


# ---------------------------------------------------------------------------- vault
class ForecastVault:
    """Holds the purchasable forecasts privately and reveals one only on ``query``.

    * ``query(source, budget_left)`` returns the hidden value, charges the cost once, and
      records the query in ``log``.
    * Raises KeyError for unknown or free sources (including ``"y0"``), RuntimeError on a
      repeated query, and RuntimeError when the cost exceeds ``budget_left``.
    * The values live in a name-mangled attribute; there is no public accessor.
    """

    def __init__(self, values: Mapping[str, float], costs: Mapping[str, float]):
        vals = {str(k): float(v) for k, v in values.items()}
        cst = {str(k): float(c) for k, c in costs.items()}
        if set(vals) != set(cst):
            raise ValueError(f"values and costs must cover the same sources: {sorted(vals)} vs {sorted(cst)}")
        for s, c in cst.items():
            if s not in SOURCES:
                raise ValueError(f"{s!r} is not a purchasable source")
            if not (math.isfinite(c) and c >= 0.0):
                raise ValueError(f"cost of {s} must be finite and >= 0, got {c}")
        for s, v in vals.items():
            if not math.isfinite(v):
                raise ValueError(f"non-finite forecast value for {s}")
        self.__values = vals
        self._costs = cst
        self._queried: set[str] = set()
        self._spent = 0.0
        self._log: list[dict[str, float | str]] = []

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(s for s in SOURCES if s in self._costs)

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def log(self) -> tuple[dict[str, float | str], ...]:
        return tuple(dict(e) for e in self._log)

    def cost_of(self, source: str) -> float:
        if source not in self._costs:
            raise KeyError(f"unknown source {source!r}")
        return self._costs[source]

    def query(self, source: str, budget_left: float) -> float:
        if source not in self._costs:
            raise KeyError(f"unknown or free source {source!r}; the vault holds {self.sources}")
        if source in self._queried:
            raise RuntimeError(f"source {source!r} was already purchased on this date")
        cost = self._costs[source]
        if cost > float(budget_left) + BUDGET_ATOL:
            raise RuntimeError(f"cannot afford {source!r}: cost {cost} > budget left {budget_left}")
        value = self.__values[source]
        self._queried.add(source)
        self._spent += cost
        self._log.append({"source": source, "cost": cost, "budget_left_before": float(budget_left)})
        return value


# ---------------------------------------------------------------------------- episode
@dataclass
class EpisodeResult:
    policy: str
    purchases: tuple[str, ...]
    steps: list[dict[str, Any]]
    total_cost: float
    budget_left: float
    initial_mean: float
    initial_var: float
    final_mean: float
    final_var: float
    G: float
    q: int
    initial_predicted_value: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _mean_var(belief: Any) -> tuple[float, float]:
    m, s2 = belief.return_moments()
    var = getattr(belief, "var", None)
    if var is None:
        var = s2 - m * m
    return float(m), float(var)


def run_episode(
    policy: Policy,
    belief: Any,
    vault: ForecastVault,
    setup: AcquisitionSetup,
    q_prev: float,
    rng: np.random.Generator | None,
) -> EpisodeResult:
    """Run one acquisition episode on one decision date.

    The loop asks the policy for an action, buys it through ``vault.query`` (the only path
    to a hidden value), conditions the belief, and stops when the policy stops or no source
    is affordable. Per step the diagnostics record the belief mean and variance before the
    step, the budget left, the net one-step VOI of every available source, the action, and
    the policy's predicted value (when it reports one).
    """
    if rng is None and not policy.deterministic:
        raise ValueError("a stochastic policy needs an rng")
    if set(vault.sources) != set(setup.costs):
        raise ValueError(f"vault sources {vault.sources} must match setup costs {tuple(setup.costs)}")
    for s in vault.sources:
        if abs(vault.cost_of(s) - setup.costs[s]) > BUDGET_ATOL:
            raise ValueError(f"vault cost of {s} differs from the setup cost")

    budget_left = float(setup.budget)
    remaining = tuple(s for s in SOURCES if s in setup.costs and s not in belief.observed)
    m0, v0 = _mean_var(belief)
    purchases: list[str] = []
    steps: list[dict[str, Any]] = []
    initial_pred: float | None = None

    while True:
        m_before, v_before = _mean_var(belief)
        avail = _available(remaining, budget_left, setup)
        vois = {
            j: voi.one_step_voi(belief, j, setup.costs[j], q_prev, setup.params) for j in avail
        }
        action, diag = policy.next_action(belief, remaining, budget_left, setup, q_prev, rng=rng)
        if action != STOP and action not in avail:
            raise ValueError(f"policy {policy.name} chose {action!r}, which is not available")
        pred = diag.get("value")
        if initial_pred is None and pred is not None:
            initial_pred = float(pred)
        steps.append(
            {
                "step": len(steps),
                "mean_before": m_before,
                "var_before": v_before,
                "budget_left_before": budget_left,
                "voi": dict(vois),
                "action": action,
                "predicted_value": None if pred is None else float(pred),
                "cost": 0.0 if action == STOP else float(setup.costs[action]),
            }
        )
        if action == STOP:
            break
        value = vault.query(action, budget_left)
        budget_left = max(budget_left - float(setup.costs[action]), 0.0)
        belief = belief.condition(action, value)
        remaining = tuple(s for s in remaining if s != action)
        purchases.append(action)

    m1, v1 = _mean_var(belief)
    s2 = belief.return_moments()[1]
    G, q = choose_position(m1, s2, q_prev, setup.params)
    return EpisodeResult(
        policy=policy.name,
        purchases=tuple(purchases),
        steps=steps,
        total_cost=float(vault.spent),
        budget_left=budget_left,
        initial_mean=m0,
        initial_var=v0,
        final_mean=m1,
        final_var=v1,
        G=float(G),
        q=int(q),
        initial_predicted_value=initial_pred,
    )


__all__ = [
    "Policy",
    "FreeOnly",
    "AlwaysBuy",
    "FixedOrder",
    "RandomAcquisition",
    "MyopicVOI",
    "OptimalVOI",
    "ForecastVault",
    "EpisodeResult",
    "run_episode",
]
