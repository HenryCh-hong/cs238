"""Shared contracts between the belief models, the planner, the policies and the replay.

Sources
-------
"A" = random-forest forecast, "B" = gradient-boosting forecast. The free linear
forecast is "y0" and is always observed before any decision. The return is "r".

Belief protocol
---------------
Any belief over the next-period return r that the planner can reason about must
implement :class:`Belief`. Two implementations exist:

* ``src.belief_model.GaussianBelief``  - joint Gaussian over (r, y0, A, B)
* ``src.toy_simulator.DiscreteBelief`` - exact finite joint pmf (validation testbed)

The planner (``src.planner``) is written only against this protocol, so the exact
discrete testbed validates the same recursion that runs on historical data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

from src.utility import UtilityParams

SOURCES: tuple[str, ...] = ("A", "B")
SOURCE_MODEL = {"A": "random_forest", "B": "gradient_boosting"}
FREE_SOURCE = "y0"
STOP = "stop"

# Budget comparisons use this absolute slack so that c_j == budget is affordable
# despite floating-point accumulation.
BUDGET_ATOL = 1e-15


@runtime_checkable
class Belief(Protocol):
    """Belief over r after observing a set of forecasts. Immutable."""

    @property
    def observed(self) -> frozenset[str]:
        """Names of the sources already observed (never includes unqueried ones)."""

    def return_moments(self) -> tuple[float, float]:
        """(E_b[r], E_b[r^2])."""

    def predictive_branches(self, source: str, n_nodes: int | None = None) -> Sequence[tuple[float, "Belief"]]:
        """Preposterior branches for querying an unobserved ``source``:
        a list of (weight, updated belief); weights are >= 0 and sum to 1.
        Exact for discrete beliefs; deterministic quadrature for Gaussian beliefs."""

    def condition(self, source: str, value: float) -> "Belief":
        """Belief after actually observing ``source`` = ``value``."""


@dataclass(frozen=True)
class AcquisitionSetup:
    """Per-decision acquisition constraints (one decision date = one episode)."""

    costs: Mapping[str, float]
    budget: float
    params: UtilityParams = field(default_factory=UtilityParams)

    def affordable(self, source: str, budget_left: float) -> bool:
        return self.costs[source] <= budget_left + BUDGET_ATOL


@dataclass(frozen=True)
class PlanResult:
    """Output of a planning call at a belief.

    value:         planner's value V(b, B, U) (or the policy-specific analogue)
    action:        STOP or a source name to query next
    stop_value:    G(b)
    stop_position: argmax position if stopping now
    action_values: value of each candidate first action (STOP included)
    """

    value: float
    action: str
    stop_value: float
    stop_position: int
    action_values: Mapping[str, float]
