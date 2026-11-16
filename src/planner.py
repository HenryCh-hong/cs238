"""Exact two-step acquisition planner: the Bellman recursion of the notes.

    V(b, B, U) = max{ G(b), max_{j in U, c_j <= B} [ -c_j + E_{Y_j|b} V(b^{j,Y_j}, B - c_j, U \\ {j}) ] }

* U is the set of sources still available on this decision date (each source at most
  once); B is the budget left on the date (affordability uses ``BUDGET_ATOL``).
* Ties: STOP wins against a query unless the query is better by more than ``PLAN_TOL``;
  between sources the order ("A", "B") wins ties (a later source must beat the earlier
  one by more than ``PLAN_TOL``).
* The last stage is exact: ``E[G(b^{j,Y_j})]`` comes from ``belief.expected_stop_value_after``
  when the belief provides it, else from ``predictive_branches``.

Two code paths give the same recursion:

* ``solve`` on a generic ``Belief`` recurses over ``predictive_branches(source, n_nodes)``.
* ``solve`` on a ``GaussianBelief`` uses a vectorized fast path. Given Y_j, the child
  mean is m + sigma_j Z with Z ~ N(0, 1) and sigma_j = Cov(r, Y_j | O)/sd(Y_j | O), and
  the child variance does not depend on Y_j. The outer expectation over Z is integrated
  by composite Gauss-Legendre on subintervals split at the kinks of the integrand
  (breakpoints of G and the stop/continue crossings). The integrand is then analytic on
  each subinterval, so the rule converges fast in ``n_nodes`` (see docs/notes_D.md).

``reference_query_value_gaussian`` is the high-accuracy reference for the Gaussian path:
adaptive ``scipy.integrate.quad`` on the same kink-split intervals with brentq-located kinks.
It is slow (seconds) and is used only by tests and the convergence study.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import roots_legendre
from scipy.stats import norm

from src import voi
from src.belief_model import GaussianBelief
from src.interfaces import BUDGET_ATOL, SOURCES, STOP, AcquisitionSetup, PlanResult
from src.utility import UtilityParams, choose_position, stop_value_array

# Absolute tolerance (utility units) for declaring a query strictly better than stopping,
# and one source strictly better than another. Far below every VOI scale in the study
# (VOIs are 1e-8 and up in return units) and far above rounding noise (~1e-17).
PLAN_TOL = 1e-12

# Truncation of the standard-normal outer integral. Beyond |z| = 12 the mass is
# below 1e-32, so the truncation is exact to machine precision for any bounded integrand.
Z_MAX = 12.0

# Gauss-Legendre order per panel used when ``n_nodes`` is None on the Gaussian path
# (measured error table in docs/notes_D.md: 1e-16 in V from order 8 on every regime).
DEFAULT_OUTER_NODES = 51

# Maximum panel width (in z units) of the composite Gauss-Legendre outer rule.
PANEL_WIDTH = 1.0
# Panel width used by the adaptive-quad reference (narrower, see _integrate_reference).
REFERENCE_PANEL_WIDTH = 0.5

# Grid used to locate stop/continue crossings in z (refined afterwards).
_KINK_GRID = 801


# ---------------------------------------------------------------------------- validation
def _ordered(sources: Iterable[str]) -> tuple[str, ...]:
    names = list(sources)
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate sources in {names}")
    for s in names:
        if s not in SOURCES:
            raise ValueError(f"{s!r} is not a purchasable source (expected one of {SOURCES})")
    return tuple(s for s in SOURCES if s in names)


def _validate_remaining(belief: Any, remaining: Iterable[str]) -> tuple[str, ...]:
    U = _ordered(remaining)
    seen = [s for s in U if s in belief.observed]
    if seen:
        raise ValueError(f"sources already observed in the belief: {seen}")
    return U


def _validate_priced(sources: Iterable[str], setup: AcquisitionSetup) -> None:
    """Every source the planner may buy must have a cost in the setup (else a clear error, not KeyError)."""
    unpriced = [s for s in sources if s not in setup.costs]
    if unpriced:
        raise ValueError(f"sources {unpriced} have no cost in the acquisition setup {sorted(setup.costs)}")


def _stop(belief: Any, q_prev: float, params: UtilityParams) -> tuple[float, int]:
    m, s2 = belief.return_moments()
    return choose_position(m, s2, q_prev, params)


def _select(stop_value: float, values: Mapping[str, float], order: Sequence[str]) -> tuple[float, str]:
    """Decide between STOP and the queries in ``values`` (candidate value per source)."""
    best: str | None = None
    for j in order:
        if j not in values:
            continue
        if best is None or values[j] > values[best] + PLAN_TOL:
            best = j
    if best is None or values[best] <= stop_value + PLAN_TOL:
        return stop_value, STOP
    return values[best], best


def _result(stop_value: float, q_star: int, values: Mapping[str, float], order: Sequence[str]) -> PlanResult:
    value, action = _select(stop_value, values, order)
    action_values: dict[str, float] = {STOP: stop_value}
    action_values.update({j: values[j] for j in order if j in values})
    return PlanResult(
        value=float(value),
        action=action,
        stop_value=float(stop_value),
        stop_position=int(q_star),
        action_values=action_values,
    )


def _affordable_after(setup: AcquisitionSetup, sources: Iterable[str], budget_after: float) -> list[str]:
    return [k for k in sources if setup.affordable(k, budget_after)]


def _budget_after(setup: AcquisitionSetup, j: str, budget: float) -> float:
    return max(float(budget) - float(setup.costs[j]), 0.0)


# ---------------------------------------------------------------------------- generic path
def _query_value_generic(
    belief: Any,
    j: str,
    U: Sequence[str],
    budget: float,
    setup: AcquisitionSetup,
    q_prev: float,
    n_nodes: int | None,
) -> float:
    """-c_j + E_{Y_j|b} V(b^{j,Y_j}, B - c_j, U \\ {j})."""
    rest = tuple(k for k in U if k != j)
    b_next = _budget_after(setup, j, budget)
    if not _affordable_after(setup, rest, b_next):
        # Last stage: the continuation is stop-only, so the expectation is E[G] (exact if available).
        EV = voi.expected_stop_after(belief, j, q_prev, setup.params, n_nodes)
    else:
        EV = 0.0
        for w, child in belief.predictive_branches(j, n_nodes):
            EV += w * solve(child, rest, b_next, setup, q_prev, n_nodes).value
    return -float(setup.costs[j]) + EV


def _solve_generic(
    belief: Any,
    U: Sequence[str],
    budget: float,
    setup: AcquisitionSetup,
    q_prev: float,
    n_nodes: int | None,
) -> PlanResult:
    G, q_star = _stop(belief, q_prev, setup.params)
    values = {
        j: _query_value_generic(belief, j, U, budget, setup, q_prev, n_nodes)
        for j in U
        if setup.affordable(j, budget)
    }
    return _result(G, q_star, values, U)


# ---------------------------------------------------------------------------- Gaussian path
def _stop_at(x: np.ndarray, v: float, q_prev: float, params: UtilityParams) -> np.ndarray:
    return np.asarray(stop_value_array(x, x * x + v, q_prev, params), dtype=float)


def _cont_at(x: np.ndarray, c_k: float, D_k: float, v: float, q_prev: float, params: UtilityParams) -> np.ndarray:
    return -c_k + voi.expected_stop_value_gaussian_array(x, D_k, v, q_prev, params)


def _refine_crossing(f, a: float, b: float, rounds: int = 3, pts: int = 33) -> float:
    """Vectorized multi-round grid refinement of a sign change of f on [a, b]."""
    for _ in range(rounds):
        t = np.linspace(a, b, pts)
        ft = f(t)
        idx = np.nonzero(np.sign(ft[:-1]) * np.sign(ft[1:]) < 0)[0]
        if idx.size == 0:
            return float(t[int(np.argmin(np.abs(ft)))])
        a, b = float(t[idx[0]]), float(t[idx[0] + 1])
    return 0.5 * (a + b)


def _kinks_z(
    m0: float,
    sigma: float,
    v_c: float,
    D_k: float,
    c_k: float,
    q_prev: float,
    params: UtilityParams,
    precise: bool = False,
) -> list[float]:
    """Kinks of z -> max{G(m0 + sigma z, v_c), -c_k + E G_k(m0 + sigma z)} in (-Z_MAX, Z_MAX).

    Two families: breakpoints of G(., v_c) (argmax changes) and stop/continue crossings.
    ``precise`` locates crossings with brentq (reference); otherwise with the fast grid.
    """
    kinks: list[float] = []
    for bp in voi.breakpoints(v_c, params.lam, params.tau, q_prev):
        z = (bp - m0) / sigma
        if -Z_MAX < z < Z_MAX:
            kinks.append(z)
    if D_k <= 0.0:
        return kinks

    def diff(z):
        x = m0 + sigma * np.asarray(z, dtype=float)
        return _cont_at(x, c_k, D_k, v_c, q_prev, params) - _stop_at(x, v_c, q_prev, params)

    zg = np.linspace(-Z_MAX, Z_MAX, _KINK_GRID)
    d = diff(zg)
    idx = np.nonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    for i in idx:
        a, b = float(zg[i]), float(zg[i + 1])
        if precise:
            z_star = brentq(lambda t: float(diff(t)), a, b, xtol=1e-15, rtol=1e-15, maxiter=300)
        else:
            z_star = _refine_crossing(diff, a, b)
        kinks.append(z_star)
    return kinks


def _gauss_rule(kinks: Sequence[float], order: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre rule for E[h(Z)], Z ~ N(0,1), on [-Z_MAX, Z_MAX].

    The interval is split at the kinks and further into panels of width <= PANEL_WIDTH
    (so no panel is a wide tail that a few nodes cannot resolve). Each panel carries an
    ``order``-point Gauss-Legendre rule. Returns (z, w) with sum_i w_i h(z_i) ~ E[h(Z)]; the
    standard-normal density is folded into w.
    """
    cuts = sorted({-Z_MAX, Z_MAX, *[float(k) for k in kinks if -Z_MAX < k < Z_MAX]})
    pts: list[float] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b > a:
            n_sub = max(1, math.ceil((b - a) / PANEL_WIDTH))
            pts.extend(np.linspace(a, b, n_sub + 1)[:-1].tolist())
    pts.append(Z_MAX)
    edges = np.asarray(pts)
    a = edges[:-1]
    b = edges[1:]
    x, wt = roots_legendre(int(order))
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    z = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    w = (half[:, None] * wt[None, :]).ravel() * norm.pdf(z)
    return z, w


def _gauss_stage_value(
    belief: GaussianBelief,
    j: str,
    k: str,
    setup: AcquisitionSetup,
    q_prev: float,
    n_nodes: int | None,
    precise: bool = False,
) -> float:
    """E_{Y_j|b}[ max{ G(b^j), -c_k + E G(b^{jk}) } ]: the value after j when k is the last source."""
    params = setup.params
    mu_j, var_j = belief.predictive(j)
    cov_j = belief.cov_with_return(j)
    child = belief.condition(j, mu_j)  # reference child: shared posterior variance and gain of k
    v_c = float(child.var)
    m0 = float(child.mean)
    D_k = float(child.gain(k))
    c_k = float(setup.costs[k])
    # Child mean under Y_j = mu_j + sqrt(var_j) z is m0 + (cov_j / sqrt(var_j)) z.
    sigma = cov_j / math.sqrt(var_j) if var_j > 0.0 else 0.0
    if sigma == 0.0:
        x = np.array([m0])
        return float(np.maximum(_stop_at(x, v_c, q_prev, params), _cont_at(x, c_k, D_k, v_c, q_prev, params))[0])
    n = DEFAULT_OUTER_NODES if n_nodes is None else int(n_nodes)
    kinks = _kinks_z(m0, sigma, v_c, D_k, c_k, q_prev, params, precise=precise)
    if precise:
        return _integrate_reference(m0, sigma, v_c, D_k, c_k, q_prev, params, kinks)
    z, w = _gauss_rule(kinks, n)
    x = m0 + sigma * z
    h = np.maximum(_stop_at(x, v_c, q_prev, params), _cont_at(x, c_k, D_k, v_c, q_prev, params))
    return float(np.dot(w, h))


def _integrate_reference(m0, sigma, v_c, D_k, c_k, q_prev, params, kinks) -> float:
    """Adaptive quad on each kink-split interval; integrand is max{stop, cont} * phi."""
    cuts = sorted({-Z_MAX, Z_MAX, *[float(k) for k in kinks if -Z_MAX < k < Z_MAX]})
    # Adaptive quad alone can miss narrow features on wide intervals (checked: an error of
    # ~7e-9 on a wide tail), so every cut interval is further split into panels of width
    # <= REFERENCE_PANEL_WIDTH before quad is applied.
    edges: list[float] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b > a:
            n_sub = max(1, math.ceil((b - a) / REFERENCE_PANEL_WIDTH))
            edges.extend(np.linspace(a, b, n_sub + 1)[:-1].tolist())
    edges.append(Z_MAX)

    def integrand(z: float) -> float:
        x = np.array([m0 + sigma * z])
        h = max(float(_stop_at(x, v_c, q_prev, params)[0]), float(_cont_at(x, c_k, D_k, v_c, q_prev, params)[0]))
        return h * float(norm.pdf(z))

    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            val, _ = quad(integrand, a, b, limit=500, epsabs=1e-30, epsrel=1e-13)
            total += val
    return total


def reference_query_value_gaussian(belief: GaussianBelief, j: str, k: str | None, setup: AcquisitionSetup, q_prev: float) -> float:
    """High-accuracy reference for -c_j + E_{Y_j|b} V(b^{j,Y_j}, *, {k}) on a Gaussian belief.

    If ``k`` is None the continuation is stop-only and the exact closed form is used.
    """
    if k is None:
        return -float(setup.costs[j]) + float(belief.expected_stop_value_after(j, q_prev, setup.params))
    return -float(setup.costs[j]) + _gauss_stage_value(belief, j, k, setup, q_prev, None, precise=True)


def _solve_gaussian(
    belief: GaussianBelief,
    U: Sequence[str],
    budget: float,
    setup: AcquisitionSetup,
    q_prev: float,
    n_nodes: int | None,
) -> PlanResult:
    G, q_star = _stop(belief, q_prev, setup.params)
    values: dict[str, float] = {}
    for j in U:
        if not setup.affordable(j, budget):
            continue
        rest = tuple(k for k in U if k != j)
        b_next = _budget_after(setup, j, budget)
        after = _affordable_after(setup, rest, b_next)
        if not after:
            values[j] = -float(setup.costs[j]) + float(belief.expected_stop_value_after(j, q_prev, setup.params))
        elif len(after) == 1:
            values[j] = -float(setup.costs[j]) + _gauss_stage_value(belief, j, after[0], setup, q_prev, n_nodes)
        else:  # not reachable with two sources; keep the recursion honest
            raise NotImplementedError("Gaussian fast path supports at most two sources")
    return _result(G, q_star, values, U)


# ---------------------------------------------------------------------------- public API
def solve(
    belief: Any,
    remaining: Iterable[str],
    budget: float,
    setup: AcquisitionSetup,
    q_prev: float,
    n_nodes: int | None = None,
) -> PlanResult:
    """Optimal first action and value V(b, budget, remaining) of the two-step recursion.

    ``remaining`` lists sources that may still be bought on this date (each at most once);
    sources already observed by the belief are rejected. ``n_nodes`` sets the outer
    quadrature: for a GaussianBelief the Gauss-Legendre order per panel on the
    kink-split intervals; for other beliefs the number of predictive branches.
    """
    U = _validate_remaining(belief, remaining)
    _validate_priced(U, setup)
    if float(budget) < 0.0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    if isinstance(belief, GaussianBelief):
        return _solve_gaussian(belief, U, budget, setup, q_prev, n_nodes)
    return _solve_generic(belief, U, budget, setup, q_prev, n_nodes)


def solve_fixed_order(
    belief: Any,
    order: Sequence[str],
    budget: float,
    setup: AcquisitionSetup,
    q_prev: float,
    n_nodes: int | None = None,
) -> PlanResult:
    """Optimal stopping along a fixed acquisition order.

    At each stage the next source is the first one in ``order`` that is still unobserved
    and affordable (unaffordable sources are skipped, since the budget only falls). The
    decision is STOP or query that source; after a query the recursion continues along
    the rest of the order. Returns the same PlanResult shape as ``solve``.
    """
    order_t = tuple(order)
    if len(set(order_t)) != len(order_t):
        raise ValueError(f"duplicate sources in order {order_t}")
    for s in order_t:
        if s not in SOURCES:
            raise ValueError(f"{s!r} is not a purchasable source")
    seen = [s for s in order_t if s in belief.observed]
    if seen:
        raise ValueError(f"sources already observed in the belief: {seen}")
    _validate_priced(order_t, setup)
    G, q_star = _stop(belief, q_prev, setup.params)
    nxt = next((s for s in order_t if setup.affordable(s, budget)), None)
    if nxt is None:
        return _result(G, q_star, {}, order_t)
    rest = tuple(s for s in order_t if s != nxt)
    b_next = _budget_after(setup, nxt, budget)
    after = [s for s in rest if setup.affordable(s, b_next)]
    if not after:
        cont = -float(setup.costs[nxt]) + voi.expected_stop_after(belief, nxt, q_prev, setup.params, n_nodes)
    elif isinstance(belief, GaussianBelief) and len(after) == 1:
        cont = -float(setup.costs[nxt]) + _gauss_stage_value(belief, nxt, after[0], setup, q_prev, n_nodes)
    else:
        EV = 0.0
        for w, child in belief.predictive_branches(nxt, n_nodes):
            EV += w * solve_fixed_order(child, rest, b_next, setup, q_prev, n_nodes).value
        cont = -float(setup.costs[nxt]) + EV
    return _result(G, q_star, {nxt: cont}, order_t)


def stop_decision(belief: Any, q_prev: float, params: UtilityParams) -> tuple[float, int]:
    """(G(b), q*) for stopping now: the terminal position by ``choose_position``."""
    return _stop(belief, q_prev, params)


def evaluate_policy(
    policy: Any,
    belief: Any,
    setup: AcquisitionSetup,
    q_prev: float,
    n_nodes: int | None = None,
) -> float:
    """Expected net utility of a deterministic policy from ``belief`` under the belief model.

    The policy is queried at every node of the purchase tree (``policy.next_action``
    with ``rng=None``) and the tree is expanded over ``predictive_branches``. Where the
    policy has no affordable continuation the last expectation is exact. Stochastic
    policies (``deterministic`` False) have no single value and are rejected.
    """
    if not getattr(policy, "deterministic", True):
        raise ValueError("evaluate_policy needs a deterministic policy")
    remaining = tuple(s for s in SOURCES if s in setup.costs and s not in belief.observed)
    return _eval_policy(policy, belief, remaining, float(setup.budget), setup, q_prev, n_nodes)


def _eval_policy(policy, belief, remaining, budget_left, setup, q_prev, n_nodes) -> float:
    action, _ = policy.next_action(belief, remaining, budget_left, setup, q_prev)
    if action == STOP:
        return voi.stop_value(belief, q_prev, setup.params)
    if action not in remaining or not setup.affordable(action, budget_left):
        raise ValueError(f"policy chose {action!r}, which is not available with budget {budget_left}")
    c = float(setup.costs[action])
    rest = tuple(s for s in remaining if s != action)
    b_next = _budget_after(setup, action, budget_left)
    if not _affordable_after(setup, rest, b_next):
        return -c + voi.expected_stop_after(belief, action, q_prev, setup.params, n_nodes)
    EV = 0.0
    for w, child in belief.predictive_branches(action, n_nodes):
        EV += w * _eval_policy(policy, child, rest, b_next, setup, q_prev, n_nodes)
    return -c + EV


__all__ = [
    "PLAN_TOL",
    "Z_MAX",
    "DEFAULT_OUTER_NODES",
    "solve",
    "solve_fixed_order",
    "stop_decision",
    "reference_query_value_gaussian",
    "evaluate_policy",
]
