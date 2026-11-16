"""Two-step Bellman planner: regression targets, reference agreement, ties and budgets."""
from __future__ import annotations

import numpy as np
import pytest

from src import planner, voi
from src.belief_model import GaussianJointModel, GaussianBelief
from src.interfaces import STOP, AcquisitionSetup
from src.utility import UtilityParams


def _belief(s, rho1, rho2, kappa, m, cov_extra=None):
    cov = np.array(
        [[s * s, 0.0, rho1 * s, rho2 * s], [0.0, 1.0, 0.0, 0.0], [rho1 * s, 0.0, 1.0, kappa], [rho2 * s, 0.0, kappa, 1.0]],
        dtype=float,
    )
    model = GaussianJointModel(mean=np.array([m, 0.0, 0.0, 0.0]), cov=cov)
    return model.belief({"y0": 0.0})


def _setup(c1, c2, lam, tau, budget=1.0):
    return AcquisitionSetup(costs={"A": c1, "B": c2}, budget=budget, params=UtilityParams(lam=lam, tau=tau))


# ---------------------------------------------------------------- regression targets (notes/03)
def test_A3_designed_instance_value_and_myopic_gap():
    # s = 1: both one-step VOIs are negative, the optimal policy queries source A.
    b = _belief(1.0, 0.15, 0.22, 0.68, 0.575)
    setup = _setup(2e-5, 8e-4, lam=0.03, tau=0.008)
    res = planner.solve(b, ["A", "B"], 1.0, setup, 0)
    assert res.stop_value == pytest.approx(0.52708125, abs=1e-12)
    assert res.value == pytest.approx(0.5275865491277, abs=1e-11)
    assert res.action == "A"
    assert voi.one_step_voi(b, "A", 2e-5, 0, setup.params) == pytest.approx(-1.300761e-5, abs=1e-10)
    assert voi.one_step_voi(b, "B", 8e-4, 0, setup.params) == pytest.approx(-1.025902e-4, abs=1e-10)


def test_A4_optimal_first_query_differs_from_myopic_argmax():
    b = _belief(1.0, 0.15, 0.22, 0.68, 0.575)
    setup = _setup(5e-6, 6.5e-4, lam=0.03, tau=0.008)
    res = planner.solve(b, ["A", "B"], 1.0, setup, 0)
    assert res.value == pytest.approx(0.5276213267930, abs=1e-11)
    assert res.action == "A"  # optimal first query
    vA = voi.one_step_voi(b, "A", 5e-6, 0, setup.params)
    vB = voi.one_step_voi(b, "B", 6.5e-4, 0, setup.params)
    assert vA == pytest.approx(1.992390e-6, abs=1e-10)
    assert vB == pytest.approx(4.740979e-5, abs=1e-10)
    assert vB > vA > 0  # the myopic argmax is B
    # Value of the myopic-first branch (B first, then optional A): notes give 0.5271286772886
    # from a dense trapezoid (about 1e-9 error); the split-panel rule here is more accurate.
    assert res.action_values["B"] == pytest.approx(0.5271286772886, abs=1e-9)


def test_A2_base_case_optimal_is_source_two_then_one():
    b = _belief(0.01, 0.05, 0.08, 0.3, 4e-4)
    setup = _setup(5e-5, 1e-4, lam=2.0, tau=2e-4)
    res = planner.solve(b, ["A", "B"], 1.0, setup, 0)
    assert res.stop_value == 0.0
    assert res.action == "B"
    assert res.value == pytest.approx(2.915057e-4, abs=2e-10)
    # Fixed orders: 1-then-2 = 2.715823e-4, 2-then-1 = 2.915057e-4 (= optimal).
    f12 = planner.solve_fixed_order(b, ("A", "B"), 1.0, setup, 0)
    f21 = planner.solve_fixed_order(b, ("B", "A"), 1.0, setup, 0)
    assert f12.value == pytest.approx(2.715823e-4, abs=2e-10)
    assert f21.value == pytest.approx(2.915057e-4, abs=2e-10)


# ---------------------------------------------------------------- reference agreement
def _random_beliefs(seed, n):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s = float(rng.choice([0.01, 1.0]))
        r1, r2 = rng.uniform(-0.4, 0.4, size=2)
        k = float(rng.uniform(-0.7, 0.7))
        C = np.array([[1, r1, r2], [r1, 1, k], [r2, k, 1]])
        if np.linalg.eigvalsh(C).min() <= 1e-4:
            continue
        m = float(rng.normal(0, 0.5 * s))
        lam = float(rng.choice([0.05, 1.0, 5.0]))
        tau = float(rng.choice([0.0, 1e-4, 5e-4]))
        c1, c2 = float(10 ** rng.uniform(-6, -3)), float(10 ** rng.uniform(-6, -3))
        out.append((_belief(s, r1, r2, k, m), _setup(c1, c2, lam, tau)))
    return out


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_solve_matches_quad_reference(idx):
    b, setup = _random_beliefs(4242, 3)[idx]
    res = planner.solve(b, ["A", "B"], 1.0, setup, 0)
    refA = planner.reference_query_value_gaussian(b, "A", "B", setup, 0)
    refB = planner.reference_query_value_gaussian(b, "B", "A", setup, 0)
    G = res.stop_value
    refV = max(G, refA, refB)
    assert res.value == pytest.approx(refV, abs=1e-10)
    assert res.action_values["A"] == pytest.approx(refA, abs=1e-9)
    assert res.action_values["B"] == pytest.approx(refB, abs=1e-9)


def test_last_stage_is_exact_closed_form():
    b = _belief(0.01, 0.1, 0.2, -0.4, 1e-3)
    setup = _setup(1e-5, 1e-4, lam=1.0, tau=1e-4)
    child = b.condition("A", b.predictive("A")[0])
    res = planner.solve(child, ["B"], 1.0, setup, 0)
    G = res.stop_value
    exact = max(G, -setup.costs["B"] + child.expected_stop_value_after("B", 0, setup.params))
    assert res.value == pytest.approx(exact, abs=1e-15)


def test_generic_path_agrees_with_fast_path():
    """Generic recursion over quantile branches (no fast path) converges to the Gaussian answer."""

    class _Generic:
        def __init__(self, inner):
            self._b = inner

        @property
        def observed(self):
            return self._b.observed

        def return_moments(self):
            return self._b.return_moments()

        def predictive_branches(self, source, n_nodes=None):
            return [(w, _Generic(c)) for w, c in self._b.predictive_branches(source, n_nodes)]

        def condition(self, source, value):
            return _Generic(self._b.condition(source, value))

    b = _belief(0.01, 0.05, 0.08, 0.3, 4e-4)
    setup = _setup(5e-5, 1e-4, lam=2.0, tau=2e-4)
    fast = planner.solve(b, ["A", "B"], 1.0, setup, 0)
    slow = planner.solve(_Generic(b), ["A", "B"], 1.0, setup, 0, n_nodes=201)
    # Quantile midpoints at n=201 are ~1e-6 absolute here (about 0.3% of this VOI).
    assert abs(slow.value - fast.value) < 5e-3 * (fast.value - fast.stop_value) + 1e-12
    assert slow.action == fast.action


# ---------------------------------------------------------------- ties, stop, budgets
def test_stop_wins_ties_with_zero_information_and_zero_cost():
    # rho = 0: neither source moves the mean, cost 0 -> VOI exactly 0 -> STOP.
    b = _belief(0.01, 0.0, 0.0, 0.0, 2e-3)
    setup = _setup(0.0, 0.0, lam=1.0, tau=1e-4)
    res = planner.solve(b, ["A", "B"], 1.0, setup, 0)
    assert res.action == STOP
    assert res.value == res.stop_value


def test_identical_sources_tie_goes_to_A():
    # A and B are the same forecast (kappa = 1): equal values, A must win.
    cov = np.array([[1e-4, 0, 5e-4, 5e-4], [0, 1, 0, 0], [5e-4, 0, 1, 1], [5e-4, 0, 1, 1]], dtype=float)
    model = GaussianJointModel(mean=np.array([1e-3, 0, 0, 0.0]), cov=cov)
    b = model.belief({"y0": 0.0})
    setup = _setup(1e-5, 1e-5, lam=1.0, tau=1e-4)
    res = planner.solve(b, ["A", "B"], 1.0, setup, 0)
    assert res.action_values["A"] == pytest.approx(res.action_values["B"], abs=1e-15)
    assert res.action == "A"


def test_budget_enforcement_and_affordability():
    b = _belief(0.01, 0.05, 0.08, 0.3, 4e-4)
    setup = _setup(5e-5, 1e-4, lam=2.0, tau=2e-4)
    # Budget covers A only: B cannot appear.
    res = planner.solve(b, ["A", "B"], 5e-5, setup, 0)
    assert "B" not in res.action_values
    assert res.action in (STOP, "A")
    # Budget covers nothing: STOP.
    res0 = planner.solve(b, ["A", "B"], 1e-6, setup, 0)
    assert res0.action == STOP and res0.action_values == {STOP: res0.stop_value}
    # Budget equal to the cost is affordable (BUDGET_ATOL slack).
    resB = planner.solve(b, ["B"], 1e-4, setup, 0)
    assert "B" in resB.action_values


def test_each_source_once_and_remaining_validation():
    b = _belief(0.01, 0.05, 0.08, 0.3, 4e-4)
    setup = _setup(5e-5, 1e-4, lam=2.0, tau=2e-4)
    with pytest.raises(ValueError):
        planner.solve(b.condition("A", 0.0), ["A", "B"], 1.0, setup, 0)  # A already observed
    with pytest.raises(ValueError):
        planner.solve(b, ["y0"], 1.0, setup, 0)  # free source is not purchasable
    with pytest.raises(ValueError):
        planner.solve(b, ["A", "A"], 1.0, setup, 0)
    with pytest.raises(ValueError):
        planner.solve_fixed_order(b.condition("B", 0.0), ("A", "B"), 1.0, setup, 0)


def test_zero_budget_stops_and_empty_order_stops():
    b = _belief(0.01, 0.05, 0.08, 0.3, 4e-4)
    setup = _setup(5e-5, 1e-4, lam=2.0, tau=2e-4)
    assert planner.solve(b, ["A", "B"], 0.0, setup, 0).action == STOP
    assert planner.solve_fixed_order(b, (), 1.0, setup, 0).action == STOP


# ---------------------------------------------------------------- structure
def test_value_dominates_stop_and_fixed_orders():
    rng_beliefs = _random_beliefs(99, 6)
    for b, setup in rng_beliefs:
        res = planner.solve(b, ["A", "B"], 1.0, setup, 0)
        f12 = planner.solve_fixed_order(b, ("A", "B"), 1.0, setup, 0).value
        f21 = planner.solve_fixed_order(b, ("B", "A"), 1.0, setup, 0).value
        assert res.value >= res.stop_value - 1e-15
        assert res.value >= max(f12, f21) - 1e-10


def test_stop_position_maximizes_posterior_expected_utility():
    b = _belief(0.01, 0.05, 0.08, 0.3, -4e-4)
    setup = _setup(5e-5, 1e-4, lam=2.0, tau=2e-4)
    res = planner.solve(b, ["A", "B"], 1.0, setup, -1)
    m, s2 = b.return_moments()
    from src.utility import expected_utility, POSITIONS

    vals = {q: expected_utility(q, m, s2, -1, setup.params) for q in POSITIONS}
    assert res.stop_position == max(vals, key=vals.get)


def test_default_outer_nodes_equal_explicit_default():
    b = _belief(0.01, 0.05, 0.08, 0.3, 4e-4)
    setup = _setup(5e-5, 1e-4, lam=2.0, tau=2e-4)
    a = planner.solve(b, ["A", "B"], 1.0, setup, 0)
    c = planner.solve(b, ["A", "B"], 1.0, setup, 0, n_nodes=planner.DEFAULT_OUTER_NODES)
    assert a.value == c.value and a.action == c.action


# ---------------------------------------------------------------- audit additions (independent reference)
def _dense_first_query_values(b, setup, q_prev, N=2001, M=401, Z=10.0):
    """Independent reference: plain double trapezoid in z (no kink detection, no closed form).

    Uses only GaussianBelief.condition for the child means/variances and the raw utility.
    Returns {"A": value of querying A first, "B": value of querying B first}.
    """
    import math

    from scipy.stats import norm

    from src.utility import stop_value_array

    p = setup.params
    zj = np.linspace(-Z, Z, N)
    wj = norm.pdf(zj) * (zj[1] - zj[0])
    zk = np.linspace(-Z, Z, M)
    wk = norm.pdf(zk) * (zk[1] - zk[0])
    out = {}
    for j, k in (("A", "B"), ("B", "A")):
        cj, ck = float(setup.costs[j]), float(setup.costs[k])
        mu_j, var_j = b.predictive(j)
        sd_j = math.sqrt(var_j)
        acc = 0.0
        for zi, w in zip(zj, wj):
            bj = b.condition(j, mu_j + sd_j * zi)
            Gj = float(stop_value_array(bj.mean, bj.mean ** 2 + bj.var, q_prev, p))
            mu_k, var_k = bj.predictive(k)
            sk = math.sqrt(var_k)
            m0 = bj.condition(k, mu_k).mean
            m1 = bj.condition(k, mu_k + sk).mean
            vk = bj.condition(k, mu_k).var
            means = m0 + (m1 - m0) * zk  # child-of-child means are affine in the k draw
            cont = -ck + float(np.dot(wk, stop_value_array(means, means ** 2 + vk, q_prev, p)))
            acc += w * max(Gj, cont)
        out[j] = -cj + acc
    return out


@pytest.mark.parametrize(
    "shape, costs, lam, tau",
    [
        ((1.0, 0.15, 0.22, 0.68, 0.575), (2e-5, 8e-4), 0.03, 0.008),  # A3
        ((1.0, 0.15, 0.22, 0.68, 0.575), (5e-6, 6.5e-4), 0.03, 0.008),  # A4
        ((0.01, 0.05, 0.08, 0.3, 4e-4), (5e-5, 1e-4), 2.0, 2e-4),  # A2
        ((0.01, 0.10, 0.20, -0.4, 1e-3), (1e-5, 1e-4), 1.0, 1e-4),  # R2
    ],
)
def test_solve_matches_independent_dense_reference(shape, costs, lam, tau):
    """Guards against a bug shared by the fast path and the quad reference (both use _kinks_z)."""
    s, r1, r2, k, m = shape
    cov = np.array([[s * s, 0, r1 * s, r2 * s], [0, 1, 0, 0], [r1 * s, 0, 1, k], [r2 * s, 0, k, 1]], float)
    b = GaussianJointModel(mean=np.array([m, 0, 0, 0.0]), cov=cov).belief({"y0": 0.0})
    setup = _setup(costs[0], costs[1], lam, tau)
    dense = _dense_first_query_values(b, setup, 0)
    res = planner.solve(b, ["A", "B"], 1.0, setup, 0)
    assert res.action_values["A"] == pytest.approx(dense["A"], abs=1e-7)
    assert res.action_values["B"] == pytest.approx(dense["B"], abs=1e-7)


def test_source_without_cost_is_rejected_with_value_error():
    b = _belief(0.01, 0.05, 0.08, 0.3, 4e-4)
    only_a = AcquisitionSetup(costs={"A": 5e-5}, budget=1.0, params=UtilityParams(lam=2.0, tau=2e-4))
    with pytest.raises(ValueError):
        planner.solve(b, ["A", "B"], 1.0, only_a, 0)  # B is not purchasable under this setup
    with pytest.raises(ValueError):
        planner.solve_fixed_order(b, ("A", "B"), 1.0, only_a, 0)
