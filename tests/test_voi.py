"""Closed-form Gaussian VOI: against adaptive quadrature, seeded Monte Carlo, and identities."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from src import voi
from src.belief_model import GaussianJointModel, GaussianBelief
from src.interfaces import AcquisitionSetup
from src.utility import POSITIONS, UtilityParams, expected_utility, stop_value_array


def _G(x, w, qp, p):
    return float(stop_value_array(x, np.asarray(x) ** 2 + w, qp, p))


def _random_cases(seed: int, n: int):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        lam = float(rng.choice([0.0, 0.03, 1.0, 5.0]))
        tau = float(rng.choice([0.0, 1e-4, 8e-3, 5e-2]))
        qp = int(rng.choice([-1, 0, 1]))
        v = float(10 ** rng.uniform(-4, 0))
        delta = float(v * rng.uniform(0.0, 1.0))
        m = float(rng.normal(0.0, math.sqrt(v)))
        cases.append((m, delta, v, qp, UtilityParams(lam=lam, tau=tau)))
    return cases


# ---------------------------------------------------------------- closed form vs quadrature
@pytest.mark.parametrize("m, delta, v, qp, p", _random_cases(20261115, 40))
def test_closed_form_matches_split_quadrature(m, delta, v, qp, p):
    if delta <= 0.0:
        pytest.skip("delta = 0 is checked exactly below")
    sig = math.sqrt(delta)
    w = v - delta
    # Split the z-line at the breakpoints of G(., w) (the integrand is smooth between them).
    bps = [(b - m) / sig for b in voi.breakpoints(w, p.lam, p.tau, qp)]
    cuts = sorted({-40.0, 40.0, *[b for b in bps if -40.0 < b < 40.0]})
    ref = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        val, _ = quad(lambda z: _G(m + sig * z, w, qp, p) * norm.pdf(z), a, b, limit=500, epsabs=1e-30, epsrel=1e-13)
        ref += val
    got = voi.expected_stop_value_gaussian(m, delta, v, qp, p)
    assert abs(got - ref) <= 1e-12 * max(1.0, abs(ref)) + 1e-15


@pytest.mark.parametrize("seed", [1, 2])
def test_closed_form_matches_seeded_monte_carlo(seed):
    rng = np.random.default_rng(seed)
    N = 400_000
    p = UtilityParams(lam=float(rng.choice([0.03, 1.0, 5.0])), tau=float(rng.choice([1e-4, 8e-3])))
    qp = int(rng.choice([-1, 0, 1]))
    v = 1.0
    delta = float(rng.uniform(0.05, 0.9))
    m = float(rng.normal(0.0, 0.5))
    z = np.random.default_rng(1000 + seed).standard_normal(N)  # common random numbers
    x = m + math.sqrt(delta) * z
    g = stop_value_array(x, x * x + (v - delta), qp, p)
    mc, se = float(np.mean(g)), float(np.std(g, ddof=1) / math.sqrt(N))
    got = voi.expected_stop_value_gaussian(m, delta, v, qp, p)
    assert abs(got - mc) <= 6.0 * se + 1e-12 * max(1.0, abs(mc))


def test_array_twin_matches_scalar():
    p = UtilityParams(lam=2.0, tau=3e-4)
    ms = np.linspace(-0.02, 0.02, 41)
    arr = voi.expected_stop_value_gaussian_array(ms, 4e-5, 1e-4, 1, p)
    scal = np.array([voi.expected_stop_value_gaussian(m, 4e-5, 1e-4, 1, p) for m in ms])
    assert arr.shape == ms.shape
    np.testing.assert_allclose(arr, scal, rtol=0, atol=1e-16)


def test_delta_zero_returns_stop_value_exactly():
    p = UtilityParams(lam=1.0, tau=1e-4)
    for m in (-0.03, -1e-4, 0.0, 2e-3, 0.05):
        got = voi.expected_stop_value_gaussian(m, 0.0, 1e-4, 0, p)
        ref = _G(m, 1e-4, 0, p)
        assert got == ref


def test_delta_equals_prior_variance_is_full_revelation():
    # Posterior variance 0: E[G(X, 0)] with X ~ N(m, v).
    p = UtilityParams(lam=0.5, tau=2e-3)
    m, v = 0.01, 4e-4
    sig = math.sqrt(v)
    ref, _ = quad(lambda z: _G(m + sig * z, 0.0, 1, p) * norm.pdf(z), -12, 12, limit=500, epsabs=1e-30, epsrel=1e-13)
    got = voi.expected_stop_value_gaussian(m, v, v, 1, p)
    assert abs(got - ref) < 1e-12


def test_delta_above_prior_variance_is_rejected():
    with pytest.raises(ValueError):
        voi.expected_stop_value_gaussian(0.0, 2.0, 1.0, 0, UtilityParams())
    with pytest.raises(ValueError):
        voi.expected_stop_value_gaussian(0.0, -1e-6, 1.0, 0, UtilityParams())


def test_linear_pieces_lam_zero_and_tau_zero():
    # lam = 0: G is piecewise linear; tau = 0: no transaction penalty. Compare to quadrature.
    for lam, tau in [(0.0, 0.0), (0.0, 1e-3), (1.0, 0.0)]:
        p = UtilityParams(lam=lam, tau=tau)
        for qp in (-1, 0, 1):
            m, delta, v = 5e-4, 2e-5, 1e-4
            sig = math.sqrt(delta)
            ref, _ = quad(lambda z: _G(m + sig * z, v - delta, qp, p) * norm.pdf(z), -12, 12, limit=500, epsabs=1e-30, epsrel=1e-13)
            assert abs(voi.expected_stop_value_gaussian(m, delta, v, qp, p) - ref) < 1e-12


# ---------------------------------------------------------------- VOI identities
def _gaussian_belief(rng, s: float, rho1: float, rho2: float, kappa: float, m: float):
    cov = np.array([[s * s, 0.0, rho1 * s, rho2 * s], [0.0, 1.0, 0.0, 0.0], [rho1 * s, 0.0, 1.0, kappa], [rho2 * s, 0.0, kappa, 1.0]])
    model = GaussianJointModel(mean=np.array([m, 0.0, 0.0, 0.0]), cov=cov)
    return model.belief({"y0": float(rng.normal(0.0, 0.1))})


def test_gross_voi_nonnegative_on_random_gaussian_beliefs():
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        s = float(rng.choice([0.01, 1.0]))
        rho1, rho2 = rng.uniform(-0.45, 0.45, size=2)
        kappa = float(rng.uniform(-0.6, 0.6))
        if np.linalg.eigvalsh(np.array([[1, rho1, rho2], [rho1, 1, kappa], [rho2, kappa, 1]])).min() <= 1e-6:
            continue
        p = UtilityParams(lam=float(rng.choice([0.0, 0.5, 5.0])), tau=float(rng.choice([0.0, 1e-4, 5e-3])))
        b = _gaussian_belief(rng, s, rho1, rho2, kappa, float(rng.normal(0, s)))
        for j in ("A", "B"):
            worst = min(worst, voi.gross_voi(b, j, int(rng.choice([-1, 0, 1])), p))
    assert worst >= -1e-14


def test_zero_information_source_gives_net_voi_minus_cost():
    rng = np.random.default_rng(3)
    p = UtilityParams(lam=2.0, tau=1e-4)
    # Source B uncorrelated with r: Delta_B = 0 exactly.
    cov = np.array([[1e-4, 0, 5e-4, 0.0], [0, 1, 0, 0], [5e-4, 0, 1, 0.0], [0.0, 0, 0.0, 1.0]])
    b = GaussianJointModel(mean=np.array([3e-4, 0, 0, 0.0]), cov=cov).belief({"y0": 0.0})
    assert b.gain("B") == 0.0
    cost = 7e-5
    assert voi.gross_voi(b, "B", 0, p) == 0.0
    assert voi.one_step_voi(b, "B", cost, 0, p) == -cost
    del rng


def test_generic_branch_path_agrees_with_fast_path():
    """one_step_voi through predictive_branches (generic) matches the closed-form fast path."""

    class _Generic:
        """Belief protocol without expected_stop_value_after: forces the generic path."""

        def __init__(self, inner):
            self._b = inner

        @property
        def observed(self):
            return self._b.observed

        def return_moments(self):
            return self._b.return_moments()

        def predictive_branches(self, source, n_nodes=None):
            return self._b.predictive_branches(source, n_nodes)

        def condition(self, source, value):
            return _Generic(self._b.condition(source, value))

    b = _gaussian_belief(np.random.default_rng(0), 0.01, 0.1, 0.2, -0.4, 1e-3)
    p = UtilityParams(lam=1.0, tau=1e-4)
    fast = voi.gross_voi(b, "B", 0, p)
    generic = voi.gross_voi(_Generic(b), "B", 0, p, n_nodes=4001)
    # Quantile midpoints converge only like ~1/n on kinked integrands (about 1e-4 relative at n=4001);
    # the planner's kink-split Gauss-Legendre rule is the accurate path (see docs/notes_D.md).
    assert abs(fast - generic) < 1e-3 * abs(fast) + 1e-12


def test_one_step_voi_uses_belief_fast_path_name():
    b = _gaussian_belief(np.random.default_rng(1), 0.01, 0.2, 0.1, 0.0, 5e-4)
    p = UtilityParams(lam=1.0, tau=1e-4)
    assert voi.one_step_voi(b, "A", 1e-5, 0, p) == pytest.approx(
        b.expected_stop_value_after("A", 0, p) - voi.stop_value(b, 0, p) - 1e-5, abs=1e-18
    )


def test_stop_value_is_argmax_of_expected_utility():
    rng = np.random.default_rng(11)
    p = UtilityParams(lam=0.7, tau=2e-3)
    for _ in range(50):
        m = float(rng.normal(0, 0.05))
        s2 = m * m + float(rng.uniform(1e-6, 1e-2))
        qp = int(rng.choice([-1, 0, 1]))
        G = voi.stop_value(_Moments(m, s2), qp, p)
        vals = {q: expected_utility(q, m, s2, qp, p) for q in POSITIONS}
        assert G == pytest.approx(max(vals.values()), abs=1e-16)


class _Moments:
    def __init__(self, m, s2):
        self._m, self._s2 = m, s2

    def return_moments(self):
        return self._m, self._s2


def test_gross_voi_generic_discrete_matches_hand_computation():
    # Two equally likely child beliefs with a known stop value each: E[G'] - G by hand.
    class _Child:
        def __init__(self, m, s2):
            self.m, self.s2 = m, s2

        def return_moments(self):
            return self.m, self.s2

    class _Parent:
        observed = frozenset()

        def return_moments(self):
            return 0.001, 0.001 ** 2 + 1e-4

        def predictive_branches(self, source, n_nodes=None):
            return [(0.5, _Child(0.0, 1e-4)), (0.5, _Child(0.002, 0.002 ** 2 + 1e-4))]

    p = UtilityParams(lam=1.0, tau=1e-4)
    parent = _Parent()
    g_children = 0.5 * voi.stop_value(_Child(0.0, 1e-4), 0, p) + 0.5 * voi.stop_value(_Child(0.002, 0.002 ** 2 + 1e-4), 0, p)
    assert voi.gross_voi(parent, "A", 0, p) == pytest.approx(g_children - voi.stop_value(parent, 0, p), abs=1e-16)


def test_setup_is_plain_dataclass_and_affordability():
    setup = AcquisitionSetup(costs={"A": 1e-5, "B": 2e-5}, budget=2e-5)
    assert setup.affordable("B", 2e-5)
    assert not setup.affordable("A", 5e-6)
