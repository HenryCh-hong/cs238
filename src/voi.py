"""Closed-form value of information for the stop-or-trade problem (Gaussian beliefs).

Setting. The belief over the next-period return r is N(m, v). Observing a source
whose preposterior shift of the mean has variance ``delta`` (Delta_j in the notes)
moves the mean to X ~ N(m, delta) and the posterior variance to v - delta. The stop
value depends on the belief only through the mean and second moment:

    G(x, w) = max_q [ q x - lam q^2 (x^2 + w) - tau |q - q_prev| ],   q in {-1, 0, 1}.

Closed form. G(., w) is a maximum of three quadratics in x, so it is piecewise
quadratic: G(x, w) = A_p x^2 + B_p x + C_p on pieces I_p of a partition of R whose
breakpoints depend only on (w, lam, tau, q_prev). Under X ~ N(m, delta), the truncated
normal moments give

    E[G(X, v - delta)] = sum_p E[(A_p X^2 + B_p X + C_p) 1{X in I_p}],

an exact finite sum (no quadrature). Ported from ``analysis/voi_core.py`` and
generalized to any UtilityParams and any q_prev in {-1, 0, 1}.

Net one-step VOI of source j: ``gross_voi - c_j`` with
``gross_voi = E[G(b^{j,Y_j})] - G(b)`` (>= 0 by Jensen and the martingale property).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import norm

from src.utility import UtilityParams, choose_position, stop_value_array

# Relative slack allowed when Delta is numerically a hair above the prior variance v.
DELTA_RTOL = 1e-12


def _check_params(params: UtilityParams) -> None:
    if not isinstance(params, UtilityParams):
        raise TypeError(f"params must be a UtilityParams, got {type(params).__name__}")


def _quad_coeffs(v: float, lam: float, tau: float, qp: float) -> dict[int, tuple[float, float, float]]:
    """f_q(x) = A x^2 + B x + C for each position q, with posterior variance v."""
    return {
        1: (-lam, 1.0, -lam * v - tau * abs(1 - qp)),
        0: (0.0, 0.0, -tau * abs(qp)),
        -1: (-lam, -1.0, -lam * v - tau * abs(-1 - qp)),
    }


def _real_roots(A: float, B: float, C: float) -> list[float]:
    """Real roots of A x^2 + B x + C = 0 (cancellation-free formula)."""
    if A == 0.0:
        return [] if B == 0.0 else [-C / B]
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return []
    sq = math.sqrt(disc)
    q = -0.5 * (B + math.copysign(sq, B)) if B != 0.0 else -0.5 * sq
    if q != 0.0:
        return [q / A, C / q]
    return [0.0]  # B == 0 and disc == 0: double root at 0


def pieces(v: float, lam: float, tau: float, qp: float) -> list[tuple[float, float, tuple[float, float, float]]]:
    """Partition of R into intervals (lo, hi, (A, B, C)) on which G(., v) = A x^2 + B x + C.

    Breakpoints depend on (v, lam, tau, qp) only. Adjacent pieces may share a quadratic.
    """
    co = _quad_coeffs(v, lam, tau, qp)
    bps: list[float] = []
    for p, q in ((1, 0), (-1, 0), (1, -1)):
        A = co[p][0] - co[q][0]
        B = co[p][1] - co[q][1]
        C = co[p][2] - co[q][2]
        bps += _real_roots(A, B, C)
    bps = sorted(set(float(x) for x in bps if math.isfinite(x)))
    edges = [-math.inf] + bps + [math.inf]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == -math.inf and hi == math.inf:
            x = 0.0
        elif lo == -math.inf:
            x = hi - 1.0
        elif hi == math.inf:
            x = lo + 1.0
        else:
            x = 0.5 * (lo + hi)
        best = max((1, 0, -1), key=lambda q: co[q][0] * x * x + co[q][1] * x + co[q][2])
        out.append((lo, hi, co[best]))
    return out


def breakpoints(v: float, lam: float, tau: float, qp: float) -> list[float]:
    """Finite breakpoints of the piecewise-quadratic G(., v): the points where the argmax changes."""
    return [lo for lo, hi, _ in pieces(v, lam, tau, qp) if math.isfinite(lo)]


def trunc_moments(mu, sig: float, lo: float, hi: float):
    """P(lo < X < hi), E[X 1], E[X^2 1] for X ~ N(mu, sig^2); ``mu`` may be an array."""
    mu = np.asarray(mu, dtype=float)
    with np.errstate(invalid="ignore", over="ignore"):
        a = (lo - mu) / sig
        b = (hi - mu) / sig
        pa = norm.pdf(a)
        pb = norm.pdf(b)
        P = np.where(a > 0, norm.sf(a) - norm.sf(b), norm.cdf(b) - norm.cdf(a))
        Aa = np.where(np.isfinite(a), a * pa, 0.0)
        Bb = np.where(np.isfinite(b), b * pb, 0.0)
    E1 = mu * P + sig * (pa - pb)
    E2 = mu * mu * P + 2.0 * mu * sig * (pa - pb) + sig * sig * (P + Aa - Bb)
    return P, E1, E2


def expected_stop_value_gaussian_array(m, delta: float, v: float, q_prev: float, params: UtilityParams) -> np.ndarray:
    """Vectorized over ``m``: E_{X ~ N(m, delta)}[ G(X, v - delta) ] in closed form.

    ``v`` is the variance before the observation; the variance after it is ``v - delta``.
    ``delta = 0`` returns G(m, v) exactly; ``delta = v`` gives a fully revealing source
    (posterior variance 0). Returns an ndarray with the shape of ``m``.
    """
    _check_params(params)
    m_arr = np.asarray(m, dtype=float)
    delta = float(delta)
    v = float(v)
    if not (math.isfinite(delta) and math.isfinite(v)):
        raise ValueError(f"delta and v must be finite, got delta={delta}, v={v}")
    if v < 0.0:
        raise ValueError(f"prior variance v must be >= 0, got {v}")
    if delta < 0.0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    if delta > v:
        if delta > v * (1.0 + DELTA_RTOL):
            raise ValueError(f"delta={delta} cannot exceed the prior variance v={v}")
        delta = v
    if delta == 0.0:
        return np.asarray(stop_value_array(m_arr, m_arr * m_arr + v, q_prev, params), dtype=float)
    vp = max(v - delta, 0.0)
    sig = math.sqrt(delta)
    total = np.zeros(m_arr.shape)
    for lo, hi, (A, B, C) in pieces(vp, params.lam, params.tau, q_prev):
        P, E1, E2 = trunc_moments(m_arr, sig, lo, hi)
        total = total + A * E2 + B * E1 + C * P
    return total


def expected_stop_value_gaussian(m: float, delta: float, v: float, q_prev: float, params: UtilityParams) -> float:
    """Scalar E_{X ~ N(m, delta)}[ G(X, v - delta) ]; see the array version."""
    return float(expected_stop_value_gaussian_array(m, delta, v, q_prev, params))


def stop_value(belief: Any, q_prev: float, params: UtilityParams) -> float:
    """G(b) = max_q E_b[u(q, r)] for any belief with ``return_moments()``."""
    _check_params(params)
    m, s2 = belief.return_moments()
    return float(choose_position(m, s2, q_prev, params)[0])


def expected_stop_after(belief: Any, source: str, q_prev: float, params: UtilityParams, n_nodes: int | None = None) -> float:
    """E_{Y_j | b}[ G(b^{j, Y_j}) ]: exact via ``expected_stop_value_after`` when the belief
    provides it, else by summing over ``predictive_branches`` (weights must sum to 1)."""
    _check_params(params)
    fast = getattr(belief, "expected_stop_value_after", None)
    if callable(fast):
        return float(fast(source, q_prev, params))
    total_w = 0.0
    acc = 0.0
    for w, child in belief.predictive_branches(source, n_nodes):
        if w < 0.0:
            raise ValueError("negative branch weight")
        total_w += w
        acc += w * stop_value(child, q_prev, params)
    if abs(total_w - 1.0) > 1e-9:
        raise ValueError(f"branch weights sum to {total_w}, expected 1")
    return acc


def gross_voi(belief: Any, source: str, q_prev: float, params: UtilityParams, n_nodes: int | None = None) -> float:
    """E[G(b^{j,Y_j})] - G(b), before the query cost. Nonnegative in exact arithmetic."""
    return expected_stop_after(belief, source, q_prev, params, n_nodes) - stop_value(belief, q_prev, params)


def one_step_voi(
    belief: Any,
    source: str,
    cost: float,
    q_prev: float,
    params: UtilityParams,
    n_nodes: int | None = None,
) -> float:
    """Net one-step VOI_j(b) = E[G(b^{j,Y_j})] - G(b) - c_j."""
    return gross_voi(belief, source, q_prev, params, n_nodes) - float(cost)


__all__: Sequence[str] = (
    "pieces",
    "breakpoints",
    "trunc_moments",
    "expected_stop_value_gaussian",
    "expected_stop_value_gaussian_array",
    "stop_value",
    "expected_stop_after",
    "gross_voi",
    "one_step_voi",
)
