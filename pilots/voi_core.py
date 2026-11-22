"""Exact Gaussian value-of-information core for the two-source query problem.

Conventions (one decision episode, after the free baseline Y0 is observed):
  * belief over next-day return r is N(m, v), v = posterior variance (s^2 initially)
  * utility u(q, r) = q r - lam q^2 r^2 - tau |q - qp|
  * stop value  G(m, v) = max_q [ q m - lam q^2 (m^2 + v) - tau |q - qp| ]
    (depends on the belief only through E[r] = m and E[r^2] = m^2 + v)
  * source j is observed with cost c_j; the conditional correlation structure given Y0 is
    Var(Y_j|Y0) = 1 (WLOG, since Delta is scale free), Corr(r, Y_j|Y0) = rho_j,
    Corr(Y1, Y2|Y0) = kappa.  Then with Cov(r, Y_j | Y0) = rho_j * s:
      Delta_1  = rho1^2 s^2                          (single-source preposterior sd of the mean)
      Delta_2  = rho2^2 s^2
      Delta_21 = s^2 (rho2 - kappa rho1)^2 / (1 - kappa^2)   (Y2 after Y1 observed)
      Delta_12 = s^2 (rho1 - kappa rho2)^2 / (1 - kappa^2)   (Y1 after Y2 observed)
    and Delta_1 + Delta_21 = Delta_2 + Delta_12 (order independence of the total).

Everything that depends on the sources is reduced to these Deltas, because under the
linear-Gaussian model the posterior variance after any set of observations is
data-independent and the posterior mean is a martingale with increments N(0, Delta).

Closed form: for X ~ N(m, Delta), E[G(X, v - Delta)] is a finite sum of truncated-normal
moments: G(., v) is piecewise quadratic in x with breakpoints that depend only on v.
"""
import math

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

INF = np.inf


def gauss_hermite_nodes(n):
    """Nodes/weights for E[g(Z)] ~ sum w_k g(z_k), Z ~ N(0,1) (stable for large n)."""
    from scipy.special import roots_hermitenorm
    z, w = roots_hermitenorm(n)
    return z, w / math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- stop value
def f_q(m, v, q, lam, tau, qp):
    """Expected utility of holding position q given belief N(m, v)."""
    return q * m - lam * q * q * (m * m + v) - tau * abs(q - qp)


def stop_value(m, v, lam, tau, qp):
    """G(m, v) = max_{q in {-1,0,1}} f_q(m, v). Vectorized over m."""
    m = np.asarray(m, dtype=float)
    best = np.full(m.shape, -np.inf)
    for q in (-1, 0, 1):
        best = np.maximum(best, f_q(m, v, q, lam, tau, qp))
    return best


def stop_argmax(m, v, lam, tau, qp):
    m = np.asarray(m, dtype=float)
    vals = np.stack([f_q(m, v, q, lam, tau, qp) for q in (-1, 0, 1)])
    return np.array([-1, 0, 1])[np.argmax(vals, axis=0)]


# --------------------------------------------------------------------------- pieces of G(., v)
def _quad_coeffs(v, lam, tau, qp):
    """f_q(x) = A x^2 + B x + C for each q, as a function of the realized mean x."""
    return {
        1: (-lam, 1.0, -lam * v - tau * abs(1 - qp)),
        0: (0.0, 0.0, -tau * abs(qp)),
        -1: (-lam, -1.0, -lam * v - tau * abs(-1 - qp)),
    }


def _real_roots(A, B, C):
    """Real roots of A x^2 + B x + C = 0 using the cancellation-free formula."""
    if A == 0.0:
        return [] if B == 0.0 else [-C / B]
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return []
    sq = math.sqrt(disc)
    q = -0.5 * (B + math.copysign(sq, B)) if B != 0.0 else -0.5 * sq
    roots = []
    if q != 0.0:
        roots.append(q / A)
        roots.append(C / q)
    else:  # B == 0 and disc == 0 -> double root at 0
        roots.append(0.0)
    return roots


def pieces(v, lam, tau, qp):
    """Partition of the real line into intervals on which a single quadratic is active.

    Returns a list of (lo, hi, (A, B, C)).  Breakpoints depend only on v, not on m.
    """
    co = _quad_coeffs(v, lam, tau, qp)
    bps = []
    for p, q in ((1, 0), (-1, 0), (1, -1)):
        A = co[p][0] - co[q][0]
        B = co[p][1] - co[q][1]
        C = co[p][2] - co[q][2]
        bps += _real_roots(A, B, C)
    bps = sorted(set(float(x) for x in bps if math.isfinite(x)))
    edges = [-INF] + bps + [INF]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == -INF and hi == INF:
            x = 0.0
        elif lo == -INF:
            x = hi - 1.0
        elif hi == INF:
            x = lo + 1.0
        else:
            x = 0.5 * (lo + hi)
        best = max((1, 0, -1), key=lambda q: co[q][0] * x * x + co[q][1] * x + co[q][2])
        out.append((lo, hi, co[best]))
    return out


# --------------------------------------------------------------------------- truncated normals
def trunc_moments(mu, sig, lo, hi):
    """P(lo<X<hi), E[X 1], E[X^2 1] for X ~ N(mu, sig^2); mu may be an array."""
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


def expected_stop(m, Delta, vb, lam, tau, qp):
    """Closed form of E_{X ~ N(m, Delta)}[ G(X, vb - Delta) ].

    vb is the variance before the observation; the post-observation variance is vb - Delta.
    Vectorized over m.  Exact up to floating point (no quadrature).
    """
    m = np.asarray(m, dtype=float)
    if Delta <= 0.0:
        return stop_value(m, vb, lam, tau, qp)
    if Delta > vb * (1 + 1e-12):
        raise ValueError("Delta cannot exceed the prior variance")
    v = max(vb - Delta, 0.0)
    sig = math.sqrt(Delta)
    total = np.zeros(m.shape)
    for lo, hi, (A, B, C) in pieces(v, lam, tau, qp):
        P, E1, E2 = trunc_moments(m, sig, lo, hi)
        total = total + A * E2 + B * E1 + C * P
    return total


def gross_voi(m, Delta, vb, lam, tau, qp):
    """E[G(m', v')] - G(m, vb): value of information before paying its cost."""
    return expected_stop(m, Delta, vb, lam, tau, qp) - stop_value(m, vb, lam, tau, qp)


# --------------------------------------------------------------------------- two-source model
class TwoSource:
    def __init__(self, s, lam, tau, qp, c1, c2, rho1, rho2, kappa):
        if not (-1.0 < kappa < 1.0):
            raise ValueError("kappa must be in (-1, 1)")
        self.s, self.lam, self.tau, self.qp = float(s), float(lam), float(tau), float(qp)
        self.c = {1: float(c1), 2: float(c2)}
        self.rho = {1: float(rho1), 2: float(rho2)}
        self.kappa = float(kappa)
        s2 = self.s * self.s
        self.v0 = s2
        self.D = {1: self.rho[1] ** 2 * s2, 2: self.rho[2] ** 2 * s2}
        # incremental (conditional) Delta of source k once source j has been observed
        self.Dc = {
            (1, 2): s2 * (self.rho[2] - kappa * self.rho[1]) ** 2 / (1 - kappa * kappa),
            (2, 1): s2 * (self.rho[1] - kappa * self.rho[2]) ** 2 / (1 - kappa * kappa),
        }
        # total variance removed by both sources (same in both orders)
        self.D_both = self.D[1] + self.Dc[(1, 2)]
        if abs(self.D_both - (self.D[2] + self.Dc[(2, 1)])) > 1e-12 * s2:
            raise AssertionError("order-independence identity violated")
        if self.D_both > s2 * (1 + 1e-12):
            raise ValueError("implied covariance is not positive semidefinite")

    # ---- helpers
    def G(self, m):
        return stop_value(m, self.v0, self.lam, self.tau, self.qp)

    def EG(self, m, Delta, vb=None):
        vb = self.v0 if vb is None else vb
        return expected_stop(m, Delta, vb, self.lam, self.tau, self.qp)

    def v_after(self, j):
        return self.v0 - self.D[j]

    # ---- one-step (myopic) quantities at the current belief
    def voi_single(self, m, j):
        return self.EG(m, self.D[j]) - self.G(m) - self.c[j]

    # ---- second stage after source j with realized mean m_j (variance v_j)
    def cont_after(self, mj, j, vj):
        """Value after observing j, with the other source k still available:
        max{ G(mj, vj), -c_k + E[G(m', vj - Dc)] } ."""
        k = 3 - j
        stop = stop_value(mj, vj, self.lam, self.tau, self.qp)
        cont = -self.c[k] + expected_stop(mj, self.Dc[(j, k)], vj, self.lam, self.tau, self.qp)
        return np.maximum(stop, cont)

    # ---- expectation over the first observation (z ~ N(0,1), m_j = m + rho_j s z)
    def branch(self, m, j, method="gh", n_gh=200):
        """-c_j + E_{Y_j}[ max{G(b_j), -c_k + E G(b_jk)} ] : value of 'query j first, then optional k'."""
        vj = self.v_after(j)
        sj = self.rho[j] * self.s
        if method == "gh":
            z, w = gauss_hermite_nodes(n_gh)
            mj = m + sj * z
            vals = self.cont_after(mj, j, vj)
            return -self.c[j] + float(np.sum(w * vals))
        if method == "dense":
            # trapezoid on a fine uniform grid of z in [-12, 12]; O(h^2) error at each kink
            z = np.linspace(-12.0, 12.0, 12001)
            h = z[1] - z[0]
            w = norm.pdf(z) * h
            w[0] *= 0.5
            w[-1] *= 0.5
            mj = m + sj * z
            vals = self.cont_after(mj, j, vj)
            return -self.c[j] + float(np.sum(w * vals))
        if method == "quad":
            return -self.c[j] + self._branch_quad_integral(m, j)
        raise ValueError(method)

    def _branch_quad_integral(self, m, j):
        """E_z[ max{G(b_j), -c_k + E G(b_jk)} ] by adaptive quadrature on sub-intervals split at the
        (scanned and bisected) kink points where the stop/continue decision switches."""
        from scipy.optimize import brentq

        vj = self.v_after(j)
        sj = self.rho[j] * self.s
        k = 3 - j

        def diff(z):  # continuation minus stop (scalar)
            mj = np.float64(m + sj * z)
            stop = float(stop_value(mj, vj, self.lam, self.tau, self.qp))
            cont = -self.c[k] + float(expected_stop(mj, self.Dc[(j, k)], vj, self.lam, self.tau, self.qp))
            return cont - stop

        grid = np.linspace(-12.0, 12.0, 4801)
        dv = np.array([diff(z) for z in grid])
        cuts = [-12.0]
        for a, b, fa, fb in zip(grid[:-1], grid[1:], dv[:-1], dv[1:]):
            if fa == 0.0:
                cuts.append(a)
            elif fa * fb < 0.0:
                cuts.append(brentq(diff, a, b, xtol=1e-15, rtol=1e-15, maxiter=200))
        cuts.append(12.0)
        cuts = sorted(set(cuts))

        def integrand(z):
            mj = np.float64(m + sj * z)
            return float(self.cont_after(mj, j, vj)) * norm.pdf(z)

        total = 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b > a:
                val, _ = quad(integrand, a, b, limit=500, epsabs=0.0, epsrel=1e-12)
                total += val
        return total

    # ---- policies / values
    def value_never(self, m):
        g = self.G(m)
        return float(g) if np.ndim(g) == 0 else g

    def value_always_both(self, m):
        return -self.c[1] - self.c[2] + float(self.EG(m, self.D_both))

    def value_fixed_order(self, m, first, method="gh", n_gh=200):
        """Fixed order (first, then second), with the optimal stop rule (stop allowed before and after)."""
        return max(float(self.G(m)), self.branch(m, first, method, n_gh))

    def value_optimal(self, m, method="gh", n_gh=200):  # method: 'gh' | 'dense' | 'quad' 
        return max(float(self.G(m)), self.branch(m, 1, method, n_gh), self.branch(m, 2, method, n_gh))

    def myopic_first(self, m):
        """Myopic argmax j (None if both one-step VOIs are <= 0)."""
        v = {j: self.voi_single(m, j) for j in (1, 2)}
        if max(v.values()) <= 0.0:
            return None, v
        j = 1 if v[1] >= v[2] else 2
        return j, v

    def value_myopic(self, m, method="gh", n_gh=200):
        j, _ = self.myopic_first(m)
        if j is None:
            return float(self.G(m))
        return self.branch(m, j, method, n_gh)

    def value_open_loop_best(self, m):
        """Best non-adaptive acquisition set (committed in advance, no stopping), incl. empty set."""
        cands = {
            "none": float(self.G(m)),
            "{1}": -self.c[1] + float(self.EG(m, self.D[1])),
            "{2}": -self.c[2] + float(self.EG(m, self.D[2])),
            "{1,2}": self.value_always_both(m),
        }
        name = max(cands, key=cands.get)
        return cands[name], name, cands
