"""T2: mean-variance 'information value' is positive when the chosen position can never change;
the quadratic expected utility gives zero (tower property).

Real units: s = 0.01 (sd of r given the current information), lam = 3, tau = 0, qprev = 0.
Belief before the purchase: m = 0.02, v = s^2 = 1e-4. Forecast with rho = 0.1 gives Delta = rho^2 s^2 = 1e-6,
so after the purchase the posterior variance is v' = v - Delta = 9.9e-5 (deterministic), and
m' ~ N(m, Delta) (sd 1e-3).
"""
import os
import math

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voi_core import expected_stop, stop_value  # noqa: E402

s = 0.01
lam = 3.0
tau = 0.0
qp = 0
m = 0.02
v = s * s
rho = 0.1
Delta = rho * rho * s * s
vp = v - Delta
sd = math.sqrt(Delta)
QS = (-1, 0, 1)


def mv_value(mu, var):
    """J(q|b) = q mu - lam q^2 var, maximized over q (tau=0)."""
    return max(q * mu - lam * q * q * var for q in QS)


def quad_value(mu, var):
    """U(q|b) = q mu - lam q^2 (mu^2+var), maximized over q."""
    return max(q * mu - lam * q * q * (mu * mu + var) for q in QS)


# ---- mean-variance criterion
mv_now = mv_value(m, v)
arg_now = [q for q in QS if abs((q * m - lam * q * q * v) - mv_now) < 1e-18]
# E over the observation of max_q J(q|b'), b' = (m', v - Delta); v' is data-independent
def integrand_mv(z):
    mp = m + sd * z
    return mv_value(mp, vp) * norm.pdf(z)
mv_expect, _ = quad(integrand_mv, -40, 40, points=[0.0], limit=500, epsabs=1e-16, epsrel=1e-13)
mv_voi = mv_expect - mv_now
# decomposition for the chosen q=1: E[J(1|b')] - J(1|b) = -lam (v' - v) = lam Delta
mv_q1_gain = -lam * (vp - v)

# ---- quadratic expected utility (genuine expected utility)
quad_now = stop_value(m, v, lam, tau, qp)
quad_expect = expected_stop(m, Delta, v, lam, tau, qp)   # E[max_q U(q|b')] in closed form
quad_voi = float(quad_expect - quad_now)
# tower property: E[U(q|b')] = U(q|b) for each fixed q
tower = {q: (q * m - lam * q * q * (m * m + v)) for q in QS}
tower_check = {}
for q in QS:
    e_pre = q * m - lam * q * q * (m * m + v)
    # E over m' of [q m' - lam q^2 (m'^2 + v')] = q m - lam q^2 (E[m'^2] + v') with E[m'^2]=m^2+Delta
    e_post = q * m - lam * q * q * ((m * m + Delta) + vp)
    tower_check[q] = (e_pre, e_post, e_post - e_pre)

# switching probability for the quadratic criterion: the optimal q=1 is replaced only if m' < lam*(v'), since
# U(1|b')-U(0|b') = m' - lam(m'^2+v') > 0 is the condition; for m' near 0 with lam v' small this has tiny mass
# Here 1-sided threshold for q=1 vs q=0 is root of m' - lam(m'^2+v')=0 near lam*v'.
thr = (1 - math.sqrt(1 - 4 * lam * lam * vp)) / (2 * lam) if 4 * lam * lam * vp < 1 else float('nan')
p_switch = norm.cdf((thr - m) / sd)

print("T2 mean-variance vs quadratic expected utility")
print(f"  parameters: s={s}, lam={lam}, tau={tau}, qprev={qp}, m={m}, v={v:.6g}, rho={rho}, Delta={Delta:.6g}, v'={vp:.6g}")
print(f"  chosen position now (quadratic and MV agree, q*=1): MV J(1|b)={m - lam*v:.10g}, U(1|b)={m - lam*(m*m+v):.10g}")
print(f"  Mean-variance: max_q J(q|b) = {mv_now:.10g}")
print(f"  Mean-variance: E[max_q J(q|b')] = {mv_expect:.12g}")
print(f"  Mean-variance 'information value' VOI_MV = E[max J(b')] - max J(b) = {mv_voi:.6e}")
print(f"  analytic: lam*Delta = {mv_q1_gain:.6e}   (E[s'^2]=v'={vp:.6g} < v={v:.6g})")
print(f"  Quadratic EU: max_q U(q|b) = {quad_now:.10g}")
print(f"  Quadratic EU: E[max_q U(q|b')] (closed form) = {quad_expect:.15g}")
print(f"  Quadratic EU VOI = {quad_voi:.6e}  (float noise scale ~ {np.spacing(abs(quad_now)):.2e})")
print(f"  Tower property check (E over m' of each fixed-q utility, before vs after):")
for q in QS:
    print(f"    q={q:+d}: E[U(q|b)] = {tower_check[q][0]:.12g}, E[U(q|b')] = {tower_check[q][1]:.12g}, diff = {tower_check[q][2]:.2e}")
print(f"  Switch threshold for quadratic q=1 vs q=0: m' = {thr:.6g} ; P(m' < threshold) = {p_switch:.3e}")
print(f"  i.e. the quadratic VOI can only come from the switch region, mass {p_switch:.1e}")
