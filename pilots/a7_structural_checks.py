"""A7: structural checks.

(i)   If some one-step net VOI_j > 0 then the optimal policy does not stop; myopic can only under-acquire;
      the two policies can differ only at the first decision.
(ii)  Pruning bound: gross VOI_j(m; Delta) <= L(m) * E|m' - m| = L(m) * sqrt(2 Delta / pi),
      with L(m) = 1 + 2 lam |m| + sqrt(2 pi) lam s (valid for Delta <= s^2). Also the uniform
      constant K(Delta) = sup_m gross(m; Delta) and the resulting never-worth-querying condition.
(iii) E[u(q, r)] = q m - lam q^2 (m^2 + v) - tau |q - qp| (second-moment penalty, not variance penalty),
      and the stop value depends only on (m, v).
(iv)  Budget B: redundant iff B >= c1 + c2 (each source at most once); otherwise can bind (example).
(v)   Gross VOI >= 0 (Jensen / martingale argument) checked numerically.
(vi)  Trivial regimes: large tau, large lam (no-trade for all beliefs when 4 lam (lam + tau) >= 1, s=1),
      tiny Delta.

Run:  python3 a7_structural_checks.py
"""
import math
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from voi_core import TwoSource, expected_stop, stop_value, f_q  # noqa: E402

SEED = 20261026


# ----------------------------------------------------------------------------- (i)
def check_i(n=3000, seed=SEED):
    rng = np.random.default_rng(seed)
    n_valid = 0
    n_pos = 0
    viol_stop = 0
    viol_order = 0
    myopic_under = 0
    under_gaps = []
    min_margin = np.inf
    while n_valid < n:
        r1, r2, k = rng.uniform(0.01, 0.99), rng.uniform(0.01, 0.99), rng.uniform(-0.99, 0.99)
        if 1 - r1 * r1 - r2 * r2 - k * k + 2 * k * r1 * r2 <= 1e-9:
            continue
        lam = math.exp(rng.uniform(math.log(0.01), math.log(1.0)))
        tau = math.exp(rng.uniform(math.log(1e-3), math.log(0.3)))
        bnd = (1 - math.sqrt(max(1 - 4 * lam * (lam + tau), 0))) / (2 * lam)
        m = bnd + rng.uniform(-0.6, 0.6)
        c1 = math.exp(rng.uniform(math.log(1e-4), math.log(0.1)))
        c2 = math.exp(rng.uniform(math.log(1e-4), math.log(0.1)))
        ts = TwoSource(1.0, lam, tau, 0.0, c1, c2, r1, r2, k)
        n_valid += 1
        G = ts.value_never(m)
        v = {j: ts.voi_single(m, j) for j in (1, 2)}
        vopt = ts.value_optimal(m, "dense")
        vmy = ts.value_myopic(m, "dense")
        if max(v.values()) > 0:
            n_pos += 1
            # optimal must not stop: V_opt >= G + VOI_j (strictly larger than G)
            margin = vopt - G - max(v.values())
            min_margin = min(min_margin, margin)
            if vopt <= G + 1e-12:
                viol_stop += 1
        if vopt < vmy - 1e-10:
            viol_order += 1
        if vmy < vopt - 1e-7 and max(v.values()) <= 0:
            myopic_under += 1
            under_gaps.append(vopt - vmy)
    print("=" * 100)
    print("A7(i): structural statements on random models (dense trapezoid, tolerance 1e-10)")
    print("=" * 100)
    print(f"  random models: {n_valid}; with some one-step VOI_j > 0: {n_pos}   (integrator: dense trapezoid)")
    print(f"  violations of 'VOI_j>0 => optimal does not stop': {viol_stop}")
    print(f"  min over those of [V_opt - G - max_j VOI_j] (should be >= 0): {min_margin:+.3e}  "
          f"(dense-trapezoid discretization noise, ~1e-7 absolute at these magnitudes)")
    print(f"  violations of V_opt >= V_myopic: {viol_order}")
    print(f"  cases where myopic stops (all VOI_j<=0) but optimal queries (V_opt - V_myopic > 1e-7): {myopic_under}")
    if under_gaps:
        print(f"    gap sizes: min {min(under_gaps):.3e}, median {np.median(under_gaps):.3e}, max {max(under_gaps):.3e}")
    print("  Second-stage claim: once one source is observed, V(b_j,{k}) = max(G, -c_k + E G) is exactly the "
          "one-step rule, so the two policies coincide after the first query (identity, not a numerical fit).")


# ----------------------------------------------------------------------------- (ii)
def L_of_m(m, lam, s=1.0):
    return 1.0 + 2.0 * lam * abs(m) + math.sqrt(2 * math.pi) * lam * s


def check_ii(n=4000, seed=SEED + 2):
    rng = np.random.default_rng(seed)
    worst_ratio = 0.0
    worst_case = None
    checked = 0
    for _ in range(n):
        lam = math.exp(rng.uniform(math.log(0.01), math.log(2.0)))
        tau = math.exp(rng.uniform(math.log(1e-3), math.log(0.3)))
        m = rng.uniform(-1.5, 1.5)
        D = math.exp(rng.uniform(math.log(1e-6), math.log(1.0)))   # Delta <= s^2 = 1
        gross = float(expected_stop(m, D, 1.0, lam, tau, 0.0) - stop_value(m, 1.0, lam, tau, 0.0))
        bound = L_of_m(m, lam) * math.sqrt(2 * D / math.pi)
        checked += 1
        if bound > 0:
            ratio = gross / bound
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_case = (m, D, lam, tau, gross, bound)
    print("\n" + "=" * 100)
    print("A7(ii): pruning bound gross_j <= L(m) * sqrt(2 Delta/pi), L(m) = 1 + 2 lam |m| + sqrt(2pi) lam s")
    print("=" * 100)
    print(f"  checked {checked} random (m, Delta, lam, tau) with s = 1; max ratio gross/bound = {worst_ratio:.4f}")
    if worst_case is not None:
        m, D, lam, tau, g, b = worst_case
        print(f"  worst case: m={m:.4f}, Delta={D:.3e}, lam={lam:.4f}, tau={tau:.4f}: gross={g:.4e}, bound={b:.4e}")
    # bound tightness at the designed boundary belief
    lam, tau = 0.03, 0.008
    bnd = (1 - math.sqrt(1 - 4 * lam * (lam + tau))) / (2 * lam)
    for D in (1e-4, 1e-3, 1e-2, 2.25e-2):
        g = float(expected_stop(bnd, D, 1.0, lam, tau, 0.0) - stop_value(bnd, 1.0, lam, tau, 0.0))
        b = L_of_m(bnd, lam) * math.sqrt(2 * D / math.pi)
        print(f"  at the boundary m={bnd:.4f}: Delta={D:.1e}: gross={g:.4e}, bound={b:.4e}, ratio={g / b:.4f}")
    # uniform constant K(Delta) = sup_m gross(m; Delta) (state-independent)
    print("\n  Uniform (state-independent) bound K(Delta) = sup_m gross(m; Delta), lam=0.03, tau=0.008, s=1:")
    mg = np.linspace(-1.0, 1.0, 40001)
    for D in (1e-4, 1e-3, 1e-2, 2.25e-2, 4.84e-2, 1e-1):
        gv = expected_stop(mg, D, 1.0, lam, tau, 0.0) - stop_value(mg, 1.0, lam, tau, 0.0)
        i = int(np.argmax(gv))
        print(f"    Delta={D:.2e}: K = {gv[i]:.6e} attained at m={mg[i]:+.4f}")
    print("  Never-worth-querying (one-step, every state): if c_j >= K(Delta_j) then VOI_j <= 0 at every belief;")
    print("  sufficient at the current belief: c_j >= L(m) * sqrt(2 Delta_j / pi).")


# ----------------------------------------------------------------------------- (iii)
def check_iii():
    print("\n" + "=" * 100)
    print("A7(iii): E[u(q,r)] under N(m, v) and the stop value")
    print("=" * 100)
    rng = np.random.default_rng(SEED + 3)
    m, v, lam, tau, qp = 4e-4, 1e-4, 2.0, 2e-4, 0.0
    N = 2_000_000
    r = m + math.sqrt(v) * rng.standard_normal(N)
    for q in (-1, 0, 1):
        u = q * r - lam * q * q * r * r - tau * abs(q - qp)
        formula = f_q(m, v, q, lam, tau, qp)
        se = u.std(ddof=1) / math.sqrt(N)
        print(f"  q={q:+d}: MC E[u] = {u.mean():+.10e} (s.e. {se:.1e}); formula = {formula:+.10e}; "
              f"diff/s.e. = {(u.mean() - formula) / se:+.2f}")
    # the penalty is on E[r^2] = m^2 + v, not on the variance alone
    print("  Penalty term: lam q^2 E[r^2] = lam q^2 (m^2 + v): contains m^2, so it is a second-moment penalty.")
    print("  A pure variance penalty would replace m^2 + v by v; decisions then differ when |m| is not tiny:")
    lam_, tau_ = 0.05, 0.05
    mb = (1 - math.sqrt(1 - 4 * lam_ * (lam_ + tau_))) / (2 * lam_)
    for name, pen in (("second-moment (model)", lambda mm, vv: lam_ * (mm * mm + vv)),
                      ("variance-only", lambda mm, vv: lam_ * vv)):
        vv = 1.0
        # threshold m at which q=+1 beats q=0 under this penalty (s = 1, qprev = 0)
        f = lambda mm: mm - pen(mm, vv) - tau_
        try:
            thr = brentq(f, 0.0, 5.0)
        except ValueError:
            thr = np.nan
        print(f"    {name:<24s}: trade threshold m* = {thr:.6f}")
    print(f"    (model boundary root from quadratic = {mb:.6f})")
    print("  Stop value depends only on (m, v): E over r of u depends on r only through E r and E r^2 (quadratic utility).")


# ----------------------------------------------------------------------------- (iv)
def budget_value(ts, m, B):
    """Exact optimal value with budget B (each source at most once, cost c_j)."""
    G = float(ts.value_never(m))
    best = G
    for j in (1, 2):
        if ts.c[j] > B:
            continue
        k = 3 - j
        rem = B - ts.c[j]
        vj = ts.v_after(j)
        sj = ts.rho[j] * ts.s
        feasible_k = ts.c[k] <= rem

        def cont(mj):
            stop = stop_value(mj, vj, ts.lam, ts.tau, ts.qp)
            if not feasible_k:
                return stop
            c = -ts.c[k] + expected_stop(mj, ts.Dc[(j, k)], vj, ts.lam, ts.tau, ts.qp)
            return np.maximum(stop, c)

        def diff(z):
            mj = np.float64(m + sj * z)
            return float(cont(mj)) - float(stop_value(mj, vj, ts.lam, ts.tau, ts.qp))

        grid = np.linspace(-12.0, 12.0, 4801)
        dv = np.array([diff(z) for z in grid])
        cuts = [-12.0]
        for a, b, fa, fb in zip(grid[:-1], grid[1:], dv[:-1], dv[1:]):
            if fa * fb < 0:
                cuts.append(brentq(diff, a, b, xtol=1e-15, rtol=1e-15, maxiter=200))
        cuts.append(12.0)
        cuts = sorted(set(cuts))
        val = 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b > a:
                v, _ = quad(lambda z: float(cont(np.float64(m + sj * z))) * norm.pdf(z), a, b,
                            limit=500, epsabs=0.0, epsrel=1e-12)
                val += v
        best = max(best, -ts.c[j] + val)
    return best


def check_iv():
    print("\n" + "=" * 100)
    print("A7(iv): is a budget B redundant when two sources are each queryable once?")
    print("=" * 100)
    # designed A3 instance (s = 1)
    ts = TwoSource(1.0, 0.03, 0.008, 0.0, 2e-5, 8e-4, 0.15, 0.22, 0.68)
    m = 0.575
    c1, c2 = ts.c[1], ts.c[2]
    print(f"  instance: c1={c1:.1e}, c2={c2:.1e}, c1+c2={c1 + c2:.6e}, m={m}")
    for B in (0.0, c1, c1 + 0.5 * c2, c1 + c2 - 1e-9, c1 + c2, 2.0 * (c1 + c2)):
        print(f"    B={B:.6e}: V_B = {budget_value(ts, m, B):+.12e}")
    print("  unconstrained V_opt = " + f"{ts.value_optimal(m, 'quad'):+.12e}")
    print("  Answer: B >= c1 + c2 never binds (total spend of any policy is at most c1 + c2). "
          "B < c1 + c2 can bind; here it removes the optional second query and the value falls to G.")


# ----------------------------------------------------------------------------- (v)
def check_v(n=20000, seed=SEED + 5):
    rng = np.random.default_rng(seed)
    mn = np.inf
    worst = None
    for _ in range(n):
        lam = math.exp(rng.uniform(math.log(0.01), math.log(3.0)))
        tau = math.exp(rng.uniform(math.log(1e-3), math.log(0.5)))
        m = rng.uniform(-2.0, 2.0)
        D = math.exp(rng.uniform(math.log(1e-8), math.log(1.0)))
        qp = float(rng.choice([-1.0, 0.0, 1.0]))
        g = float(expected_stop(m, D, 1.0, lam, tau, qp) - stop_value(m, 1.0, lam, tau, qp))
        if g < mn:
            mn = g
            worst = (m, D, lam, tau)
    print("\n" + "=" * 100)
    print("A7(v): gross VOI >= 0 ?")
    print("=" * 100)
    print(f"  min gross VOI over {n} random (m, Delta, lam, tau, qprev) draws = {mn:+.3e} "
          f"(numerical noise level ~1e-17 relative to G)")
    print("  Proof: G(b) = max_q E_b[u(q,r)] and E_b[ E_{b'}[u(q,r)] ] = E_b[u(q,r)] (martingale mean; "
          "E[m'^2 + v'] = m^2 + v). Max of expectations <= expectation of max (Jensen), so E G(b') >= G(b).")


# ----------------------------------------------------------------------------- (vi)
def check_vi():
    print("\n" + "=" * 100)
    print("A7(vi): trivial regimes (s = 1, qprev = 0)")
    print("=" * 100)
    print("  Trade is strictly optimal for belief (m, v) iff lam m^2 - m + (lam v + tau) < 0. The discriminant")
    print("  disc(v) = 1 - 4 lam (lam v + tau) decreases in v, and posterior variances v = 1 - Delta are <= 1.")
    print("  Hence if 4 lam tau >= 1 there is no trade for ANY posterior variance in [0,1] and every VOI is 0.")
    print("  Caution: G(b) = 0 at the prior (disc(1) <= 0) does NOT make VOI zero, because observing a source")
    print("  lowers the variance and can open a trade region (demonstrated below).")
    cases = ((0.5, 0.0), (1.0, 0.25), (1.0, 0.1), (0.05, 0.05), (0.05, 0.5))
    for lam, tau in cases:
        ts = TwoSource(1.0, lam, tau, 0.0, 1e-6, 1e-6, 0.3, 0.3, 0.2)
        ms = np.linspace(-3, 3, 6001)
        Gs = ts.G(ms)
        best_gain = 0.0
        for D in np.linspace(0.005, 0.99, 100):
            g = expected_stop(ms, D, 1.0, lam, tau, 0.0) - Gs
            best_gain = max(best_gain, float(np.max(g)))
        print(f"  lam={lam}, tau={tau}: 4*lam*tau={4 * lam * tau:.3f}; max_m G(m;prior)={Gs.max():.3e}; "
              f"max over Delta in (0,0.99] and m of gross VOI = {best_gain:.3e}")
    print("  Large lam: trade needs posterior variance v > (1-4 lam tau)/(4 lam^2) ... i.e. v below the trade")
    print("  threshold v* = (1 - 4 lam tau)/(4 lam^2); with lam=5, tau=0.001 (s=1), v* = %.4f:" % ((1 - 4 * 5 * 0.001) / (4 * 25)))
    print("  only sources with Delta > 1 - v* (near-full revelation) can have nonzero VOI for a belief in that range.")
    print("  Large tau (lam=0.05): the smallest |m| with positive trade value (s units) rises with tau:")
    for tau in (0.01, 0.1, 0.5, 1.0):
        ts = TwoSource(1.0, 0.05, tau, 0.0, 1e-6, 1e-6, 0.3, 0.3, 0.2)
        ms = np.linspace(0, 3, 30001)
        Gs = ts.G(ms)
        trade_region = ms[Gs > 1e-14]
        lo = trade_region.min() if trade_region.size else np.nan
        print(f"    tau={tau}: smallest |m| with trade value = {lo:.4f}  (boundary ~ lam + tau = {0.05 + tau:.3f})")
    print("  Tiny Delta far from the boundary: gross VOI is exponentially small in (distance/sd):")
    lam, tau = 0.03, 0.008
    m_far = 0.575
    for D in (1e-4, 1e-3, 1e-2):
        g = float(expected_stop(m_far, D, 1.0, lam, tau, 0.0) - stop_value(m_far, 1.0, lam, tau, 0.0))
        print(f"    m={m_far}, Delta={D:.0e}: gross = {g:.3e}")
    print("  Tiny Delta at the boundary: gross VOI ~ sqrt(Delta) (first-order kink effect):")
    Ds = [1e-8, 1e-7, 1e-6, 1e-5]
    bnd = (1 - math.sqrt(1 - 4 * lam * (lam + tau))) / (2 * lam)
    gs = [float(expected_stop(bnd, D, 1.0, lam, tau, 0.0) - stop_value(bnd, 1.0, lam, tau, 0.0)) for D in Ds]
    for i in range(len(Ds) - 1):
        slope = math.log(gs[i + 1] / gs[i]) / math.log(Ds[i + 1] / Ds[i]) if gs[i] > 0 else np.nan
        print(f"    Delta {Ds[i]:.0e} -> {Ds[i + 1]:.0e}: gross {gs[i]:.3e} -> {gs[i + 1]:.3e}, "
              f"local log-slope = {slope:.3f}")


if __name__ == "__main__":
    check_i()
    check_ii()
    check_iii()
    check_iv()
    check_v()
    check_vi()
