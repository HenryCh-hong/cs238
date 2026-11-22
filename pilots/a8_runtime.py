"""A8: runtime per decision.

Compares
  * closed-form E[G(m', v')] (truncated-normal moments; one scalar call)
  * deterministic 1-D quadrature over the realized mean (scipy quad with kink breakpoints)
  * Monte Carlo VOI with common random numbers (N = 1e3, 1e4, 1e5, 1e6) and its standard error
  * the full two-source optimal policy (exact Bellman with closed-form inner step and 1-D outer step)
    with three outer integrators: Gauss-Hermite (200 nodes), dense trapezoid (12001 nodes), split quad.

Timings are wall-clock medians on this machine (Python 3.14, numpy 2.4, scipy 1.17).

Run:  python3 a8_runtime.py
"""
import math
import os
import sys
import time

import numpy as np
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from voi_core import TwoSource, expected_stop, stop_value, pieces  # noqa: E402

SEED = 20261026


def timeit(fn, repeats):
    ts = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def voi_quad_1d(m, D, v_before, lam, tau, qp):
    """E[G(X, v_before - D)] for X ~ N(m, D) by adaptive quadrature over x (no closed form used)."""
    sig = math.sqrt(D)
    v = v_before - D
    pts = [hi for (_, hi, _) in pieces(v, lam, tau, qp) if math.isfinite(hi) and m - 12 * sig < hi < m + 12 * sig]
    f = lambda x: float(stop_value(x, v, lam, tau, qp)) * math.exp(-0.5 * ((x - m) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
    val, _ = quad(f, m - 12 * sig, m + 12 * sig, points=pts or None, limit=2000, epsabs=0.0, epsrel=1e-11)
    return val


def main():
    lam, tau, qp = 0.03, 0.008, 0.0
    s2 = 1.0
    m, D = 0.575, 0.0225          # designed A3 state: Delta_1 = 0.0225
    G = float(stop_value(m, s2, lam, tau, qp))
    print("=" * 100)
    print("A8: runtime per decision (s = 1 units; designed state m=0.575, Delta=0.0225, lam=0.03, tau=0.008)")
    print("=" * 100)

    # 1) closed form
    t, cf = timeit(lambda: float(expected_stop(m, D, s2, lam, tau, qp)), 2000)
    print(f"closed form E[G] (scalar)            : {t * 1e6:10.1f} us   value {cf:.12e}")
    # 2) 1-D quadrature
    t, qv = timeit(lambda: voi_quad_1d(m, D, s2, lam, tau, qp), 50)
    print(f"1-D adaptive quadrature E[G]         : {t * 1e3:10.3f} ms   value {qv:.12e}   |diff vs closed| = {abs(qv - cf):.2e}")
    # 3) Monte Carlo with common random numbers
    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal(1_000_000)
    for N in (1_000, 10_000, 100_000, 1_000_000):
        def mc():
            x = m + math.sqrt(D) * Z[:N]
            g = stop_value(x, s2 - D, lam, tau, qp)
            return g.mean(), g.std(ddof=1) / math.sqrt(N)
        t, (mean, se) = timeit(mc, 20 if N < 1_000_000 else 5)
        print(f"Monte Carlo N={N:>9,d} (CRN)        : {t * 1e3:10.3f} ms   estimate {mean:.12e} +/- {se:.1e}  "
              f"(closed-form within {abs(mean - cf) / se:.2f} s.e.)")

    # 4) full two-source optimal policy
    ts = TwoSource(1.0, lam, tau, qp, 2e-5, 8e-4, 0.15, 0.22, 0.68)
    print("\nFull two-source optimal policy (exact Bellman; one decision = V_opt and the myopic argmax):")
    for name, meth, rep in (("Gauss-Hermite 200 nodes", "gh", 200), ("dense trapezoid 12001 nodes", "dense", 20),
                            ("split adaptive quadrature", "quad", 3)):
        t, vopt = timeit(lambda meth=meth: ts.value_optimal(m, meth), rep)
        tm, vmy = timeit(lambda meth=meth: ts.value_myopic(m, meth), rep)
        tv, _ = timeit(lambda: ts.myopic_first(m), 200)
        print(f"  {name:<28s}: V_opt {t * 1e3:9.2f} ms ({vopt:.12e}); myopic policy {tm * 1e3:9.2f} ms "
              f"({vmy:.12e}); one-step VOIs {tv * 1e6:8.1f} us")
    t, vb = timeit(lambda: ts.value_optimal(m, "dense"), 20)
    print(f"\n  closed-form VOI for both sources + argmax (no tree) : "
          f"{timeit(lambda: ts.myopic_first(m), 500)[0] * 1e6:8.1f} us")

    # 5) tiny cost of the non-myopic tree for a sweep of scale
    print("\nRuntime scaling of the exact tree with the number of outer nodes (dense trapezoid):")
    import voi_core as V
    orig = V.TwoSource.branch

    def branch_nodes(self, mm, j, method="gh", n_gh=200):
        if method == "dense_n":
            return self._branch_dense_n(mm, j, n_gh)
        return orig(self, mm, j, method, n_gh)

    def _branch_dense_n(self, mm, j, n):
        z = np.linspace(-12.0, 12.0, n)
        h = z[1] - z[0]
        w = V.norm.pdf(z) * h
        w[0] *= 0.5
        w[-1] *= 0.5
        sj = self.rho[j] * self.s
        vals = self.cont_after(mm + sj * z, j, self.v_after(j))
        return -self.c[j] + float(np.sum(w * vals))

    V.TwoSource._branch_dense_n = _branch_dense_n
    V.TwoSource.branch = branch_nodes
    ref = ts.branch(m, 1, "quad")
    for n in (501, 2001, 12001, 48001):
        t, val = timeit(lambda n=n: ts.branch(m, 1, "dense_n", n), 10)
        print(f"  n={n:>6d}: branch(1) {t * 1e3:8.2f} ms, value {val:.12e}, error vs split quad {val - ref:+.2e}")
    print(f"  split quad reference branch(1) = {ref:.12e}")


if __name__ == "__main__":
    main()
