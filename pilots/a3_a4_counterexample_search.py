"""A3 and A4.

A3: a case where BOTH one-step net VOIs are negative (myopic stops) but the exact optimal policy queries.
A4: a case where BOTH one-step net VOIs are positive but the optimal first query differs from the
    myopic argmax.

Normalization: s = 1 in the search and in the designed instances.  The problem is scale covariant:
with real s, lam_real = lam'/s, tau_real = tau' * s, m_real = m' * s, c_real = c' * s, and every value
(G, V, VOI) is multiplied by s.  Real-unit versions are printed for the designed instances.

Run:  python3 a3_a4_counterexample_search.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voi_core import TwoSource, expected_stop  # noqa: E402

SEED = 20261026
LAM, TAU, QP = 0.03, 0.008, 0.0   # normalized (s = 1): realistic-scale analogue lam=3, tau=8e-5 at s=0.01


def psd_ok(r1, r2, k):
    return (1 - r1 * r1 - r2 * r2 - k * k + 2 * k * r1 * r2) > 1e-9


def make(r1, r2, k, c1, c2, lam=LAM, tau=TAU, s=1.0):
    return TwoSource(s, lam, tau, QP, c1, c2, r1, r2, k)


def report_instance(tag, r1, r2, k, m, c1, c2, lam=LAM, tau=TAU):
    print(f"\n[{tag}] s = 1 units")
    print(f"  parameters: rho1={r1}, rho2={r2}, kappa={k}, lam={lam}, tau={tau}, m={m}, qprev=0")
    print(f"  costs: c1={c1:.6g}, c2={c2:.6g}")
    ts = make(r1, r2, k, c1, c2, lam, tau)
    G = ts.value_never(m)
    v1 = ts.voi_single(m, 1)
    v2 = ts.voi_single(m, 2)
    b1 = ts.branch(m, 1, "quad")
    b2 = ts.branch(m, 2, "quad")
    vopt = ts.value_optimal(m, "quad")
    vmy = ts.value_myopic(m, "quad")
    print(f"  Delta_1={ts.D[1]:.10g}  Delta_2={ts.D[2]:.10g}  Delta_21={ts.Dc[(1, 2)]:.10g}  "
          f"Delta_12={ts.Dc[(2, 1)]:.10g}")
    print(f"  G(b)                                 = {G:+.12e}")
    print(f"  one-step net VOI_1 = g(D1)-c1        = {v1:+.12e}   (g(D1) = {v1 + c1:.6e})")
    print(f"  one-step net VOI_2 = g(D2)-c2        = {v2:+.12e}   (g(D2) = {v2 + c2:.6e})")
    print(f"  branch: query 1, then optional 2     = {b1:+.12e}   (minus G: {b1 - G:+.6e})")
    print(f"  branch: query 2, then optional 1     = {b2:+.12e}   (minus G: {b2 - G:+.6e})")
    print(f"  V_opt = {vopt:+.12e}; V_myopic = {vmy:+.12e}; regret = {vopt - vmy:+.6e}")
    print(f"  myopic policy: {'stop' if max(v1, v2) <= 0 else ('query ' + str(1 if v1 >= v2 else 2))}; "
          f"optimal: {'stop' if vopt <= G else 'query (first = ' + str(1 if b1 >= b2 else 2) + ')'}")
    # real units: s = 0.01
    s = 0.01
    tsr = TwoSource(s, lam / s, tau * s, QP, c1 * s, c2 * s, r1, r2, k)
    mr = m * s
    Gr = tsr.value_never(mr)
    Vr = tsr.value_optimal(mr, "quad")
    print(f"  real units s=0.01: lam={lam / s:.4g}, tau={tau * s:.4g} (={tau * s * 1e4:.3g} bp), "
          f"m={mr:.4e}, c1={c1 * s:.4e}, c2={c2 * s:.4e}")
    print(f"    G={Gr:+.6e}  V_opt={Vr:+.6e}  V_opt/s={Vr / s:+.12e} (matches s=1 value {vopt:+.12e})")
    return dict(G=G, v1=v1, v2=v2, b1=b1, b2=b2, vopt=vopt, vmy=vmy)


def convexity_table(r1, r2, k, m, lam=LAM, tau=TAU):
    """Gross VOI g(Delta) = E[G(m', v-Delta)] - G(m, v) at the designed belief, as a function of Delta."""
    ts = make(r1, r2, k, 1.0, 1.0, lam, tau)
    G = ts.value_never(m)
    g = lambda D: float(expected_stop(m, D, 1.0, lam, tau, QP) - G)
    print("\nNon-concavity check: gross VOI g(Delta) at the designed belief (s = 1)")
    for D in (1e-3, 1e-2, 1.5e-2, 2e-2, 2.5e-2, 3e-2, 4e-2, 5e-2, 1e-1):
        print(f"  Delta = {D:.4g}: g = {g(D):.6e}")
    D1, D21, D2, D12 = ts.D[1], ts.Dc[(1, 2)], ts.D[2], ts.Dc[(2, 1)]
    print(f"  g(D1) + g(D21)  = {g(D1) + g(D21):.6e}   vs   g(D1 + D21) = {g(D1 + D21):.6e}  (D1+D21 = D_both)")
    print(f"  g(D2) + g(D12)  = {g(D2) + g(D12):.6e}   vs   g(D2 + D12) = {g(D2 + D12):.6e}")
    print(f"  Delta_1 = {D1:.5f} and Delta_21 = {D21:.5f}: each alone is below the threshold of g "
          f"(g = 0 for Delta <~ 0.01), but their sum is above it.")


def random_search(n, seed):
    """Frequency of A3-type and A4-type configurations among random models with log-uniform costs."""
    rng = np.random.default_rng(seed)
    n_a3 = n_a4 = n_valid = 0  # all candidate tests use split quadrature (no GH screening)
    first_a3 = first_a4 = None
    while n_valid < n:
        r1 = rng.uniform(0.01, 0.99)
        r2 = rng.uniform(0.01, 0.99)
        k = rng.uniform(-0.99, 0.99)
        if not psd_ok(r1, r2, k):
            continue
        lam = math.exp(rng.uniform(math.log(0.01), math.log(1.0)))
        tau = math.exp(rng.uniform(math.log(1e-3), math.log(0.3)))
        bnd = (1 - math.sqrt(max(1 - 4 * lam * (lam + tau), 0))) / (2 * lam)
        m = bnd + rng.uniform(-0.6, 0.6)
        c1 = math.exp(rng.uniform(math.log(1e-6), math.log(1e-1)))
        c2 = math.exp(rng.uniform(math.log(1e-6), math.log(1e-1)))
        ts = TwoSource(1.0, lam, tau, QP, c1, c2, r1, r2, k)
        n_valid += 1
        v1 = ts.voi_single(m, 1)
        v2 = ts.voi_single(m, 2)
        G = ts.value_never(m)
        if v1 < 0 and v2 < 0:
            # screen with GH-200, confirm with split adaptive quadrature
            vopt = ts.value_optimal(m, "dense")
            if vopt - G > 1e-7:
                n_a3 += 1
                if first_a3 is None:
                    first_a3 = (r1, r2, k, lam, tau, m, c1, c2, vopt - G)
        elif v1 > 0 and v2 > 0:
            b1 = ts.branch(m, 1, "dense")
            b2 = ts.branch(m, 2, "dense")
            myo = 1 if v1 >= v2 else 2
            opt = 1 if b1 >= b2 else 2
            if opt != myo and abs(b1 - b2) > 1e-7:
                n_a4 += 1
                if first_a4 is None:
                    first_a4 = (r1, r2, k, lam, tau, m, c1, c2)
    print(f"\nRandom search over {n_valid} feasible models (log-uniform lam in [0.01,1], tau in [1e-3,0.3], "
          f"costs in [1e-6,1e-1], m within +-0.6 of the boundary, s=1):")
    print(f"  both one-step VOI<0 and V_opt>G (A3-type): {n_a3} ({n_a3 / n_valid:.3%})")
    print(f"  both one-step VOI>0 and argmax mismatch (A4-type): {n_a4} ({n_a4 / n_valid:.3%})")
    return first_a3, first_a4


def main():
    print("=" * 100)
    print("A3: both one-step VOIs negative, optimal policy queries (designed exact instance)")
    print("=" * 100)
    # designed instance: cheap weak source 1 and a source 2 whose stand-alone value is below its cost
    rho1, rho2, kappa, m = 0.15, 0.22, 0.68, 0.575
    res3 = report_instance("A3 designed instance", rho1, rho2, kappa, m, c1=2e-5, c2=8e-4)
    convexity_table(rho1, rho2, kappa, m)

    print("\n" + "=" * 100)
    print("A4: both one-step VOIs positive, optimal first query differs from the myopic argmax")
    print("=" * 100)
    res4 = report_instance("A4 designed instance", rho1, rho2, kappa, m, c1=5e-6, c2=6.5e-4)

    print("\n" + "=" * 100)
    print("Random-search frequencies")
    print("=" * 100)
    first_a3, first_a4 = random_search(1500, SEED)
    if first_a3 is not None:
        r1, r2, k, lam, tau, m_, c1, c2, gain = first_a3
        print(f"  first random A3-type example (exact re-check below): rho1={r1:.4f}, rho2={r2:.4f}, "
              f"kappa={k:+.4f}, lam={lam:.4f}, tau={tau:.4f}, m={m_:.4f}, c1={c1:.3e}, c2={c2:.3e}, "
              f"V_opt-G={gain:.3e}")
        report_instance("A3 first random example", r1, r2, k, m_, c1, c2, lam, tau)
    if first_a4 is not None:
        r1, r2, k, lam, tau, m_, c1, c2 = first_a4
        print(f"  first random A4-type example: rho1={r1:.4f}, rho2={r2:.4f}, kappa={k:+.4f}, lam={lam:.4f}, "
              f"tau={tau:.4f}, m={m_:.4f}, c1={c1:.3e}, c2={c2:.3e}")
        report_instance("A4 first random example", r1, r2, k, m_, c1, c2, lam, tau)


if __name__ == "__main__":
    main()
