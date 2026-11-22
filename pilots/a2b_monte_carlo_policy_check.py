"""Independent Monte Carlo check of the exact two-source policy values (A2 / A3 / A4 instances).

Simulates belief paths with common random numbers: z1, z2 ~ N(0,1) i.i.d.; the first observation moves the
mean by rho1 * s * z1 (variance s^2 - D1), the second by sqrt(D21) * z2 (variance s^2 - D1 - D21).  Each policy
makes its decisions from the realized belief only; the realized value of a path is G(m_final, v_final)
minus the acquisition costs, whose conditional expectation given the path equals the policy's exact value.
Policies are defined by the same rules as the exact tree; the optimal policy's first query is chosen by
comparing the exact branch values (closed-form inner step, quadrature outer step).

Run:  python3 a2b_monte_carlo_policy_check.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from voi_core import TwoSource, expected_stop, stop_value  # noqa: E402

SEED = 20261026
N = 2_000_000


def simulate(ts, m, z1, z2, policy):
    """policy in {'never','both','fixed12','fixed21','myopic','optimal'}; returns path values (vectors)."""
    s2 = ts.v0
    D1, D2 = ts.D[1], ts.D[2]
    D21, D12 = ts.Dc[(1, 2)], ts.Dc[(2, 1)]
    n = len(z1)
    value = np.empty(n)
    G0 = float(ts.G(m))

    def stage(first, second, Dfirst, Dsecond_given_first):
        """Query 'first' (always), then optional 'second' by the optimal stop rule, realized along z1, z2."""
        s_first = ts.rho[first] * ts.s
        m1 = m + s_first * z1
        v1 = s2 - Dfirst
        G1 = stop_value(m1, v1, ts.lam, ts.tau, ts.qp)
        cont = -ts.c[second] + expected_stop(m1, Dsecond_given_first, v1, ts.lam, ts.tau, ts.qp)
        query2 = cont > G1
        m2 = m1 + np.sqrt(Dsecond_given_first) * z2
        v2 = v1 - Dsecond_given_first
        G2 = stop_value(m2, v2, ts.lam, ts.tau, ts.qp)
        final = np.where(query2, G2 - ts.c[second], G1)
        return final - ts.c[first], query2

    if policy == "never":
        return np.full(n, G0)
    if policy == "both":
        # acquire both: terminal mean ~ N(m, D_both) and variance s^2 - D_both (law-level check only)
        mb = m + np.sqrt(ts.D_both) * z1
        vb = s2 - ts.D_both
        return stop_value(mb, vb, ts.lam, ts.tau, ts.qp) - ts.c[1] - ts.c[2]
    if policy == "fixed12":
        return stage(1, 2, D1, D21)[0]
    if policy == "fixed21":
        return stage(2, 1, D2, D12)[0]
    if policy in ("myopic", "optimal"):
        if policy == "myopic":
            v = {j: ts.voi_single(m, j) for j in (1, 2)}
            if max(v.values()) <= 0:
                return np.full(n, G0)
            first = 1 if v[1] >= v[2] else 2
        else:
            b1 = ts.branch(m, 1, "quad")
            b2 = ts.branch(m, 2, "quad")
            if max(b1, b2) <= G0:
                return np.full(n, G0)
            first = 1 if b1 >= b2 else 2
        second = 3 - first
        Dfirst = ts.D[first]
        Dsec = ts.Dc[(first, second)]
        return stage(first, second, Dfirst, Dsec)[0]
    raise ValueError(policy)


def run(label, ts, m, exact):
    rng = np.random.default_rng(SEED)
    z1 = rng.standard_normal(N)
    z2 = rng.standard_normal(N)
    print(f"\n[{label}] (s = 1 units), N = {N:,} common random draws; 'fixed*' = forced first query, optional second")
    for pol, ex in exact.items():
        vals = simulate(ts, m, z1, z2, pol)
        mean = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(N)
        print(f"  {pol:<9s}: MC {mean:+.10e} +/- {se:.1e}   exact {ex:+.10e}   (diff/s.e. = {(mean - ex) / se:+.2f})")


def main():
    # designed A3 instance
    ts = TwoSource(1.0, 0.03, 0.008, 0.0, 2e-5, 8e-4, 0.15, 0.22, 0.68)
    m = 0.575
    exact = {
        "never": ts.value_never(m),
        "both": ts.value_always_both(m),  # forced: both queries, then stop
        # 'fixed*' in the simulation force the first query (no initial stop), so compare with the branch values
        "fixed12": ts.branch(m, 1, "quad"),
        "fixed21": ts.branch(m, 2, "quad"),
        "myopic": ts.value_myopic(m, "quad"),
        "optimal": ts.value_optimal(m, "quad"),
    }
    run("A3 designed instance", ts, m, exact)

    # designed A4 instance (cheap source 1 first is optimal, myopic picks source 2)
    ts4 = TwoSource(1.0, 0.03, 0.008, 0.0, 5e-6, 6.5e-4, 0.15, 0.22, 0.68)
    exact4 = {
        "never": ts4.value_never(m),
        "fixed12": ts4.branch(m, 1, "quad"),
        "fixed21": ts4.branch(m, 2, "quad"),
        "myopic": ts4.value_myopic(m, "quad"),
        "optimal": ts4.value_optimal(m, "quad"),
    }
    run("A4 designed instance", ts4, m, exact4)


if __name__ == "__main__":
    main()
