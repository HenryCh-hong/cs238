"""A2: exact expected net utility of each policy under the known joint Gaussian, plus
value of adaptivity.  Also cross-checks the 1-D quadrature tree against Gauss-Hermite.

Run:  python3 a2_exact_policy_values.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voi_core import TwoSource  # noqa: E402


def policy_table(ts, m):
    """All exact values at belief mean m (variance s^2 after the free forecast Y0)."""
    G = ts.value_never(m)
    out = {
        "never acquire": G,
        "always both": ts.value_always_both(m),
        "fixed 1 then 2 (optimal stop)": ts.value_fixed_order(m, 1, "quad"),
        "fixed 2 then 1 (optimal stop)": ts.value_fixed_order(m, 2, "quad"),
        "myopic VOI": ts.value_myopic(m, "quad"),
        "exact optimal (Bellman)": ts.value_optimal(m, "quad"),
    }
    j, vo = ts.myopic_first(m)
    ol, name, cands = ts.value_open_loop_best(m)
    info = dict(
        one_step_voi_1=vo[1], one_step_voi_2=vo[2], myopic_first=j,
        open_loop_best=name, open_loop_value=ol,
        value_of_adaptivity=out["exact optimal (Bellman)"] - ol,
    )
    gh = {
        "fixed 1 then 2 (optimal stop)": ts.value_fixed_order(m, 1, "gh", 400),
        "fixed 2 then 1 (optimal stop)": ts.value_fixed_order(m, 2, "gh", 400),
        "myopic VOI": ts.value_myopic(m, "gh", 400),
        "exact optimal (Bellman)": ts.value_optimal(m, "gh", 400),
    }
    return out, info, gh, cands


def main():
    s, lam, tau, qp = 0.01, 2.0, 2e-4, 0.0
    cases = {
        "base (realistic, positively correlated errors)": dict(
            rho1=0.05, rho2=0.08, kappa=0.3, c1=5e-5, c2=1e-4),
        "base, cheaper sources": dict(rho1=0.05, rho2=0.08, kappa=0.3, c1=1e-5, c2=2e-5),
        "high-signal synthetic (rho up to 0.5)": dict(
            rho1=0.5, rho2=0.4, kappa=0.3, c1=2e-3, c2=3e-3),
    }
    m_list = {
        "base": {"m=4.0e-4 (at boundary)": 4.0e-4, "m=6.0e-4": 6.0e-4},
        "high-signal": {"m=0.0 (no-trade center)": 0.0, "m=2e-3": 2e-3},
    }
    for cname, kw in cases.items():
        ts = TwoSource(s, lam, tau, qp, kw["c1"], kw["c2"], kw["rho1"], kw["rho2"], kw["kappa"])
        print("=" * 100)
        print(f"CASE: {cname}")
        print(f"  s={s}, lam={lam}, tau={tau}, qprev={qp}; rho1={kw['rho1']}, rho2={kw['rho2']}, "
              f"kappa={kw['kappa']}, c1={kw['c1']}, c2={kw['c2']}")
        print(f"  Delta_1={ts.D[1]:.6e}  Delta_2={ts.D[2]:.6e}  Delta_21={ts.Dc[(1, 2)]:.6e}  "
              f"Delta_12={ts.Dc[(2, 1)]:.6e}  Delta_both={ts.D_both:.6e}  (s^2={s*s:.6e})")
        mset = m_list["high-signal"] if "high-signal" in cname else m_list["base"]
        for mname, m in mset.items():
            out, info, gh, cands = policy_table(ts, m)
            print(f"  --- belief {mname}: G(b) = {ts.value_never(m):.10e}")
            for k, v in out.items():
                extra = ""
                if k in gh:
                    extra = f"   [GH400 - quad = {gh[k] - v:+.2e}]"
                print(f"    {k:<34s} {v:+.12e}{extra}")
            print(f"    one-step VOI_1 = {info['one_step_voi_1']:+.6e}, VOI_2 = {info['one_step_voi_2']:+.6e}; "
                  f"myopic first = {info['myopic_first']}")
            print(f"    open-loop (non-adaptive) candidates: "
                  + ", ".join(f"{k}: {v:+.6e}" for k, v in cands.items()))
            print(f"    best open-loop set = {info['open_loop_best']} ({info['open_loop_value']:+.6e}); "
                  f"value of adaptivity V_opt - open-loop = {info['value_of_adaptivity']:+.3e}")
            opt = out["exact optimal (Bellman)"]
            gap = opt - out["myopic VOI"]
            print(f"    myopic regret V_opt - V_myopic = {gap:+.3e}  "
                  f"({'myopic == optimal' if abs(gap) < 1e-15 else 'differs'})")


if __name__ == "__main__":
    main()
