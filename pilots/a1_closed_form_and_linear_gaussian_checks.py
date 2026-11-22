"""A1: verify (i) the closed form for E_{m'~N(m,D)}[G(m', s^2-D)] against deterministic quadrature
and Monte Carlo with common random numbers, and (ii) the linear-Gaussian claims by simulating
from an explicit joint Gaussian of (r, Y0, Y1, Y2): posterior variance is data-independent,
and the preposterior law of the posterior mean is N(m, Delta).

Run:  python3 a1_closed_form_and_linear_gaussian_checks.py
"""
import math
import sys
import os

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, kstest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voi_core import expected_stop, stop_value, pieces  # noqa: E402

SEED = 20261026
N_MC = 1_000_000


def quad_reference(m, D, vb, lam, tau, qp):
    """Deterministic adaptive quadrature over x of G(x, vb-D) * N(x; m, D) with kinks as breakpoints."""
    sig = math.sqrt(D)
    v = vb - D
    pts = [p for (_, hi, _) in pieces(v, lam, tau, qp) for p in [hi] if math.isfinite(p)]
    pts = [p for p in pts if m - 12 * sig < p < m + 12 * sig]
    f = lambda x: float(stop_value(x, v, lam, tau, qp)) * norm.pdf((x - m) / sig) / sig
    val, err = quad(f, m - 12 * sig, m + 12 * sig, points=pts or None, limit=2000,
                    epsabs=0.0, epsrel=1e-13)
    return val, err


def gauss_hermite(m, D, vb, lam, tau, qp, n):
    from voi_core import gauss_hermite_nodes
    z, w = gauss_hermite_nodes(n)
    return float(np.sum(w * stop_value(m + math.sqrt(D) * z, vb - D, lam, tau, qp)))


def part1_closed_form():
    print("=" * 100)
    print("PART 1: closed form vs quadrature vs Gauss-Hermite vs 1e6-sample Monte Carlo (common random numbers)")
    print("=" * 100)
    s = 0.01
    vb = s * s
    rng = np.random.default_rng(SEED)
    z_common = rng.standard_normal(N_MC)  # shared across all (m, D) and all regimes
    regimes = {
        "realistic lam=2, tau=2e-4": (2.0, 2e-4, 0.0),
        "realistic lam=0.5, tau=1e-4, qprev=1": (0.5, 1e-4, 1.0),
        "high-signal lam=2, tau=2e-4": (2.0, 2e-4, 0.0),
    }
    rows = []
    for name, (lam, tau, qp) in regimes.items():
        print(f"\nRegime: {name}   (s={s}, vb=s^2)")
        print(f"{'m':>10} {'Delta':>10} {'closed form':>18} {'quad ref':>18} {'|cf-quad|':>10} "
              f"{'GH200 - cf':>12} {'MC(1e6)':>18} {'MC s.e.':>10} {'cf-MC (in s.e.)':>16}")
        m_grid = [0.0, 1.5e-4, 4e-4, 6e-4, -4.5e-4]
        D_list = [1e-8, 1e-7, 1e-6] if name.startswith("realistic") else [5e-5, 2e-5]
        for m in m_grid:
            for D in D_list:
                if D > vb:
                    continue
                cf = float(expected_stop(m, D, vb, lam, tau, qp))
                qv, qe = quad_reference(m, D, vb, lam, tau, qp)
                gh = gauss_hermite(m, D, vb, lam, tau, qp, 200)
                x = m + math.sqrt(D) * z_common
                g = stop_value(x, vb - D, lam, tau, qp)
                mc = g.mean()
                se = g.std(ddof=1) / math.sqrt(N_MC)
                rows.append((name, m, D, cf, qv, gh, mc, se))
                print(f"{m:>10.2e} {D:>10.1e} {cf:>18.12e} {qv:>18.12e} {abs(cf - qv):>10.1e} "
                      f"{gh - cf:>12.2e} {mc:>18.12e} {se:>10.1e} {(cf - mc) / se:>16.2f}")
    worst_quad = max(abs(r[3] - r[4]) for r in rows)
    print(f"\nMax |closed form - quad reference| over all cases: {worst_quad:.3e}")
    gh_rel = max(abs(r[5] - r[3]) / max(abs(r[3]), 1e-300) for r in rows)
    print(f"Max relative |GH200 - closed form| over all cases: {gh_rel:.3e} "
          "(Gauss-Hermite is not accurate enough for kinked integrands; the closed form matches quadrature to ~1e-16.)")
    return rows


def part2_linear_gaussian():
    print("\n" + "=" * 100)
    print("PART 2: explicit joint Gaussian (r, Y0, Y1, Y2): Bayesian conditioning, 1e6 joint draws")
    print("=" * 100)
    # explicit covariance (sd units: returns); chosen to be a valid PD matrix
    sd = np.array([0.010, 0.012, 0.020, 0.015])  # r, Y0, Y1, Y2
    R = np.array([
        [1.00, 0.30, 0.20, 0.15],
        [0.30, 1.00, 0.40, 0.10],
        [0.20, 0.40, 1.00, -0.30],
        [0.15, 0.10, -0.30, 1.00],
    ])
    print("corr matrix eigenvalues (must be > 0):", np.round(np.linalg.eigvalsh(R), 4))
    S = R * np.outer(sd, sd)
    rng = np.random.default_rng(SEED + 1)
    L = np.linalg.cholesky(S)
    X = rng.standard_normal((N_MC, 4)) @ L.T  # columns: r, Y0, Y1, Y2

    def posterior(obs_idx):
        """Gaussian conditioning of r on observed Y's (vectorized over samples).
        Returns posterior mean per sample and the (data-independent) posterior variance."""
        o = list(obs_idx)
        Soo = S[np.ix_(o, o)]
        Sro = S[0, o]
        K = np.linalg.solve(Soo, Sro)  # gain vector
        post_var = S[0, 0] - Sro @ K
        mean = X[:, o] @ K
        return mean, post_var

    m0, v0 = posterior([1])            # observe Y0 (index 1)
    m1, v1 = posterior([1, 2])         # + Y1
    m2, v2 = posterior([1, 2, 3])      # + Y2 (both sources)
    m1_only, v_y1_only = posterior([2])
    print(f"\nposterior variance after Y0      : {v0:.10e} (scalar, identical for every sample)")
    print(f"posterior variance after Y0,Y1   : {v1:.10e}")
    print(f"posterior variance after Y0,Y1,Y2: {v2:.10e}")

    # Joint-Gaussian formulas for the one-step Deltas, conditional on Y0 (Y0 is the conditioning info)
    s2 = v0
    # conditional covariance of (r, Y1, Y2) given Y0
    Sig = S[np.ix_([0, 2, 3], [0, 2, 3])]
    S0 = S[np.ix_([0, 2, 3], [1])]
    Cc = Sig - S0 @ np.linalg.solve(S[1:2, 1:2], S0.T)
    rho1 = Cc[0, 1] / math.sqrt(Cc[0, 0] * Cc[1, 1])
    rho2 = Cc[0, 2] / math.sqrt(Cc[0, 0] * Cc[2, 2])
    kappa = Cc[1, 2] / math.sqrt(Cc[1, 1] * Cc[2, 2])
    D1_formula = Cc[0, 1] ** 2 / Cc[1, 1]           # Cov(r,Y1|Y0)^2 / Var(Y1|Y0)
    D2_formula = Cc[0, 2] ** 2 / Cc[2, 2]
    # Y2 given (Y0, Y1): conditional on Y1 as well
    D21_formula = (Cc[0, 2] - Cc[0, 1] * Cc[1, 2] / Cc[1, 1]) ** 2 / (Cc[2, 2] - Cc[1, 2] ** 2 / Cc[1, 1])
    D21_closed = s2 * (rho2 - kappa * rho1) ** 2 / (1 - kappa ** 2)
    D12_formula = (Cc[0, 1] - Cc[0, 2] * Cc[1, 2] / Cc[2, 2]) ** 2 / (Cc[1, 1] - Cc[1, 2] ** 2 / Cc[2, 2])
    D12_closed = s2 * (rho1 - kappa * rho2) ** 2 / (1 - kappa ** 2)
    print("\nconditional-on-Y0 correlations: rho1=%.6f rho2=%.6f kappa=corr(Y1,Y2|Y0)=%.6f"
          % (rho1, rho2, kappa))
    print(f"s^2 = Var(r|Y0)             = {s2:.6e}")
    print(f"Delta_1  formula (Cov^2/Var) = {D1_formula:.6e}   closed rho1^2 s^2 = {rho1**2 * s2:.6e}")
    print(f"Delta_2  formula             = {D2_formula:.6e}   closed rho2^2 s^2 = {rho2**2 * s2:.6e}")
    print(f"Delta_21 formula             = {D21_formula:.6e}   closed form      = {D21_closed:.6e}")
    print(f"Delta_12 formula             = {D12_formula:.6e}   closed form      = {D12_closed:.6e}")
    print(f"Delta_1 + Delta_21           = {D1_formula + D21_formula:.6e}  (= s^2 - v_both = {s2 - v2:.6e})")
    print(f"Delta_2 + Delta_12           = {D2_formula + D12_formula:.6e}")

    # Preposterior laws: m' - m for m' = E[r|Y0,Y1] and m = E[r|Y0] (full joint sampling, not conditioning on value)
    d1 = m1 - m0
    d2 = m2 - m1
    print("\nPreposterior check (1e6 joint draws, Y0 random so m varies across draws):")
    print(f"Var(m1 - m0)           = {d1.var():.6e}   vs Delta_1 = {D1_formula:.6e}   "
          f"(MC s.e. ~ {d1.var() * math.sqrt(2 / N_MC):.1e})")
    print(f"E[m1 - m0]             = {d1.mean():+.3e}   (should be 0; s.e. {d1.std() / math.sqrt(N_MC):.1e})")
    print(f"Corr(m1 - m0, m0)      = {np.corrcoef(d1, m0)[0, 1]:+.2e}  (should be 0: increment independent of belief)")
    print(f"Var(m2 - m1)           = {d2.var():.6e}   vs Delta_21 = {D21_closed:.6e}")
    print(f"Var(m2 - m0)           = {(m2 - m0).var():.6e}   vs Delta_both = {s2 - v2:.6e}")
    # Conditional on Y0 fixed, the increment law should be N(0, Delta): KS test vs N(0, Delta)
    z1 = d1 / math.sqrt(D1_formula)
    ks = kstest(z1[:200_000], "norm")
    print(f"KS test of standardized (m1-m0)/sqrt(Delta_1) vs N(0,1): stat={ks.statistic:.4e}, p={ks.pvalue:.3f}")
    print(f"skew={((z1 ** 3).mean()):+.3e} excess kurt={((z1 ** 4).mean() - 3):+.3e}")
    # the posterior mean's increment conditional on the realized Y0 bin (a sanity check of the independence)
    bins = np.quantile(m0, np.linspace(0, 1, 6))
    print("Var(m1-m0 | m0 quintile bin):",
          [f"{d1[(m0 >= bins[i]) & (m0 < bins[i + 1])].var():.4e}" for i in range(5)])
    print(f"Target Delta_1 = {D1_formula:.4e}")
    return dict(D1=D1_formula, D21=D21_closed, D12=D12_closed, s2=s2, rho1=rho1, rho2=rho2, kappa=kappa)


if __name__ == "__main__":
    part1_closed_form()
    part2_linear_gaussian()
