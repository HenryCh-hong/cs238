"""A6: correlated forecast errors.

(a) Y_i = r + eps_i, Var(eps_i) = a_i^2 (a1 = a <= a2 = b), Corr(eps1, eps2) = rho_e.
    Verify the incremental posterior precision from Y2 given Y1:
        (a - rho_e b)^2 / (a^2 b^2 (1 - rho_e^2)),
    zero at rho_e = a/b, divergent as rho_e -> 1 when a != b (vanishes when a = b).
    Verify it is the same object as the Delta of the decision model:
        Delta_{2|1} = Cov(r,Y2|Y1)^2/Var(Y2|Y1) = v1 - v12  (posterior variance drop),
    and that the gross VOI of Y2 given Y1 saturates (does not diverge) because Delta <= v1.
(b) Finance-like factor model: r = f + u (u large, unpredictable), Y_j = alpha_j f + eta_j.
    Show that raw error correlation Corr(r - Y1, r - Y2) -> 1 as the unpredictable part grows,
    while the decision-relevant quantities are the conditional correlations
    rho_j = Corr(r, Y_j) and kappa = Corr(Y1, Y2), which stay small.

Run:  python3 a6_correlated_errors.py
"""
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from voi_core import TwoSource, expected_stop  # noqa: E402
from a5_myopic_regret_grid import boundary_root  # noqa: E402

FIG_A = os.path.join(HERE, "figures", "a6_incremental_precision_and_gross_voi.png")
FIG_B = os.path.join(HERE, "figures", "a6_factor_model_error_correlation.png")
LAM, TAU, QP = 0.05, 0.05, 0.0  # same normalized decision parameters as A5 (s = 1)


def incr_precision(a, b, rho):
    return (a - rho * b) ** 2 / (a * a * b * b * (1 - rho * rho))


def incr_precision_from_matrix(a, b, rho):
    """Independent check: information about r from (Y1, Y2) minus from Y1 alone (noise covariance)."""
    Sig = np.array([[a * a, rho * a * b], [rho * a * b, b * b]])
    one = np.ones(2)
    I_tot = one @ np.linalg.solve(Sig, one)
    return I_tot - 1.0 / (a * a)


def part_a(verbose=True):
    out = {}
    if verbose:
        print("=" * 100)
        print("A6(a): incremental precision formula and its link to Delta")
        print("=" * 100)
    # 1) algebra check over a grid of (a, b, rho)
    max_err = 0.0
    for a in np.linspace(0.2, 3.0, 15):
        for b in np.linspace(0.2, 3.0, 15):
            for rho in np.linspace(-0.95, 0.95, 39):
                f = incr_precision(a, b, rho)
                g = incr_precision_from_matrix(a, b, rho)
                max_err = max(max_err, abs(f - g) / max(1.0, abs(g)))
    if verbose:
        print(f"max relative |formula - (I_total - I_1)| over grid: {max_err:.3e}")
    # 2) zero and divergence
    a, b = 1.0, 2.0
    if verbose:
        print(f"zero at rho = a/b = {a/b}: formula = {incr_precision(a, b, a / b):.3e}")
        for rho in (0.9, 0.99, 0.999, 0.9999):
            print(f"  a={a}, b={b}, rho={rho}: incr precision = {incr_precision(a, b, rho):.6e}")
        print(f"a=b case: a=b=1 at rho=0.9,0.999,0.99999: "
              + ", ".join(f"{incr_precision(1, 1, r):.4e}" for r in (0.9, 0.999, 0.99999)))
    out["zero_at_a_over_b"] = incr_precision(a, b, a / b)

    # 3) Delta link: prior variance s^2 = 1 for r (no Y0; r ~ N(0,1) is the belief), a and b given.
    s2 = 1.0
    if verbose:
        print("\nLink to the decision model (prior r ~ N(0, s^2), s = 1):")
    for (a, b) in [(3.0, 5.0), (1.0, 2.0), (0.5, 1.0)]:
        rows = []
        for rho in (-0.5, 0.0, 0.3, 0.6, 0.9, 0.99, 0.999):
            v1 = 1.0 / (1.0 / s2 + 1.0 / a ** 2)
            I_tot = (a * a + b * b - 2 * rho * a * b) / (a * a * b * b * (1 - rho * rho))
            v12 = 1.0 / (1.0 / s2 + I_tot)
            Delta_post_drop = v1 - v12
            # decision-model formula: rho_j, kappa from covariances of (r, Y1, Y2)
            cov_r1, cov_r2 = s2, s2
            var1, var2 = s2 + a * a, s2 + b * b
            cov12 = s2 + rho * a * b
            r1 = cov_r1 / math.sqrt(s2 * var1)
            r2 = cov_r2 / math.sqrt(s2 * var2)
            kap = cov12 / math.sqrt(var1 * var2)
            D21_model = s2 * (r2 - kap * r1) ** 2 / (1 - kap * kap)
            # Cov(r,Y2|Y1)^2/Var(Y2|Y1) computed directly from covariances (independent of the closed form)
            c_r2_given_1 = cov_r2 - cov_r1 * cov12 / var1
            v_2_given_1 = var2 - cov12 ** 2 / var1
            D21_direct = c_r2_given_1 ** 2 / v_2_given_1
            rows.append((rho, incr_precision(a, b, rho), Delta_post_drop, D21_direct, D21_model))
        if verbose:
            print(f"  (a,b)=({a},{b}); columns rho_e | incr precision | v1-v12 | Cov^2/Var (direct) | model closed form")
            for row in rows:
                print("   %6.3f | %.6e | %.6e | %.6e | %.6e" % row)
        out[(a, b)] = rows
    # 4) saturation: gross VOI of Y2 given Y1 as rho -> 1 is bounded by v1 (Delta <= v1)
    if verbose:
        print("\nSaturation: as rho_e -> 1 (a<b), Delta_{2|1} -> v1 = s^2 a^2/(s^2+a^2):")
        a, b = 1.0, 2.0
        v1 = 1.0 / (1.0 + 1.0 / a ** 2)
        for rho in (0.9, 0.999, 0.999999):
            I_tot = (a * a + b * b - 2 * rho * a * b) / (a * a * b * b * (1 - rho * rho))
            v12 = 1.0 / (1.0 + I_tot)
            print(f"  rho={rho}: Delta_(2|1) = {v1 - v12:.10f}  (v1 = {v1:.10f})")
    return out


def decision_setup_for_pair(a, b, rho_e, c1=0.0, c2=0.0):
    """Two sources observing r (prior N(0,1)): returns TwoSource object with (rho1, rho2, kappa) from
    the explicit (a, b, rho_e) model."""
    s2 = 1.0
    var1, var2 = s2 + a * a, s2 + b * b
    rho1 = s2 / math.sqrt(s2 * var1)
    rho2 = s2 / math.sqrt(s2 * var2)
    kappa = (s2 + rho_e * a * b) / math.sqrt(var1 * var2)
    return TwoSource(1.0, LAM, TAU, QP, c1, c2, rho1, rho2, kappa)


def gross_voi_second_given_first(a, b, rho_e, m):
    """E over Y1 of [E G(m12) - G(m1)] at prior belief mean m (note E G(m1) is over Y1)."""
    ts = decision_setup_for_pair(a, b, rho_e)
    D1 = ts.D[1]
    D12 = ts.Dc[(1, 2)]  # Delta of Y2 given Y1
    v0 = ts.v0
    # E_{m1} [ G(m1 + d2) ] - E_{m1}[G(m1)]  with m1 ~ N(m, D1) and d2 ~ N(0, D12): two closed forms
    # E[G(m12, v0 - D1 - D12)] over m12 ~ N(m, D1 + D12) equals the joint expectation (martingale)
    E_m12 = float(expected_stop(m, D1 + D12, v0, LAM, TAU, QP))
    E_m1 = float(expected_stop(m, D1, v0, LAM, TAU, QP))
    return E_m12 - E_m1, ts


def part_a_figure():
    m = boundary_root(LAM, TAU)
    pairs = [(3.0, 5.0), (1.0, 2.0), (0.5, 1.0), (1.0, 1.0)]
    rho_e = np.linspace(-0.95, 0.995, 400)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for (a, b) in pairs:
        ip = np.array([incr_precision(a, b, r) for r in rho_e])
        ax[0].plot(rho_e, ip, label=f"a={a}, b={b}")
        gv = []
        for r in rho_e:
            g, _ = gross_voi_second_given_first(a, b, r, m)
            gv.append(g)
        ax[1].plot(rho_e, gv, label=f"a={a}, b={b}")
        ts_g = decision_setup_for_pair(a, b, 0.0)
        g2 = float(ts_g.EG(m, ts_g.D[2]) - ts_g.G(m))
        ax[1].axhline(g2, ls=":", lw=0.8, color=ax[1].lines[-1].get_color())
    ax[0].set_yscale("log")
    ax[0].set_xlabel("rho_e = Corr(eps1, eps2)")
    ax[0].set_ylabel("incremental precision (Y2 | Y1)")
    ax[0].axvline(0, color="k", lw=0.4)
    ax[0].set_title("A6(a) incremental precision; zero at rho_e = a/b")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("rho_e = Corr(eps1, eps2)")
    ax[1].set_ylabel("gross VOI of Y2 given Y1  (utility units, s = 1)")
    ax[1].set_title("gross VOI of 2nd source vs error correlation\n(state at boundary; dotted: Y2 alone, rho_e-free)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_A, dpi=130)
    print(f"saved {FIG_A}")


def part_b(verbose=True):
    if verbose:
        print("\n" + "=" * 100)
        print("A6(b): finance-like factor model (shared factor + large unpredictable component)")
        print("=" * 100)
    sf2 = 1.0           # Var(f): predictable factor
    om1, om2 = 0.5, 0.5 # idiosyncratic forecast noise sd (eta_j)
    al1, al2 = 1.0, 1.0 # loadings of forecasts on the factor
    res = []
    for ratio in (0.0, 1.0, 3.0, 10.0, 30.0, 100.0):
        su2 = ratio * sf2   # Var(u) = unpredictable part
        var_r = sf2 + su2
        # forecasts Y_j = al_j f + eta_j ; errors e_j = r - Y_j = u + (1-al_j) f - eta_j
        cov_r_y1 = al1 * sf2
        cov_r_y2 = al2 * sf2
        var_y1 = al1 * al1 * sf2 + om1 * om1
        var_y2 = al2 * al2 * sf2 + om2 * om2
        cov_y12 = al1 * al2 * sf2
        rho1 = cov_r_y1 / math.sqrt(var_r * var_y1)
        rho2 = cov_r_y2 / math.sqrt(var_r * var_y2)
        kappa = cov_y12 / math.sqrt(var_y1 * var_y2)
        # naive error correlation Corr(r - Y1, r - Y2)
        cov_e = su2 + (1 - al1) * (1 - al2) * sf2
        var_e1 = su2 + (1 - al1) ** 2 * sf2 + om1 ** 2
        var_e2 = su2 + (1 - al2) ** 2 * sf2 + om2 ** 2
        naive = cov_e / math.sqrt(var_e1 * var_e2)
        # decision quantities: the belief is the prior N(0, var_r) in this factor model (no Y0)
        s = math.sqrt(var_r)
        ts = TwoSource(s, LAM / s, TAU * s, QP, 0.0, 0.0, rho1, rho2, kappa)
        D1 = ts.D[1]
        D21 = ts.Dc[(1, 2)]
        frac1 = D1 / (s * s)
        res.append((ratio, rho1, rho2, kappa, naive, frac1, D21 / (s * s)))
        if verbose:
            print(f"  Var(u)/Var(f)={ratio:7.1f}: rho1={rho1:.4f} rho2={rho2:.4f} kappa=Corr(Y1,Y2)={kappa:.4f} | "
                  f"naive Corr(r-Y1,r-Y2)={naive:.4f} | Delta_1/s^2={frac1:.4e}, Delta_21/s^2={D21 / (s * s):.4e}")
    if verbose:
        print("  Interpretation: raw error correlation approaches 1 as the unpredictable part dominates, "
              "but Delta depends on rho_j = Corr(r,Y_j) and kappa = Corr(Y1,Y2), which shrink. "
              "Plugging the naive error correlation in as kappa changes the answer:")
    # misspecification illustration at ratio 30
    ratio = 30.0
    su2 = ratio
    var_r = 1.0 + su2
    var_y1 = 1.0 + om1 ** 2
    var_y2 = 1.0 + om2 ** 2
    rho1 = 1.0 / math.sqrt(var_r * var_y1)
    rho2 = 1.0 / math.sqrt(var_r * var_y2)
    kap_true = 1.0 / math.sqrt(var_y1 * var_y2)
    # alpha = 1: var(e_j) = su2 + om_j^2 and corr(e1,e2) = su2 / sqrt((su2+om1^2)(su2+om2^2))
    naive = su2 / math.sqrt((su2 + om1 ** 2) * (su2 + om2 ** 2))
    s = math.sqrt(var_r)
    ts_true = TwoSource(s, LAM / s, TAU * s, QP, 0.0, 0.0, rho1, rho2, kap_true)
    ts_naive = TwoSource(s, LAM / s, TAU * s, QP, 0.0, 0.0, rho1, rho2, naive)
    if verbose:
        print(f"  ratio=30: proper kappa={kap_true:.4f} -> Delta_21/s^2={ts_true.Dc[(1, 2)] / s ** 2:.5f}; "
              f"naive error-corr kappa={naive:.4f} -> Delta_21/s^2={ts_naive.Dc[(1, 2)] / s ** 2:.5f}")
    return res


def part_b_figure(res):
    ratios = [r[0] for r in res]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogx([max(x, 1e-2) for x in ratios], [r[4] for r in res], "o-", label="naive Corr(r-Y1, r-Y2)")
    ax[0].semilogx([max(x, 1e-2) for x in ratios], [r[3] for r in res], "s-", label="kappa = Corr(Y1,Y2)")
    ax[0].semilogx([max(x, 1e-2) for x in ratios], [r[1] for r in res], "^-", label="rho1 = Corr(r,Y1)")
    ax[0].set_xlabel("Var(u)/Var(f)  (unpredictable part / predictable factor)")
    ax[0].set_ylabel("correlation")
    ax[0].legend(fontsize=8)
    ax[0].set_title("A6(b) error correlation vs decision-relevant correlations")
    ax[1].semilogx([max(x, 1e-2) for x in ratios], [r[6] for r in res], "o-", label="Delta_21 / s^2")
    ax[1].semilogx([max(x, 1e-2) for x in ratios], [r[5] for r in res], "s-", label="Delta_1 / s^2 = rho1^2")
    ax[1].set_xlabel("Var(u)/Var(f)")
    ax[1].set_ylabel("Delta / s^2")
    ax[1].legend(fontsize=8)
    ax[1].set_title("information increments in the factor model")
    fig.tight_layout()
    fig.savefig(FIG_B, dpi=130)
    print(f"saved {FIG_B}")


if __name__ == "__main__":
    part_a()
    part_a_figure()
    res = part_b()
    part_b_figure(res)
