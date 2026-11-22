"""T3: value of adaptivity under finance-like parameters (real units, s = 0.01).

Free-forecast state: Y0 is observed; m = E[r|Y0] ~ N(0, sdm^2) with sdm^2 = s^2 rho0^2/(1-rho0^2), and the
posterior variance of r given Y0 is s^2.  rho0 = Corr(r, Y0) is NOT in the brief; we fix rho0 = 0.3 (stated
assumption) and report it.  The integral over m is a trapezoid on [-6 sdm, 6 sdm] (Gaussian weights).

For each grid point and each m:
  V_cl(m) = V_opt(m)  : exact adaptive optimum (stop/continue at every stage, both orders)
  V_ol(m)             : best pre-committed set in {none, {1}, {2}, {1,2}} with no observation-dependent stopping
  G(m)                : stop value
Conditional structure: Delta_j = rho_j^2 s^2, Dc(j->k) from voi_core, kappa = Corr(Y1,Y2|Y0).
Inner expectation over the first observation: trapezoid on z in [-10, 10] (kinks make this O(h^2) accurate).
"""
import os
import itertools
import math
import sys
import time

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _cache_path(name):
    """Scratch files written by this script (git-ignored)."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)

from voi_core import TwoSource, expected_stop, stop_value  # noqa: E402

S = 0.01
RHO0 = 0.3
SDM = S * RHO0 / math.sqrt(1 - RHO0 ** 2)
BP = 1e-4


def make_model(lam, tau_bp, rho1, rho2, kappa, c1_bp, c2_bp):
    return TwoSource(s=S, lam=lam, tau=tau_bp * BP, qp=0.0,
                     c1=c1_bp * BP, c2=c2_bp * BP, rho1=rho1, rho2=rho2, kappa=kappa)


def branch_vec(md, j, Mg, zg, wz, n_z_unused=None):
    """Vectorized over m-grid Mg: -c_j + sum_z w_z cont_after(m_j(z), j) ; also returns the continue indicator."""
    vj = md.v_after(j)
    sj = md.rho[j] * md.s
    k = 3 - j
    mj = Mg[:, None] + sj * zg[None, :]
    stop = stop_value(mj, vj, md.lam, md.tau, md.qp)
    cont = -md.c[k] + expected_stop(mj, md.Dc[(j, k)], vj, md.lam, md.tau, md.qp)
    cont_after = np.maximum(stop, cont)
    val = -md.c[j] + cont_after @ wz
    return val, (cont > stop), stop


def value_grids(md, Mg, wm, zg, wz):
    """Return G(m), V_opt(m), V_ol(m), and per-m optimal first source and second-buy indicator over z."""
    G = stop_value(Mg, md.v0, md.lam, md.tau, md.qp)
    b1, ind1, _ = branch_vec(md, 1, Mg, zg, wz)
    b2, ind2, _ = branch_vec(md, 2, Mg, zg, wz)
    Vopt = np.maximum(G, np.maximum(b1, b2))
    # open-loop candidates
    ol = np.maximum.reduce([
        G,
        -md.c[1] + expected_stop(Mg, md.D[1], md.v0, md.lam, md.tau, md.qp),
        -md.c[2] + expected_stop(Mg, md.D[2], md.v0, md.lam, md.tau, md.qp),
        -md.c[1] - md.c[2] + expected_stop(Mg, md.D_both, md.v0, md.lam, md.tau, md.qp),
    ])
    first = np.where(b1 >= b2, 1, 2)
    first = np.where(np.maximum(b1, b2) > G, first, 0)
    return G, Vopt, ol, first, b1, b2, ind1, ind2


def second_purchase_dependence(md, Mg, first, ind1, ind2, zg, wz, wm):
    """For m-states where the optimal policy buys a first source, classify the second-purchase decision as a
    function of the first observation z1: 'never' (never buys second), 'always' (buys for every z1 in
    support), or 'depends' (buys for some z1 and not others).  Weights: density of z1 restricted to the support."""
    supp = wz > 1e-12
    out = {"first_buy_states_w": 0.0, "never_w": 0.0, "always_w": 0.0, "depends_w": 0.0}
    for i in range(len(Mg)):
        if first[i] == 0:
            continue
        ind = ind1[i] if first[i] == 1 else ind2[i]
        sub = ind[supp]
        wi = wm[i]
        out["first_buy_states_w"] += wi
        if not sub.any():
            out["never_w"] += wi
        elif sub.all():
            out["always_w"] += wi
        else:
            out["depends_w"] += wi
    return out


def run(grid_lam, grid_tau, grid_rho, grid_kappa, grid_cost, M=361, Z=1001, zmax=8.0, log=None):
    Mm = np.linspace(-6 * SDM, 6 * SDM, M)
    wm = norm.pdf(Mm / SDM) * (Mm[1] - Mm[0]) / SDM
    wm = wm / wm.sum()  # normalized weights (sum of trapezoid weights over the prior)
    zg = np.linspace(-zmax, zmax, Z)
    wz = norm.pdf(zg) * (zg[1] - zg[0])
    wz[0] *= 0.5
    wz[-1] *= 0.5
    rows = []
    t0 = time.time()
    for lam, tau_bp, (r1, r2), kappa, (c1, c2) in itertools.product(
            grid_lam, grid_tau, grid_rho, grid_kappa, grid_cost):
        md = make_model(lam, tau_bp, r1, r2, kappa, c1, c2)
        G, Vopt, ol, first, b1, b2, ind1, ind2 = value_grids(md, Mm, wm, zg, wz)
        EG = float(wm @ G)
        Vo = float(wm @ Vopt)
        Vl = float(wm @ ol)
        adapt = Vo - Vl
        voi_tot = Vo - EG
        dep = second_purchase_dependence(md, Mm, first, ind1, ind2, zg, wz, wm)
        rows.append(dict(lam=lam, tau_bp=tau_bp, r1=r1, r2=r2, kappa=kappa, c1=c1, c2=c2,
                         EG=EG, Vopt=Vo, Vol=Vl, adapt=adapt, voi=voi_tot,
                         frac=(adapt / voi_tot) if voi_tot > 1e-18 else float('nan'),
                         **dep))
    el = time.time() - t0
    return rows, el


if __name__ == "__main__":
    grid_lam = [1.0, 5.0]
    grid_tau = [1.0, 5.0]
    grid_rho = list(itertools.product([0.05, 0.1, 0.2, 0.4], repeat=2))
    grid_kappa = [-0.5, 0.0, 0.5]
    grid_cost = list(itertools.product([0.1, 1.0, 10.0], repeat=2))
    print(f"SDM={SDM:.6g} (sd of m, rho0={RHO0}), n_configs={len(grid_lam)*len(grid_tau)*len(grid_rho)*len(grid_kappa)*len(grid_cost)}")
    rows, el = run(grid_lam, grid_tau, grid_rho, grid_kappa, grid_cost)
    print(f"runtime {el:.1f}s for {len(rows)} configs ({el/len(rows)*1000:.1f} ms/config)")
    import pickle
    with open(_cache_path("t3_rows.pkl"), "wb") as f:
        pickle.dump(rows, f)
    print("saved rows")
