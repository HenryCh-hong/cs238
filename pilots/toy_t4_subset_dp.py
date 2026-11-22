"""T4: exact-structure DP over the subset lattice for K = 2, 3, 4 optional sources (real units, s = 0.01).

State (S, m): S = set of purchased sources, m = posterior mean of r given Y0 and the purchased observations.
Posterior variance v_S depends only on S (Schur complement of the conditional correlation matrix).
Transition: buying j from S moves m to m + sqrt(Delta_{j|S}) z, z ~ N(0,1), with Delta_{j|S} = v_S - v_{S+j}.
Bellman:  V(S,m) = max{ G(m, v_S), max_{j not in S} [ -c_j + E_z V(S+j, m + sqrt(Delta_{j|S}) z) ] },  V(full,m)=G(m,v_full).
Myopic policy: at (S,m) buy argmax_j of the one-step net VOI (closed form via voi_core.expected_stop) if it is > 0,
else stop; its value V^pi is evaluated by the same recursion along its own actions.

Inner expectation: linear interpolation of V(S+j,.) on a uniform m-grid, z-trapezoid on [-8,8] (161 nodes).
Validation: for K=2 the DP is compared with voi_core.TwoSource (closed form + quadrature).
"""
import os
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

S_SD = 0.01
RHO0 = 0.3
BP = 1e-4
SDM = S_SD * RHO0 / math.sqrt(1 - RHO0 ** 2)     # sd of prior m = E[r|Y0]
SIGR = S_SD / math.sqrt(1 - RHO0 ** 2)           # unconditional sd of r


def subsets(K):
    return [frozenset(c) for n in range(K + 1) for c in __import__("itertools").combinations(range(K), n)]


class Model:
    def __init__(self, K, lam, tau, costs, rho, R):
        self.K, self.lam, self.tau = K, lam, tau
        self.c = np.asarray(costs, dtype=float)
        self.rho = np.asarray(rho, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.vS = {}
        for n in range(K + 1):
            pass
        import itertools
        for n in range(K + 1):
            for comb in itertools.combinations(range(K), n):
                S = frozenset(comb)
                if not S:
                    self.vS[S] = S_SD ** 2
                else:
                    idx = sorted(S)
                    rS = self.rho[idx]
                    RSS = self.R[np.ix_(idx, idx)]
                    self.vS[S] = S_SD ** 2 * (1.0 - rS @ np.linalg.solve(RSS, rS))
        self.full = frozenset(range(K))

    def delta(self, S, j):
        return self.vS[S] - self.vS[S | {j}]


def grid_setup(N=2001, Z=161, zmax=8.0):
    Lm = 7.0 * SIGR
    mg = np.linspace(-Lm, Lm, N)
    h = mg[1] - mg[0]
    z = np.linspace(-zmax, zmax, Z)
    wz = norm.pdf(z) * (z[1] - z[0])
    wz[0] *= 0.5
    wz[-1] *= 0.5
    wp = norm.pdf(mg / SDM) * h
    wp = wp / wp.sum()
    return mg, z, wz, wp


def exp_over_z(Vn, mg, sd, z, wz):
    """E_z[ Vn(m + sd z) ] for every m on the grid mg, linear interpolation of Vn on mg."""
    pts = (mg[:, None] + sd * z[None, :]).ravel()
    vals = np.interp(pts, mg, Vn).reshape(len(mg), len(z))
    return vals @ wz


def solve_dp(md, mg, z, wz, myopic=False):
    K = md.K
    Vs, pol, Vmyo = {}, {}, {}
    order = sorted(md.vS.keys(), key=lambda S: -len(S))
    for S in order:
        G = stop_value(mg, md.vS[S], md.lam, md.tau, 0.0)
        if S == md.full:
            Vs[S] = G
            Vmyo[S] = G
            pol[S] = np.full(len(mg), -1)
            continue
        conts = {}
        cont_myo = {}
        for j in range(K):
            if j in S:
                continue
            d = md.delta(S, j)
            Vn = Vs[S | {j}]
            conts[j] = -md.c[j] + exp_over_z(Vn, mg, math.sqrt(d), z, wz)
            # one-step net VOI (closed form), used by the myopic policy
            voi1 = expected_stop(mg, d, md.vS[S], md.lam, md.tau, 0.0) - G - md.c[j]
            cont_myo[j] = voi1
        best = np.maximum(G, np.max(np.stack(list(conts.values())), axis=0))
        Vs[S] = best
        # optimal action index: -1 = stop, else j
        jlist = list(conts.keys())
        stack = np.stack([conts[j] for j in jlist])
        arg = np.array(jlist)[np.argmax(stack, axis=0)]
        pol[S] = np.where(best > G, arg, -1)
        # myopic policy
        vo = np.stack([cont_myo[j] for j in jlist])
        jm = np.array(jlist)[np.argmax(vo, axis=0)]
        buy = np.max(vo, axis=0) > 0
        act_m = np.where(buy, jm, -1)
        Vm = G.copy()
        for j in jlist:
            mask = act_m == j
            if mask.any():
                d = md.delta(S, j)
                cj = -md.c[j] + exp_over_z(Vmyo[S | {j}], mg, math.sqrt(d), z, wz)
                Vm[mask] = cj[mask]
        Vmyo[S] = Vm
    return Vs, Vmyo, pol


def random_factor_instance(rng, K):
    while True:
        rho = rng.uniform(0.05, 0.4, size=K)
        L = rng.normal(size=(K, 2))
        Rraw = L @ L.T
        d = np.sqrt(np.diag(Rraw))
        R = Rraw / np.outer(d, d)
        R = 0.6 * R + 0.4 * np.eye(K)  # shrink towards identity to keep Y-correlations moderate
        C = np.block([[np.ones((1, 1)), rho[None, :]], [rho[:, None], R]])
        if np.linalg.eigvalsh(C).min() > 1e-6:
            return rho, R


def equicorr_instance(K, kappa, rng):
    rho = rng.uniform(0.05, 0.4, size=K)
    R = (1 - kappa) * np.eye(K) + kappa * np.ones((K, K))
    C = np.block([[np.ones((1, 1)), rho[None, :]], [rho[:, None], R]])
    if np.linalg.eigvalsh(C).min() <= 1e-9:
        return None
    return rho, R


def validate_K2(lam, tau, costs_real, rho, R):
    """Compare DP (K=2) with voi_core exact values at several m."""
    md = Model(2, lam, tau, costs_real, rho, R)
    mg, z, wz, wp = grid_setup()
    Vs, Vmyo, pol = solve_dp(md, mg, z, wz)
    ts = TwoSource(s=S_SD, lam=lam, tau=tau, qp=0.0, c1=costs_real[0], c2=costs_real[1],
                   rho1=rho[0], rho2=rho[1], kappa=R[0, 1])
    errs, errs_myo = [], []
    test_m = [-0.3 * SDM, 0.0, 0.5 * SDM, 1.0 * SDM, 2.0 * SDM]
    for m in test_m:
        i = int(np.argmin(np.abs(mg - m)))
        mm = mg[i]
        ref = ts.value_optimal(mm, method="dense")
        dp = Vs[frozenset()][i]
        myo_ref = ts.value_myopic(mm, method="dense")
        myo_dp = Vmyo[frozenset()][i]
        errs.append(abs(ref - dp))
        errs_myo.append(abs(myo_ref - myo_dp))
    return max(errs), max(errs_myo), ts.voi_single(0.0, 1), ts.D_both


if __name__ == "__main__":
    NGRID = int(sys.argv[1]) if len(sys.argv) > 1 else 4001
    rng = np.random.default_rng(20261026)
    # ---- validation against closed form for K=2
    print("VALIDATION (K=2 DP vs voi_core closed form + quadrature)")
    for lam, tau_bp, costs_bp, rho, R_off in [
        (3.0, 3.0, (1.0, 2.0), (0.2, 0.3), 0.4),
        (5.0, 1.0, (0.5, 5.0), (0.1, 0.4), -0.5),
        (1.0, 5.0, (0.1, 0.3), (0.05, 0.4), 0.5),
    ]:
        R = np.array([[1.0, R_off], [R_off, 1.0]])
        e_opt, e_myo, _, _ = validate_K2(lam, tau_bp * BP, [c * BP for c in costs_bp], rho, R)
        print(f"  lam={lam} tau={tau_bp}bp costs={costs_bp}bp rho={rho} kappa={R_off}: "
              f"max|V_DP-V_closedform| = {e_opt:.3e} (real units), max|V_myopic_DP - V_myopic_ref| = {e_myo:.3e}")
    print()

    # ---- random instances
    results = []
    tol_rel = 1e-6
    mg, z, wz, wp = grid_setup(N=NGRID)
    print(f'm-grid N={NGRID}')
    for K in (3, 4):
        n_fac = 60
        n_eq = 20
        cases = []
        for _ in range(n_fac):
            rho, R = random_factor_instance(rng, K)
            cases.append(("factor", rho, R))
        kappas = [-0.5, 0.0, 0.5] if K == 3 else [-0.3, 0.0, 0.5]
        for kap in kappas:
            for _ in range(n_eq // len(kappas)):
                inst = equicorr_instance(K, kap, rng)
                if inst is not None:
                    cases.append((f"equi{kap}", inst[0], inst[1]))
        for fam, rho, R in cases:
            lam = rng.uniform(1.0, 5.0)
            tau = rng.uniform(1.0, 5.0) * BP
            costs = np.exp(rng.uniform(np.log(0.1 * BP), np.log(10 * BP), size=K))
            md = Model(K, lam, tau, costs, rho, R)
            t0 = time.perf_counter()
            Vs, Vmyo, pol = solve_dp(md, mg, z, wz)
            t_dp = time.perf_counter() - t0
            V0 = Vs[frozenset()]
            Vm0 = Vmyo[frozenset()]
            G0 = stop_value(mg, md.vS[frozenset()], md.lam, md.tau, 0.0)
            gross = float(wp @ (V0 - G0))
            regret_pt = V0 - Vm0
            reg_prior = float(wp @ regret_pt)
            max_reg = float(regret_pt.max())
            # per-decision time: one myopic decision (K closed-form one-step VOIs at one m)
            t1 = time.perf_counter()
            for _ in range(200):
                for j in range(K):
                    expected_stop(0.0, md.delta(frozenset(), j), md.vS[frozenset()], md.lam, md.tau, 0.0)
            t_dec = (time.perf_counter() - t1) / 200
            results.append(dict(K=K, fam=fam, t_dp=t_dp, t_dec=t_dec, n_states=(2 ** K) * len(mg),
                                gross=gross, reg_prior=reg_prior, max_reg=max_reg,
                                reg_rel=(reg_prior / gross) if gross > 1e-15 else float('nan'),
                                regret=(max_reg > tol_rel * max(gross, 1e-15)),
                                regret_strict=(max_reg > 1e-12)))
    import pickle
    with open(_cache_path(f"t4_results_N{NGRID}.pkl"), "wb") as f:
        pickle.dump(results, f)
    for K in (3, 4):
        rs = [r for r in results if r["K"] == K]
        n = len(rs)
        n_reg = sum(r["regret"] for r in rs)
        n_reg_strict = sum(r["regret_strict"] for r in rs)
        rel = [r["reg_rel"] for r in rs if np.isfinite(r["reg_rel"])]
        print(f"K={K}: instances={n}, DP time per instance mean={np.mean([r['t_dp'] for r in rs]):.3f}s "
              f"(per state={np.mean([r['t_dp']/r['n_states'] for r in rs])*1e6:.2f} us), "
              f"one myopic decision (K closed-form VOIs) = {np.mean([r['t_dec'] for r in rs])*1e6:.1f} us")
        print(f"   regret instances (max_m regret > 1e-6 x gross VOI): {n_reg}/{n} = {n_reg/n:.3f}; "
              f"(max_m regret > 1e-12 real units): {n_reg_strict}/{n} = {n_reg_strict/n:.3f}")
        print(f"   prior-averaged regret / gross VOI: mean={np.mean(rel):.4f}, median={np.median(rel):.4f}, max={np.max(rel):.4f}")
