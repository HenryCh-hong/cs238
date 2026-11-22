"""B1 (policy level): true expected value of the agent's full two-source policy vs the true-model optimum.

Setup (4-dim Gaussian over (r, Y0, Y1, Y2)); Y0 is free; sources Y1 and Y2 each at most once.
Policies (all decide from a covariance M: either the agent's estimate or the truth):
  stop      : never query; act on the free baseline
  always    : query both sources
  myopic(M) : query argmax_j VOI_j(M) if positive (one-step VOI, the proposal's rule), then the
              remaining source iff its one-step VOI is positive
  bellman(M): exact finite-horizon DP under M (value of continuing includes the option to query
              the other source later)
  optimal   : bellman(truth) = the true-model optimal policy

Evaluation (the TRUE expected value of any policy): fixed common-random-number draws of
(r, Y0, Y1, Y2) from the truth. Each draw's realized acquisitions follow the policy; the value is the
TRUE posterior expected utility of the chosen position (Rao-Blackwellized over r) minus the costs paid.
Using the same draws for every policy gives paired differences (common random numbers).

Computation: every decision depends on the observed data only through the posterior mean after the
observed set (a linear function of the data) and a Gaussian preposterior increment with data-independent
variance. So the continuation values are 1-D functions of the mean, tabulated on a grid and
interpolated per draw. The root Bellman continuation uses an outer expectation over N(m0, Delta) and an
inner one, both midpoint-quantile quadrature.

Replications: each (rho, n) cell uses R fresh calibration samples of size n (the draws are fixed).
"""
import json
import sys
import time
import numpy as np
from pilot_common import (build_cov4, sample_cov, ledoit_wolf, post_weights, delta_info,
                          stop_value, EG_after, qmid, util_mom, RHOS, NS, SEED, LAM, TAU,
                          RES_DIR, FIG_DIR)

COST = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0e-4
R = int(sys.argv[2]) if len(sys.argv) > 2 else 60
NDRAW = 50_000
TAG = f"_cost{COST:.0e}".replace("e-0", "e-").replace("+", "")
CA = CB = COST
KZ = 512           # quantile nodes for one-step expectations (tables)
KZ1 = 512          # outer nodes for the root continuation
NGRID = 1201
t0 = time.time()


class Model:
    """Decision tables for a covariance M (agent's estimate or truth)."""

    def __init__(self, M, L):
        self.M = M
        self.w0 = post_weights(M, [1])[0][0]
        self.s0 = post_weights(M, [1])[1]
        self.w12, self.s12 = post_weights(M, [1, 2])
        self.w13, self.s13 = post_weights(M, [1, 3])
        self.Da0 = delta_info(M, [1], 2)            # Y1 at root
        self.Db0 = delta_info(M, [1], 3)            # Y2 at root
        self.Db_a = delta_info(M, [1, 2], 3)        # Y2 after Y1
        self.Da_b = delta_info(M, [1, 3], 2)        # Y1 after Y2
        self.mg = np.linspace(-L, L, NGRID)
        mg = self.mg
        s0, Da0, Db0 = self.s0, self.Da0, self.Db0
        v1a, v1b = s0 - Da0, s0 - Db0
        z, w = qmid(KZ), np.full(KZ, 1.0 / KZ)
        self.G0 = stop_value(mg, s0)[0]
        self.voi_a = EG_after(mg, s0, Da0, z, w) - self.G0 - CA
        self.voi_b = EG_after(mg, s0, Db0, z, w) - self.G0 - CB
        # after the first query a: state (m1, v1a); stop or query b
        G_after_a = stop_value(mg, v1a)[0]
        self.voi_b_after_a = EG_after(mg, v1a, self.Db_a, z, w) - G_after_a - CB
        Ha = np.maximum(G_after_a, G_after_a + self.voi_b_after_a)
        # after the first query b: stop or query a
        G_after_b = stop_value(mg, v1b)[0]
        self.voi_a_after_b = EG_after(mg, v1b, self.Da_b, z, w) - G_after_b - CA
        Hb = np.maximum(G_after_b, G_after_b + self.voi_a_after_b)
        # root continuation values: E over the preposterior increment of H
        z1, w1 = qmid(KZ1), np.full(KZ1, 1.0 / KZ1)
        self.Ca = np.zeros(NGRID)
        self.Cb = np.zeros(NGRID)
        for zz, ww in zip(z1, w1):
            self.Ca += ww * np.interp(mg + np.sqrt(Da0) * zz, mg, Ha)
            self.Cb += ww * np.interp(mg + np.sqrt(Db0) * zz, mg, Hb)
        # option values at the root (before the first query): -c + E[H(m1)]
        self.optA = -CA + self.Ca
        self.optB = -CB + self.Cb

    def interp(self, arr, x):
        return np.interp(x, self.mg, arr)


def decide(X, T, kind):
    """Return boolean arrays (qa, qb): which sources each draw queries under policy kind."""
    N = X.shape[0]
    if kind == "stop":
        return np.zeros(N, bool), np.zeros(N, bool)
    if kind == "always":
        return np.ones(N, bool), np.ones(N, bool)
    m0 = X[:, 1] * T.w0
    if kind == "myopic":
        va = T.interp(T.voi_a, m0)
        vb = T.interp(T.voi_b, m0)
        first_a = va >= vb
        go = np.maximum(va, vb) > 0
    elif kind == "bellman":
        G0 = T.interp(T.G0, m0)
        oa = T.interp(T.optA, m0)
        ob = T.interp(T.optB, m0)
        first_a = oa >= ob
        go = np.maximum(oa, ob) > G0
    else:
        raise ValueError(kind)
    qa_first = go & first_a
    qb_first = go & ~first_a
    # stage 2 after a (decided on the mean after Y0,Y1)
    m1a = X[:, 1] * T.w12[0] + X[:, 2] * T.w12[1]
    qb_after_a = T.interp(T.voi_b_after_a, m1a) > 0
    # stage 2 after b
    m1b = X[:, 1] * T.w13[0] + X[:, 3] * T.w13[1]
    qa_after_b = T.interp(T.voi_a_after_b, m1b) > 0
    qa = qa_first | (qb_first & qa_after_b)
    qb = qb_first | (qa_first & qb_after_a)
    return qa, qb


def policy_values(X, Ttrue, Tdec, qa, qb, Mdec, Strue):
    """True expected value (per draw) of the policy's final position and costs, for common draws X.

    The final position is chosen with the decision model's posterior (Mdec) and evaluated with
    the true posterior mean/variance for the same observed set.
    """
    sets = {(False, False): [1], (True, False): [1, 2], (False, True): [1, 3], (True, True): [1, 2, 3]}
    m_dec = np.zeros(X.shape[0])
    v_dec = np.zeros(X.shape[0])
    m_tru = np.zeros(X.shape[0])
    v_tru = np.zeros(X.shape[0])
    for key, idx in sets.items():
        mask = (qa == key[0]) & (qb == key[1])
        if not mask.any():
            continue
        wd, sd = post_weights(Mdec, idx)
        wt, st = post_weights(Strue, idx)
        m_dec[mask] = X[mask][:, idx] @ wd
        v_dec[mask] = sd
        m_tru[mask] = X[mask][:, idx] @ wt
        v_tru[mask] = st
    _, q = stop_value(m_dec, v_dec)
    R2 = m_tru ** 2 + v_tru
    val = util_mom(q, m_tru, R2) - CA * qa - CB * qb
    return val


def run(rho):
    S4 = build_cov4(rho)
    L = np.linalg.cholesky(S4)
    rng = np.random.default_rng([SEED, 7, int(round(rho * 1000))])
    Z = rng.standard_normal((NDRAW, 4))
    X = Z @ L.T
    # grid range covering all posterior means
    wAll = post_weights(S4, [1, 2, 3])[0]
    Lgrid = 12.0 * np.sqrt(wAll @ S4[np.ix_([1, 2, 3], [1, 2, 3])] @ wAll)
    Ttrue = Model(S4, Lgrid)
    tr = {}
    for kind in ("stop", "always"):
        qa, qb = decide(X, Ttrue, kind)
        tr[kind] = policy_values(X, Ttrue, Ttrue, qa, qb, S4, S4)
    qa_o, qb_o = decide(X, Ttrue, "bellman")
    V_opt = policy_values(X, Ttrue, Ttrue, qa_o, qb_o, S4, S4)
    qa_m, qb_m = decide(X, Ttrue, "myopic")
    V_myo_true = policy_values(X, Ttrue, Ttrue, qa_m, qb_m, S4, S4)
    diff_true = V_opt - V_myo_true
    truth = dict(
        V_stop=float(tr["stop"].mean()), V_always=float(tr["always"].mean()),
        V_optimal=float(V_opt.mean()), V_optimal_se_MC=float(V_opt.std(ddof=1) / np.sqrt(NDRAW)),
        V_true_myopic=float(V_myo_true.mean()),
        diff_bellman_minus_myopic=float(diff_true.mean()),
        diff_bellman_minus_myopic_se_MC=float(diff_true.std(ddof=1) / np.sqrt(NDRAW)),
        diff_bellman_minus_myopic_sd_per_draw=float(diff_true.std(ddof=1)),
        frac_draws_myopic_differs=float(np.mean((qa_o != qa_m) | (qb_o != qb_m))),
        P_optimal_query_any=float(np.mean(qa_o | qb_o)),
        P_optimal_query_both=float(np.mean(qa_o & qb_o)),
        mean_draw_value_stop=float(tr["stop"].mean()),
    )
    cells = {}
    for n in NS:
        acc = {k: [] for k in ("myopic_plain", "bellman_plain", "myopic_LW", "bellman_LW")}
        overq = {k: [] for k in acc}
        for rep in range(R):
            rr = np.random.default_rng([SEED, 11, int(round(rho * 1000)), n, rep])
            Xc = rr.standard_normal((n, 4)) @ L.T
            for est_name, Sh in (("plain", sample_cov(Xc)), ("LW", ledoit_wolf(Xc))):
                Mt = Model(Sh, Lgrid)
                for kind in ("myopic", "bellman"):
                    key = f"{kind}_{'plain' if est_name == 'plain' else 'LW'}"
                    qa, qb = decide(X, Mt, kind)
                    val = policy_values(X, Ttrue, Mt, qa, qb, Sh, S4)
                    acc[key].append(float(V_opt.mean() - val.mean()))
                    # over-acquisition at the policy level: queries when the true optimum does not
                    overq[key].append(float(np.mean((qa | qb) & ~(qa_o | qb_o))))
        for k in acc:
            v = np.array(acc[k])
            cells.setdefault(n, {})[k] = dict(
                regret_mean=float(v.mean()), regret_se_reps=float(v.std(ddof=1) / np.sqrt(R)),
                regret_sd_reps=float(v.std(ddof=1)),
                frac_reps_regret_pos=float(np.mean(v > 0)),
                P_draws_queries_when_opt_does_not=float(np.mean(overq[k])),
            )
    return truth, cells


results = {"config": dict(cost=COST, R=R, NDRAW=NDRAW, KZ=KZ, KZ1=KZ1, NGRID=NGRID,
                          lam=LAM, tau=TAU, sigma=0.01, rhos=RHOS, n_grid=NS,
                          note="regret = V_true(optimal) - V_true(policy), paired on common draws; "
                               "E over draws of the Rao-Blackwellized true utility minus costs")}
for rho in RHOS:
    truth, cells = run(rho)
    results[str(rho)] = dict(truth=truth, cells={str(k): v for k, v in cells.items()})
    print(f"rho={rho} elapsed {time.time() - t0:.1f}s | optimal V={truth['V_optimal']:.4e} "
          f"stop V={truth['V_stop']:.4e} always V={truth['V_always']:.4e} "
          f"bellman-myopic diff={truth['diff_bellman_minus_myopic']:.3e}"
          f"+-{truth['diff_bellman_minus_myopic_se_MC']:.1e} "
          f"frac draws differ={truth['frac_draws_myopic_differs']:.4f}", flush=True)
    for n in NS:
        c = cells[n]
        print(f"   n={n:<5} " + "  ".join(
            f"{k}: regret={c[k]['regret_mean']:.3e}(se {c[k]['regret_se_reps']:.1e}) "
            f"P(extra query)={c[k]['P_draws_queries_when_opt_does_not']:.4f}"
            for k in ("myopic_plain", "bellman_plain", "myopic_LW", "bellman_LW")), flush=True)

with open(RES_DIR + f"/pilot_03_b1_two_source_policies{TAG}.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"done in {time.time() - t0:.1f}s")
