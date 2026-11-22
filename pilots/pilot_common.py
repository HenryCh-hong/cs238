"""Shared Gaussian belief-state utilities for the VOI pilot checks (B1-B4).

Conventions
-----------
* Variables are ordered (r, Y0, Y1, Y2). Index 0 is the next-day return r, 1 the free
  baseline forecast Y0 (always observed), 2 and 3 the optional sources.
* Utility u(q, r) = q r - lam q^2 r^2 - tau |q - qp|. Given a posterior with mean m and
  variance v, E[u | q] = q m - lam q^2 (m^2 + v) - tau |q - qp|, so the stop value
  G(m, v) = max_q E[u | q] depends only on the first two posterior moments.
* Linear-Gaussian conditioning: after observing a set O of forecasts, the posterior mean is
  m = y_O . w_O and the posterior variance is data independent. The preposterior increment
  for one more source j is m' - m ~ N(0, Delta_j) with Delta_j = Cov(r, Y_j | O)^2 / Var(Y_j | O).
"""
import os
import numpy as np
from scipy.stats import norm
from numpy.polynomial.hermite_e import hermegauss

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = ANALYSIS_DIR + "/figures"
RES_DIR = ANALYSIS_DIR + "/results"

SEED = 20260126
SIG = 0.01          # daily return sd (1% per day, SPY-like)
LAM = 1.0           # risk penalty, return units (lam * r^2 ~ 1e-4 at sigma = 1%)
TAU = 2.0e-4        # transaction cost per unit |dq| (2 bp)
QPREV = 0.0         # previous position
COST = 1.0e-4       # query cost in utility units (1 bp)
QS = np.array([-1.0, 0.0, 1.0])
RHOS = [0.02, 0.05, 0.1, 0.2, 0.4]
NS = [60, 250, 1000, 5000]
A0 = 0.05           # Corr(Y0, r): baseline forecast weakly informative
C01 = 0.6           # Corr(Y1, Y0): sources correlated with the baseline
C12 = 0.4           # Corr(Y1, Y2)
C02 = 0.6           # Corr(Y2, Y0)


def gh(K):
    """Probabilists' Gauss-Hermite nodes and weights normalized to sum to one."""
    x, w = hermegauss(K)
    return x, w / w.sum()


def qmid(K):
    """Midpoint quantile nodes of N(0,1): equal-weight deterministic quadrature."""
    return norm.ppf((np.arange(1, K + 1) - 0.5) / K)


# Default nodes for E over a Gaussian increment. The stop value has kinks, so Gauss-Hermite
# converges slowly (errors ~1e-5 at 32 nodes); equal-weight quantile midpoints converge like
# O(1/K) with abs. error ~1.5e-7 at K=1024 (see pilot_01 output).
ZK_DEFAULT = 1024
ZN = qmid(ZK_DEFAULT)
ZW = np.full(ZK_DEFAULT, 1.0 / ZK_DEFAULT)


def build_cov4(rho1, rho2=None, sig=SIG):
    """Covariance over (r, Y0, Y1, Y2) with Var(Y_j) = Var(r) = sig^2 (forecasts on return scale)."""
    if rho2 is None:
        rho2 = rho1
    R = np.array([[1.0, A0, rho1, rho2],
                  [A0, 1.0, C01, C02],
                  [rho1, C01, 1.0, C12],
                  [rho2, C02, C12, 1.0]])
    return sig ** 2 * R


def build_cov3(rho1, sig=SIG):
    """Covariance over (r, Y0, Y1)."""
    return build_cov4(rho1, rho1, sig)[:3, :3]


def is_pd(S):
    return bool(np.all(np.linalg.eigvalsh(S) > 0))


# ----------------------------------------------------------------------------- utilities
def util_mom(q, M, R2, lam=LAM, tau=TAU, qp=QPREV):
    """E[u(q,r)] given first moment M = E[r] and second moment R2 = E[r^2]."""
    return q * M - lam * q * q * R2 - tau * np.abs(q - qp)


def stop_value(m, v, lam=LAM, tau=TAU, qp=QPREV):
    """G(m, v) and maximizing q, for posterior N(m, v) with v the posterior variance."""
    m = np.asarray(m, dtype=float)
    R2 = m * m + v
    V = np.stack([util_mom(q, m, R2, lam, tau, qp) for q in (-1.0, 0.0, 1.0)], axis=-1)
    return V.max(-1), QS[V.argmax(-1)]


def EG_after(m, v, Delta, z=None, w=None):
    """E[G(m', v - Delta)] with m' ~ N(m, Delta): expected stop value after one more source."""
    if z is None:
        z, w = ZN, ZW
    mp = np.asarray(m, dtype=float)[..., None] + np.sqrt(Delta) * z
    G, _ = stop_value(mp, v - Delta)
    return G @ w


def voi_gross(m, v, Delta):
    """One-step gross value of information E[G(b')] - G(b)."""
    G0, _ = stop_value(m, v)
    return EG_after(m, v, Delta) - G0


# ----------------------------------------------------------------------------- conditioning
def post_weights(S, O):
    """Posterior r | y_O is N(y_O . w, s2): returns (w, s2)."""
    O = list(O)
    A = S[np.ix_(O, O)]
    b = S[0, O]
    w = np.linalg.solve(A, b)
    return w, S[0, 0] - b @ w


def delta_info(S, O, j):
    """Delta_j = Cov(r, Y_j | O)^2 / Var(Y_j | O); data independent for Gaussians."""
    O = list(O)
    A = S[np.ix_(O, O)]
    bj = S[j, O]
    cj = S[0, j] - S[0, O] @ np.linalg.solve(A, bj)
    vj = S[j, j] - bj @ np.linalg.solve(A, bj)
    return cj * cj / vj


def ledoit_wolf(X):
    """Ledoit-Wolf (2004) shrinkage covariance, shrinking the correlation matrix toward I.

    Shrinkage toward the scaled identity drives off-diagonal (cross-) correlations toward zero.
    """
    n, p = X.shape
    Xc = X - X.mean(0)
    sd = Xc.std(0, ddof=1)
    Z = Xc / sd
    S = Z.T @ Z / n
    mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2)
    outer = np.einsum("ki,kj->kij", Z, Z)
    b2 = np.sum((outer - S[None]) ** 2) / n ** 2
    b2 = min(b2, d2)
    Sig = (b2 / d2) * mu * np.eye(p) + ((d2 - b2) / d2) * S
    return Sig * np.outer(sd, sd)


def sample_cov(X):
    Xc = X - X.mean(0)
    return Xc.T @ Xc / (X.shape[0] - 1)


# ----------------------------------------------------------------------------- exact one-step
class TrueOneStep:
    """Exact one-step (one optional source Y1) evaluator for a Gaussian-regime truth.

    regimes: list of 3x3 covariances over (r, Y0, Y1); probs: regime probabilities.
    One regime = a plain Gaussian truth. Two regimes = a latent-regime mixture truth.
    Expectations over Y0 use a fine uniform grid weighted by the exact mixture density.
    Expectations over Y1 | Y0, regime use midpoint quantile nodes (deterministic).
    """

    def __init__(self, regimes, probs, n0=1201, K1=800, halfwidth=7.0, cost=COST):
        self.S = [np.asarray(S, float) for S in regimes]
        self.p = np.asarray(probs, float)
        self.K = len(self.S)
        self.cost = cost
        smax = max(np.sqrt(S[1, 1]) for S in self.S)
        self.y0 = np.linspace(-halfwidth * smax, halfwidth * smax, n0)
        dy = self.y0[1] - self.y0[0]
        dens = np.stack([pk * norm.pdf(self.y0, 0.0, np.sqrt(S[1, 1]))
                         for S, pk in zip(self.S, self.p)])
        pmix = dens.sum(0)
        self.wq = pmix * dy                 # weights for the Y0 grid
        self.pi = dens / pmix               # P(regime k | Y0 = y0), shape (K, n0)
        zq = qmid(K1)
        # per-regime conditional structure
        self.w0, self.s20, self.w01, self.s201, self.mu1, self.v1 = [], [], [], [], [], []
        for S in self.S:
            w0, s20 = post_weights(S, [1])
            w01, s201 = post_weights(S, [1, 2])
            self.w0.append(w0[0])
            self.s20.append(s20)
            self.w01.append(w01)
            self.s201.append(s201)
            self.mu1.append(S[2, 1] / S[1, 1] * self.y0)
            self.v1.append(S[2, 2] - S[2, 1] ** 2 / S[1, 1])
        # true posterior moments after Y0 only (mixture over regimes)
        M0 = np.zeros_like(self.y0)
        R20 = np.zeros_like(self.y0)
        for k in range(self.K):
            m0k = self.w0[k] * self.y0
            M0 += self.pi[k] * m0k
            R20 += self.pi[k] * (m0k ** 2 + self.s20[k])
        self.M0, self.R20 = M0, R20
        self.Gtrue0, self.q_true0 = None, None
        self.Gtrue0 = np.max([util_mom(q, M0, R20) for q in (-1.0, 0.0, 1.0)], axis=0)
        # regime-branch quantities after Y1
        self.y1 = []
        self.Mpost = []
        self.R2post = []
        self.Gtrue_after_k = []
        self.U_true_nodes = []
        for k in range(self.K):
            y1 = self.mu1[k][:, None] + np.sqrt(self.v1[k]) * zq[None, :]
            rho = np.zeros((self.K,) + y1.shape)
            for j in range(self.K):
                rho[j] = self.pi[j][:, None] * norm.pdf(
                    y1, self.mu1[j][:, None], np.sqrt(self.v1[j]))
            rho /= rho.sum(0)[None]
            Mp = np.zeros(y1.shape)
            R2p = np.zeros(y1.shape)
            for j in range(self.K):
                mj = self.w01[j][0] * self.y0[:, None] + self.w01[j][1] * y1
                Mp += rho[j] * mj
                R2p += rho[j] * (mj ** 2 + self.s201[j])
            self.y1.append(y1)
            self.Mpost.append(Mp)
            self.R2post.append(R2p)
            self.Gtrue_after_k.append(
                np.max([util_mom(q, Mp, R2p) for q in (-1.0, 0.0, 1.0)], axis=0))
        # oracle (true-model) value of looking, per y0
        self.oracle_after = np.zeros_like(self.y0)
        for k in range(self.K):
            self.oracle_after += self.pi[k] * self.Gtrue_after_k[k].mean(1)
        self.oracle_gross = self.oracle_after - self.Gtrue0
        self.oracle_net = self.oracle_gross - cost
        self.oracle_query = self.oracle_net > 0

    def evaluate(self, SA):
        """Agent with a single Gaussian belief fitted to covariance SA (3x3) acts on it.

        Returns per-y0 arrays and scalar summaries. All true values use the true mixture.
        """
        SA = np.asarray(SA, float)
        w0A, s20A = post_weights(SA, [1])
        wA, s2A = post_weights(SA, [1, 2])
        DeltaA = delta_info(SA, [1], 2)
        m0A = w0A[0] * self.y0
        G0A, qA0 = stop_value(m0A, s20A)
        pred_gross = EG_after(m0A, s20A, DeltaA) - G0A
        pred_net = pred_gross - self.cost
        query = pred_net > 0
        # agent's true value of its stop action
        U0 = util_mom(qA0, self.M0, self.R20)
        # agent's true value of acting after looking (expectation over Y1 | Y0, regimes)
        agent_after = np.zeros_like(self.y0)
        for k in range(self.K):
            mA = wA[0] * self.y0[:, None] + wA[1] * self.y1[k]
            _, qAk = stop_value(mA, s2A)
            U = util_mom(qAk, self.Mpost[k], self.R2post[k])
            agent_after += self.pi[k] * U.mean(1)
        agent_gross = agent_after - U0
        V_star = np.maximum(self.Gtrue0, self.oracle_after - self.cost)
        V_agent = np.where(query, agent_after - self.cost, U0)
        regret = V_star - V_agent
        return dict(
            y0=self.y0, wq=self.wq, pred_gross=pred_gross, pred_net=pred_net,
            oracle_gross=self.oracle_gross, oracle_net=self.oracle_net,
            agent_gross=agent_gross, query=query, regret=regret,
            pi_H=self.pi[-1], G0A=G0A, qA0=qA0,
            baseline_loss=self.Gtrue0 - U0,   # true loss of the agent's stop action (no VOI involved)
        )

    def summaries(self, out):
        w = out["wq"]
        q = out["query"]
        net_p = out["pred_net"]
        net_o = out["oracle_net"]
        g_ag = out["agent_gross"]
        Pq = float(np.sum(w * q))
        s = dict(
            mean_pred_net=float(np.sum(w * net_p)),
            mean_oracle_net=float(np.sum(w * net_o)),
            bias_pred_minus_oracle=float(np.sum(w * (net_p - net_o))),
            P_query=Pq,
            P_overacquire=float(np.sum(w * (q & (net_o < 0)))),
            P_underacquire=float(np.sum(w * ((~q) & (net_o > 0)))),
            P_overacq_given_query=float(np.sum(w * (q & (net_o < 0))) / max(Pq, 1e-300)),
            E_regret=float(np.sum(w * out["regret"])),
            E_baseline_loss=float(np.sum(w * out["baseline_loss"])),
            E_oracle_gross=float(np.sum(w * out["oracle_gross"])),
            min_oracle_gross=float(np.min(out["oracle_gross"][w > 1e-12])),
            E_agent_gross_forced=float(np.sum(w * g_ag)),
            min_agent_gross_forced=float(np.min(g_ag[w > 1e-12])),
            P_agent_gross_neg_forced=float(np.sum(w * (g_ag < 0))),
            E_agent_gross_given_query=float(np.sum(w * q * g_ag) / max(Pq, 1e-300)),
            P_agent_gross_neg_given_query=float(np.sum(w * (q & (g_ag < 0))) / max(Pq, 1e-300)),
        )
        return s
