"""B2: model misspecification of the belief model (no estimation noise).

The agent uses a single Gaussian fitted to the TRUE second moments (infinite calibration data), so the
only error is the shape of the truth. The agent's decisions are its Gaussian posterior actions.

(a) Heavy tails: (r, Y0, Y1) multivariate Student-t with nu = 4, covariance matched to the Gaussian.
    The true expected value of the agent's policy is estimated by Monte Carlo from the truth with
    N = 2e6 draws (realized utility u(q, r) with the realized r is an unbiased estimate of the true
    expected utility). A Gaussian-truth control with the same draws procedure checks the MC code.
    Predicted gross VOI is the Gaussian one-step value E[G(b')] - G(b) at the realized Y0.
(b) Two-regime mixture (low / high volatility; the forecast is informative only in the high regime).
    Exact computation (deterministic quadrature over Y0, Y1 within each regime) with the TRUE mixture
    posterior; the agent's Gaussian belief uses the mixture covariance (moment matched).

Reported: error = true minus predicted, by Y0 state (bins), with Monte Carlo standard errors for (a).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pilot_common import (build_cov3, post_weights, delta_info, stop_value, EG_after, util_mom,
                          TrueOneStep, LAM, TAU, COST, SEED, FIG_DIR, RES_DIR)

NU = 4
N = 2_000_000
NBIN = 20
RHOS_T = [0.05, 0.2]
out = {"config": dict(nu=NU, N=N, NBIN=NBIN, rhos_t=RHOS_T, cost=COST, lam=LAM, tau=TAU)}


def util_real(q, r):
    return q * r - LAM * q * q * r * r - TAU * np.abs(q)


def draws(S, n, rng, nu=None):
    """Zero-mean Gaussian (nu=None) or multivariate Student-t with covariance S (matched)."""
    Z = rng.standard_normal((n, S.shape[0]))
    if nu is None:
        return Z @ np.linalg.cholesky(S).T
    Ls = np.linalg.cholesky(S * (nu - 2.0) / nu)     # Cov = Ls Ls^T * nu/(nu-2) = S
    W = rng.chisquare(nu, size=n)
    return (Z @ Ls.T) / np.sqrt(W / nu)[:, None]


def run_mc(S, nu, seed):
    rng = np.random.default_rng(seed)
    X = draws(S, N, rng, nu)
    r, y0, y1 = X[:, 0], X[:, 1], X[:, 2]
    w0, s0 = post_weights(S, [1])
    w01, s01 = post_weights(S, [1, 2])
    D0 = delta_info(S, [1], 2)
    grid = np.linspace(-6, 6, 2001) * np.sqrt(S[1, 1])
    G0g = stop_value(grid * w0[0], s0)[0]
    pred_grid = EG_after(grid * w0[0], s0, D0) - G0g                  # Gaussian predicted gross VOI
    pred = np.interp(y0, grid, pred_grid)
    q0 = stop_value(y0 * w0[0], s0)[1]                                  # agent's Gaussian actions
    q1 = stop_value(w01[0] * y0 + w01[1] * y1, s01)[1]
    d = util_real(q1, r) - util_real(q0, r)                             # realized gross benefit (forced look)
    edges = np.quantile(y0, np.linspace(0, 1, NBIN + 1))
    idx = np.clip(np.searchsorted(edges, y0, side="right") - 1, 0, NBIN - 1)
    rows = []
    for b in range(NBIN):
        mk = idx == b
        nb = mk.sum()
        dd = d[mk]
        rows.append(dict(bin=b, y0_mid=float(np.median(y0[mk])), n=int(nb),
                         pred_gross=float(pred[mk].mean()),
                         true_gross=float(dd.mean()), true_se=float(dd.std(ddof=1) / np.sqrt(nb)),
                         err=float(dd.mean() - pred[mk].mean()),
                         err_se=float(dd.std(ddof=1) / np.sqrt(nb)),
                         err_over_pred=float((dd.mean() - pred[mk].mean()) / max(abs(pred[mk].mean()), 1e-300))))
    overall = dict(pred_mean=float(pred.mean()), true_mean=float(d.mean()),
                   true_se=float(d.std(ddof=1) / np.sqrt(N)), err=float(d.mean() - pred.mean()),
                   err_se=float(d.std(ddof=1) / np.sqrt(N)))
    return rows, overall


print("== (a) Student-t (nu=4) truth vs Gaussian belief, Monte Carlo from the truth (N=%d)" % N)
out["a_student_t"] = {}
out["a_gauss_control"] = {}
for rho in RHOS_T:
    S = build_cov3(rho)
    rows_t, ov_t = run_mc(S, NU, SEED + int(rho * 1000))
    rows_g, ov_g = run_mc(S, None, SEED + 7 + int(rho * 1000))
    out["a_student_t"][str(rho)] = dict(overall=ov_t, bins=rows_t)
    out["a_gauss_control"][str(rho)] = dict(overall=ov_g, bins=rows_g)
    print(f"\nrho={rho}  Gaussian-truth control: pred mean={ov_g['pred_mean']:.4e} true mean={ov_g['true_mean']:.4e} "
          f"+- {ov_g['true_se']:.1e}  err={ov_g['err']:+.2e} (z={ov_g['err'] / ov_g['err_se']:+.2f})")
    print(f"rho={rho}  Student-t(nu=4) truth: pred mean={ov_t['pred_mean']:.4e} true mean={ov_t['true_mean']:.4e} "
          f"+- {ov_t['true_se']:.1e}  err={ov_t['err']:+.2e} (+-{ov_t['err_se']:.1e})")
    print("   bin  y0_mid      n     pred_gross     true_gross (se)        err (se)     err/pred")
    for rr in rows_t:
        print(f"   {rr['bin']:>3}  {rr['y0_mid']:+.3e}  {rr['n']:>6}  {rr['pred_gross']:+.4e}  "
              f"{rr['true_gross']:+.4e} ({rr['true_se']:.1e})  {rr['err']:+.2e} ({rr['err_se']:.1e})  "
              f"{rr['err_over_pred']:+.3f}")

# ----------------------------------------------------------------- (b) regime mixture, exact
print("\n== (b) two-regime mixture (exact quadrature) vs moment-matched Gaussian belief")
sig_L, sig_H = 0.006, 0.014
S_L = build_cov3(0.0, sig=sig_L)          # low volatility: forecast uninformative
S_H = build_cov3(0.3, sig=sig_H)          # high volatility: Corr(Y1,r)=0.3
probs = [0.5, 0.5]
S_mix = probs[0] * S_L + probs[1] * S_H   # moment-matched Gaussian (agent's belief)
ev = TrueOneStep([S_L, S_H], probs, n0=1201, K1=800)
res = ev.evaluate(S_mix)
w = res["wq"]
pi_H = res["pi_H"]
err_pred = res["pred_gross"] - res["oracle_gross"]        # misspecified prediction vs true-optimal value
err_act = res["oracle_gross"] - res["agent_gross"]        # cost of acting on the Gaussian posterior
err_total = res["pred_gross"] - res["agent_gross"]        # predicted vs true value of the agent's actions
def wmean(x, mask):
    return float(np.sum(w[mask] * x[mask]) / max(np.sum(w[mask]), 1e-300))
bins = [(0.0, 0.5), (0.5, 0.95), (0.95, 1.0001)]  # P(high regime | Y0) >= 0.3 by construction
sum_b = []
for lo, hi in bins:
    mk = (pi_H >= lo) & (pi_H < hi)
    sum_b.append(dict(pi_H_range=[lo, hi], mass=float(np.sum(w[mk])),
                      mean_err_pred_minus_oracle=wmean(err_pred, mk),
                      mean_err_oracle_minus_agent=wmean(err_act, mk),
                      mean_err_pred_minus_agent=wmean(err_total, mk),
                      mean_oracle_gross=wmean(res["oracle_gross"], mk),
                      mean_pred_gross=wmean(res["pred_gross"], mk)))
k_max = int(np.argmax(np.abs(err_pred) * (w > 1e-12)))
k_min_agent = int(np.argmin(np.where(w > 1e-12, res["agent_gross"], np.inf)))
mix = dict(
    weights=probs, sig_L=sig_L, sig_H=sig_H,
    corr_Y1_r_mixture=float(S_mix[0, 2] / np.sqrt(S_mix[0, 0] * S_mix[2, 2])),
    E_oracle_gross=float(np.sum(w * res["oracle_gross"])),
    E_pred_gross=float(np.sum(w * res["pred_gross"])),
    E_agent_gross=float(np.sum(w * res["agent_gross"])),
    mean_err_pred_minus_oracle=float(np.sum(w * err_pred)),
    mean_abs_err_pred_minus_oracle=float(np.sum(w * np.abs(err_pred))),
    max_abs_err_pred_minus_oracle=float(np.abs(err_pred[k_max])),
    location_max_err=dict(y0=float(ev.y0[k_max]), pi_H=float(pi_H[k_max]), pred=float(res["pred_gross"][k_max]),
                          oracle=float(res["oracle_gross"][k_max]), err=float(err_pred[k_max])),
    mean_err_pred_minus_agent=float(np.sum(w * err_total)),
    E_err_oracle_minus_agent=float(np.sum(w * err_act)),
    P_query_pred=float(np.sum(w * res["query"])),
    P_query_oracle=float(np.sum(w * ev.oracle_query)),
    min_agent_gross=float(res["agent_gross"][k_min_agent]),
    location_min_agent_gross=dict(y0=float(ev.y0[k_min_agent]), pi_H=float(pi_H[k_min_agent])),
    bins_by_pi_H=sum_b,
)
# decompose: how much of the error comes from the posterior variance vs the posterior mean
# (compare the true mixture moments with the Gaussian belief's moments at the same y0)
out["b_mixture"] = mix
print(json.dumps(mix, indent=1))

# ----------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
ax.plot(ev.y0 / np.sqrt(S_mix[1, 1]), res["oracle_gross"], label="true (mixture) one-step VOI", color="k")
ax.plot(ev.y0 / np.sqrt(S_mix[1, 1]), res["pred_gross"], label="Gaussian predicted VOI", ls="--", color="#d95f02")
ax.plot(ev.y0 / np.sqrt(S_mix[1, 1]), res["agent_gross"], label="true value of Gaussian agent's actions", ls=":", color="#7570b3")
ax.set_xlabel("baseline forecast Y0 / sd(Y0)")
ax.set_ylabel("gross VOI (utility units)")
ax.set_title("Mixture truth: predicted vs true value of looking")
ax.legend(fontsize=8)
ax2 = axes[1]
ax2.plot(ev.y0 / np.sqrt(S_mix[1, 1]), err_pred, color="k", label="pred - true-optimal")
ax2.plot(ev.y0 / np.sqrt(S_mix[1, 1]), -err_act, color="#7570b3", ls=":", label="-(optimal - agent actions)")
ax3 = ax2.twinx()
ax3.plot(ev.y0 / np.sqrt(S_mix[1, 1]), pi_H, color="grey", alpha=0.6, lw=1, label="P(high regime | Y0)")
ax3.set_ylabel("P(high regime | Y0)", color="grey")
ax2.set_xlabel("baseline forecast Y0 / sd(Y0)")
ax2.set_ylabel("error (utility units)")
ax2.set_title("Error by state")
ax2.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(FIG_DIR + "/fig_b2_misspecification.png", dpi=150)

with open(RES_DIR + "/pilot_04_b2_misspecification.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved")
