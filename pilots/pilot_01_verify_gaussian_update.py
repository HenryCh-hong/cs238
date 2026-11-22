"""Check 1: verify the linear-Gaussian preposterior claims and the quadrature used everywhere.

Claims checked
  (a) posterior variance after observing Y_j is data independent: s'^2 = s^2 - Delta_j
  (b) preposterior law of the posterior mean increment: m' - m ~ N(0, Delta_j)
  (c) order independence: s^2 - Delta_1 - Delta_2|1 = s^2 - Delta_2 - Delta_1|2
  (d) Gauss-Hermite quadrature for E[G(m', v - Delta)] vs a fine midpoint-quantile reference
  (e) sanity: the one-step evaluator with the true covariance reproduces its own predictions.
"""
import json
import numpy as np
from scipy.stats import norm
from pilot_common import (build_cov3, build_cov4, post_weights, delta_info, stop_value,
                          EG_after, gh, qmid, TrueOneStep, SEED, RES_DIR, COST)

out = {}
rng = np.random.default_rng(SEED)

# ---- (a),(b) Monte Carlo on the joint Gaussian (3-dim: r, Y0, Y1), rho = 0.1
S3 = build_cov3(0.1)
L = np.linalg.cholesky(S3)
N = 2_000_000
X = rng.standard_normal((N, 3)) @ L.T
w0, s20 = post_weights(S3, [1])
w01, s201 = post_weights(S3, [1, 2])
D1_0 = delta_info(S3, [1], 2)               # Delta for Y1 given Y0
m0 = X[:, 1] * w0[0]
m01 = X[:, 1] * w01[0] + X[:, 2] * w01[1]
inc = m01 - m0
out["a_posterior_var_after_Y0_closed"] = float(s20)
out["a_posterior_var_after_Y0_Y1_closed"] = float(s201)
out["a_s2_minus_Delta_matches_posterior_var"] = bool(np.isclose(s20 - D1_0, s201, rtol=1e-12, atol=0))
out["b_Delta_closed"] = float(D1_0)
out["b_Var_increment_MC"] = float(inc.var())
out["b_Var_increment_MC_se_rel"] = float(np.sqrt(2.0 / N))
out["b_mean_increment_MC"] = float(inc.mean())
# conditional on Y0 = y0 (fixed), increment should be N(0, Delta): check the conditional law
y0_fix = 0.5 * np.sqrt(S3[1, 1])
cond_var_Y1 = S3[2, 2] - S3[2, 1] ** 2 / S3[1, 1]
Y1c = (S3[2, 1] / S3[1, 1] * y0_fix) + np.sqrt(cond_var_Y1) * rng.standard_normal(N)
m01c = y0_fix * w01[0] + Y1c * w01[1]
m0c = y0_fix * w0[0]
out["b_cond_var_increment_given_y0_MC"] = float((m01c - m0c).var())
out["b_cond_mean_increment_given_y0_MC"] = float((m01c - m0c).mean())
# residual variance of r given (Y0, Y1) by simulation (should equal s201, data independent)
r_res = X[:, 0] - (X[:, 1:] @ w01)
out["a_residual_var_r_given_Y0Y1_MC"] = float(r_res.var())
out["a_residual_var_closed"] = float(s201)

# ---- (c) order independence with the 4-dim covariance
S4 = build_cov4(0.2, 0.1)
s20_4 = post_weights(S4, [1])[1]
D_a = delta_info(S4, [1], 2)
D_b_after_a = delta_info(S4, [1, 2], 3)
D_b = delta_info(S4, [1], 3)
D_a_after_b = delta_info(S4, [1, 3], 2)
out["c_total_reduction_order_ab"] = float(D_a + D_b_after_a)
out["c_total_reduction_order_ba"] = float(D_b + D_a_after_b)
out["c_total_reduction_abs_diff"] = float(abs((D_a + D_b_after_a) - (D_b + D_a_after_b)))
out["c_posterior_var_0123_closed"] = float(post_weights(S4, [1, 2, 3])[1])
out["c_posterior_var_0123_from_order_ab"] = float(s20_4 - D_a - D_b_after_a)

# ---- (d) quadrature check of E[G(m', v - Delta)]
rows = []
for rho in (0.02, 0.1, 0.4):
    S = build_cov3(rho)
    Dl = delta_info(S, [1], 2)
    _, v0 = post_weights(S, [1])
    sd_m0 = np.sqrt(S[0, 1] ** 2 / S[1, 1])
    for mm in (0.0, 0.5 * sd_m0, 1.5 * sd_m0):
        ref_z = qmid(20000)                              # fine midpoint reference
        G_ref = stop_value(mm + np.sqrt(Dl) * ref_z, v0 - Dl)[0].mean()
        row = dict(rho=rho, m=float(mm), Delta=float(Dl))
        for K in (8, 16, 32, 64, 128):
            z, w = gh(K)
            row[f"GH{K}_minus_ref"] = float(EG_after(mm, v0, Dl, z, w) - G_ref)
        row["ref_EG"] = float(G_ref)
        rows.append(row)
out["d_quadrature_rows"] = rows

# ---- (e) sanity of the exact one-step evaluator
ev = TrueOneStep([S3], [1.0], n0=1201, K1=800)
res_true = ev.evaluate(S3)
s_true = ev.summaries(res_true)
out["e_true_model_pred_minus_oracle_max_abs"] = float(np.max(np.abs(res_true["pred_gross"] - res_true["oracle_gross"])))
out["e_true_model_agent_minus_oracle_max_abs"] = float(np.max(np.abs(res_true["agent_gross"] - res_true["oracle_gross"])))
out["e_true_model_min_oracle_gross"] = float(s_true["min_oracle_gross"])
out["e_weight_sum"] = float(ev.wq.sum())
ev_fine = TrueOneStep([S3], [1.0], n0=1201, K1=1600)
out["e_K1_400_vs_1600_oracle_gross_max_abs"] = float(np.max(np.abs(ev_fine.oracle_gross - ev.oracle_gross)))

with open(RES_DIR + "/pilot_01_verify_gaussian_update.json", "w") as f:
    json.dump(out, f, indent=2)
for k, v in out.items():
    if k != "d_quadrature_rows":
        print(f"{k}: {v}")
print("quadrature rows (GH nodes minus fine reference, in utility units):")
for r in rows:
    print({k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in r.items()})
