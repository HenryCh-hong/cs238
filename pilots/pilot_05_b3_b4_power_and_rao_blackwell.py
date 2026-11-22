"""B3 (historical-replay statistical power) and B4 (Rao-Blackwellized calibration), single-source model.

Truth: Gaussian over (r, Y0, Y1), Corr(Y1, r) = rho (SPY-like scale, sigma = 1%).
Decisions: the oracle (true-model) one-step policy. Each day: the agent holds q0 (optimal given Y0),
observes Y1, and then picks q1 (optimal given Y0, Y1). Forced looking (all days) is used so that the
paired benefit is defined on every day, as in the replay design.

Per-day quantities (N = 1e6 common draws):
  d = u(q1, r) - u(q0, r)                  realized paired benefit (gross)
  h = E[d | Y0, Y1]                        Rao-Blackwellized benefit, computed from the true posterior
                                            h = G(m01, s01) - E[u(q0, r) | Y0, Y1]
  g = E[d | Y0]                            the predicted (one-step) gross VOI, exact quadrature
  x = g - c                                predicted net VOI, the regressor for calibration
Also: E[h | Y0] = E[d | Y0] = g (verified numerically).

B3 uses the raw d (what a historical replay can observe).
B4 compares Var(d) and the regression residual of d about g with the same quantities for h.
"""
import json
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pilot_common import (build_cov3, post_weights, delta_info, stop_value, EG_after, util_mom,
                          RHOS, COST, LAM, TAU, SEED, FIG_DIR, RES_DIR)

N = 1_000_000
Z95 = norm.ppf(0.975)
Z80 = norm.ppf(0.80)


def util_real(q, r):
    return q * r - LAM * q * q * r * r - TAU * np.abs(q)


def power_two_sided(mu, sd, n):
    delta = mu * np.sqrt(n) / sd
    return norm.cdf(delta - Z95) + norm.cdf(-delta - Z95)


out = {"config": dict(N=N, cost=COST, lam=LAM, tau=TAU, sigma=0.01, rhos=RHOS,
                      alpha=0.05, power=0.80, test_days=[1250, 2500])}
rows = []
for rho in RHOS:
    S = build_cov3(rho)
    L = np.linalg.cholesky(S)
    rng = np.random.default_rng([SEED, 55, int(round(rho * 1000))])
    X = rng.standard_normal((N, 3)) @ L.T
    r, y0, y1 = X[:, 0], X[:, 1], X[:, 2]
    w0, s0 = post_weights(S, [1])
    w01, s01 = post_weights(S, [1, 2])
    m0 = w0[0] * y0
    m01 = w01[0] * y0 + w01[1] * y1
    G0, q0 = stop_value(m0, s0)
    G1, q1 = stop_value(m01, s01)
    d = util_real(q1, r) - util_real(q0, r)
    h = G1 - util_mom(q0, m01, m01 * m01 + s01)
    # predicted gross VOI g(y0) = E[G(b')] - G(b): tabulated on a fine y0 grid, then interpolated
    D = delta_info(S, [1], 2)
    ygrid = np.linspace(-8, 8, 4001) * np.sqrt(S[1, 1])
    Gg = stop_value(ygrid * w0[0], s0)[0]
    vg = EG_after(ygrid * w0[0], s0, D) - Gg
    g = np.interp(y0, ygrid, vg)
    x = g - COST
    # checks
    chk_h_mean = float(h.mean() - g.mean())
    chk_d_mean = float(d.mean() - g.mean())
    # ---- summary statistics
    mu_d, sd_d = float(d.mean()), float(d.std(ddof=1))
    mu_exact = float(g.mean())   # exact E[d] (= E[g] by the tower property); MC mean has error ~ sd/sqrt(N)
    mu_h, sd_h = float(h.mean()), float(h.std(ddof=1))
    e_raw = d - g
    e_rb = h - g
    sd_e_raw, sd_e_rb = float(e_raw.std(ddof=1)), float(e_rb.std(ddof=1))
    sd_g = float(g.std(ddof=1))
    var_removed = float(np.var(d - h, ddof=1))
    rec = dict(
        rho=rho,
        mean_gross_VOI_exact_g=float(g.mean()), mean_gross_d=mu_d, mean_gross_h=mu_h,
        check_mean_d_minus_g=chk_d_mean, check_mean_h_minus_g=chk_h_mean,
        sd_d=sd_d, sd_h=sd_h, sd_g=sd_g,
        sd_residual_raw_d_about_g=sd_e_raw, sd_residual_rb_h_about_g=sd_e_rb,
        removed_noise_var_E_Var_d_given_Y=var_removed,
        VRF_mean_Var_d_over_Var_h=float(sd_d ** 2 / sd_h ** 2),
        VRF_residual=float(sd_e_raw ** 2 / sd_e_rb ** 2),
        frac_query_oracle=float(np.mean(x > 0)),
    )
    # B4: episodes for a tight calibration curve (95% CI half-width 0.10 on the slope, i.e. SE <= 0.051)
    se_target = 0.10 / (2 * 1.96)
    rec["N_episodes_slope_raw_for_halfwidth_0.1"] = float((sd_e_raw / (se_target * sd_g)) ** 2)
    rec["N_episodes_slope_RB_for_halfwidth_0.1"] = float((sd_e_rb / (se_target * sd_g)) ** 2)
    rec["N_episodes_mean_raw_for_rel_10pct"] = float((1.96 * sd_d / (0.10 * abs(mu_exact))) ** 2)
    rec["N_episodes_mean_RB_for_rel_10pct"] = float((1.96 * sd_h / (0.10 * abs(mu_exact))) ** 2)
    # B3(a): detect nonzero mean paired benefit (gross) from querying one forecast
    rec["mean_gross_d_MC_se"] = sd_d / np.sqrt(N)
    rec["B3a_N_days_detect_mean"] = float(((Z95 + Z80) * sd_d / abs(mu_exact)) ** 2)
    rec["B3a_power_1250"] = float(power_two_sided(mu_exact, sd_d, 1250))
    rec["B3a_power_2500"] = float(power_two_sided(mu_exact, sd_d, 2500))
    rec["B3a_net_mean_after_cost"] = float(mu_exact - COST)
    # B3(b): distinguish calibration slope 1 from 0 when regressing realized benefit on predicted VOI
    se_per_n = lambda n: sd_e_raw / (np.sqrt(n) * sd_g)
    rec["B3b_N_days_slope1_vs_0"] = float(((Z95 + Z80) * sd_e_raw / sd_g) ** 2)
    rec["B3b_power_slope1_vs_0_1250"] = float(norm.cdf(1 / se_per_n(1250) - Z95) + norm.cdf(-1 / se_per_n(1250) - Z95))
    rec["B3b_power_slope1_vs_0_2500"] = float(norm.cdf(1 / se_per_n(2500) - Z95) + norm.cdf(-1 / se_per_n(2500) - Z95))
    rec["B3b_SE_slope_1250"] = float(se_per_n(1250))
    rows.append(rec)
    print(f"rho={rho}: E[g]={g.mean():.4e}  mean d={mu_d:.4e}  mean h={mu_h:.4e}  sd_d={sd_d:.4e}  "
          f"sd_g={sd_g:.4e}  sd_resid_raw={sd_e_raw:.4e}  sd_resid_RB={sd_e_rb:.4e}  "
          f"VRF_mean={rec['VRF_mean_Var_d_over_Var_h']:.2f}  VRF_resid={rec['VRF_residual']:.2f}")
    print(f"       B3a N(detect mean)={rec['B3a_N_days_detect_mean']:.3e}  power@1250={rec['B3a_power_1250']:.3f} "
          f"power@2500={rec['B3a_power_2500']:.3f}   B3b N(slope 1 vs 0)={rec['B3b_N_days_slope1_vs_0']:.3e}  "
          f"power@1250={rec['B3b_power_slope1_vs_0_1250']:.3f}  power@2500={rec['B3b_power_slope1_vs_0_2500']:.3f}")
    print(f"       B4 N(slope halfwidth 0.1) raw={rec['N_episodes_slope_raw_for_halfwidth_0.1']:.3e}  "
          f"RB={rec['N_episodes_slope_RB_for_halfwidth_0.1']:.3e}")

out["by_rho"] = rows

# ---- B3(c): optimal vs myopic net utility difference, from the two-source run (pilot_03)
try:
    p3 = json.load(open(RES_DIR + f"/pilot_03_b1_two_source_policies_cost{COST:.0e}".replace("e-0", "e-").replace("+", "") + ".json"))
    c_rows = []
    for rho in RHOS:
        t = p3[str(rho)]["truth"]
        mu_c, sd_c = t["diff_bellman_minus_myopic"], t["diff_bellman_minus_myopic_sd_per_draw"]
        c_rows.append(dict(
            rho=rho, mean_diff_optimal_minus_myopic=mu_c, mean_diff_se_MC=t["diff_bellman_minus_myopic_se_MC"],
            sd_per_day=sd_c,
            N_days_detect=float(((Z95 + Z80) * sd_c / abs(mu_c)) ** 2) if mu_c != 0 else np.inf,
            power_1250=float(power_two_sided(mu_c, sd_c, 1250)) if sd_c > 0 else np.nan,
            power_2500=float(power_two_sided(mu_c, sd_c, 2500)) if sd_c > 0 else np.nan,
            frac_days_policies_differ=t["frac_draws_myopic_differs"]))
    out["B3c_optimal_vs_myopic"] = c_rows
    print("\nB3(c) optimal minus myopic (two-source, true model), per day:")
    for cr in c_rows:
        print(f"   rho={cr['rho']}: mean={cr['mean_diff_optimal_minus_myopic']:+.3e} (MC se {cr['mean_diff_se_MC']:.1e}) "
              f"sd/day={cr['sd_per_day']:.3e}  N_detect={cr['N_days_detect']:.3e}  "
              f"power@1250={cr['power_1250']:.3f}  power@2500={cr['power_2500']:.3f}  "
              f"days differing={cr['frac_days_policies_differ']:.4f}")
except FileNotFoundError:
    print("pilot_03 results not found; B3(c) skipped in this run")

# ---- illustrative autocorrelation factor (qualitative caveat in the report)
phis = [0.0, 0.05, 0.1, 0.2]
out["ar1_effective_sample_factor"] = {str(p): float((1 - p) / (1 + p)) for p in phis}

with open(RES_DIR + "/pilot_05_b3_b4_power_and_rao_blackwell.json", "w") as f:
    json.dump(out, f, indent=1)

# ---- figure
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
rh = [r_["rho"] for r_ in rows]
axes[0].loglog(rh, [r_["B3a_N_days_detect_mean"] for r_ in rows], "o-", label="B3a: detect mean paired benefit")
axes[0].loglog(rh, [r_["B3b_N_days_slope1_vs_0"] for r_ in rows], "s-", label="B3b: slope 1 vs 0 (raw benefit)")
axes[0].loglog(rh, [r_["N_episodes_slope_raw_for_halfwidth_0.1"] for r_ in rows], "^--", label="B4: slope CI +-0.10 (raw)")
axes[0].loglog(rh, [r_["N_episodes_slope_RB_for_halfwidth_0.1"] for r_ in rows], "v--", label="B4: slope CI +-0.10 (Rao-Blackwell)")
axes[0].axhline(1250, color="k", ls=":", lw=1)
axes[0].axhline(2500, color="k", ls="-.", lw=1)
axes[0].text(rh[0], 1300, "1,250 days", fontsize=8)
axes[0].text(rh[0], 2600, "2,500 days", fontsize=8)
axes[0].set_xlabel("Corr(Y1, r)")
axes[0].set_ylabel("days / episodes required (log scale)")
axes[0].set_title("Sample size needed (alpha 0.05, power 0.8 where applicable)")
axes[0].legend(fontsize=8)
axes[1].semilogx(rh, [r_["VRF_mean_Var_d_over_Var_h"] for r_ in rows], "o-", label="variance ratio Var(d)/Var(h) (mean)")
axes[1].semilogx(rh, [r_["VRF_residual"] for r_ in rows], "s-", label="residual ratio (slope regression)")
axes[1].set_xlabel("Corr(Y1, r)")
axes[1].set_ylabel("variance reduction factor")
axes[1].set_title("Rao-Blackwell variance reduction (oracle policy)")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR + "/fig_b3_b4_power_and_rao_blackwell.png", dpi=150)
print("saved")
