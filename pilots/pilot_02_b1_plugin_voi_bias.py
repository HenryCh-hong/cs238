"""B1 (one-step): plug-in VOI bias from estimated covariance, over-acquisition, and LW remedy.

Setup
  * True model: Gaussian over (r, Y0, Y1) with Corr(Y1, r) = rho in {0.02,...,0.4}.
  * Agent observes the free baseline Y0, then estimates the 3x3 covariance from n i.i.d.
    calibration draws from the TRUE model (n in {60, 250, 1000, 5000}).
  * Agent plans with the estimate: predicted one-step net VOI for Y1, and its stop action.
  * Truth values are computed exactly (deterministic quadrature) from the true model:
    oracle VOI (true-model optimal behavior) and the true value of the agent's own actions.
  * Remedy: Ledoit-Wolf covariance (shrinks cross-correlations toward zero).

Each (rho, n) cell is repeated R times with fresh calibration samples (seeded, reproducible).
Standard errors are across replications (the truth is deterministic, so no truth MC error).
"""
import json
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pilot_common import (build_cov3, sample_cov, ledoit_wolf, TrueOneStep, RHOS, NS,
                          SEED, LAM, TAU, FIG_DIR, RES_DIR)

# query cost in utility units; default 1e-4 (1 bp). A second run uses 1e-5 so that the weak-signal
# cells (rho <= 0.05) are not trivially never-worth-acquiring.
COST = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0e-4
TAG = "" if COST == 1.0e-4 else f"_cost{COST:.0e}".replace("e-0", "e-").replace("+", "")
R = 200
N_MAX = 5000
t0 = time.time()
results = {"config": dict(R=R, n_grid=NS, rhos=RHOS, cost=COST, lam=LAM, tau=TAU,
                          sigma=0.01, A0=0.05, C01=0.6, quadrature="deterministic: y0 uniform grid "
                          "(n0=801, +-7 sd, exact mixture weights); Y1 midpoint quantiles K1=800; "
                          "increment expectation midpoint quantiles K=1024")}
cells = {}
truth_info = {}
for rho in RHOS:
    S3 = build_cov3(rho)
    ev = TrueOneStep([S3], [1.0], n0=801, K1=800, cost=COST)
    truth_info[rho] = dict(
        E_oracle_gross=float(np.sum(ev.wq * ev.oracle_gross)),
        E_oracle_net=float(np.sum(ev.wq * ev.oracle_net)),
        P_oracle_query=float(np.sum(ev.wq * ev.oracle_query)),
        sd_pred_state_gross=float(np.sqrt(np.sum(ev.wq * (ev.oracle_gross - np.sum(ev.wq * ev.oracle_gross)) ** 2))),
        max_oracle_gross=float(np.max(ev.oracle_gross)),
    )
    L = np.linalg.cholesky(S3)
    for n in NS:
        acc = {"plain": [], "LW": []}
        for rep in range(R):
            rng = np.random.default_rng([SEED, int(round(rho * 1000)), n, rep])
            X = rng.standard_normal((n, 3)) @ L.T
            for name, Sh in (("plain", sample_cov(X)), ("LW", ledoit_wolf(X))):
                out = ev.evaluate(Sh)
                acc[name].append(ev.summaries(out))
        cells[(rho, n)] = {}
        for name in acc:
            keys = acc[name][0].keys()
            d = {}
            for k in keys:
                v = np.array([a[k] for a in acc[name]], float)
                d[k] = float(v.mean())
                d[k + "_se"] = float(v.std(ddof=1) / np.sqrt(R))
                d[k + "_sd"] = float(v.std(ddof=1))
            d["frac_reps_gross_neg_forced"] = float(np.mean([a["min_agent_gross_forced"] < 0 for a in acc[name]]))
            cells[(rho, n)][name] = d
    print(f"rho={rho} done, elapsed {time.time() - t0:.1f}s", flush=True)

results["truth_by_rho"] = {str(k): v for k, v in truth_info.items()}
results["cells"] = {f"rho={k[0]},n={k[1]}": v for k, v in cells.items()}
with open(RES_DIR + f"/pilot_02_b1_plugin_voi_bias{TAG}.json", "w") as f:
    json.dump(results, f, indent=1)

# ----------------------------------------------------------------- printed tables
print("\nTRUE-MODEL (oracle) single-source VOI, per rho (utility units; cost = %.1e):" % COST)
for rho in RHOS:
    t = truth_info[rho]
    print(f"  rho={rho:<5} E[oracle gross VOI]={t['E_oracle_gross']:.3e}  E[oracle net]={t['E_oracle_net']:.3e}"
          f"  P(oracle queries)={t['P_oracle_query']:.3f}  max gross={t['max_oracle_gross']:.3e}")
print("\nPLUG-IN (plain sample covariance) and LW-shrunk results, mean over R=%d replications (+-SE):" % R)
hdr = ("rho    n     bias(pred-orc net)   P(overacq)    P(underacq)  P(query)   E[regret]        "
       "E[agent gross | query]  P(gross<0|query)  P(min gross<0)")
print(hdr)
for rho in RHOS:
    for n in NS:
        for name in ("plain", "LW"):
            c = cells[(rho, n)][name]
            print(f"{rho:<6} {n:<5} {c['bias_pred_minus_oracle']:+.3e}({c['bias_pred_minus_oracle_se']:.1e})  "
                  f"{c['P_overacquire']:.4f}({c['P_overacquire_se']:.4f})  {c['P_underacquire']:.4f}  "
                  f"{c['P_query']:.4f}  {c['E_regret']:.3e}({c['E_regret_se']:.1e})  "
                  f"{c['E_agent_gross_given_query']:+.3e}  {c['P_agent_gross_neg_given_query']:.4f}  "
                  f"{c['frac_reps_gross_neg_forced']:.3f}  [{name}]")

# ----------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
colors = {0.02: "#1b9e77", 0.05: "#d95f02", 0.1: "#7570b3", 0.2: "#e7298a", 0.4: "#66a61e"}
for rho in RHOS:
    b = [cells[(rho, n)]["plain"]["bias_pred_minus_oracle"] for n in NS]
    be = [cells[(rho, n)]["plain"]["bias_pred_minus_oracle_se"] for n in NS]
    axes[0].errorbar(NS, b, yerr=be, marker="o", color=colors[rho], label=f"rho={rho}")
    po = [cells[(rho, n)]["plain"]["P_overacquire"] for n in NS]
    pl = [cells[(rho, n)]["LW"]["P_overacquire"] for n in NS]
    axes[1].plot(NS, po, marker="o", color=colors[rho], label=f"plain, rho={rho}")
    axes[1].plot(NS, pl, marker="x", ls="--", color=colors[rho])
    rg = [cells[(rho, n)]["plain"]["E_regret"] for n in NS]
    rl = [cells[(rho, n)]["LW"]["E_regret"] for n in NS]
    axes[2].plot(NS, rg, marker="o", color=colors[rho])
    axes[2].plot(NS, rl, marker="x", ls="--", color=colors[rho])
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_xscale("log")
axes[0].set_title("Mean bias: predicted minus oracle net VOI")
axes[0].set_xlabel("calibration sample size n")
axes[0].set_ylabel("utility units")
axes[0].legend(fontsize=8)
axes[1].set_xscale("log")
axes[1].set_title("P(over-acquire): predicted net>0, oracle net<0\n(solid: plain sample cov; dashed: Ledoit-Wolf)")
axes[1].set_xlabel("calibration sample size n")
axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_title("Expected one-step regret vs oracle\n(solid: plain; dashed: Ledoit-Wolf)")
axes[2].set_xlabel("calibration sample size n")
plt.tight_layout()
plt.savefig(FIG_DIR + f"/fig_b1_plugin_voi_bias{TAG}.png", dpi=150)
print(f"\nSaved figure. Total elapsed {time.time() - t0:.1f}s")
