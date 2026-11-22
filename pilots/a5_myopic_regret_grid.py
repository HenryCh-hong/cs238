"""A5: map the myopic-vs-optimal regret over a grid.

Axes (normalized, s = 1; real units follow by scaling, see voi_core/a3_a4 docstrings):
  d      : distance of the belief mean m from the nearest stop/trade boundary, in units of s
  c      : per-source query cost in units of s (c_j / s; the same for both sources)
  rho    : conditional correlation of source 1 with r given Y0; Delta_1/s^2 = rho^2,
           source 2 has rho2 = 1.5 rho (capped by PSD feasibility), Delta_2/s^2 = rho2^2
  kappa  : conditional correlation Corr(Y1, Y2 | Y0) (conditional dependence between sources)
Fixed: lam' = lam*s = 0.05 and tau' = tau/s = 0.05 (the realistic mapping lam=5, tau=5e-4, s=0.01),
qprev = 0.  Boundary at m ~ 0.1010 (s units).

Regret = V_opt - V_myopic (both exact, quadrature split at kinks).  Relative regret = regret / (V_opt - G).
Heatmap: analysis/figures/a5_myopic_regret_heatmap.png

Run:  python3 a5_myopic_regret_grid.py
"""
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from voi_core import TwoSource  # noqa: E402

FIG = os.path.join(HERE, "figures", "a5_myopic_regret_heatmap.png")
LAM, TAU, QP = 0.05, 0.05, 0.0


def boundary_root(lam, tau):
    disc = 1.0 - 4 * lam * (lam + tau)
    return (1.0 - np.sqrt(disc)) / (2 * lam)


def evaluate(d, c, rho, kappa, method="dense", rho2_factor=1.5):
    rho2 = min(rho2_factor * rho, 0.95)
    if 1 - rho * rho - rho2 * rho2 - kappa * kappa + 2 * kappa * rho * rho2 <= 1e-9:
        return None  # infeasible (correlation matrix not PSD)
    xb = boundary_root(LAM, TAU)
    m = xb + d
    ts = TwoSource(1.0, LAM, TAU, QP, c, c, rho, rho2, kappa)
    G = ts.value_never(m)
    j, vo = ts.myopic_first(m)
    V_opt = ts.value_optimal(m, method)
    V_my = ts.value_myopic(m, method)
    gain = V_opt - G
    regret = V_opt - V_my
    return dict(G=G, V_opt=V_opt, V_my=V_my, gain=gain, regret=regret,
                myopic_first=j, voi1=vo[1], voi2=vo[2], rho2=rho2)


def main():
    t0 = time.time()
    d_grid = np.linspace(0.0, 1.0, 21)
    c_grid = np.logspace(-5, -0.5, 25)
    rho_grid = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
    kappa_grid = [-0.8, -0.4, 0.0, 0.4, 0.8]
    xb = boundary_root(LAM, TAU)
    print(f"boundary (s units) = {xb:.6f}; lam'={LAM}, tau'={TAU}, qprev={QP}")
    print(f"grid: d in [0,1] ({len(d_grid)} pts), c in [1e-5, 0.316] log ({len(c_grid)} pts), "
          f"rho in {rho_grid} (rho2=1.5 rho), kappa in {kappa_grid}")

    rows = []
    infeasible = 0
    for rho in rho_grid:
        for kappa in kappa_grid:
            for d in d_grid:
                for c in c_grid:
                    r = evaluate(d, c, rho, kappa)
                    if r is None:
                        infeasible += 1
                        continue
                    rows.append(dict(d=d, c=c, rho=rho, kappa=kappa, **r))
    print(f"evaluated {len(rows)} feasible grid points; infeasible (non-PSD) skipped: {infeasible}; "
          f"time {time.time() - t0:.1f}s")

    n = len(rows)
    gain = np.array([r["gain"] for r in rows])
    reg = np.array([r["regret"] for r in rows])
    rel = np.where(gain > 0, reg / np.maximum(gain, 1e-300), 0.0)
    # Integrator: dense trapezoid in z (12001 pts), agrees with split adaptive quadrature to ~1e-8 abs;
    # Gauss-Hermite-200 was found to be off by up to ~1e-3 on the same kinked integrals, so it is not used.
    # 'regret > 0' is declared above 1e-7 of the optimal gain (tie-breaking noise excluded).
    pos = reg > 1e-7 * np.maximum(gain, 1e-300)
    big = pos & (reg > 0.01 * gain)
    print(f"\nFRACTION OF GRID (all {n} points):")
    print(f"  regret > 0 (gap > 1e-7 of gain):       {pos.sum():6d}  = {pos.mean():.4%}")
    print(f"  regret > 1% of optimal gain:           {big.sum():6d}  = {big.mean():.4%}")
    has_gain = gain > 1e-12 * 0 + 1e-300
    print(f"FRACTION AMONG POINTS WITH POSITIVE OPTIMAL GAIN (n={has_gain.sum()}):")
    print(f"  regret > 0:                            {(pos & has_gain).sum() / has_gain.sum():.4%}")
    print(f"  regret > 1% of optimal gain:           {(big & has_gain).sum() / has_gain.sum():.4%}")
    print(f"  points with optimal gain > 0 at all:   {has_gain.sum()} of {n}")
    print(f"\nMAXIMUM regret (absolute, s=1 units):  {reg.max():.6e}")
    i = int(np.argmax(reg))
    r = rows[i]
    print(f"  at d={r['d']:.3f}, c={r['c']:.3e}, rho={r['rho']}, rho2={r['rho2']:.3f}, kappa={r['kappa']}, "
          f"gain={r['gain']:.6e}, myopic first={r['myopic_first']}, VOI1={r['voi1']:+.4e}, VOI2={r['voi2']:+.4e}")
    print(f"  relative regret at that point: {rel[i]:.4%}")
    print(f"MAXIMUM relative regret (regret/gain): {rel.max():.4%}")
    j = int(np.argmax(rel))
    r = rows[j]
    print(f"  at d={r['d']:.3f}, c={r['c']:.3e}, rho={r['rho']}, rho2={r['rho2']:.3f}, kappa={r['kappa']}, "
          f"regret={r['regret']:.6e}, gain={r['gain']:.6e}")

    # where regret concentrates
    print("\nCONCENTRATION (fraction of points with regret > 1% of gain, by factor):")
    for name, key, vals in (("kappa", "kappa", kappa_grid), ("rho", "rho", rho_grid)):
        for v in vals:
            sel = np.array([r[key] == v for r in rows])
            print(f"  {name}={v:+.2f}: {(big & sel).sum() / sel.sum():.3%}  "
                  f"(points with regret>0: {(pos & sel).sum()})")
    dvals = np.array([r["d"] for r in rows])
    cvals = np.array([r["c"] for r in rows])
    print("  by distance d (bin: frac regret>1% of gain | regret>0 count):")
    for lo, hi in ((0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.7), (0.7, 1.01)):
        sel = (dvals >= lo) & (dvals < hi)
        print(f"    d in [{lo:.2f},{hi:.2f}): {(big & sel).sum() / max(sel.sum(), 1):.3%} | {(pos & sel).sum()}")
    print("  by cost c (decade bins):")
    for lo in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        sel = (cvals >= lo) & (cvals < lo * 10)
        print(f"    c in [{lo:.0e},{lo*10:.0e}): {(big & sel).sum() / max(sel.sum(), 1):.3%} | {(pos & sel).sum()}")

    # regret-positive points: check which branch optimal picked
    neg_voi = np.array([(r["voi1"] <= 0 and r["voi2"] <= 0) for r in rows])
    print(f"\nregret>0 cases with both one-step VOI <= 0 (myopic stops, optimal queries): "
          f"{(pos & neg_voi).sum()}")
    print(f"regret>0 cases with a positive one-step VOI (argmax mismatch): {(pos & ~neg_voi).sum()}")

    # spot check of the GH-200 classification with split adaptive quadrature on a random subsample
    rng = np.random.default_rng(20261026)
    idx = rng.choice(len(rows), size=min(30, len(rows)), replace=False)
    agree = flips = 0
    max_abs_dev = 0.0
    for i in idx:
        rq = evaluate(rows[i]["d"], rows[i]["c"], rows[i]["rho"], rows[i]["kappa"], method="quad")
        max_abs_dev = max(max_abs_dev, abs(rq["regret"] - rows[i]["regret"]), abs(rq["V_opt"] - rows[i]["V_opt"]))
        pq = rq["regret"] > 1e-7 * max(rq["gain"], 1e-300)
        agree += int(pq == pos[i])
        flips += int(pq != pos[i])
    print(f"\nquadrature spot check on {len(idx)} random grid points (dense trapezoid vs split quadrature): "
          f"regret>0 classification agrees in {agree}, disagrees in {flips}; max |dense - quad| on V and regret = {max_abs_dev:.2e}")

    # heatmap: relative regret over (d, c) for kappa x rho panels
    import matplotlib.colors as mcolors
    fig, axes = plt.subplots(len(kappa_grid), len(rho_grid), figsize=(15, 12), sharex=True, sharey=True)
    lookup = {(r["rho"], r["kappa"], r["d"], r["c"]): rel[k] for k, r in enumerate(rows)}
    norm = mcolors.Normalize(vmin=0, vmax=max(0.2, float(rel.max())))
    im = None
    for a, kappa in enumerate(kappa_grid):
        for b, rho in enumerate(rho_grid):
            ax = axes[a, b]
            Z = np.full((len(c_grid), len(d_grid)), np.nan)
            for ii, c in enumerate(c_grid):
                for jj, d in enumerate(d_grid):
                    v = lookup.get((rho, kappa, d, c))
                    if v is not None:
                        Z[ii, jj] = v
            im = ax.pcolormesh(d_grid, c_grid, Z, shading="nearest", cmap="magma", norm=norm)
            ax.set_yscale("log")
            if a == 0:
                ax.set_title(f"rho={rho}", fontsize=9)
            if b == 0:
                ax.set_ylabel(f"kappa={kappa:+.1f}\nc/s")
            if a == len(kappa_grid) - 1:
                ax.set_xlabel("d = |m - boundary| / s")
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("relative regret = (V_opt - V_myopic)/(V_opt - G)   [clipped at color max]")
    fig.suptitle("A5: myopic VOI regret over (distance, cost) for each signal rho (columns) and "
                 "conditional source dependence kappa (rows); rho2 = 1.5 rho; s = 1 units", fontsize=10)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=130, bbox_inches="tight")
    print(f"\nsaved figure: {FIG}")
    print(f"total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
