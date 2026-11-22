"""Robustness of the 'second purchase depends on z1' classification: uses P_z(buy second | first purchase, m)
and classifies with tolerances 1e-6 on the z-probability, on a random subset of the T3 grid."""
import itertools
import numpy as np
from scipy.stats import norm
import toy_t3_adaptivity as t

rng = np.random.default_rng(7)
grid = list(itertools.product([1.0, 5.0], [1.0, 5.0], list(itertools.product([0.05, 0.1, 0.2, 0.4], repeat=2)),
                              [-0.5, 0.0, 0.5], list(itertools.product([0.1, 1.0, 10.0], repeat=2))))
idx = rng.choice(len(grid), size=150, replace=False)
M, Z, zmax = 361, 1001, 8.0
Mm = np.linspace(-6 * t.SDM, 6 * t.SDM, M)
wm = norm.pdf(Mm / t.SDM) * (Mm[1] - Mm[0]) / t.SDM
wm = wm / wm.sum()
zg = np.linspace(-zmax, zmax, Z)
wz = norm.pdf(zg) * (zg[1] - zg[0]); wz[0] *= 0.5; wz[-1] *= 0.5
pz_tot = wz.sum()
n_states_w = 0.0
buckets = {"never": 0.0, "always": 0.0, "interior_1e-6": 0.0, "dependent_any": 0.0}
cfg_dep_strict = 0
cfg_dep_loose = 0
probs_B = []
for k in idx:
    lam, tau_bp, (r1, r2), kap, (c1, c2) = grid[k]
    md = t.make_model(lam, tau_bp, r1, r2, kap, c1, c2)
    G, Vopt, ol, first, b1, b2, ind1, ind2 = t.value_grids(md, Mm, wm, zg, wz)
    dep_here = False
    dep_loose = False
    for i in range(M):
        if first[i] == 0:
            continue
        ind = ind1[i] if first[i] == 1 else ind2[i]
        pB = float(wz[ind].sum() / pz_tot)
        n_states_w += wm[i]
        if pB <= 1e-12:
            buckets["never"] += wm[i]
        elif pB >= 1 - 1e-12:
            buckets["always"] += wm[i]
        else:
            buckets["dependent_any"] += wm[i]
            if 1e-6 < pB < 1 - 1e-6:
                buckets["interior_1e-6"] += wm[i]
                dep_here = True
            probs_B.append(pB)
    cfg_dep_strict += int(dep_here)
print("configs checked:", len(idx))
print("first-purchase m-weight:", round(n_states_w, 4))
for kk, vv in buckets.items():
    print(f"  {kk}: {vv / n_states_w:.4f} of first-purchase m-weight")
print("configs with a genuine z-dependent second purchase (P_z(B) in (1e-6,1-1e-6) at some m):",
      cfg_dep_strict, "/", len(idx))
pb = np.array(probs_B)
if len(pb):
    print("P_z(buy second | first purchase, m) among dependent states: quantiles 5/25/50/75/95% =",
          np.round(np.quantile(pb, [0.05, 0.25, 0.5, 0.75, 0.95]), 4))
