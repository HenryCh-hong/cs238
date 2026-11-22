"""Summarize T3 grid results (rows saved by toy_t3_adaptivity.py)."""
import os
import pickle
import numpy as np

rows = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache", "t3_rows.pkl"), "rb"))
n = len(rows)
adapt = np.array([r["adapt"] for r in rows])
voi = np.array([r["voi"] for r in rows])
EG = np.array([r["EG"] for r in rows])
Vopt = np.array([r["Vopt"] for r in rows])
Vol = np.array([r["Vol"] for r in rows])
print(f"configs = {n}")
print(f"configs with positive mean VOI (V_opt - G > 1e-12): {(voi > 1e-12).sum()}")
print(f"adapt: min={adapt.min():.3e}, max={adapt.max():.3e}, negative count (< -1e-12) = {(adapt < -1e-12).sum()}")
print(f"mean adapt (over grid) = {adapt.mean():.6e} real units; mean VOI = {voi.mean():.6e}")
print(f"RATIO of grid means: sum(adapt)/sum(VOI) = {adapt.sum()/voi.sum():.4f}")
pos = voi > 1e-12
frac = adapt[pos] / voi[pos]
print(f"among configs with VOI>0 (n={pos.sum()}): mean of per-config fraction = {frac.mean():.4f}; "
      f"median = {np.median(frac):.4f}; 10%/90% = {np.quantile(frac,0.1):.4f}/{np.quantile(frac,0.9):.4f}; "
      f"max = {frac.max():.4f}")
rel_gap = adapt[pos] / Vopt[pos]
print(f"adapt / V_opt (VOI>0 configs): mean = {rel_gap.mean():.3e}, max = {rel_gap.max():.3e}")
# dependence of second purchase on first observation
fb = np.array([r["first_buy_states_w"] for r in rows])
dep = np.array([r["depends_w"] for r in rows])
nev = np.array([r["never_w"] for r in rows])
alw = np.array([r["always_w"] for r in rows])
print(f"m-weighted share of states where the optimal policy makes a first purchase = {fb.sum()/n:.4f} (per config average)")
tot_fb = fb.sum()
print(f"Among first-purchase states (m-weighted): second purchase depends on z1 = {dep.sum()/tot_fb:.4f}, "
      f"always buys second = {alw.sum()/tot_fb:.4f}, never buys second = {nev.sum()/tot_fb:.4f}")
cfg_dep = (dep > 1e-9)
print(f"configs where the second purchase depends on the first observation for some m (weight > 1e-9): {cfg_dep.sum()}/{n} = {cfg_dep.mean():.3f}")
pos_dep = cfg_dep[pos]
print(f"  among VOI>0 configs: {pos_dep.sum()}/{pos.sum()} = {pos_dep.mean():.3f}")
# by parameter
import collections
for key in ["kappa", "lam", "tau_bp"]:
    d = collections.defaultdict(list)
    for r, a, v in zip(rows, adapt, voi):
        if v > 1e-12:
            d[r[key]].append(a / v)
    print(f"mean adapt/VOI by {key}: " + ", ".join(f"{k}: {np.mean(vv):.4f}" for k, vv in sorted(d.items())))
d = collections.defaultdict(list)
for r, a, v in zip(rows, adapt, voi):
    if v > 1e-12:
        d[(min(r["c1"], r["c2"]), max(r["c1"], r["c2"]))].append(a / v)
print("mean adapt/VOI by (min,max) cost (bp): " + ", ".join(f"{k}: {np.mean(vv):.4f}" for k, vv in sorted(d.items())))
d = collections.defaultdict(list)
for r, a, v in zip(rows, adapt, voi):
    if v > 1e-12:
        d[max(r["r1"], r["r2"])].append(a / v)
print("mean adapt/VOI by max(rho): " + ", ".join(f"{k}: {np.mean(vv):.4f}" for k, vv in sorted(d.items())))
# top adaptivity configs
idx = np.argsort(-(adapt / np.where(voi > 1e-12, voi, np.nan)))
print("largest fraction configs (top 5):")
for i in idx[:5]:
    r = rows[i]
    print(f"  lam={r['lam']} tau={r['tau_bp']}bp rho=({r['r1']},{r['r2']}) kappa={r['kappa']} c=({r['c1']},{r['c2']})bp "
          f"adapt={r['adapt']:.3e} VOI={r['voi']:.3e} frac={r['frac']:.4f}")
print("example config list with VOI>0 and largest absolute adaptivity (top 3):")
for i in np.argsort(-adapt)[:3]:
    r = rows[i]
    print(f"  lam={r['lam']} tau={r['tau_bp']}bp rho=({r['r1']},{r['r2']}) kappa={r['kappa']} c=({r['c1']},{r['c2']})bp "
          f"adapt={r['adapt']:.3e} VOI={r['voi']:.3e} frac={r['frac']:.4f}")
