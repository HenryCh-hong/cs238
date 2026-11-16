"""Experiments 1-4 on the SPY replay (plus a labelled synthetic sanity run).

Runs on GitHub Actions, NOT on the shared development laptop:

    python -m experiments.run_experiments --workers 4            # everything
    python -m experiments.run_experiments --only exp1,exp2       # subset
    python -m experiments.run_experiments --smoke --max-dates 30 # tiny check (seconds)

Outputs go to results/tables/exp*_*.csv and results/figures/ (via experiments.make_figures).

Exp0  synthetic well-specified world (belief = true law): sanity of values vs realized.
Exp1  policy comparison on SPY (lam in {1,5}): net utility, costs, frequencies, CPU time.
Exp2  VOI calibration: predicted one-step gross VOI (A, B, A+B) and full-policy value
      vs paired realized improvement on held-out dates, with block-bootstrap CIs.
Exp3  cost sensitivity: equal costs c in {0,...,10} bp for lam in {1,5}.
Exp4  calibration sample size: belief fitted on the last n calibration rows.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from experiments.run_replay import POLICY_ORDER, make_policies, prepare, run, settings_from_cfg
from src.config import load_config, resolve, with_overrides
from src.evaluation import calibration_table, mean_ci, voi_counterfactuals
from src.utility import choose_position, realized_utility

TABLES = resolve("results/tables")
COSTS_BP = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
LAMS = (1.0, 5.0)
CAL_SIZES = (250, 500, 1000, 2000, 0)  # 0 = all calibration rows


def _save(df: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False)


def _job(args):
    cfg, tag, kw = args
    t0 = time.time()
    log, summary, paired, manifest = run(cfg, tag=tag, progress_every=0, **kw)
    s = summary.reset_index().rename(columns={"index": "policy"})
    s["tag"] = tag
    s["runtime_s"] = time.time() - t0
    if not paired.empty:
        s = s.merge(paired[["policy", "mean_diff_bp", "ci_lo_bp", "ci_hi_bp"]], on="policy", how="left")
    return tag, s, manifest


def _map(jobs, workers: int):
    if workers <= 1:
        return [_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job, jobs))


# ---------------------------------------------------------------------- experiments
def exp0_synthetic(cfg, workers, max_dates):
    from src.synthetic_archive import make_synthetic_archive, spy_like_model

    truth = spy_like_model(rho_y0=0.05, rho_A=0.15, rho_B=0.12, corr_AB=0.5)
    arch = make_synthetic_archive(truth, 8000, seed=int(cfg["seed"]))
    arch["synthetic"] = True
    c = with_overrides(cfg, decision={"lam": 5.0, "cost_A": 2e-5, "cost_B": 2e-5})
    out = _map([(c, "exp0_synthetic", {"archive": arch, "max_dates": max_dates})], 1)
    _save(out[0][1], "exp0_synthetic_summary")
    return out[0][1]


def exp1_policy_comparison(cfg, workers, max_dates):
    jobs = [(with_overrides(cfg, decision={"lam": lam}), f"exp1_lam{lam:g}", {"max_dates": max_dates}) for lam in LAMS]
    res = _map(jobs, workers)
    tab = pd.concat([r[1] for r in res], ignore_index=True)
    _save(tab, "exp1_policy_comparison")
    return tab


def exp2_voi_calibration(cfg, workers, max_dates):
    from src.planner import solve
    from src.policies import ForecastVault, run_episode

    rows = []
    for lam in LAMS:
        c = with_overrides(cfg, decision={"lam": lam})
        tag = f"exp1_lam{lam:g}"
        log_path = resolve("results/logs") / f"{tag}_log.csv.gz"
        archive, eval_mask, schedule, info = prepare(c)
        ev = archive.loc[eval_mask].reset_index(drop=True)
        if max_dates:
            ev = ev.iloc[:max_dates]
        # reference position path: the free-only policy's own q_prev (same for every arm)
        if log_path.exists():
            lg = pd.read_csv(log_path, parse_dates=["date"])
            qp = dict(zip(lg.loc[lg.policy == "free_only", "date"], lg.loc[lg.policy == "free_only", "q_prev"]))
        else:
            qp = {}
        settings = settings_from_cfg(c)
        p, setup = settings.params, settings.setup
        cf = voi_counterfactuals(ev, schedule, qp, p)
        # full adaptive policies from the same reference position
        opt, myo = make_policies(c, ("optimal_voi", "myopic_voi"))
        full = []
        for i, row in enumerate(ev.itertuples(index=False)):
            t = pd.Timestamp(row.date)
            q_prev = int(qp.get(t, 0))
            b0 = schedule.model_for(t, i).belief({"y0": float(row.free_forecast)})
            G0, q0 = choose_position(*b0.return_moments(), q_prev, p)
            r = float(row.realized_return)
            u0 = realized_utility(q0, r, q_prev, p)
            plan = solve(b0, frozenset(("A", "B")), settings.budget, setup, q_prev, n_nodes=settings.n_quad)
            rec = {"date": t, "pred_full_opt": plan.value - G0}
            for name, pol in (("opt", opt), ("myo", myo)):
                vault = ForecastVault({"A": float(row.rf_forecast), "B": float(row.gb_forecast)}, dict(settings.costs))
                ep = run_episode(pol, b0, vault, setup, q_prev, np.random.default_rng(0))
                rec[f"real_full_{name}"] = realized_utility(ep.q, r, q_prev, p) - ep.total_cost - u0
                rec[f"buys_{name}"] = len(ep.purchases)
            full.append(rec)
        cf = cf.merge(pd.DataFrame(full), on="date")
        cf["lam"] = lam
        cf.to_csv(TABLES / f"exp2_counterfactuals_lam{lam:g}.csv.gz", index=False)
        for arm in ("A", "B", "AB"):
            t = calibration_table(cf, arm=arm, n_bins=5, seed=int(c["seed"]))
            t["lam"] = lam
            rows.append(t)
        pred = cf["pred_full_opt"].to_numpy() * 1e4
        real = cf["real_full_opt"].to_numpy() * 1e4
        ci = mean_ci(real, n_boot=2000, seed=int(c["seed"]))
        rows.append(pd.DataFrame([{"arm": "full_optimal_policy(net)", "bin": "all", "n": len(cf), "lam": lam,
                                   "pred_mean_bp": pred.mean(), "real_mean_bp": ci["estimate"],
                                   "real_ci_lo_bp": ci["lo"], "real_ci_hi_bp": ci["hi"],
                                   "frac_position_changed": float((cf["buys_opt"] > 0).mean())}]))
    tab = pd.concat(rows, ignore_index=True)
    _save(tab, "exp2_voi_calibration")
    return tab


def exp3_cost_sensitivity(cfg, workers, max_dates):
    jobs = []
    for lam in LAMS:
        for c_bp in COSTS_BP:
            c = with_overrides(cfg, decision={"lam": lam, "cost_A": c_bp * 1e-4, "cost_B": c_bp * 1e-4})
            jobs.append((c, f"exp3_lam{lam:g}_c{c_bp:g}", {"max_dates": max_dates}))
    res = _map(jobs, workers)
    out = []
    for (cfg_j, tag, _), (_, s, _) in zip(jobs, res):
        s = s.copy()
        s["lam"] = cfg_j["decision"]["lam"]
        s["cost_bp"] = cfg_j["decision"]["cost_A"] * 1e4
        out.append(s)
    tab = pd.concat(out, ignore_index=True)
    _save(tab, "exp3_cost_sensitivity")
    return tab


def exp4_calibration_size(cfg, workers, max_dates):
    from src.belief_model import GaussianJointModel

    jobs = []
    for lam in LAMS:
        for n in CAL_SIZES:
            c = with_overrides(cfg, decision={"lam": lam}, belief={"mode": "fixed", "calibration_days": n})
            jobs.append((c, f"exp4_lam{lam:g}_n{n if n else 'all'}", {"max_dates": max_dates}))
    res = _map(jobs, workers)
    out = []
    for (cfg_j, tag, _), (_, s, manifest) in zip(jobs, res):
        s = s.copy()
        s["lam"] = cfg_j["decision"]["lam"]
        s["calibration_days_requested"] = cfg_j["belief"]["calibration_days"]  # 0 = all
        s["calibration_days"] = int(manifest["belief_fits"][0]["n"])  # rows actually used by the fit
        fm = manifest["first_belief_model"]
        m = GaussianJointModel(np.array(fm["mean"]), np.array(fm["cov"]))
        b = m.belief({"y0": float(m.mean[1])})
        sd = np.sqrt(np.diag(b._S_u))  # conditional sds of (r, A, B) given y0
        C = b._S_u / np.outer(sd, sd)
        s["rho_A_given_y0"], s["rho_B_given_y0"], s["kappa_AB_given_y0"] = C[0, 1], C[0, 2], C[1, 2]
        s["delta_A_bp2"], s["delta_B_bp2"] = b.gain("A") * 1e8, b.gain("B") * 1e8
        out.append(s)
    tab = pd.concat(out, ignore_index=True)
    _save(tab, "exp4_calibration_size")
    return tab


EXPERIMENTS = {"exp0": exp0_synthetic, "exp1": exp1_policy_comparison, "exp2": exp2_voi_calibration,
               "exp3": exp3_cost_sensitivity, "exp4": exp4_calibration_size}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.toml")
    ap.add_argument("--only", default="exp0,exp1,exp2,exp3,exp4")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-dates", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run: --max-dates 30 if unset, 1 worker")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    max_dates = args.max_dates or (30 if args.smoke else 0)
    workers = 1 if args.smoke else args.workers
    timings = {}
    for key in [k.strip() for k in args.only.split(",") if k.strip()]:
        t0 = time.time()
        print(f"=== {key} ===", flush=True)
        tab = EXPERIMENTS[key](cfg, workers, max_dates)
        timings[key] = time.time() - t0
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(tab.round(5).to_string(index=False)[:6000], flush=True)
        print(f"--- {key} done in {timings[key]:.1f}s", flush=True)
    with open(TABLES / "experiment_timings.json", "w") as f:
        json.dump({"timings_s": timings, "workers": workers, "max_dates": max_dates,
                   "cpu_count": os.cpu_count()}, f, indent=2)


if __name__ == "__main__":
    main()
