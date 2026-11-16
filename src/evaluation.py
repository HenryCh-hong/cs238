"""Evaluation statistics: block-bootstrap intervals, paired policy comparisons,
and VOI calibration (predicted value of information vs realized benefit).

All quantities here are computed AFTER the replay, by the evaluator, which may
look at every archived forecast and the realized return. Nothing in this module
is visible to a policy.

Why the paired realized improvement estimates the predicted gross VOI
-----------------------------------------------------------------------
Let q0 = argmax_q E_{b0}[u(q,r)] and q_j(y) the argmax after observing Y_j = y.
For a fixed q, E_{b^{j,y}}[u(q, r)] averages back to E_{b0}[u(q, r)] (the belief
is a martingale), so under the belief model

    E[ u(q_j(Y_j), r) - u(q0, r) | b0 ] = E[G(b^{j,Y_j})] - G(b0)  (gross VOI).

The realized paired improvement on one date is therefore an unbiased but very
noisy estimate of the predicted gross VOI; only averages over many dates are
informative, and only if the belief model is correctly specified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.utility import UtilityParams, choose_position, realized_utility


# ---------------------------------------------------------------------- bootstrap
def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap resample of indices 0..n-1."""
    block = max(1, min(block, n // 2))  # a block >= n would make every resample a rotation of the sample
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.reshape(-1)[:n]


def block_bootstrap(
    data: Sequence[np.ndarray] | np.ndarray,
    stat: Callable[..., float],
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile CI for ``stat(*arrays)`` under a circular moving-block bootstrap.

    ``data`` is one array or a sequence of equal-length arrays resampled jointly
    (keeps pairs and serial dependence within blocks).
    """
    arrays = [np.asarray(data, dtype=float)] if isinstance(data, np.ndarray) else [np.asarray(a, dtype=float) for a in data]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("arrays must have equal length")
    if n == 0:
        return {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "se": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    est = float(stat(*arrays))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = _block_indices(n, block, rng)
        boots[b] = stat(*[a[idx] for a in arrays])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": est, "lo": float(lo), "hi": float(hi), "se": float(np.nanstd(boots, ddof=1)), "n": n}


def mean_ci(x: np.ndarray, **kw) -> dict[str, float]:
    return block_bootstrap(np.asarray(x, dtype=float), np.mean, **kw)


def ols_slope(pred: np.ndarray, real: np.ndarray) -> float:
    p = pred - pred.mean()
    d = float(p @ p)
    return float(p @ (real - real.mean()) / d) if d > 0 else np.nan


# ---------------------------------------------------------------------- policy comparison
def paired_differences(log: pd.DataFrame, baseline: str = "free_only", value: str = "net_utility",
                       block: int = 21, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Mean per-date difference (policy - baseline) in bp with block-bootstrap CIs."""
    wide = log.pivot(index="date", columns="policy", values=value).sort_index()
    if baseline not in wide:
        raise KeyError(f"baseline policy {baseline!r} not in log")
    out = []
    for pol in wide.columns:
        d = (wide[pol] - wide[baseline]).to_numpy() * 1e4
        ci = mean_ci(d, block=block, n_boot=n_boot, seed=seed)
        out.append({"policy": pol, "baseline": baseline, "mean_diff_bp": ci["estimate"],
                    "ci_lo_bp": ci["lo"], "ci_hi_bp": ci["hi"], "se_bp": ci["se"], "n_dates": ci["n"]})
    return pd.DataFrame(out)


def policy_means(log: pd.DataFrame, value: str = "net_utility", block: int = 21, n_boot: int = 2000,
                 seed: int = 0) -> pd.DataFrame:
    out = []
    for pol, g in log.groupby("policy", sort=False):
        ci = mean_ci(g.sort_values("date")[value].to_numpy() * 1e4, block=block, n_boot=n_boot, seed=seed)
        out.append({"policy": pol, "mean_bp": ci["estimate"], "ci_lo_bp": ci["lo"], "ci_hi_bp": ci["hi"],
                    "n_dates": ci["n"]})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------- VOI calibration
@dataclass(frozen=True)
class CounterfactualRow:
    date: pd.Timestamp
    q_prev: int
    G0: float
    q0: int
    pred_gross_A: float
    pred_gross_B: float
    pred_gross_AB: float
    real_impr_A: float
    real_impr_B: float
    real_impr_AB: float


def voi_counterfactuals(archive_rows: pd.DataFrame, schedule, q_prev_by_date: dict, params: UtilityParams) -> pd.DataFrame:
    """Predicted gross VOI vs realized paired improvement, one row per date.

    archive_rows: evaluation rows of the archive (evaluator has full access).
    schedule: the same BeliefSchedule the replay used (so predictions use only
    information available at each date).
    q_prev_by_date: reference previous position per date (e.g. the free-only
    policy's path), held fixed across the counterfactual arms.
    """
    from src.voi import expected_stop_value_gaussian

    recs = []
    for i, row in enumerate(archive_rows.itertuples(index=False)):
        t = pd.Timestamp(row.date)
        qp = int(q_prev_by_date.get(t, 0))
        model = schedule.model_for(t, i)
        b0 = model.belief({"y0": float(row.free_forecast)})
        G0, q0 = choose_position(*b0.return_moments(), qp, params)
        r = float(row.realized_return)
        u0 = realized_utility(q0, r, qp, params)
        y = {"A": float(row.rf_forecast), "B": float(row.gb_forecast)}
        rec = {"date": t, "q_prev": qp, "G0": G0, "q0": q0, "m0": b0.mean, "v0": b0.var, "realized_return": r}
        for j in ("A", "B"):
            rec[f"pred_gross_{j}"] = b0.expected_stop_value_after(j, qp, params) - G0
            bj = b0.condition(j, y[j])
            _, qj = choose_position(*bj.return_moments(), qp, params)
            rec[f"q_{j}"] = qj
            rec[f"real_impr_{j}"] = realized_utility(qj, r, qp, params) - u0
        bA = b0.condition("A", y["A"])
        bAB = bA.condition("B", y["B"])
        delta_total = max(b0.var - bAB.var, 0.0)
        rec["pred_gross_AB"] = float(expected_stop_value_gaussian(b0.mean, delta_total, b0.var, qp, params)) - G0
        _, qab = choose_position(*bAB.return_moments(), qp, params)
        rec["q_AB"] = qab
        rec["real_impr_AB"] = realized_utility(qab, r, qp, params) - u0
        recs.append(rec)
    return pd.DataFrame(recs)


def calibration_table(cf: pd.DataFrame, arm: str = "A", n_bins: int = 5, block: int = 21, n_boot: int = 1000,
                      seed: int = 0) -> pd.DataFrame:
    """Bin dates by predicted gross VOI (quantile bins) and compare means (bp)."""
    pred = cf[f"pred_gross_{arm}"].to_numpy() * 1e4
    real = cf[f"real_impr_{arm}"].to_numpy() * 1e4
    order = np.argsort(pred, kind="stable")
    bins = np.array_split(order, n_bins)
    rows = []
    for k, idx in enumerate(bins):
        idx = np.sort(idx)  # keep chronological order inside the bin for the block bootstrap
        ci = mean_ci(real[idx], block=block, n_boot=n_boot, seed=seed + k)
        rows.append({"arm": arm, "bin": k, "n": len(idx), "pred_mean_bp": float(pred[idx].mean()),
                     "pred_min_bp": float(pred[idx].min()), "pred_max_bp": float(pred[idx].max()),
                     "real_mean_bp": ci["estimate"], "real_ci_lo_bp": ci["lo"], "real_ci_hi_bp": ci["hi"],
                     "frac_position_changed": float((cf[f"q_{arm}"].to_numpy()[idx] != cf["q0"].to_numpy()[idx]).mean())})
    allci = mean_ci(real, block=block, n_boot=n_boot, seed=seed + 99)
    slope = block_bootstrap([pred, real], ols_slope, block=block, n_boot=n_boot, seed=seed + 7)
    rows.append({"arm": arm, "bin": "all", "n": len(pred), "pred_mean_bp": float(pred.mean()),
                 "pred_min_bp": float(pred.min()), "pred_max_bp": float(pred.max()),
                 "real_mean_bp": allci["estimate"], "real_ci_lo_bp": allci["lo"], "real_ci_hi_bp": allci["hi"],
                 "frac_position_changed": float((cf[f"q_{arm}"] != cf["q0"]).mean()),
                 "slope": slope["estimate"], "slope_ci_lo": slope["lo"], "slope_ci_hi": slope["hi"]})
    return pd.DataFrame(rows)
