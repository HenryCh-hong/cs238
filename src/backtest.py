"""Chronological historical replay of the forecast-purchase decision.

Data flow per evaluation date t (decision at the close of t):

1. The belief model is the one calibrated on archive rows whose return was
   realized no later than t (``BeliefSchedule``); no held-out return after t is
   ever used for calibration.
2. Every policy starts from the same belief b0 = law of r given the free forecast
   y0(t). Hidden forecasts A(t), B(t) sit in a ``ForecastVault`` and are revealed
   only by a paid query.
3. The policy acquires (or not), then picks q by maximizing posterior expected
   utility given its own previous position q_prev.
4. Only then is r_{t+1} read from the archive to settle the trade.

Each policy carries its own position history. Every policy sees the same dates
and the same underlying forecasts. Randomized policies use a generator seeded by
(seed, policy index, date) so results are reproducible and independent of order.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.belief_model import GaussianJointModel, fit_joint_gaussian
from src.interfaces import SOURCES, AcquisitionSetup
from src.utility import UtilityParams, trading_utility, transaction_cost

ARCHIVE_COLS = ("free_forecast", "rf_forecast", "gb_forecast", "realized_return")
BELIEF_COLS = ("realized_return", "free_forecast", "rf_forecast", "gb_forecast")  # -> (r, y0, A, B)
SOURCE_COL = {"A": "rf_forecast", "B": "gb_forecast"}


# ---------------------------------------------------------------------- archive
def load_archive(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return validate_archive(df)


def validate_archive(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and enforce the temporal-validity invariants of the archive."""
    df = df.copy()
    for c in ("date", "return_date", "forecast_origin", "feature_timestamp", "train_end", "train_label_end", "train_start"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    missing = [c for c in ("date", "return_date", *ARCHIVE_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"archive is missing columns {missing}")
    if df["date"].duplicated().any():
        raise ValueError("archive has duplicate dates")
    if not df["date"].is_monotonic_increasing:
        raise ValueError("archive dates are not sorted")
    if df[list(ARCHIVE_COLS)].isna().any().any() or not np.isfinite(df[list(ARCHIVE_COLS)].to_numpy()).all():
        raise ValueError("archive has missing or non-finite forecasts/returns")
    if not (df["return_date"] > df["date"]).all():
        raise ValueError("return_date must be strictly after the decision date")
    if "train_end" in df.columns and not (df["train_end"] < df["date"]).all():
        raise ValueError("a model was trained on a feature row dated at or after its forecast origin")
    if "train_label_end" in df.columns and not (df["train_label_end"] <= df["date"]).all():
        raise ValueError("a model was trained on a label realized after its forecast origin")
    if "feature_timestamp" in df.columns and not (df["feature_timestamp"] <= df["date"]).all():
        raise ValueError("features dated after the decision date")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------- belief calibration
@dataclass
class BeliefSchedule:
    """Supplies the belief model to use at each decision date without look-ahead.

    mode="fixed":     one model fitted on calibration rows (date <= calibration_end)
                      whose returns are realized by the first evaluation date.
    mode="expanding": refit every ``refit_every`` evaluation dates on every archive
                      row whose return was realized at or before the refit date.
    calibration_days: if > 0, use only the most recent n eligible rows.
    """

    archive: pd.DataFrame
    calibration_end: pd.Timestamp
    method: str = "ledoit_wolf"
    mode: str = "fixed"
    refit_every: int = 21
    calibration_days: int = 0
    fit_fn: Callable[..., GaussianJointModel] | None = None
    _cache: dict = field(default_factory=dict, repr=False)
    fit_log: list = field(default_factory=list, repr=False)

    def _eligible(self, t: pd.Timestamp, date_cap: pd.Timestamp | None) -> pd.DataFrame:
        a = self.archive
        rows = a[a["return_date"] <= t]
        if date_cap is not None:
            rows = rows[rows["date"] <= date_cap]
        if self.calibration_days and self.calibration_days > 0:
            rows = rows.iloc[-self.calibration_days:]
        return rows

    def _fit(self, rows: pd.DataFrame, t: pd.Timestamp) -> GaussianJointModel:
        X = rows[list(BELIEF_COLS)].to_numpy(dtype=float)
        if len(rows) == 0 and self.fit_fn is None:
            raise ValueError(f"no calibration rows with a return realized by {t.date()}")
        fit = self.fit_fn or fit_joint_gaussian
        model = fit(X, method=self.method)
        last_ret = rows["return_date"].max() if len(rows) else None
        assert last_ret is None or last_ret <= t, "calibration used a return realized after the decision date"
        self.fit_log.append({"decision_date": t, "n": len(rows), "first_row": rows["date"].min(),
                             "last_row": rows["date"].max(), "last_return_date": last_ret})
        return model

    def model_for(self, t: pd.Timestamp, eval_index: int) -> GaussianJointModel:
        if self.mode == "fixed":
            if "fixed" not in self._cache:
                self._cache["fixed"] = self._fit(self._eligible(t, self.calibration_end), t)
            return self._cache["fixed"]
        if self.mode == "expanding":
            key = eval_index // self.refit_every
            if key not in self._cache:
                self._cache.clear()
                self._cache[key] = self._fit(self._eligible(t, None), t)
            return self._cache[key]
        raise ValueError(f"unknown belief mode {self.mode!r}")


# ---------------------------------------------------------------------- replay
@dataclass(frozen=True)
class ReplaySettings:
    params: UtilityParams
    costs: Mapping[str, float]
    budget: float
    n_quad: int | None = None
    seed: int = 0

    @property
    def setup(self) -> AcquisitionSetup:
        return AcquisitionSetup(costs=dict(self.costs), budget=self.budget, params=self.params)


def _rng_for(seed: int, policy_index: int, date: pd.Timestamp) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(policy_index), int(date.toordinal())])


def replay(
    archive: pd.DataFrame,
    eval_dates_mask: np.ndarray,
    schedule: BeliefSchedule,
    policies: Sequence,
    settings: ReplaySettings,
    q_prev0: int = 0,
    progress_every: int = 0,
) -> pd.DataFrame:
    """Run every policy over the evaluation dates; returns one log row per (date, policy)."""
    from src.policies import ForecastVault, run_episode

    rows = archive.loc[eval_dates_mask].reset_index(drop=True)
    if rows.empty:
        raise ValueError("no evaluation dates")
    setup = settings.setup
    p = settings.params
    q_prev = {pol.name: q_prev0 for pol in policies}
    log: list[dict] = []
    for i, row in enumerate(rows.itertuples(index=False)):
        t = pd.Timestamp(row.date)
        model = schedule.model_for(t, i)
        b0 = model.belief({"y0": float(row.free_forecast)})
        hidden = {"A": float(row.rf_forecast), "B": float(row.gb_forecast)}
        for k, pol in enumerate(policies):
            vault = ForecastVault(hidden, dict(settings.costs))
            rng = _rng_for(settings.seed, k, t)
            t0 = time.perf_counter()
            ep = run_episode(pol, b0, vault, setup, q_prev[pol.name], rng)
            dt = time.perf_counter() - t0
            r = float(row.realized_return)  # revealed only for settlement
            gross = trading_utility(ep.q, r, p)
            tc = transaction_cost(ep.q, q_prev[pol.name], p)
            acq = float(ep.total_cost)
            rec = {
                "date": t, "policy": pol.name, "q_prev": q_prev[pol.name], "y0": float(row.free_forecast),
                "m_prior": b0.mean, "v_prior": b0.var,
            }
            rec.update(_episode_fields(ep))
            rec.update({
                "realized_return": r, "gross_utility": gross, "transaction_cost": tc,
                "acquisition_cost": acq, "net_utility": gross - tc - acq, "decision_seconds": dt,
            })
            # Revealed forecast values, logged only if purchased (never hidden ones).
            for s in SOURCES:
                rec[f"revealed_{s}"] = hidden[s] if s in ep.purchases else np.nan
            log.append(rec)
            q_prev[pol.name] = ep.q
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  replay {i + 1}/{len(rows)} dates", flush=True)
    out = pd.DataFrame(log)
    _check_log(out, settings)
    return out


def _episode_fields(ep) -> dict:
    """Flatten an EpisodeResult into log columns (tolerant to optional diagnostics)."""
    steps = list(getattr(ep, "steps", []) or [])
    first = steps[0] if steps else {}
    voi0 = dict(first.get("voi", {}) or {}) if isinstance(first, dict) else {}
    purchases = list(ep.purchases)
    rec = {
        "purchases": ">".join(purchases),
        "n_purchases": len(purchases),
        "bought_A": int("A" in purchases),
        "bought_B": int("B" in purchases),
        "first_purchase": purchases[0] if purchases else "",
        "voi_A_0": float(voi0.get("A", np.nan)),
        "voi_B_0": float(voi0.get("B", np.nan)),
        "predicted_value": float(np.nan if ep.initial_predicted_value is None else ep.initial_predicted_value),
        "budget_left": float(getattr(ep, "budget_left", np.nan)),
        "m_post": float(getattr(ep, "final_mean", np.nan)),
        "v_post": float(getattr(ep, "final_var", np.nan)),
        "G_final": float(getattr(ep, "G", np.nan)),
        "q": int(ep.q),
    }
    if len(steps) > 1 and isinstance(steps[1], dict):
        voi1 = dict(steps[1].get("voi", {}) or {})
        rec["voi_next_1"] = float(next(iter(voi1.values()))) if voi1 else np.nan
    else:
        rec["voi_next_1"] = np.nan
    return rec


def _check_log(log: pd.DataFrame, settings: ReplaySettings) -> None:
    """Accounting and sanity invariants (Gate E). Raises on violation."""
    num = log.select_dtypes(include=[np.number])
    must_be_finite = ["q", "q_prev", "realized_return", "gross_utility", "transaction_cost",
                      "acquisition_cost", "net_utility", "m_prior", "v_prior"]
    bad = [c for c in must_be_finite if c in num and not np.isfinite(num[c]).all()]
    if bad:
        raise FloatingPointError(f"non-finite values in {bad}")
    recon = log["gross_utility"] - log["transaction_cost"] - log["acquisition_cost"]
    if not np.allclose(recon, log["net_utility"], rtol=0, atol=1e-15):
        raise AssertionError("net utility does not reconcile with its components")
    exp_cost = log["bought_A"] * settings.costs["A"] + log["bought_B"] * settings.costs["B"]
    if not np.allclose(exp_cost, log["acquisition_cost"], rtol=0, atol=1e-15):
        raise AssertionError("acquisition cost does not equal the sum of purchased source costs")
    if (log["acquisition_cost"] > settings.budget + 1e-12).any():
        raise AssertionError("budget exceeded")
    if not log["q"].isin([-1, 0, 1]).all():
        raise AssertionError("invalid position")
    if (log["n_purchases"] > len(SOURCES)).any():
        raise AssertionError("a source was purchased more than once")


def summarize(log: pd.DataFrame) -> pd.DataFrame:
    """Per-policy summary in basis points per day (1 bp = 1e-4)."""
    g = log.groupby("policy", sort=False)
    bp = 1e4
    out = pd.DataFrame({
        "n_dates": g.size(),
        "mean_net_bp": g["net_utility"].mean() * bp,
        "mean_gross_bp": g["gross_utility"].mean() * bp,
        "mean_tc_bp": g["transaction_cost"].mean() * bp,
        "mean_acq_cost_bp": g["acquisition_cost"].mean() * bp,
        "freq_buy_any": g["n_purchases"].apply(lambda s: float((s > 0).mean())),
        "freq_buy_A": g["bought_A"].mean(),
        "freq_buy_B": g["bought_B"].mean(),
        "freq_buy_both": g["n_purchases"].apply(lambda s: float((s == 2).mean())),
        "mean_purchases": g["n_purchases"].mean(),
        "frac_long": g["q"].apply(lambda s: float((s == 1).mean())),
        "frac_flat": g["q"].apply(lambda s: float((s == 0).mean())),
        "frac_short": g["q"].apply(lambda s: float((s == -1).mean())),
        "ms_per_decision": g["decision_seconds"].mean() * 1e3,
    })
    return out
