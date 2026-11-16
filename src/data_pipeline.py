"""Historical daily prices: download, validation, target and causal features.

Timing convention (see docs/notes_B.md)
---------------------------------------
* A row dated ``t`` is the daily bar of trading day ``t``. The forecast is made at the
  close of ``t``.
* Target: ``target_t = AdjClose_{t+1} / AdjClose_t - 1`` (total return including
  dividends), with ``return_date_t = t+1`` (the next row of the table).
* Features at ``t`` use only rows dated ``<= t``. Adjusted-close *levels* are not used
  because Yahoo back-adjusts all earlier prices for dividends paid later, so a level at
  ``t`` changes when a future dividend is paid. Only ratios of prices (or log ratios) over
  windows ending at ``t`` are used; a constant rescaling of the whole adjusted series, and
  any dividend back-adjustment whose ex-date is after ``t``, leave them unchanged.

The module never sorts or silently repairs data: ``validate_prices`` reports every defect
and ``assert_valid`` refuses to continue when there are hard errors.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

RAW_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
PRICE_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Adj Close")

RETURN_WINDOWS: tuple[int, ...] = (1, 5, 10, 21, 63, 126, 252)
VOL_WINDOWS: tuple[int, ...] = (5, 21, 63)
DRAWDOWN_WINDOW = 252
MA_WINDOWS: tuple[int, ...] = (50, 200)
VOLUME_Z_WINDOW = 63

FEATURE_COLUMNS: tuple[str, ...] = (
    tuple(f"logret_{k}d" for k in RETURN_WINDOWS)
    + tuple(f"rv_{k}d" for k in VOL_WINDOWS)
    + ("vol_ratio_5d_63d", "drawdown_252d", "log_price_ma50", "log_price_ma200", "log_hl_range", "volume_z_63d")
)
TARGET_COLUMN = "target"
RETURN_DATE_COLUMN = "return_date"

DEFAULT_MAX_GAP_DAYS = 7
DEFAULT_IMPLAUSIBLE_RETURN = 0.25
# Adjusted/Close ratio steps: a dividend back-adjustment multiplies all earlier adjusted
# prices by (1 - d/P) < 1, so the ratio steps UP by a small factor on the ex-date. Steps with
# |log step| <= RATIO_NOISE_TOL (1 bp) are float rounding in the vendor series (about 1e-6 on
# SPY, both signs) and are counted separately, not as adjustment events.
RATIO_NOISE_TOL = 1e-4
DIVIDEND_MAX_LOG_STEP = float(np.log(1.5))


# ----------------------------------------------------------------------------- download
def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance >= 0.2.5x returns (Price, Ticker) MultiIndex columns; keep the price level."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
        df.columns.name = None
    return df


def manifest_path_for(csv_path: str | Path) -> Path:
    p = Path(csv_path)
    return p.with_name(f"{p.stem}_manifest.json")


def download_prices(ticker: str, start: str, end: str, path: str | Path, *, now: datetime | None = None) -> pd.DataFrame:
    """Download daily bars for ``[start, end]`` (both inclusive) and write CSV + JSON manifest.

    yfinance treats ``end`` as exclusive, so ``end + 1 day`` is requested. ``auto_adjust=False``
    keeps the raw OHLC and the separate ``Adj Close`` column. One HTTP request is made.
    """
    path = Path(path)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        raise ValueError(f"end {end} precedes start {start}")
    end_exclusive = end_ts + pd.Timedelta(days=1)
    raw = yf.download(
        ticker,
        start=start_ts.strftime("%Y-%m-%d"),
        end=end_exclusive.strftime("%Y-%m-%d"),
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker} {start}..{end}")
    df = flatten_columns(raw)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"yfinance response lacks columns {missing}")
    df = df.loc[:, list(RAW_COLUMNS)].astype(float)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize().rename("Date")
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]
    if df.empty:
        raise RuntimeError("no rows left after restricting to [start, end]")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="Date", date_format="%Y-%m-%d")
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "ticker": ticker,
        "source": "Yahoo Finance via yfinance",
        "auto_adjust": False,
        "columns": list(RAW_COLUMNS),
        "start_requested": start_ts.strftime("%Y-%m-%d"),
        "end_requested_inclusive": end_ts.strftime("%Y-%m-%d"),
        "yfinance_end_argument_exclusive": end_exclusive.strftime("%Y-%m-%d"),
        "rows": int(len(df)),
        "first_date": df.index.min().strftime("%Y-%m-%d"),
        "last_date": df.index.max().strftime("%Y-%m-%d"),
        "csv_file": path.name,
        "sha256": file_sha256(path),
        "download_utc": stamp,
        "yfinance_version": yf.__version__,
    }
    manifest_path_for(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return df


# ------------------------------------------------------------------------------ load
def load_prices(path: str | Path) -> pd.DataFrame:
    """Read the CSV written by :func:`download_prices`. Order and content are NOT repaired."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df.loc[:, list(RAW_COLUMNS)].astype(float)
    df.index = pd.DatetimeIndex(df.index, name="Date")
    return df


# ------------------------------------------------------------------------- validation
@dataclass
class PriceReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors), "warnings": list(self.warnings), "stats": dict(self.stats)}


def _fmt_dates(dates, limit: int = 5) -> str:
    dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]
    more = f" (+{len(dates) - limit} more)" if len(dates) > limit else ""
    return ", ".join(dates[:limit]) + more


def validate_prices(
    df: pd.DataFrame,
    *,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
    implausible_return: float = DEFAULT_IMPLAUSIBLE_RETURN,
) -> PriceReport:
    """Check a price table and report every defect.

    Errors (the table must not be used): missing columns, non-DatetimeIndex, duplicated dates,
    unsorted dates, NaN values, nonpositive prices or volume, High < Low, weekend dates.
    Warnings (inspect, usually kept): calendar gaps longer than ``max_gap_days``, daily
    adjusted-close simple returns larger than ``implausible_return`` in absolute value,
    and Adjusted/Close ratio steps that are not small upward dividend steps (split-like or
    other). Dividend-like ratio steps are counted and listed in ``stats``.
    """
    rep = PriceReport()
    if not isinstance(df.index, pd.DatetimeIndex):
        rep.errors.append("index is not a DatetimeIndex")
        return rep
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        rep.errors.append(f"missing columns: {missing}")
        return rep
    n = len(df)
    rep.stats["rows"] = int(n)
    if n == 0:
        rep.errors.append("no rows")
        return rep
    idx = df.index
    rep.stats["first_date"] = pd.Timestamp(idx.min()).strftime("%Y-%m-%d")
    rep.stats["last_date"] = pd.Timestamp(idx.max()).strftime("%Y-%m-%d")

    dup = np.asarray(idx.duplicated(keep=False))
    if dup.any():
        rep.errors.append(f"{int(dup.sum())} rows have duplicated dates, e.g. {_fmt_dates(idx[dup])}")
    if not idx.is_monotonic_increasing:
        rep.errors.append("dates are not sorted in ascending order")

    vals = df.loc[:, list(RAW_COLUMNS)]
    nan_counts = {c: int(v) for c, v in vals.isna().sum().items() if v > 0}
    if nan_counts:
        rep.errors.append(f"NaN values: {nan_counts}")
    for c in PRICE_COLUMNS:
        bad = np.asarray(vals[c] <= 0)
        if bad.any():
            rep.errors.append(f"{int(bad.sum())} nonpositive '{c}' values, e.g. {_fmt_dates(idx[bad])}")
    bad_vol = np.asarray(vals["Volume"] <= 0)
    if bad_vol.any():
        rep.errors.append(f"{int(bad_vol.sum())} nonpositive Volume values, e.g. {_fmt_dates(idx[bad_vol])}")
    hl = np.asarray(vals["High"] < vals["Low"])
    if hl.any():
        rep.errors.append(f"{int(hl.sum())} rows with High < Low, e.g. {_fmt_dates(idx[hl])}")
    weekend = np.asarray(idx.dayofweek >= 5)
    if weekend.any():
        rep.errors.append(f"{int(weekend.sum())} weekend dates, e.g. {_fmt_dates(idx[weekend])}")

    # Calendar gaps (on the sorted, de-duplicated dates).
    uniq = pd.DatetimeIndex(sorted(set(idx)))
    if len(uniq) > 1:
        gap_days = np.diff(uniq.values).astype("timedelta64[D]").astype(int)
        big = gap_days > max_gap_days
        rep.stats["max_calendar_gap_days"] = int(gap_days.max())
        if big.any():
            starts = uniq[:-1][big]
            rep.warnings.append(
                f"{int(big.sum())} calendar gaps longer than {max_gap_days} days, starting "
                f"{_fmt_dates(starts)} (gaps of {gap_days[big].tolist()} days)"
            )

    # Implausible simple returns of the adjusted close (row order).
    adj = vals["Adj Close"].to_numpy(dtype=float)
    if len(adj) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = adj[1:] / adj[:-1] - 1.0
        big_ret = np.isfinite(ret) & (np.abs(ret) > implausible_return)
        rep.stats["max_abs_adj_return"] = float(np.nanmax(np.abs(ret))) if np.isfinite(ret).any() else float("nan")
        if big_ret.any():
            rep.warnings.append(
                f"{int(big_ret.sum())} daily Adj Close returns exceed {implausible_return:.0%} in absolute value, "
                f"e.g. {_fmt_dates(idx[1:][big_ret])}"
            )

    # Adjusted/Close ratio steps (dividend and split days).
    close = vals["Close"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(adj / close)
    valid = np.isfinite(log_ratio)
    if len(log_ratio) > 1:
        both = valid[1:] & valid[:-1]
        step = np.diff(log_ratio)
        # Above float64 round-off (1e-10) but within the noise tolerance.
        noise = both & (np.abs(step) <= RATIO_NOISE_TOL) & (np.abs(step) > 1e-10)
        rep.stats["n_ratio_rounding_steps"] = int(noise.sum())
        events = both & (np.abs(step) > RATIO_NOISE_TOL)
        dividend_like = events & (step > 0) & (step <= DIVIDEND_MAX_LOG_STEP)
        split_like = events & (np.abs(step) > DIVIDEND_MAX_LOG_STEP)
        other = events & ~dividend_like & ~split_like
        dates_next = idx[1:]
        rep.stats["n_ratio_steps"] = int(events.sum())
        rep.stats["n_dividend_adjustment_days"] = int(dividend_like.sum())
        rep.stats["dividend_adjustment_dates"] = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates_next[dividend_like]]
        rep.stats["n_split_like_ratio_steps"] = int(split_like.sum())
        rep.stats["n_other_ratio_steps"] = int(other.sum())
        if split_like.any():
            rep.warnings.append(f"{int(split_like.sum())} split-like Adj/Close ratio steps, e.g. {_fmt_dates(dates_next[split_like])}")
        if other.any():
            rep.warnings.append(f"{int(other.sum())} unexplained Adj/Close ratio steps, e.g. {_fmt_dates(dates_next[other])}")
    return rep


def assert_valid(report: PriceReport) -> None:
    if not report.ok:
        raise ValueError("price data failed validation: " + "; ".join(report.errors))


# ----------------------------------------------------------------------- target, features
def compute_target(df: pd.DataFrame) -> pd.Series:
    """target_t = AdjClose_{t+1}/AdjClose_t - 1 indexed by t. Last row is NaN (no label yet)."""
    px = df["Adj Close"].astype(float)
    return (px.shift(-1) / px - 1.0).rename(TARGET_COLUMN)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal, ratio-type features at each row t (uses rows dated <= t only).

    Columns (see FEATURE_COLUMNS):
      logret_{k}d      log(AdjClose_t / AdjClose_{t-k}), k in RETURN_WINDOWS
      rv_{k}d          std (ddof=1) of the last k daily log returns, k in VOL_WINDOWS
      vol_ratio_5d_63d rv_5d / rv_63d
      drawdown_252d    AdjClose_t / max(AdjClose over last 252 rows incl. t) - 1  (<= 0)
      log_price_ma{w}  log(AdjClose_t / mean of last w AdjClose, incl. t), w in MA_WINDOWS
      log_hl_range     log(High_t / Low_t) (same-day ratio, adjustment-invariant)
      volume_z_63d     z-score of log(Volume_t) against the last 63 rows incl. t
    Warm-up rows (first 252) are NaN.
    """
    px = df["Adj Close"].astype(float)
    lp = np.log(px)
    out: dict[str, pd.Series] = {}
    for k in RETURN_WINDOWS:
        out[f"logret_{k}d"] = lp - lp.shift(k)
    lr1 = lp.diff()
    for k in VOL_WINDOWS:
        out[f"rv_{k}d"] = lr1.rolling(k, min_periods=k).std(ddof=1)
    out["vol_ratio_5d_63d"] = out["rv_5d"] / out["rv_63d"]
    out["drawdown_252d"] = px / px.rolling(DRAWDOWN_WINDOW, min_periods=DRAWDOWN_WINDOW).max() - 1.0
    for w in MA_WINDOWS:
        out[f"log_price_ma{w}"] = lp - np.log(px.rolling(w, min_periods=w).mean())
    out["log_hl_range"] = np.log(df["High"].astype(float) / df["Low"].astype(float))
    lv = np.log(df["Volume"].astype(float))
    m = lv.rolling(VOLUME_Z_WINDOW, min_periods=VOLUME_Z_WINDOW).mean()
    s = lv.rolling(VOLUME_Z_WINDOW, min_periods=VOLUME_Z_WINDOW).std(ddof=1)
    out["volume_z_63d"] = (lv - m) / s
    feats = pd.DataFrame(out, index=df.index)
    return feats.loc[:, list(FEATURE_COLUMNS)]


def make_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """Supervised table: index = feature date t, columns = FEATURE_COLUMNS, ``target`` = r_{t+1},
    ``return_date`` = t+1. Drops warm-up rows and the final row (no label yet)."""
    if not (df.index.is_monotonic_increasing and df.index.is_unique):
        raise ValueError("make_supervised needs strictly increasing, unique dates (run validate_prices)")
    sup = build_features(df)
    sup[TARGET_COLUMN] = compute_target(df)
    sup[RETURN_DATE_COLUMN] = pd.Series(df.index, index=df.index).shift(-1)
    sup = sup.dropna(subset=list(FEATURE_COLUMNS) + [TARGET_COLUMN, RETURN_DATE_COLUMN])
    sup.index.name = "date"
    return sup
