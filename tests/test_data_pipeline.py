"""Tests for src/data_pipeline.py on synthetic price paths (no network)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src import data_pipeline as dp


def make_prices(n: int = 400, seed: int = 0, start: str = "2018-01-01", dividends=()) -> pd.DataFrame:
    """Synthetic daily bars. ``dividends`` = [(row_position_of_ex_date, amount), ...].

    Adjusted close mimics Yahoo: Adj_t = Close_t * F_t with F_t = prod over ex-dates e > t of
    (1 - d_e / Close_{e-1}); the ratio Adj/Close therefore steps up on each ex-date.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n, name="Date")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    F = np.ones(n)
    for pos, amt in dividends:
        F[:pos] *= 1.0 - amt / close[pos - 1]
    adj = close * F
    open_ = close * (1.0 + rng.normal(0.0, 0.002, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, n)))
    volume = rng.integers(50_000_000, 100_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Adj Close": adj, "Volume": volume},
        index=dates,
    )


def test_target_alignment_and_return_date():
    df = make_prices(120)
    tgt = dp.compute_target(df)
    px = df["Adj Close"].to_numpy()
    for t in (0, 10, 50, 118):
        assert tgt.iloc[t] == pytest.approx(px[t + 1] / px[t] - 1.0, rel=0, abs=1e-15)
    assert np.isnan(tgt.iloc[-1])
    sup_like = pd.Series(df.index, index=df.index).shift(-1)
    assert sup_like.iloc[10] == df.index[11]
    assert pd.isna(sup_like.iloc[-1])


def test_make_supervised_drops_warmup_and_last_row():
    df = make_prices(400)
    sup = dp.make_supervised(df)
    assert sup.index[0] == df.index[252]  # first row with ret_252d and drawdown defined
    assert sup.index[-1] == df.index[-2]  # last row has no label
    assert len(sup) == 400 - 252 - 1
    assert not sup.isna().any().any()
    t = sup.index[5]
    pos = df.index.get_loc(t)
    assert sup.loc[t, "return_date"] == df.index[pos + 1]
    assert sup.loc[t, dp.TARGET_COLUMN] == pytest.approx(df["Adj Close"].iloc[pos + 1] / df["Adj Close"].iloc[pos] - 1)


def test_features_truncation_invariance():
    df = make_prices(330, seed=1)
    full = dp.build_features(df)
    for t in (251, 260, 300, 329):
        trunc = dp.build_features(df.iloc[: t + 1])
        pd.testing.assert_frame_equal(full.iloc[: t + 1], trunc, rtol=1e-13, atol=0)


def test_target_truncation_invariance():
    df = make_prices(200, seed=2)
    full = dp.compute_target(df)
    for t in (30, 100, 150):
        trunc = dp.compute_target(df.iloc[: t + 2])  # series truncated at t+1
        assert trunc.iloc[t] == full.iloc[t]
        np.testing.assert_array_equal(trunc.iloc[:t].to_numpy(), full.iloc[:t].to_numpy())


def test_features_unchanged_by_perturbing_prices_after_t():
    df = make_prices(330, seed=3)
    base = dp.build_features(df)
    rng = np.random.default_rng(7)
    for t in (260, 300):
        pert = df.copy()
        after = pert.index > df.index[t]
        shocks = np.exp(rng.normal(0, 0.05, after.sum()))
        for c in dp.RAW_COLUMNS:
            pert.loc[after, c] = pert.loc[after, c].to_numpy() * shocks
        feats = dp.build_features(pert)
        pd.testing.assert_frame_equal(base.iloc[: t + 1], feats.iloc[: t + 1], rtol=1e-13, atol=0)


def test_features_unchanged_by_future_dividend_backadjustment():
    """A later ex-date multiplies every earlier adjusted price by a constant (1 - d/P).

    With the ex-date after t, all windows ending at or before t are affected by a common factor
    and the features at rows <= t do not move. (The target at t does move, by design.)
    """
    df = make_prices(330, seed=4)
    t = 280
    ex_pos = 300
    base = dp.build_features(df)
    adjusted = df.copy()
    factor = 1.0 - 0.8 / df["Close"].iloc[ex_pos - 1]
    adjusted.iloc[:ex_pos, adjusted.columns.get_loc("Adj Close")] *= factor
    feats = dp.build_features(adjusted)
    pd.testing.assert_frame_equal(base.iloc[: t + 1], feats.iloc[: t + 1], rtol=1e-12, atol=1e-14)


def test_features_are_ratio_type_under_constant_rescaling():
    df = make_prices(330, seed=5)
    scaled = df.copy()
    scaled["Adj Close"] *= 37.5
    pd.testing.assert_frame_equal(dp.build_features(df), dp.build_features(scaled), rtol=1e-12, atol=1e-14)


def test_features_closed_form_at_one_date():
    df = make_prices(300, seed=6)
    f = dp.build_features(df)
    a = df["Adj Close"].to_numpy()
    h, lo, v = df["High"].to_numpy(), df["Low"].to_numpy(), df["Volume"].to_numpy()
    t = 290
    lr = np.log(a[1:] / a[:-1])  # lr[k-1] = log(a[k]/a[k-1])
    assert f.iloc[t]["logret_5d"] == pytest.approx(np.log(a[t] / a[t - 5]), rel=1e-12)
    assert f.iloc[t]["logret_252d"] == pytest.approx(np.log(a[t] / a[t - 252]), rel=1e-12)
    rv21 = np.std(lr[t - 21 : t], ddof=1)  # last 21 log returns ending at t
    rv5 = np.std(lr[t - 5 : t], ddof=1)
    rv63 = np.std(lr[t - 63 : t], ddof=1)
    assert f.iloc[t]["rv_21d"] == pytest.approx(rv21, rel=1e-10)
    assert f.iloc[t]["rv_5d"] == pytest.approx(rv5, rel=1e-10)
    assert f.iloc[t]["vol_ratio_5d_63d"] == pytest.approx(rv5 / rv63, rel=1e-10)
    assert f.iloc[t]["drawdown_252d"] == pytest.approx(a[t] / a[t - 251 : t + 1].max() - 1.0, rel=1e-12)
    assert f.iloc[t]["log_price_ma50"] == pytest.approx(np.log(a[t] / a[t - 49 : t + 1].mean()), rel=1e-12)
    assert f.iloc[t]["log_price_ma200"] == pytest.approx(np.log(a[t] / a[t - 199 : t + 1].mean()), rel=1e-12)
    assert f.iloc[t]["log_hl_range"] == pytest.approx(np.log(h[t] / lo[t]), rel=1e-12)
    lv = np.log(v)
    z = (lv[t] - lv[t - 62 : t + 1].mean()) / lv[t - 62 : t + 1].std(ddof=1)
    assert f.iloc[t]["volume_z_63d"] == pytest.approx(z, rel=1e-10)


def test_warmup_is_nan_and_rest_finite():
    f = dp.build_features(make_prices(300, seed=8))
    assert f.iloc[:252].isna().any(axis=1).all()
    assert not f.iloc[252:].isna().any().any()
    assert list(f.columns) == list(dp.FEATURE_COLUMNS)


# ------------------------------------------------------------------ validator
def test_clean_series_passes_and_counts_dividend_days():
    df = make_prices(300, seed=9, dividends=[(60, 0.8), (150, 0.9), (240, 0.7)])
    rep = dp.validate_prices(df)
    assert rep.ok, rep.errors
    assert rep.warnings == []
    assert rep.stats["n_dividend_adjustment_days"] == 3
    assert rep.stats["dividend_adjustment_dates"][0] == df.index[60].strftime("%Y-%m-%d")
    assert rep.stats["n_split_like_ratio_steps"] == 0
    assert rep.stats["n_other_ratio_steps"] == 0
    assert rep.stats["n_ratio_rounding_steps"] == 0
    json.dumps(rep.to_dict())  # JSON serializable


def test_validator_duplicate_dates():
    df = make_prices(50)
    df = pd.concat([df, df.iloc[[10]]]).sort_index(kind="stable")
    rep = dp.validate_prices(df)
    assert any("duplicated" in e for e in rep.errors)


def test_validator_unsorted_dates():
    df = make_prices(50).iloc[::-1]
    rep = dp.validate_prices(df)
    assert any("not sorted" in e for e in rep.errors)


def test_validator_nan_values():
    df = make_prices(50)
    df.iloc[7, df.columns.get_loc("Close")] = np.nan
    rep = dp.validate_prices(df)
    assert any("NaN" in e and "Close" in e for e in rep.errors)


def test_validator_nonpositive_prices_and_volume():
    df = make_prices(50)
    df.iloc[7, df.columns.get_loc("Adj Close")] = 0.0
    df.iloc[8, df.columns.get_loc("Volume")] = -1.0
    rep = dp.validate_prices(df)
    assert any("nonpositive 'Adj Close'" in e for e in rep.errors)
    assert any("nonpositive Volume" in e for e in rep.errors)


def test_validator_high_below_low():
    df = make_prices(50)
    df.iloc[5, df.columns.get_loc("High")] = df.iloc[5]["Low"] - 1.0
    rep = dp.validate_prices(df)
    assert any("High < Low" in e for e in rep.errors)


def test_validator_weekend_dates():
    df = make_prices(50)
    dates = df.index.to_list()
    dates[10] = pd.Timestamp("2018-01-06")  # Saturday
    df.index = pd.DatetimeIndex(dates, name="Date")
    rep = dp.validate_prices(df)
    assert any("weekend" in e for e in rep.errors)


def test_validator_large_calendar_gap_is_warning():
    df = make_prices(60)
    df = pd.concat([df.iloc[:30], df.iloc[30:].set_axis(df.index[30:] + pd.Timedelta(days=14))])
    rep = dp.validate_prices(df)
    assert rep.ok
    assert any("calendar gaps" in w for w in rep.warnings)


def test_validator_implausible_return_is_warning():
    df = make_prices(60)
    df.iloc[20:, df.columns.get_loc("Adj Close")] *= 0.6
    df.iloc[20:, df.columns.get_loc("Close")] *= 0.6
    rep = dp.validate_prices(df)
    assert rep.ok
    assert any("exceed" in w for w in rep.warnings)


def test_validator_unexplained_ratio_step_is_warning():
    df = make_prices(60)
    df.iloc[30:, df.columns.get_loc("Adj Close")] *= 0.97  # ratio drops, not a dividend step
    rep = dp.validate_prices(df)
    assert rep.ok
    assert rep.stats["n_other_ratio_steps"] >= 1
    assert any("unexplained" in w for w in rep.warnings)


def test_validator_rounding_level_ratio_steps_are_not_events():
    df = make_prices(60)
    df.iloc[30:, df.columns.get_loc("Adj Close")] *= 1.0 + 1e-6  # float-rounding size step
    rep = dp.validate_prices(df)
    assert rep.ok and rep.warnings == []
    assert rep.stats["n_ratio_rounding_steps"] == 1
    assert rep.stats["n_dividend_adjustment_days"] == 0


def test_validator_split_like_ratio_step_is_warning():
    df = make_prices(60)
    df.iloc[30:, df.columns.get_loc("Adj Close")] *= 2.0  # ratio jumps by factor 2
    rep = dp.validate_prices(df)
    assert rep.stats["n_split_like_ratio_steps"] >= 1
    assert any("split-like" in w for w in rep.warnings)


def test_validator_missing_column_and_bad_index():
    df = make_prices(20)
    rep = dp.validate_prices(df.drop(columns=["Volume"]))
    assert any("missing columns" in e for e in rep.errors)
    rep2 = dp.validate_prices(df.reset_index(drop=True))
    assert any("DatetimeIndex" in e for e in rep2.errors)
    with pytest.raises(ValueError):
        dp.assert_valid(rep)


# ------------------------------------------------------------------ download / load
def test_flatten_columns_and_download_manifest(tmp_path, monkeypatch):
    calls = {}

    def fake_download(ticker, **kwargs):
        calls.update(kwargs, ticker=ticker)
        raw = make_prices(20, start="2024-01-01")
        raw = raw.iloc[:, [0, 1, 2, 3, 4, 5]]
        raw.columns = pd.MultiIndex.from_product([raw.columns, [ticker]], names=["Price", "Ticker"])
        raw.index = raw.index.rename("Date")
        return raw

    monkeypatch.setattr(dp.yf, "download", fake_download)
    path = tmp_path / "SPY_daily.csv"
    out = dp.download_prices("SPY", "2024-01-01", "2024-02-02", path, now=datetime(2026, 11, 15, 12, tzinfo=timezone.utc))
    # yfinance end is exclusive: the request must end the day after the inclusive end date.
    assert calls["end"] == "2024-02-03"
    assert calls["auto_adjust"] is False
    assert isinstance(out.columns, pd.Index) and list(out.columns) == list(dp.RAW_COLUMNS)
    assert len(out) == 20 and out.index.max() <= pd.Timestamp("2024-02-02")
    man = json.loads((tmp_path / "SPY_daily_manifest.json").read_text())
    assert man["rows"] == 20
    assert man["sha256"] == dp.file_sha256(path)
    assert man["download_utc"] == "2026-11-15T12:00:00Z"
    assert man["yfinance_end_argument_exclusive"] == "2024-02-03"
    assert man["auto_adjust"] is False
    assert man["yfinance_version"]
    loaded = dp.load_prices(path)
    pd.testing.assert_frame_equal(loaded, out, check_freq=False)


def test_load_prices_does_not_sort(tmp_path):
    df = make_prices(20).iloc[::-1]
    path = tmp_path / "x.csv"
    df.to_csv(path, index_label="Date", date_format="%Y-%m-%d")
    loaded = dp.load_prices(path)
    assert not loaded.index.is_monotonic_increasing
    assert not dp.validate_prices(loaded).ok
