"""Tests for src/forecast_archive.py and src/forecast_models.py on synthetic prices (no network).

Tiny models (5 trees) and short schedules keep these tests fast; the timing assertions are
the same ones the full build runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src import data_pipeline as dp
from src.forecast_archive import (
    ARCHIVE_COLUMNS,
    _hard_check,
    build_archive,
    load_config,
    walk_forward,
    write_archive,
)
from src.forecast_models import MODEL_KEYS, make_models
from tests.test_data_pipeline import make_prices

ROOT = Path(__file__).resolve().parents[1]
MIN_TRAIN = 100
RETRAIN = 25


def tiny_cfg() -> dict:
    cfg = load_config(ROOT / "configs" / "default.toml")
    cfg["forecast"]["random_forest"]["n_estimators"] = 5
    cfg["forecast"]["gradient_boosting"]["n_estimators"] = 5
    cfg["forecast"]["min_train_days"] = MIN_TRAIN
    cfg["forecast"]["retrain_every"] = RETRAIN
    return cfg


def tiny_models(n_jobs: int = 1) -> dict:
    return make_models(tiny_cfg(), n_jobs=n_jobs)


def run(sup: pd.DataFrame, n_jobs: int = 1) -> pd.DataFrame:
    return walk_forward(sup, tiny_models(), dp.FEATURE_COLUMNS, min_train_days=MIN_TRAIN, retrain_every=RETRAIN, n_jobs=n_jobs)


@pytest.fixture(scope="module")
def sup520() -> pd.DataFrame:
    return dp.make_supervised(make_prices(520, seed=11))


# --------------------------------------------------------------------------- models
def test_make_models_follow_config():
    cfg = load_config(ROOT / "configs" / "default.toml")
    m = make_models(cfg, n_jobs=1)
    assert tuple(m) == MODEL_KEYS
    assert m["free"].named_steps["ridge"].alpha == cfg["forecast"]["ridge"]["alpha"]
    assert m["free"].steps[0][0] == "scaler"  # scaler is inside the pipeline
    rf = m["rf"]
    assert rf.n_estimators == cfg["forecast"]["random_forest"]["n_estimators"]
    assert rf.max_depth == cfg["forecast"]["random_forest"]["max_depth"]
    assert rf.min_samples_leaf == cfg["forecast"]["random_forest"]["min_samples_leaf"]
    assert rf.max_features == cfg["forecast"]["random_forest"]["max_features"]
    assert rf.random_state == cfg["forecast"]["random_forest"]["random_state"]
    assert rf.n_jobs == 1
    gb = m["gb"]
    assert gb.n_estimators == cfg["forecast"]["gradient_boosting"]["n_estimators"]
    assert gb.learning_rate == cfg["forecast"]["gradient_boosting"]["learning_rate"]
    assert gb.subsample == cfg["forecast"]["gradient_boosting"]["subsample"]
    assert make_models(cfg, n_jobs=3)["rf"].n_jobs == 3


# ------------------------------------------------------------------- timing contract
def test_walk_forward_columns_origins_and_timestamps(sup520):
    arc = run(sup520)
    assert list(arc.columns) == list(ARCHIVE_COLUMNS)
    assert len(arc) == len(sup520) - MIN_TRAIN
    assert not arc.isna().any().any()
    assert (arc["date"] == arc["forecast_origin"]).all()
    assert (arc["train_end"] < arc["forecast_origin"]).all()
    assert (arc["train_label_end"] <= arc["forecast_origin"]).all()
    assert (arc["return_date"] > arc["date"]).all()
    assert (arc["feature_timestamp"] <= arc["forecast_origin"]).all()
    # realized return of origin t is the label of row t, dated t+1
    assert arc["date"].iloc[0] == sup520.index[MIN_TRAIN]
    assert arc["return_date"].iloc[0] == sup520.index[MIN_TRAIN + 1]
    assert arc["realized_return"].iloc[0] == pytest.approx(sup520[dp.TARGET_COLUMN].iloc[MIN_TRAIN])


def test_hard_check_raises_on_violations(sup520):
    arc = run(sup520)
    _hard_check(arc, MIN_TRAIN)  # passes on the real output
    bad = arc.copy()
    bad.loc[bad.index[3], "train_end"] = bad.loc[bad.index[3], "forecast_origin"]
    with pytest.raises(AssertionError, match="train_end"):
        _hard_check(bad)
    bad = arc.copy()
    bad.loc[bad.index[4], "train_label_end"] = bad.loc[bad.index[4], "forecast_origin"] + pd.Timedelta(days=1)
    with pytest.raises(AssertionError, match="train_label_end"):
        _hard_check(bad)
    bad = arc.copy()
    bad.loc[bad.index[5], "return_date"] = bad.loc[bad.index[5], "date"]
    with pytest.raises(AssertionError, match="return_date"):
        _hard_check(bad)
    bad = arc.copy()
    bad.loc[bad.index[6], "rf_forecast"] = np.nan
    with pytest.raises(AssertionError, match="NaN"):
        _hard_check(bad)


def test_refit_schedule_is_anchored_and_expanding(sup520):
    arc = run(sup520)
    pos = np.arange(len(arc))
    fit_pos = MIN_TRAIN + (pos // RETRAIN) * RETRAIN  # anchored at the first origin
    # n_train equals the refit row, which grows over time
    np.testing.assert_array_equal(arc["n_train"].to_numpy(), fit_pos)
    assert (np.diff(arc["n_train"].to_numpy()) >= 0).all()
    assert arc["n_train"].iloc[0] == MIN_TRAIN
    assert arc["n_train"].nunique() == int(np.ceil(len(arc) / RETRAIN))
    # within a block: one fit_id; train_end and train_label_end are the fit row's neighbours
    for fid, block in arc.groupby("fit_id", sort=False):
        f = int(block["n_train"].iloc[0])
        assert (block["n_train"] == f).all()
        assert block["train_start"].iloc[0] == sup520.index[0]
        assert block["train_end"].iloc[0] == sup520.index[f - 1]
        assert block["train_label_end"].iloc[0] == sup520.index[f]
        assert fid == f"fit_{sup520.index[f].strftime('%Y-%m-%d')}"


def test_forecast_equals_manual_fit_on_past_rows_only(sup520):
    """Recompute one origin by hand: fit on rows [0, f) only, predict row i."""
    arc = run(sup520)
    X = sup520[list(dp.FEATURE_COLUMNS)].to_numpy(float)
    y = sup520[dp.TARGET_COLUMN].to_numpy(float)
    models = tiny_models()
    i = MIN_TRAIN + 2 * RETRAIN + 7  # third block, not a refit origin
    f = MIN_TRAIN + 2 * RETRAIN
    row = arc.loc[arc["date"] == sup520.index[i]].iloc[0]
    for key, col in (("free", "free_forecast"), ("rf", "rf_forecast"), ("gb", "gb_forecast")):
        est = clone(models[key]).fit(X[:f], y[:f])
        assert row[col] == pytest.approx(est.predict(X[i : i + 1])[0], rel=0, abs=1e-15)


def test_parallel_refit_blocks_match_serial(sup520):
    a = run(sup520, n_jobs=1)
    b = run(sup520, n_jobs=2)
    pd.testing.assert_frame_equal(a, b)


def test_bad_inputs_are_rejected(sup520):
    shuffled = sup520.iloc[::-1]
    with pytest.raises(ValueError, match="strictly increasing"):
        run(shuffled)
    with pytest.raises(ValueError, match="no origins"):
        walk_forward(sup520.iloc[:MIN_TRAIN], tiny_models(), min_train_days=MIN_TRAIN, retrain_every=RETRAIN)
    broken = sup520.copy()
    broken.iloc[5, broken.columns.get_loc("return_date")] = sup520.index[7]
    with pytest.raises(ValueError, match="next row"):
        run(broken)


# ------------------------------------------------------------ information invariance
def test_forecasts_unchanged_when_future_prices_are_perturbed():
    prices = make_prices(520, seed=12)
    sup = dp.make_supervised(prices)
    t = sup.index[150]
    arc = run(sup)
    rng = np.random.default_rng(3)
    pert = prices.copy()
    after = pert.index > t
    shocks = np.exp(rng.normal(0.0, 0.08, after.sum()))
    for c in dp.RAW_COLUMNS:
        pert.loc[after, c] = pert.loc[after, c].to_numpy() * shocks
    sup_pert = dp.make_supervised(pert)
    assert sup_pert.index.equals(sup.index)
    arc_pert = run(sup_pert)
    keep = arc["forecast_origin"] <= t
    cols = ["free_forecast", "rf_forecast", "gb_forecast", "n_train", "fit_id", "train_end", "train_label_end"]
    pd.testing.assert_frame_equal(arc.loc[keep, cols].reset_index(drop=True), arc_pert.loc[keep, cols].reset_index(drop=True))


def test_forecasts_unchanged_when_future_rows_are_removed():
    prices = make_prices(520, seed=13)
    sup = dp.make_supervised(prices)
    t_pos = 150
    t = sup.index[t_pos]
    arc = run(sup)
    arc_trunc = run(sup.iloc[: t_pos + 1])
    keep = arc["forecast_origin"] <= t
    assert keep.sum() == len(arc_trunc)
    cols = ["free_forecast", "rf_forecast", "gb_forecast", "n_train", "fit_id"]
    pd.testing.assert_frame_equal(arc.loc[keep, cols].reset_index(drop=True), arc_trunc[cols])


def test_schedule_does_not_move_when_data_is_truncated():
    sup = dp.make_supervised(make_prices(520, seed=14))
    full_fit = run(sup)[["forecast_origin", "fit_id"]]
    short_fit = run(sup.iloc[:200])[["forecast_origin", "fit_id"]]
    pd.testing.assert_frame_equal(full_fit.iloc[: len(short_fit)].reset_index(drop=True), short_fit.reset_index(drop=True))


# ------------------------------------------------------------------ build and write
def test_build_archive_smoke_and_full_write(tmp_path):
    prices = make_prices(600, seed=15)
    cfg = tiny_cfg()
    cfg["data"] = {"ticker": "SPY", "start": "1993-01-29", "end": "2026-11-13", "raw_path": "unused.csv"}

    arc_s, man_s = build_archive(prices, cfg, smoke=True, smoke_rows=120, n_jobs=1)
    assert man_s["smoke"] is True
    assert man_s["min_train_days_used"] == 80 and man_s["retrain_every_used"] == 20
    assert len(arc_s) == 40
    assert man_s["rf_n_estimators_used"] == 10
    csv_s = write_archive(arc_s, man_s, tmp_path / "forecast_archive_smoke.csv")
    assert csv_s.name == "forecast_archive_smoke.csv"
    man_file = json.loads((tmp_path / "forecast_archive_smoke_manifest.json").read_text())
    assert man_file["smoke"] is True
    assert man_file["archive_csv_sha256"] == dp.file_sha256(csv_s)
    assert man_file["row_counts"]["archive_rows"] == 40
    assert man_file["validation"]["ok"] is True
    back = pd.read_csv(csv_s)
    assert list(back.columns) == list(ARCHIVE_COLUMNS)
    assert back["date"].iloc[0] == arc_s["date"].iloc[0].strftime("%Y-%m-%d")

    arc_f, man_f = build_archive(prices, cfg, smoke=False, n_jobs=1)
    assert man_f["smoke"] is False
    sup = dp.make_supervised(prices)
    assert len(arc_f) == len(sup) - MIN_TRAIN
    write_archive(arc_f, man_f, tmp_path / "forecast_archive.csv")
    assert (tmp_path / "forecast_archive_manifest.json").exists()
    assert json.loads((tmp_path / "forecast_archive_manifest.json").read_text())["smoke"] is False


def test_build_archive_refuses_invalid_prices():
    prices = make_prices(400, seed=16).iloc[::-1]
    with pytest.raises(ValueError, match="validation"):
        build_archive(prices, tiny_cfg())
