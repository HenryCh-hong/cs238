"""Walk-forward (expanding-window) out-of-sample forecast archive.

Indexing. Rows of the supervised table are trading days ``d_0 < d_1 < ...``. Row ``i`` holds
the features at ``d_i`` and the label ``r_{d_{i+1}}`` (``return_date = d_{i+1}``).

Origins. The first origin is row ``i0 = min_train_days``; every later row up to the last
labelled row is an origin. Origin ``i`` is a decision at the close of ``d_i``.

Refit schedule. Refits happen at ``f(i) = i0 + floor((i - i0) / retrain_every) * retrain_every``,
anchored at the first origin, so truncating the data at the end never moves a refit. The
model used at origin ``i`` was fit at ``f(i) <= i`` on rows ``0 .. f-1``, i.e. on pairs
``(x_s, r_{d_{s+1}})`` with ``d_{s+1} <= d_f <= d_i``. Its label dates are therefore known
at origin ``i``. Nothing after ``d_i`` enters any forecast at ``d_i``.

Output columns: ``date``, ``free_forecast``, ``rf_forecast``, ``gb_forecast``,
``realized_return``, ``return_date``, ``feature_timestamp``, ``forecast_origin``,
``train_start``, ``train_end``, ``train_label_end``, ``fit_id``, ``n_train``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import tomllib
from joblib import Parallel, delayed
from sklearn.base import clone

from src import data_pipeline as dp
from src.forecast_models import MODEL_KEYS, make_models

ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_COLUMNS: tuple[str, ...] = (
    "date",
    "free_forecast",
    "rf_forecast",
    "gb_forecast",
    "realized_return",
    "return_date",
    "feature_timestamp",
    "forecast_origin",
    "train_start",
    "train_end",
    "train_label_end",
    "fit_id",
    "n_train",
)
_DATE_COLUMNS = ("date", "return_date", "feature_timestamp", "forecast_origin", "train_start", "train_end", "train_label_end")
_FORECAST_COLUMN = {"free": "free_forecast", "rf": "rf_forecast", "gb": "gb_forecast"}

# Smoke-run settings: last SMOKE_ROWS usable rows, first origin after 2/3 of them, refit every 1/6.
SMOKE_ROWS = 1500
SMOKE_RF_TREES = 10


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def config_sha256(cfg: Mapping) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def _hard_check(arc: pd.DataFrame, min_train_days: int | None = None) -> None:
    """Raise AssertionError on any timing or NaN violation. Explicit raise, so it survives -O."""

    def must(cond: pd.Series | bool, what: str) -> None:
        bad = int((~np.asarray(cond, dtype=bool)).sum()) if not isinstance(cond, bool) else (0 if cond else 1)
        if bad:
            raise AssertionError(f"archive check failed: {what} ({bad} rows)")

    if arc.isna().any().any():
        cols = arc.columns[arc.isna().any()].tolist()
        raise AssertionError(f"archive contains NaN in {cols}")
    must(arc["date"] == arc["forecast_origin"], "date == forecast_origin")
    must(arc["feature_timestamp"] <= arc["forecast_origin"], "feature_timestamp <= forecast_origin")
    must(arc["train_end"] < arc["forecast_origin"], "train_end < forecast_origin")
    must(arc["train_label_end"] <= arc["forecast_origin"], "train_label_end <= forecast_origin")
    must(arc["return_date"] > arc["date"], "return_date > date")
    must(arc["n_train"] >= 1, "n_train >= 1")
    if min_train_days is not None:
        must(arc["n_train"] >= int(min_train_days), "n_train >= min_train_days")


def _fit_predict_block(
    f: int,
    rows: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    models: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Fit fresh clones on rows [0, f) (labels up to row f) and predict the origins ``rows``."""
    preds: dict[str, np.ndarray] = {}
    for key in MODEL_KEYS:
        est = clone(models[key])
        est.fit(X[:f], y[:f])
        preds[key] = np.asarray(est.predict(X[rows]), dtype=float)
    return preds


def walk_forward(
    sup: pd.DataFrame,
    models: Mapping[str, object],
    feature_cols: Sequence[str] = dp.FEATURE_COLUMNS,
    *,
    min_train_days: int,
    retrain_every: int,
    target_col: str = dp.TARGET_COLUMN,
    return_date_col: str = dp.RETURN_DATE_COLUMN,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Expanding-window walk-forward archive over a supervised table (see module docstring).

    ``sup`` must be indexed by the feature date (strictly increasing), contain ``feature_cols``,
    ``target_col`` (realized return) and ``return_date_col`` (= the next row's date). ``models``
    maps ``"free"``, ``"rf"``, ``"gb"`` to unfitted scikit-learn estimators. Refit blocks run in
    parallel with joblib when ``n_jobs != 1``; results do not depend on ``n_jobs``.
    """
    if not isinstance(sup.index, pd.DatetimeIndex):
        raise TypeError("sup must be indexed by a DatetimeIndex of feature dates")
    if not (sup.index.is_monotonic_increasing and sup.index.is_unique):
        raise ValueError("sup index must be strictly increasing")
    missing_keys = [k for k in MODEL_KEYS if k not in models]
    if missing_keys:
        raise ValueError(f"models lack keys {missing_keys}")
    i0, R = int(min_train_days), int(retrain_every)
    N = len(sup)
    if i0 < 1 or R < 1:
        raise ValueError("min_train_days and retrain_every must be >= 1")
    if i0 >= N:
        raise ValueError(f"no origins: min_train_days={i0} but only {N} labelled rows")

    idx = pd.DatetimeIndex(sup.index)
    X = sup.loc[:, list(feature_cols)].to_numpy(dtype=float)
    y = sup[target_col].to_numpy(dtype=float)
    rd = pd.DatetimeIndex(sup[return_date_col])
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("features and targets must be finite for every row")
    if N > 1 and not (rd[:-1] == idx[1:]).all():
        raise ValueError("return_date must equal the date of the next row (consecutive trading days)")

    origins = np.arange(i0, N)
    fit_pos = i0 + ((origins - i0) // R) * R
    blocks = [(int(f), origins[fit_pos == f]) for f in np.unique(fit_pos)]
    results = Parallel(n_jobs=n_jobs)(delayed(_fit_predict_block)(f, rows, X, y, models) for f, rows in blocks)

    pred = {k: np.full(len(origins), np.nan) for k in MODEL_KEYS}
    for (f, rows), res in zip(blocks, results):
        pos = rows - i0
        for k in MODEL_KEYS:
            pred[k][pos] = res[k]

    f_idx = fit_pos
    arc = pd.DataFrame(
        {
            "date": idx[origins],
            "free_forecast": pred["free"],
            "rf_forecast": pred["rf"],
            "gb_forecast": pred["gb"],
            "realized_return": y[origins],
            "return_date": rd[origins],
            "feature_timestamp": idx[origins],
            "forecast_origin": idx[origins],
            "train_start": idx[np.zeros(len(origins), dtype=int)],
            "train_end": idx[f_idx - 1],
            "train_label_end": rd[f_idx - 1],
            "fit_id": [f"fit_{idx[f].strftime('%Y-%m-%d')}" for f in f_idx],
            "n_train": f_idx.astype(int),
        }
    )
    arc = arc.loc[:, list(ARCHIVE_COLUMNS)]
    _hard_check(arc, i0)
    return arc.reset_index(drop=True)


def _package_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "yfinance", "joblib"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:  # pragma: no cover
            out[name] = "unknown"
    return out


def build_archive(
    prices: pd.DataFrame,
    cfg: Mapping,
    *,
    n_jobs: int = 1,
    smoke: bool = False,
    smoke_rows: int = SMOKE_ROWS,
    raw_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Validate prices, build the supervised table and run the walk-forward archive.

    Returns (archive frame, manifest dict). The archive is not written; see :func:`write_archive`.
    Smoke mode keeps the last ``smoke_rows`` labelled rows, sets the first origin to 2/3 of them,
    refits every 1/6 of them and uses ``SMOKE_RF_TREES`` trees. It is a plumbing check only.
    """
    fc = cfg["forecast"]
    data = cfg["data"]
    report = dp.validate_prices(prices)
    dp.assert_valid(report)

    start, end = pd.Timestamp(data["start"]), pd.Timestamp(data["end"])
    px = prices.loc[(prices.index >= start) & (prices.index <= end)]
    sup_full = dp.make_supervised(px)
    run_cfg = copy.deepcopy(dict(cfg))
    sup = sup_full
    min_train = int(fc["min_train_days"])
    retrain = int(fc["retrain_every"])
    if smoke:
        if smoke_rows <= 2:
            raise ValueError("smoke_rows too small")
        sup = sup_full.iloc[-int(smoke_rows):]
        min_train = int(smoke_rows) * 2 // 3
        retrain = max(1, int(smoke_rows) // 6)
        run_cfg["forecast"]["random_forest"]["n_estimators"] = SMOKE_RF_TREES
        run_cfg["forecast"]["min_train_days"] = min_train
        run_cfg["forecast"]["retrain_every"] = retrain

    models = make_models(run_cfg, n_jobs=1)
    arc = walk_forward(
        sup,
        models,
        dp.FEATURE_COLUMNS,
        min_train_days=min_train,
        retrain_every=retrain,
        n_jobs=n_jobs,
    )
    corr = {
        key: float(np.corrcoef(arc[col].to_numpy(float), arc["realized_return"].to_numpy(float))[0, 1])
        for key, col in _FORECAST_COLUMN.items()
    }
    manifest = {
        "smoke": bool(smoke),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": run_cfg,
        "config_sha256": config_sha256(run_cfg),
        "full_config_sha256": config_sha256(cfg),
        "raw_data_sha256": raw_sha256,
        "package_versions": _package_versions(),
        "validation": report.to_dict(),
        "n_jobs_walk_forward": int(n_jobs),
        "min_train_days_used": int(min_train),
        "retrain_every_used": int(retrain),
        "rf_n_estimators_used": int(run_cfg["forecast"]["random_forest"]["n_estimators"]),
        "row_counts": {
            "price_rows_in_window": int(len(px)),
            "labelled_feature_rows_full": int(len(sup_full)),
            "labelled_feature_rows_used": int(len(sup)),
            "archive_rows": int(len(arc)),
            "refits": int(arc["fit_id"].nunique()),
        },
        "period": {
            "feature_first_date": pd.Timestamp(sup.index[0]).strftime("%Y-%m-%d"),
            "feature_last_date": pd.Timestamp(sup.index[-1]).strftime("%Y-%m-%d"),
            "first_origin": arc["forecast_origin"].iloc[0].strftime("%Y-%m-%d"),
            "last_origin": arc["forecast_origin"].iloc[-1].strftime("%Y-%m-%d"),
            "first_return_date": arc["return_date"].iloc[0].strftime("%Y-%m-%d"),
            "last_return_date": arc["return_date"].iloc[-1].strftime("%Y-%m-%d"),
        },
        "diagnostics_not_for_selection": {
            "corr_forecast_with_realized_return": corr,
            "label": "smoke-run plumbing check" if smoke else "descriptive only",
        },
    }
    return arc, manifest


def write_archive(arc: pd.DataFrame, manifest: dict, csv_path: str | Path) -> Path:
    """Write the archive CSV and its manifest (``<stem>_manifest.json``) with the CSV hash."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = arc.copy()
    for c in _DATE_COLUMNS:
        out[c] = pd.DatetimeIndex(out[c]).strftime("%Y-%m-%d")
    out.to_csv(csv_path, index=False)
    manifest = dict(manifest)
    manifest["archive_csv"] = csv_path.name
    manifest["archive_csv_sha256"] = dp.file_sha256(csv_path)
    manifest["archive_columns"] = list(ARCHIVE_COLUMNS)
    dp.manifest_path_for(csv_path).write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return csv_path
