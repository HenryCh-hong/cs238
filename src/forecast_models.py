"""Forecast models for the walk-forward archive.

* ``free``: ridge regression on standardized features (the linear forecast y0). The
  StandardScaler lives inside the Pipeline, so it only ever sees training rows.
* ``rf``:   random forest regressor (source "A").
* ``gb``:   gradient boosting regressor (source "B").

Hyperparameters are read from ``configs/default.toml`` and fixed a priori: nothing is tuned
on the data (tuning on the realized sample would be look-ahead). Every model is cloned and
refit from scratch at each refit origin; no state is carried between fits.
"""
from __future__ import annotations

from typing import Mapping

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_KEYS: tuple[str, ...] = ("free", "rf", "gb")


def make_models(cfg: Mapping, n_jobs: int = 1) -> dict:
    """Return unfitted estimators ``{"free", "rf", "gb"}`` built from ``cfg["forecast"]``.

    ``n_jobs`` is passed to the random forest only (the gradient boosting and ridge fits are
    single-threaded). Use ``n_jobs=1`` when the walk-forward itself runs in parallel, to avoid
    nested oversubscription.
    """
    fc = cfg["forecast"]
    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(fc["ridge"]["alpha"]))),
        ]
    )
    rf_cfg = fc["random_forest"]
    rf = RandomForestRegressor(
        n_estimators=int(rf_cfg["n_estimators"]),
        max_depth=int(rf_cfg["max_depth"]),
        min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
        max_features=float(rf_cfg["max_features"]),
        random_state=int(rf_cfg["random_state"]),
        n_jobs=int(n_jobs),
    )
    gb_cfg = fc["gradient_boosting"]
    gb = GradientBoostingRegressor(
        n_estimators=int(gb_cfg["n_estimators"]),
        max_depth=int(gb_cfg["max_depth"]),
        learning_rate=float(gb_cfg["learning_rate"]),
        subsample=float(gb_cfg["subsample"]),
        random_state=int(gb_cfg["random_state"]),
    )
    return {"free": ridge, "rf": rf, "gb": gb}
