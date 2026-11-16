"""SYNTHETIC forecast archives drawn from a known joint Gaussian law.

These are NOT market data. They exist to (1) integration-test the replay before
or without the real SPY archive, and (2) provide a correctly specified world in
which the belief model equals the truth, so the replay's predicted values of
information can be checked against realized averages. Every synthetic archive
carries ``synthetic = True`` and is never written to the real archive path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.belief_model import GaussianJointModel


def spy_like_model(rho_y0: float = 0.03, rho_A: float = 0.06, rho_B: float = 0.05, corr_y0A: float = 0.5,
                   corr_y0B: float = 0.5, corr_AB: float = 0.6, sd_r: float = 0.011, sd_f: float = 0.001,
                   mean_r: float = 4e-4) -> GaussianJointModel:
    """A weak-signal joint law with SPY-like scale (daily sd ~1.1%, drift ~4 bp)."""
    R = np.array([
        [1.0, rho_y0, rho_A, rho_B],
        [rho_y0, 1.0, corr_y0A, corr_y0B],
        [rho_A, corr_y0A, 1.0, corr_AB],
        [rho_B, corr_y0B, corr_AB, 1.0],
    ])
    if np.linalg.eigvalsh(R).min() <= 0:
        raise ValueError("correlation matrix is not positive definite")
    sd = np.array([sd_r, sd_f, sd_f, sd_f])
    mean = np.array([mean_r, mean_r, mean_r, mean_r])
    return GaussianJointModel(mean=mean, cov=R * np.outer(sd, sd), meta={"synthetic": True})


def make_synthetic_archive(model: GaussianJointModel, n: int, seed: int, start: str = "2000-01-03") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = model.sample(n + 0, rng)
    dates = pd.bdate_range(start, periods=n + 1)
    df = pd.DataFrame({
        "date": dates[:-1],
        "free_forecast": X[:, 1],
        "rf_forecast": X[:, 2],
        "gb_forecast": X[:, 3],
        "realized_return": X[:, 0],
        "return_date": dates[1:],
    })
    df["feature_timestamp"] = df["date"]
    df["forecast_origin"] = df["date"]
    df["train_start"] = df["date"].iloc[0] - pd.Timedelta(days=3650)
    df["train_end"] = df["date"] - pd.Timedelta(days=1)
    df["train_label_end"] = df["date"]
    df["fit_id"] = -1
    df["n_train"] = 0
    df["synthetic"] = True
    return df
