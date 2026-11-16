"""Integration tests for the historical replay (Gates C, D, E) on SYNTHETIC archives."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("src.policies")

from experiments.run_replay import POLICY_ORDER, make_policies, run  # noqa: E402
from src.backtest import BeliefSchedule, ReplaySettings, replay, validate_archive  # noqa: E402
from src.config import load_config, with_overrides  # noqa: E402
from src.evaluation import block_bootstrap, calibration_table, paired_differences, voi_counterfactuals  # noqa: E402
from src.synthetic_archive import make_synthetic_archive, spy_like_model  # noqa: E402
from src.utility import UtilityParams  # noqa: E402

N_ARCHIVE = 300
N_EVAL = 40


@pytest.fixture(scope="module")
def cfg():
    # cheap costs so that VOI-based policies do buy sometimes; smaller quadrature for speed
    return with_overrides(load_config(), decision={"cost_A": 2e-6, "cost_B": 2e-6, "lam": 5.0, "n_quad": 41})


@pytest.fixture(scope="module")
def archive():
    model = spy_like_model(rho_y0=0.05, rho_A=0.25, rho_B=0.2)
    return make_synthetic_archive(model, N_ARCHIVE, seed=7)


@pytest.fixture(scope="module")
def result(cfg, archive):
    return run(cfg, tag="test", archive=archive, max_dates=N_EVAL, save=False, progress_every=0, n_boot=50)


def test_all_policies_same_dates_and_no_nans(result):
    log = result[0]
    assert set(log["policy"]) == set(POLICY_ORDER)
    dates = log.groupby("policy")["date"].apply(lambda s: tuple(s))
    assert len(set(dates)) == 1 and len(dates.iloc[0]) == N_EVAL
    for c in ("q", "net_utility", "gross_utility", "transaction_cost", "acquisition_cost", "m_prior", "v_prior"):
        assert np.isfinite(log[c]).all(), c


def test_accounting_reconciles(result, cfg):
    log = result[0]
    recon = log["gross_utility"] - log["transaction_cost"] - log["acquisition_cost"]
    np.testing.assert_allclose(recon, log["net_utility"], atol=1e-15, rtol=0)
    d = cfg["decision"]
    np.testing.assert_allclose(log["acquisition_cost"], log["bought_A"] * d["cost_A"] + log["bought_B"] * d["cost_B"],
                               atol=1e-15, rtol=0)
    p = UtilityParams(d["lam"], d["tau"])
    np.testing.assert_allclose(log["gross_utility"], log["q"] * log["realized_return"]
                               - p.lam * log["q"] ** 2 * log["realized_return"] ** 2, atol=1e-15, rtol=0)
    np.testing.assert_allclose(log["transaction_cost"], p.tau * (log["q"] - log["q_prev"]).abs(), atol=1e-15, rtol=0)


def test_policy_rules(result):
    log = result[0]
    g = log.groupby("policy")
    assert g.get_group("free_only")["n_purchases"].eq(0).all()
    assert g.get_group("always_buy")["n_purchases"].eq(2).all()  # budget is non-binding by default
    assert log["n_purchases"].between(0, 2).all()
    assert (log["acquisition_cost"] >= 0).all()


def test_each_policy_keeps_its_own_position_history(result):
    log = result[0]
    for pol, g in log.groupby("policy"):
        g = g.sort_values("date")
        assert g["q_prev"].iloc[0] == 0
        np.testing.assert_array_equal(g["q_prev"].to_numpy()[1:], g["q"].to_numpy()[:-1])


def test_revealed_values_only_when_purchased(result, archive):
    log = result[0].merge(archive[["date", "rf_forecast", "gb_forecast"]], on="date")
    for s, col in (("A", "rf_forecast"), ("B", "gb_forecast")):
        bought = log[f"bought_{s}"] == 1
        assert log.loc[~bought, f"revealed_{s}"].isna().all()
        np.testing.assert_array_equal(log.loc[bought, f"revealed_{s}"], log.loc[bought, col])


def test_positions_maximize_posterior_expected_utility(result, cfg):
    log = result[0]
    d = cfg["decision"]
    p = UtilityParams(d["lam"], d["tau"])
    for row in log.itertuples(index=False):
        m, v = row.m_post, row.v_post
        eu = {q: q * m - p.lam * q * q * (m * m + v) - p.tau * abs(q - row.q_prev) for q in (-1, 0, 1)}
        assert eu[row.q] >= max(eu.values()) - 1e-15


def test_budget_binding_limits_purchases(cfg, archive):
    tight = with_overrides(cfg, decision={"budget": cfg["decision"]["cost_A"]})
    log = run(tight, archive=archive, max_dates=15, save=False, progress_every=0, n_boot=20)[0]
    assert log["n_purchases"].le(1).all()
    assert (log["acquisition_cost"] <= tight["decision"]["budget"] + 1e-15).all()
    ab = log[log["policy"] == "always_buy"]
    assert ab["bought_A"].eq(1).all() and ab["bought_B"].eq(0).all()


def test_replay_is_reproducible(cfg, archive):
    a = run(cfg, archive=archive, max_dates=12, save=False, progress_every=0, n_boot=20)[0]
    b = run(cfg, archive=archive, max_dates=12, save=False, progress_every=0, n_boot=20)[0]
    cols = [c for c in a.columns if c != "decision_seconds"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_no_lookahead_future_data_cannot_change_past_decisions(cfg, archive):
    """Perturb every forecast and return dated after a cutoff: decisions up to the cutoff
    (and their realized utilities, which use r_{t+1} <= cutoff's return) must be identical."""
    base = run(cfg, archive=archive, max_dates=30, save=False, progress_every=0, n_boot=20)[0]
    eval_dates = sorted(base["date"].unique())
    cutoff = eval_dates[14]
    pert = archive.copy()
    rng = np.random.default_rng(123)
    later = pert["date"] > cutoff
    for c in ("free_forecast", "rf_forecast", "gb_forecast", "realized_return"):
        pert.loc[later, c] = pert.loc[later, c] + rng.normal(0, 0.05, later.sum())
    new = run(cfg, archive=pert, max_dates=30, save=False, progress_every=0, n_boot=20)[0]
    cols = ["policy", "date", "q", "purchases", "m_prior", "v_prior", "m_post", "net_utility"]
    a = base[base["date"] <= cutoff][cols].reset_index(drop=True)
    b = new[new["date"] <= cutoff][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("mode", ["fixed", "expanding"])
def test_belief_schedule_never_uses_unrealized_returns(archive, mode):
    a = validate_archive(archive)
    cal_end = a["date"].iloc[150]
    sch = BeliefSchedule(archive=a, calibration_end=cal_end, mode=mode, refit_every=5)
    eval_rows = a[a["date"] > cal_end].iloc[:20]
    for i, t in enumerate(eval_rows["date"]):
        sch.model_for(t, i)
    assert sch.fit_log
    for f in sch.fit_log:
        assert f["last_return_date"] <= f["decision_date"]
        if mode == "fixed":
            assert f["last_row"] <= cal_end


def test_expanding_refit_invariant_to_future_returns(archive):
    a = validate_archive(archive)
    t = a["date"].iloc[200]
    s1 = BeliefSchedule(archive=a, calibration_end=a["date"].iloc[150], mode="expanding")
    m1 = s1.model_for(t, 0)
    b = a.copy()
    b.loc[b["return_date"] > t, "realized_return"] += 1.0
    s2 = BeliefSchedule(archive=b, calibration_end=a["date"].iloc[150], mode="expanding")
    m2 = s2.model_for(t, 0)
    np.testing.assert_array_equal(m1.mean, m2.mean)
    np.testing.assert_array_equal(m1.cov, m2.cov)


def test_validate_archive_rejects_leaky_rows(archive):
    bad = archive.copy()
    bad.loc[5, "train_end"] = bad.loc[5, "date"]
    with pytest.raises(ValueError):
        validate_archive(bad)
    bad = archive.copy()
    bad.loc[7, "return_date"] = bad.loc[7, "date"]
    with pytest.raises(ValueError):
        validate_archive(bad)
    bad = archive.copy()
    bad.loc[3, "rf_forecast"] = np.nan
    with pytest.raises(ValueError):
        validate_archive(bad)


def test_paired_difference_of_baseline_is_zero(result):
    pdiff = paired_differences(result[0], n_boot=50)
    row = pdiff[pdiff["policy"] == "free_only"].iloc[0]
    assert row["mean_diff_bp"] == 0 and row["ci_lo_bp"] == 0 and row["ci_hi_bp"] == 0


def test_block_bootstrap_covers_known_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1.0, 400)
    ci = block_bootstrap(x, np.mean, block=10, n_boot=500, seed=1)
    assert ci["lo"] < 1.0 < ci["hi"]


def test_voi_calibration_well_specified_world():
    """When the archive is drawn from the belief law itself, the mean realized paired
    improvement must agree with the mean predicted gross VOI (within a bootstrap CI)."""
    truth = spy_like_model(rho_y0=0.05, rho_A=0.35, rho_B=0.3, corr_AB=0.5)
    arch = validate_archive(make_synthetic_archive(truth, 6000, seed=11))
    sch = BeliefSchedule(archive=arch, calibration_end=arch["date"].iloc[0], fit_fn=lambda X, method=None: truth)
    p = UtilityParams(lam=5.0, tau=1e-4)
    cf = voi_counterfactuals(arch, sch, {}, p)
    for arm in ("A", "B", "AB"):
        pred = cf[f"pred_gross_{arm}"].to_numpy()
        resid = cf[f"real_impr_{arm}"].to_numpy() - pred
        assert pred.mean() > 0
        se = resid.std(ddof=1) / np.sqrt(len(resid))  # synthetic draws are iid
        assert abs(resid.mean()) <= 3 * se, (arm, resid.mean(), se, pred.mean())
        tab = calibration_table(cf, arm=arm, n_bins=4, block=1, n_boot=100, seed=3)
        assert len(tab) == 5 and np.isfinite(tab["real_mean_bp"]).all()
