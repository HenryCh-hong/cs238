"""Tests for the joint Gaussian belief model (src/belief_model.py).

Reference implementations here are deliberately independent of the module: conditioning
uses explicit block solves (not the pseudo-inverse), and regression uses least squares on
a large seeded sample. Runtime is kept well under 20 s.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf, ledoit_wolf_shrinkage

from src.belief_model import (
    GaussianBelief,
    GaussianJointModel,
    conditional_summary,
    ewma_volatility,
    fit_from_frame,
    fit_joint_gaussian,
    fit_standardized_model,
    scale_model,
)

NAMES = ("r", "y0", "A", "B")
SD = np.array([0.010, 0.012, 0.008, 0.011])
CORR = np.array(
    [
        [1.00, 0.30, 0.50, 0.40],
        [0.30, 1.00, 0.45, 0.20],
        [0.50, 0.45, 1.00, 0.60],
        [0.40, 0.20, 0.60, 1.00],
    ]
)
MEAN = np.array([0.0004, 0.0002, 0.0003, 0.0005])


def fixed_model() -> GaussianJointModel:
    return GaussianJointModel(MEAN, CORR * np.outer(SD, SD), NAMES)


def schur_reference(mean, cov, names, obs):
    """Independent Gaussian conditioning: explicit block formulas with linear solves."""
    io = [names.index(k) for k in obs]
    iu = [i for i in range(len(names)) if names[i] not in obs]
    yo = np.array([obs[k] for k in obs], dtype=float)
    Soo = cov[np.ix_(io, io)]
    Suo = cov[np.ix_(iu, io)]
    Suu = cov[np.ix_(iu, iu)]
    mu_u = mean[iu] + Suo @ np.linalg.solve(Soo, yo - mean[io])
    Su = Suu - Suo @ np.linalg.solve(Soo, Suo.T)
    return mu_u, Su, [names[i] for i in iu]


def random_correlation(rng, p=4):
    G = rng.standard_normal((p, p))
    C = G @ G.T + 0.5 * np.eye(p)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = float(res @ res) / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return beta, se, s2


# ----------------------------------------------------------------- conditional moments
@pytest.mark.parametrize("obs_names", [(), ("y0",), ("A",), ("B",), ("y0", "A"), ("A", "B"), ("y0", "A", "B")])
def test_conditional_moments_match_schur_reference(obs_names):
    model = fixed_model()
    obs = {k: 0.37 * SD[model.index(k)] + MEAN[model.index(k)] for k in obs_names}
    b = model.belief(obs)
    mu_ref, S_ref, unobs = schur_reference(model.mean, model.cov, model.names, obs)
    assert b.unobserved == tuple(unobs)
    np.testing.assert_allclose(b._mu_u, mu_ref, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(b.conditional_cov, S_ref, rtol=1e-9, atol=1e-18)
    assert b.mean == pytest.approx(mu_ref[0], rel=1e-12, abs=1e-16)
    assert b.var == pytest.approx(S_ref[0, 0], rel=1e-9, abs=1e-18)


def test_conditional_moments_random_models_match_schur():
    rng = np.random.default_rng(7)
    for _ in range(25):
        C = random_correlation(rng)
        sd = rng.uniform(0.002, 0.02, size=4)
        cov = C * np.outer(sd, sd)
        model = GaussianJointModel(rng.normal(0, 1e-3, 4), cov, NAMES)
        for k in range(1, 4):
            for subset in combinations(("y0", "A", "B"), k):
                obs = {s: rng.normal(0, 0.02) for s in subset}
                b = model.belief(obs)
                mu_ref, S_ref, _ = schur_reference(model.mean, model.cov, model.names, obs)
                np.testing.assert_allclose(b._mu_u, mu_ref, rtol=1e-9, atol=1e-16)
                np.testing.assert_allclose(b.conditional_cov, S_ref, rtol=1e-8, atol=1e-18)


def test_conditional_moments_match_least_squares_on_large_sample():
    model = fixed_model()
    rng = np.random.default_rng(20261115)
    N = 200_000
    X = model.sample(N, rng)
    r, y0, A, B = X.T

    # Regress r on (y0, A): OLS coefficients must equal the Schur gain matrix.
    b_yA = model.belief({"y0": 0.0, "A": 0.0})
    Soo = model.cov[np.ix_([1, 2], [1, 2])]
    slope = np.linalg.solve(Soo, model.cov[[1, 2], 0])
    intercept = model.mean[0] - slope @ model.mean[[1, 2]]
    beta, se, s2 = ols(r, np.column_stack([np.ones(N), y0, A]))
    np.testing.assert_array_less(np.abs(beta - np.r_[intercept, slope]), 5 * se)
    # The residual variance of r is the (data-independent) posterior variance.
    assert s2 == pytest.approx(b_yA.var, rel=0.02)

    # Predict B from y0 alone: OLS intercept and slope against the belief's predictive mean.
    beta_B, se_B, s2_B = ols(B, np.column_stack([np.ones(N), y0]))
    v = 0.012
    b_y0 = model.belief({"y0": v})
    mu_B, var_B = b_y0.predictive("B")
    slope_B = model.cov[3, 1] / model.cov[1, 1]
    intercept_B = model.mean[3] - slope_B * model.mean[1]
    assert beta_B[0] == pytest.approx(intercept_B, abs=5 * se_B[0])
    assert beta_B[1] == pytest.approx(slope_B, abs=5 * se_B[1])
    assert intercept_B + slope_B * v == pytest.approx(mu_B, rel=1e-12, abs=1e-16)
    var_B_given_y0 = model.cov[3, 3] - model.cov[3, 1] ** 2 / model.cov[1, 1]
    assert var_B == pytest.approx(var_B_given_y0, rel=1e-12)
    assert s2_B == pytest.approx(var_B_given_y0, rel=0.02)


def test_sampling_moments_match_model():
    model = fixed_model()
    rng = np.random.default_rng(11)
    N = 200_000
    X = model.sample(N, rng)
    se_mean = np.sqrt(np.diag(model.cov) / N)
    np.testing.assert_array_less(np.abs(X.mean(axis=0) - model.mean), 5 * se_mean)
    S_emp = np.cov(X, rowvar=False)
    se_cov = np.sqrt((np.outer(np.diag(model.cov), np.diag(model.cov)) + model.cov**2) / N)
    np.testing.assert_array_less(np.abs(S_emp - model.cov), 5 * se_cov)


# ------------------------------------------------------------------- order independence
def test_order_independence_two_acquisitions():
    model = fixed_model()
    a, bval, v = 0.0021, -0.0016, 0.0009
    base = model.belief({"y0": v})
    ab = base.condition("A", a).condition("B", bval)
    ba = base.condition("B", bval).condition("A", a)
    joint = model.belief({"y0": v, "A": a, "B": bval})
    for other in (ba, joint):
        assert ab.mean == pytest.approx(other.mean, rel=1e-12, abs=1e-18)
        assert ab.var == pytest.approx(other.var, rel=1e-12, abs=1e-18)
        np.testing.assert_allclose(ab.conditional_cov, other.conditional_cov, rtol=1e-12, atol=1e-20)


# -------------------------------------------------------------------- Delta identities
@pytest.mark.parametrize("y0_obs", [None, 0.0031])
def test_gain_identity_two_sources(y0_obs):
    model = fixed_model()
    base_obs = {} if y0_obs is None else {"y0": y0_obs}
    b0 = model.belief(base_obs)
    a, bval = 0.0012, -0.0025
    bA = b0.condition("A", a)
    bB = b0.condition("B", bval)
    bAB = bA.condition("B", bval)
    lhs1 = b0.gain("A") + bA.gain("B")
    lhs2 = b0.gain("B") + bB.gain("A")
    rhs = b0.var - bAB.var
    assert lhs1 == pytest.approx(rhs, rel=1e-10, abs=1e-20)
    assert lhs2 == pytest.approx(rhs, rel=1e-10, abs=1e-20)
    assert lhs1 == pytest.approx(lhs2, rel=1e-12, abs=1e-20)


def test_gain_equals_squared_conditional_covariance_over_variance():
    model = fixed_model()
    b = model.belief({"y0": 0.001})
    for j in ("A", "B"):
        c = b.cov_with_return(j)
        _, var_j = b.predictive(j)
        assert b.gain(j) == pytest.approx(c * c / var_j, rel=1e-12)
        assert 0.0 <= b.gain(j) <= b.var


def test_posterior_variance_independent_of_observed_values():
    model = fixed_model()
    rng = np.random.default_rng(3)
    for names in (("y0",), ("A",), ("y0", "A"), ("A", "B")):
        variances = []
        for _ in range(6):
            obs = {k: rng.normal(0, 0.03) for k in names}
            variances.append(model.belief(obs).var)
        spread = max(variances) - min(variances)
        assert spread <= 1e-14 * max(variances)


# ------------------------------------------------------------- predictive and branches
def test_predictive_moments_match_schur():
    model = fixed_model()
    obs = {"y0": 0.0045}
    b = model.belief(obs)
    mu_ref, S_ref, unobs = schur_reference(model.mean, model.cov, model.names, obs)
    for j in ("A", "B"):
        i = unobs.index(j)
        m, v = b.predictive(j)
        assert m == pytest.approx(mu_ref[i], rel=1e-12, abs=1e-18)
        assert v == pytest.approx(S_ref[i, i], rel=1e-9, abs=1e-20)


def test_predictive_branches_preserve_mean_and_match_gain():
    model = fixed_model()
    b = model.belief({"y0": 0.002})
    for j in ("A", "B"):
        branches = b.predictive_branches(j)
        weights = np.array([w for w, _ in branches])
        assert np.all(weights >= 0) and weights.sum() == pytest.approx(1.0, abs=1e-14)
        means = np.array([nb.mean for _, nb in branches])
        assert float(weights @ means) == pytest.approx(b.mean, rel=1e-12, abs=1e-16)
        # Equal-weight quantile midpoints reproduce the preposterior variance of the mean up to
        # a discretization bias of about 1.28/n (0.64% at n=201; see docs/notes_C.md).
        increment_var = float(weights @ (means - b.mean) ** 2)
        assert increment_var == pytest.approx(b.gain(j), rel=1.5 / 201)
        fine = b.predictive_branches(j, n_nodes=10001)
        fine_w = np.array([w for w, _ in fine])
        fine_m = np.array([nb.mean for _, nb in fine])
        assert float(fine_w @ (fine_m - b.mean) ** 2) == pytest.approx(b.gain(j), rel=2e-4)
        for _, nb in branches[:5]:
            assert nb.var == pytest.approx(b.var - b.gain(j), rel=1e-12, abs=1e-20)
            assert set(nb.observed) == set(b.observed) | {j}


def test_predictive_branches_n_nodes_validation():
    b = fixed_model().belief({"y0": 0.0})
    with pytest.raises(ValueError):
        b.predictive_branches("A", n_nodes=0)
    assert len(b.predictive_branches("A", n_nodes=7)) == 7


def test_branch_values_are_consistent_with_the_belief_only():
    model = fixed_model()
    b = model.belief({"y0": 0.001})
    first = [nb.mean for _, nb in b.predictive_branches("A", n_nodes=11)]
    second = [nb.mean for _, nb in model.belief({"y0": 0.001}).predictive_branches("A", n_nodes=11)]
    np.testing.assert_array_equal(first, second)


# ------------------------------------------------------------- PSD and degenerate cases
def test_fit_exact_duplicate_stays_pd_and_gives_zero_gain():
    rng = np.random.default_rng(5)
    n = 600
    r = rng.standard_normal(n) * 0.01
    A = 0.6 * r + rng.standard_normal(n) * 0.008
    y0 = 0.3 * r + rng.standard_normal(n) * 0.01
    data = np.column_stack([r, y0, A, A.copy()])
    for method in ("ledoit_wolf", "sample"):
        m = fit_joint_gaussian(data, method=method)
        assert np.all(np.isfinite(m.cov)) and np.all(np.isfinite(m.mean))
        assert np.linalg.eigvalsh(m.cov).min() > 0
        assert m.meta["duplicates"] == (("B", "A", 1),)
        b = m.belief({"y0": 0.0, "A": 0.0})
        assert b.gain("B") / b.var < 1e-8
        assert b.gain("B") >= 0.0
        assert np.isfinite(b.predictive("B")[1])


def test_fit_near_duplicate_noise_1e9():
    rng = np.random.default_rng(6)
    n = 600
    r = rng.standard_normal(n) * 0.01
    A = 0.6 * r + rng.standard_normal(n) * 0.008
    y0 = 0.3 * r + rng.standard_normal(n) * 0.01
    B = A + 1e-9 * rng.standard_normal(n)
    data = np.column_stack([r, y0, A, B])
    m = fit_joint_gaussian(data, method="ledoit_wolf")
    assert np.linalg.eigvalsh(m.cov).min() > 0
    b = m.belief({"y0": 0.0, "A": 0.0})
    assert np.isfinite(b.gain("B"))
    assert b.gain("B") / b.var < 1e-8


@pytest.mark.parametrize("method", ["sample", "ledoit_wolf"])
def test_rank_deficient_linear_dependence_is_handled(method):
    rng = np.random.default_rng(9)
    n = 400
    r = rng.standard_normal(n) * 0.01
    A = 0.5 * r + rng.standard_normal(n) * 0.009
    B = -0.2 * r + rng.standard_normal(n) * 0.01
    y0 = A + B  # exact linear dependence, not a duplicate
    m = fit_joint_gaussian(np.column_stack([r, y0, A, B]), method=method)
    assert np.all(np.isfinite(m.cov)) and np.linalg.eigvalsh(m.cov).min() > 0
    b = m.belief({"A": 0.001, "B": -0.002})
    assert np.isfinite(b.var) and np.isfinite(b.gain("y0"))
    assert np.isfinite(b.predictive_branches("y0")[0][1].mean)
    # The sample estimator keeps the exact dependence (residual ~ floor). Ledoit-Wolf shrinks
    # toward the identity, which leaves Var(y0 | A, B) at about delta * Var(y0) (documented
    # limitation; exact duplicates are handled separately by collapsing them).
    if method == "sample":
        assert b.predictive("y0")[1] <= 1e-6 * m.cov[1, 1]
    else:
        assert b.predictive("y0")[1] <= 0.2 * m.cov[1, 1]


def test_direct_singular_model_duplicate_branches_are_degenerate():
    S = CORR * np.outer(SD, SD)
    S[3, :] = S[2, :]
    S[:, 3] = S[:, 2]  # B duplicates A exactly; the model is PSD but singular
    model = GaussianJointModel(MEAN, S, NAMES)
    b = model.belief({"A": 0.0015})
    assert b.is_degenerate("B")
    branches = b.predictive_branches("B")
    assert len(branches) == 1 and branches[0][0] == 1.0
    assert b.gain("B") == 0.0


def test_inconsistent_observations_on_singular_block_raise():
    S = CORR * np.outer(SD, SD)
    S[3, :] = S[2, :]
    S[:, 3] = S[:, 2]
    mean = MEAN.copy()
    mean[3] = mean[2]  # B duplicates A in mean too, so B = A is consistent with the model
    model = GaussianJointModel(mean, S, NAMES)
    b = model.belief({"A": 0.002})
    with pytest.raises(ValueError, match="inconsistent"):
        b.condition("B", 0.02)
    # Consistent value (B equals A) is accepted and gives the same law as observing A alone.
    same = b.condition("B", 0.002)
    assert same.var == pytest.approx(b.var, rel=1e-12)
    assert same.mean == pytest.approx(b.mean, rel=1e-12, abs=1e-18)


def test_non_psd_covariance_is_rejected():
    S = CORR * np.outer(SD, SD)
    S[2, 3] = S[3, 2] = 1.5 * np.sqrt(S[2, 2] * S[3, 3])
    with pytest.raises(ValueError):
        GaussianJointModel(MEAN, S, NAMES)


def test_near_perfect_correlation_numerics():
    C = np.array(
        [
            [1.0, 0.5, 0.5, 0.5],
            [0.5, 1.0, 0.9999, 0.9999],
            [0.5, 0.9999, 1.0, 0.9999],
            [0.5, 0.9999, 0.9999, 1.0],
        ]
    )
    model = GaussianJointModel(MEAN, C * np.outer(SD, SD), NAMES)
    obs = {"y0": 0.0007}
    b = model.belief(obs)
    mu_ref, S_ref, _ = schur_reference(model.mean, model.cov, model.names, obs)
    np.testing.assert_allclose(b.conditional_cov, S_ref, rtol=1e-8, atol=1e-20)
    b0 = model.belief({})
    b_A = b0.condition("A", 0.0002)
    b_AB = b_A.condition("B", -0.0001)
    lhs = b0.gain("A") + b_A.gain("B")
    rhs = b0.var - b_AB.var
    assert lhs == pytest.approx(rhs, rel=1e-8)
    assert np.all(np.isfinite([b.var, b.gain("A"), b.gain("B")]))


def test_near_perfect_correlation_fit_is_finite_and_pd():
    rng = np.random.default_rng(13)
    C = np.array(
        [
            [1.0, 0.5, 0.5, 0.5],
            [0.5, 1.0, 0.9999, 0.9999],
            [0.5, 0.9999, 1.0, 0.9999],
            [0.5, 0.9999, 0.9999, 1.0],
        ]
    )
    model = GaussianJointModel(MEAN, C * np.outer(SD, SD), NAMES)
    X = model.sample(4000, rng)
    for method in ("ledoit_wolf", "sample"):
        m = fit_joint_gaussian(X, method=method)
        assert np.linalg.eigvalsh(m.cov).min() > 0
        b = m.belief({"y0": 0.0})
        assert np.isfinite(b.gain("A")) and np.isfinite(b.gain("B"))


def test_low_variance_nonconstant_signal_is_scale_invariant():
    rng = np.random.default_rng(17)
    model = fixed_model()
    X = model.sample(3000, rng)
    Xs = X.copy()
    Xs[:, 1] *= 1e-9  # y0 at a tiny scale; correlations and gains are scale free in y0
    m1 = fit_joint_gaussian(X)
    m2 = fit_joint_gaussian(Xs)
    b1 = m1.belief({"y0": 0.001})
    b2 = m2.belief({"y0": 0.001 * 1e-9})
    assert b2.gain("A") == pytest.approx(b1.gain("A"), rel=1e-9)
    assert b2.gain("B") == pytest.approx(b1.gain("B"), rel=1e-9)
    assert b2.var == pytest.approx(b1.var, rel=1e-9)


def test_scaling_all_columns_scales_gains_quadratically():
    rng = np.random.default_rng(19)
    X = fixed_model().sample(2000, rng)
    m1 = fit_joint_gaussian(X)
    m2 = fit_joint_gaussian(X * 1e-6)
    b1 = m1.belief({"y0": 0.0005})
    b2 = m2.belief({"y0": 0.0005 * 1e-6})
    assert b2.gain("A") == pytest.approx(1e-12 * b1.gain("A"), rel=1e-9)
    assert b2.var == pytest.approx(1e-12 * b1.var, rel=1e-9)


# ----------------------------------------------------------------- constant columns
@pytest.mark.parametrize("value", [0.0, 0.7, 1.0 / 3.0, 0.1234567])
def test_constant_column_raises_clear_error(value):
    rng = np.random.default_rng(21)
    n = 300
    data = np.column_stack([rng.standard_normal(n), np.full(n, value), rng.standard_normal(n), rng.standard_normal(n)])
    with pytest.raises(ValueError, match="constant"):
        fit_joint_gaussian(data)


def test_too_few_rows_raises():
    data = np.random.default_rng(0).standard_normal((5, 4))
    with pytest.raises(ValueError, match="at least"):
        fit_joint_gaussian(data)


def test_nonfinite_data_raises():
    data = np.random.default_rng(0).standard_normal((50, 4))
    data[3, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_joint_gaussian(data)


# --------------------------------------------------------------- one and two acquisitions
def test_one_and_two_acquisition_updates_agree_with_joint_conditioning():
    model = fixed_model()
    v, a, bval = 0.0013, 0.0008, -0.0004
    one = model.belief({"y0": v}).condition("A", a)
    one_joint = model.belief({"y0": v, "A": a})
    assert one.mean == pytest.approx(one_joint.mean, rel=1e-12, abs=1e-18)
    assert one.var == pytest.approx(one_joint.var, rel=1e-12, abs=1e-20)
    two = one.condition("B", bval)
    assert two.observed == frozenset({"y0", "A", "B"})
    assert two.var <= one.var <= model.belief({"y0": v}).var


# -------------------------------------------------------------- Ledoit-Wolf vs sklearn
def test_ledoit_wolf_intensity_matches_sklearn_on_standardized_data():
    rng = np.random.default_rng(23)
    X = fixed_model().sample(500, rng)
    sd = X.std(axis=0, ddof=1)
    Z = (X - X.mean(axis=0)) / sd
    m = fit_joint_gaussian(X, method="ledoit_wolf")
    assert m.meta["shrinkage"] == pytest.approx(ledoit_wolf_shrinkage(Z), rel=1e-10, abs=1e-14)


def test_ledoit_wolf_correlation_matches_sklearn_covariance():
    rng = np.random.default_rng(29)
    X = fixed_model().sample(500, rng)
    sd = X.std(axis=0, ddof=1)
    Z = (X - X.mean(axis=0)) / sd
    sk_cov = LedoitWolf().fit(Z).covariance_
    d = np.sqrt(np.diag(sk_cov))
    sk_corr = sk_cov / np.outer(d, d)
    m = fit_joint_gaussian(X, method="ledoit_wolf")
    ours = m.cov / np.outer(np.sqrt(np.diag(m.cov)), np.sqrt(np.diag(m.cov)))
    np.testing.assert_allclose(ours, sk_corr, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------- fit_from_frame
def _frame(n=400, seed=31):
    rng = np.random.default_rng(seed)
    X = fixed_model().sample(n, rng)
    return pd.DataFrame(
        {
            "date": pd.date_range("2015-01-02", periods=n, freq="B"),
            "realized_return": X[:, 0],
            "free_forecast": X[:, 1],
            "rf_forecast": X[:, 2],
            "gb_forecast": X[:, 3],
        }
    )


def test_fit_from_frame_maps_columns_to_names():
    df = _frame()
    m = fit_from_frame(df)
    ref = fit_joint_gaussian(df[["realized_return", "free_forecast", "rf_forecast", "gb_forecast"]].to_numpy())
    assert m.names == ("r", "y0", "A", "B")
    np.testing.assert_allclose(m.mean, ref.mean, rtol=1e-14)
    np.testing.assert_allclose(m.cov, ref.cov, rtol=1e-12, atol=1e-20)


def test_fit_from_frame_last_n_uses_tail_rows_only():
    df = _frame(n=400)
    m = fit_from_frame(df, last_n=150)
    ref = fit_joint_gaussian(df.iloc[-150:][["realized_return", "free_forecast", "rf_forecast", "gb_forecast"]].to_numpy())
    np.testing.assert_allclose(m.cov, ref.cov, rtol=1e-12, atol=1e-20)
    assert m.meta["last_n"] == 150


def test_fit_from_frame_rejects_nan_few_rows_bad_order_and_bad_columns():
    df = _frame()
    bad = df.copy()
    bad.loc[10, "gb_forecast"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        fit_from_frame(bad)
    with pytest.raises(ValueError, match="at least"):
        fit_from_frame(_frame(n=5))
    with pytest.raises(ValueError, match="at least"):
        fit_from_frame(df, last_n=4)
    shuffled = df.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="chronological"):
        fit_from_frame(shuffled)
    with pytest.raises(ValueError, match="exceeds"):
        fit_from_frame(df, last_n=10_000)
    with pytest.raises(KeyError):
        fit_from_frame(df, cols=("realized_return", "free_forecast", "rf_forecast", "nope"))


# ----------------------------------------------------------- volatility and scaling
def test_ewma_matches_direct_double_sum():
    rng = np.random.default_rng(37)
    x = rng.standard_normal(40) * 0.01
    h = 5.0
    lam = 0.5 ** (1.0 / h)
    sig = ewma_volatility(x, h)
    for t in range(len(x)):
        w = np.array([lam ** (t - s) for s in range(t + 1)])
        ref = np.sqrt(np.sum(w * x[: t + 1] ** 2) / np.sum(w))
        assert sig[t] == pytest.approx(ref, rel=1e-12)


def test_ewma_is_causal():
    rng = np.random.default_rng(41)
    x = rng.standard_normal(200) * 0.01
    base = ewma_volatility(x, 21.0)
    for t in (0, 10, 99):
        y = x.copy()
        y[t + 1 :] = rng.standard_normal(len(x) - t - 1) * 10.0  # wildly different future
        np.testing.assert_array_equal(ewma_volatility(y, 21.0)[: t + 1], base[: t + 1])


def test_ewma_known_two_point_value_and_constant_input():
    sig = ewma_volatility(np.array([0.02, -0.01]), halflife=1.0)
    lam = 0.5
    assert sig[0] == pytest.approx(0.02)
    assert sig[1] == pytest.approx(np.sqrt((lam * 0.02**2 + 0.01**2) / (lam + 1.0)), rel=1e-14)
    const = ewma_volatility(np.full(30, -0.003), halflife=10.0)
    np.testing.assert_allclose(const, 0.003, rtol=1e-12)


def test_ewma_min_periods_and_validation():
    sig = ewma_volatility(np.array([0.01, 0.02, 0.03]), halflife=3.0, min_periods=2)
    assert np.isnan(sig[0]) and np.isfinite(sig[1]) and np.isfinite(sig[2])
    with pytest.raises(ValueError):
        ewma_volatility(np.array([0.01]), halflife=0.0)
    with pytest.raises(ValueError):
        ewma_volatility(np.array([0.01, np.nan]), halflife=3.0)


def test_scale_model_is_date_t_law():
    model = fixed_model()
    s = 0.02
    sm = scale_model(model, s)
    np.testing.assert_allclose(sm.mean, model.mean * s, rtol=1e-14)
    np.testing.assert_allclose(sm.cov, model.cov * s * s, rtol=1e-14)
    obs_std = {"y0": 0.5, "A": -0.3}
    b_std = model.belief(obs_std)
    b_sc = sm.belief({k: v * s for k, v in obs_std.items()})
    assert b_sc.mean == pytest.approx(s * b_std.mean, rel=1e-12, abs=1e-18)
    assert b_sc.var == pytest.approx(s * s * b_std.var, rel=1e-12)
    assert b_sc.gain("B") == pytest.approx(s * s * b_std.gain("B"), rel=1e-10, abs=1e-30)
    with pytest.raises(ValueError):
        scale_model(model, 0.0)


def test_standardized_fit_recovers_standardized_moments_and_matches_direct_fit():
    rng = np.random.default_rng(43)
    z = fixed_model().sample(60_000, rng)
    sigma = np.exp(rng.normal(np.log(0.01), 0.4, size=len(z)))
    data = z * sigma[:, None]
    std = fit_standardized_model(data, sigma, method="sample")
    direct = fit_joint_gaussian(data / sigma[:, None], method="sample")
    np.testing.assert_allclose(std.cov, direct.cov, rtol=1e-12, atol=1e-20)
    assert std.meta["standardized"] is True
    np.testing.assert_allclose(std.mean, fixed_model().mean, atol=5e-4)
    np.testing.assert_allclose(std.cov, fixed_model().cov, rtol=0.1, atol=1e-8)
    with pytest.raises(ValueError):
        fit_standardized_model(data, np.where(np.arange(len(z)) == 0, 0.0, sigma))


# -------------------------------------------------------------- conditional summaries
def test_conditional_summary_matches_definitions():
    model = fixed_model()
    s = conditional_summary(model, ["y0"])
    mu_ref, S_ref, unobs = schur_reference(model.mean, model.cov, model.names, {"y0": 0.0})
    idx = {n: i for i, n in enumerate(unobs)}
    assert s["var_r"] == pytest.approx(S_ref[0, 0], rel=1e-12)
    for j in ("A", "B"):
        rho = S_ref[0, idx[j]] / np.sqrt(S_ref[0, 0] * S_ref[idx[j], idx[j]])
        assert s[f"rho_{j}"] == pytest.approx(rho, rel=1e-10)
        assert s[f"Delta_{j}"] == pytest.approx(S_ref[0, idx[j]] ** 2 / S_ref[idx[j], idx[j]], rel=1e-10)
    kappa = S_ref[idx["A"], idx["B"]] / np.sqrt(S_ref[idx["A"], idx["A"]] * S_ref[idx["B"], idx["B"]])
    assert s["kappa_AB"] == pytest.approx(kappa, rel=1e-10)
    # Gain identity inside the summary: Delta_A + Delta_{B|A} = Delta_B + Delta_{A|B}.
    assert s["Delta_A"] + s["Delta_B|A"] == pytest.approx(s["Delta_B"] + s["Delta_A|B"], rel=1e-10)
    assert s["Delta_A"] + s["Delta_B|A"] == pytest.approx(s["var_r"] - model.belief({"y0": 0.0, "A": 0.0, "B": 0.0}).var, rel=1e-10)


# -------------------------------------------------------------------- no hidden access
def test_no_hidden_access_to_unqueried_values():
    model = fixed_model()
    b1 = model.belief({"y0": 0.0021})
    b2 = model.belief({"y0": 0.0021})
    # The belief stores only what was passed in; the observed-value dict is the whole state.
    assert b1.observations == {"y0": 0.0021}
    assert b1.observed == frozenset({"y0"})
    assert b1.mean == b2.mean and b1.var == b2.var
    np.testing.assert_array_equal(b1.conditional_cov, b2.conditional_cov)
    # Branches depend only on the belief: identical beliefs give identical branch laws.
    for (w1, n1), (w2, n2) in zip(b1.predictive_branches("A", 9), b2.predictive_branches("A", 9)):
        assert w1 == w2 and n1.mean == n2.mean and n1.observations == n2.observations
    # Values of unqueried forecasts enter only through condition(): an observed B changes the
    # belief, and the untouched belief is unchanged afterwards.
    with_B = model.belief({"y0": 0.0021, "B": 0.0100})
    assert with_B.mean != b1.mean
    assert b1.observations == {"y0": 0.0021}
    assert b1.observed == frozenset({"y0"})
    conditioned = b1.condition("A", 0.0007)
    assert conditioned.observations == {"y0": 0.0021, "A": 0.0007}
    assert b1.observations == {"y0": 0.0021}
    # Constructing a belief with an unknown or return observation is refused.
    with pytest.raises(KeyError):
        model.belief({"r": 0.01})
    with pytest.raises(KeyError):
        model.belief({"Z": 0.01})
    with pytest.raises(KeyError):
        b1.condition("y0", 0.0)
    assert isinstance(b1, GaussianBelief)


# ------------------------------------------------------- scale invariance of conditioning
def test_conditioning_is_invariant_to_rescaling_an_observed_source():
    # Rescaling y0 by 1e-9 (a units change) must not change the posterior of r given {y0, A}.
    # Regression for the pseudo-inverse cut-off that was applied to unscaled covariances.
    model = fixed_model()
    sc = np.array([1.0, 1e-9, 1.0, 1.0])
    scaled = GaussianJointModel(model.mean * sc, model.cov * np.outer(sc, sc), NAMES)
    for y0v, Av in [(0.003, -0.002), (0.0005, 0.001)]:
        b1 = model.belief({"y0": y0v, "A": Av})
        b2 = scaled.belief({"y0": y0v * 1e-9, "A": Av})
        assert b2.mean == pytest.approx(b1.mean, rel=1e-10, abs=1e-18)
        assert b2.var == pytest.approx(b1.var, rel=1e-10)
        assert b2.gain("B") == pytest.approx(b1.gain("B"), rel=1e-8, abs=1e-30)
