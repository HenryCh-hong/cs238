"""Joint Gaussian belief over X = (r, y0, A, B).

r  : next-period simple return
y0 : free linear-model forecast (always observed)
A  : random-forest forecast (purchasable)
B  : gradient-boosting forecast (purchasable)

The joint law N(mu, Sigma) is estimated on a calibration period of out-of-sample
forecasts and subsequently realized returns. A belief is the conditional law of
the unobserved coordinates given the observed ones. Because the model is
linear-Gaussian, the posterior variance of r after observing a set O does not
depend on the observed values, and the posterior mean after one more source j
is distributed (preposterior) as N(m, Delta_j) with

    Delta_j = Cov(r, Y_j | O)^2 / Var(Y_j | O).

Conditioning is order independent: a belief is always recomputed from the joint
model and the full set of observations, so "A then B", "B then A" and "{A, B} at
once" give the same law. Under the model, Delta_j + Delta_{k|j} equals
Delta_k + Delta_{j|k} equals Var(r|O) - Var(r|O, j, k).

Volatility scaling (context-dependent belief)
--------------------------------------------
If the standardized quantities x_t / sigma_t are jointly N(mu, Sigma), where
sigma_t > 0 is an ex-ante volatility known at origin t, then the law of x_t given
the date-t information is N(mu sigma_t, Sigma sigma_t^2). ``scale_model`` builds
that date-t model. Alignment: ``ewma_volatility(returns, halflife)[t]`` uses only
``returns[0..t]``, where ``returns[t]`` is the return realized at the close of day
t. At origin t the target is the next-day return, so the volatility of the date-t
belief is ``sigma[t]`` and never depends on the target return.

This module never sees unqueried forecast values: a belief stores only the
observations passed to it explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import lfilter
from scipy.stats import norm

from src.utility import UtilityParams

VAR_NAMES: tuple[str, ...] = ("r", "y0", "A", "B")

# Default column names of a calibration frame, mapped position-wise to VAR_NAMES.
FRAME_COLS: tuple[str, ...] = ("realized_return", "free_forecast", "rf_forecast", "gb_forecast")

# Conditional variances below this fraction of the marginal variance are treated
# as zero (the forecast is fully determined by what was already observed).
DEGENERATE_RTOL = 1e-10

# A column whose spread is below this fraction of its magnitude is constant up to
# rounding error (a rounding-level sd must not be standardized into noise).
CONSTANT_RTOL = 1e-12

# The observed block is treated as singular when its smallest eigenvalue is below
# this fraction of its largest. Pseudo-inverse cut-off for the same test.
SINGULAR_RTOL = 1e-13

# Relative residual above which observations are inconsistent with a singular
# observed block (their conditional law is then undefined).
CONSISTENCY_RTOL = 1e-8

# Two standardized columns with |sample correlation| > 1 - DUPLICATE_RTOL are treated
# as exact duplicates (up to sign) and collapsed before shrinkage/eigen-flooring.
DUPLICATE_RTOL = 1e-10


@dataclass(frozen=True)
class GaussianJointModel:
    mean: np.ndarray
    cov: np.ndarray
    names: tuple[str, ...] = VAR_NAMES
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float).reshape(-1)
        cov = np.asarray(self.cov, dtype=float)
        p = len(self.names)
        if mean.shape != (p,) or cov.shape != (p, p):
            raise ValueError(f"shape mismatch: mean {mean.shape}, cov {cov.shape}, names {p}")
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))):
            raise ValueError("non-finite mean or covariance")
        if not np.allclose(cov, cov.T, rtol=0, atol=1e-14 * max(1.0, np.abs(cov).max())):
            raise ValueError("covariance is not symmetric")
        cov = 0.5 * (cov + cov.T)
        if np.linalg.eigvalsh(cov).min() < -1e-12 * np.trace(cov):
            raise ValueError("covariance is not positive semidefinite")
        if self.names[0] != "r":
            raise ValueError("the first variable must be the return 'r'")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "cov", cov)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def belief(self, observations: Mapping[str, float]) -> "GaussianBelief":
        return GaussianBelief(self, dict(observations))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n joint samples (rows ordered like ``names``)."""
        return rng.multivariate_normal(self.mean, self.cov, size=n, method="eigh")


class GaussianBelief:
    """Conditional law of the unobserved coordinates given ``observations``."""

    def __init__(self, model: GaussianJointModel, observations: Mapping[str, float]):
        for k, v in observations.items():
            if k not in model.names or k == "r":
                raise KeyError(f"cannot observe {k!r}")
            if not np.isfinite(v):
                raise ValueError(f"non-finite observation {k}={v}")
        self.model = model
        self._obs = dict(observations)
        names = model.names
        o = [model.index(k) for k in self._obs]
        u = [i for i in range(len(names)) if names[i] not in self._obs]
        self._u_names = tuple(names[i] for i in u)
        mu, S = model.mean, model.cov
        if o:
            S_oo = S[np.ix_(o, o)]
            S_uo = S[np.ix_(u, o)]
            y_o = np.array([self._obs[names[i]] for i in o], dtype=float)
            d = y_o - mu[o]
            # Conditioning is done on the correlation scale of the observed block: with
            # D = diag(S_oo)^{1/2}, S_uo S_oo^+ = (S_uo D^-1) R_oo^+ D^-1 with R_oo = D^-1 S_oo D^-1.
            # This makes the singularity test and the pseudo-inverse cut-off invariant to the
            # units of each observed source (rescaling y0 by 1e-9 must not change the posterior).
            # Zero-variance observations have dinv = 0; they must be exactly consistent (checked).
            dvar = np.clip(np.diag(S_oo), 0.0, None)
            dinv = np.where(dvar > 0.0, 1.0 / np.sqrt(np.where(dvar > 0.0, dvar, 1.0)), 0.0)
            R_oo = S_oo * np.outer(dinv, dinv)
            R_oo = 0.5 * (R_oo + R_oo.T)
            z = dinv * d
            P = np.linalg.pinv(R_oo, rcond=SINGULAR_RTOL, hermitian=True)
            w = np.linalg.eigvalsh(R_oo)
            Q = S_uo * dinv[None, :]
            if w[0] <= SINGULAR_RTOL * max(w[-1], 0.0) or np.any(dvar <= 0.0):
                # Singular observed block: the observations must lie in its range.
                resid = z - R_oo @ (P @ z)
                scale = max(1.0, float(np.max(np.abs(z))))
                zero_var_bad = np.any((dvar <= 0.0) & (np.abs(d) > CONSISTENCY_RTOL * max(1.0, float(np.max(np.abs(d))))))
                if zero_var_bad or np.max(np.abs(resid)) > CONSISTENCY_RTOL * scale:
                    raise ValueError(
                        "observations are inconsistent with the singular covariance of the "
                        f"observed block {sorted(self._obs)}: the conditional law is undefined"
                    )
            self._mu_u = mu[u] + Q @ (P @ z)
            S_c = S[np.ix_(u, u)] - Q @ P @ Q.T
        else:
            self._mu_u = mu[u].copy()
            S_c = S[np.ix_(u, u)].copy()
        S_c = 0.5 * (S_c + S_c.T)
        # Clip tiny negative diagonal entries produced by rounding.
        d_diag = np.clip(np.diag(S_c), 0.0, None)
        np.fill_diagonal(S_c, d_diag)
        self._S_u = S_c

    # ------------------------------------------------------------------ protocol
    @property
    def observed(self) -> frozenset[str]:
        return frozenset(self._obs)

    @property
    def observations(self) -> Mapping[str, float]:
        return dict(self._obs)

    @property
    def unobserved(self) -> tuple[str, ...]:
        """Names of the unobserved coordinates, in model order (``r`` first)."""
        return self._u_names

    @property
    def conditional_cov(self) -> np.ndarray:
        """Conditional covariance of the unobserved coordinates (copy). Data independent."""
        return self._S_u.copy()

    def _ui(self, name: str) -> int:
        if name in self._obs:
            raise KeyError(f"{name!r} is already observed")
        return self._u_names.index(name)

    @property
    def mean(self) -> float:
        """E_b[r]."""
        return float(self._mu_u[0])

    @property
    def var(self) -> float:
        """Var_b(r) (data independent under the linear-Gaussian model)."""
        return float(self._S_u[0, 0])

    def return_moments(self) -> tuple[float, float]:
        m = self.mean
        return m, m * m + self.var

    def predictive(self, source: str) -> tuple[float, float]:
        """Mean and variance of the unobserved forecast ``source`` under b."""
        i = self._ui(source)
        return float(self._mu_u[i]), float(self._S_u[i, i])

    def cov_with_return(self, source: str) -> float:
        i = self._ui(source)
        return float(self._S_u[0, i])

    def is_degenerate(self, source: str) -> bool:
        _, var_j = self.predictive(source)
        marg = self.model.cov[self.model.index(source), self.model.index(source)]
        return var_j <= DEGENERATE_RTOL * marg

    def gain(self, source: str) -> float:
        """Delta_j: variance of the preposterior shift of E[r] from observing j.

        Equals Cov(r, Y_j|O)^2 / Var(Y_j|O) and is clipped to [0, Var_b(r)]
        (Cauchy-Schwarz bound; the clip only removes rounding excess).
        """
        if self.is_degenerate(source):
            return 0.0
        _, var_j = self.predictive(source)
        c = self.cov_with_return(source)
        return float(min(max(c * c / var_j, 0.0), self.var))

    def condition(self, source: str, value: float) -> "GaussianBelief":
        if source in self._obs:
            raise KeyError(f"{source!r} already observed")
        new = dict(self._obs)
        new[source] = float(value)
        return GaussianBelief(self.model, new)

    def predictive_branches(self, source: str, n_nodes: int | None = None) -> Sequence[tuple[float, "GaussianBelief"]]:
        """Equal-weight quantile-midpoint nodes of the predictive law of ``source``.

        Midpoint nodes in probability space handle the kinks of the stop value
        better than Gauss-Hermite (see docs/mathematical_assumptions.md).
        Degenerate sources return a single branch at the predictive mean.
        """
        n = 201 if n_nodes is None else int(n_nodes)
        if n < 1:
            raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
        mu_j, var_j = self.predictive(source)
        if var_j <= 0.0 or self.is_degenerate(source):
            return [(1.0, self.condition(source, mu_j))]
        z = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        sd = np.sqrt(var_j)
        return [(1.0 / n, self.condition(source, mu_j + sd * zk)) for zk in z]

    def expected_stop_value_after(self, source: str, q_prev: float, params: UtilityParams) -> float:
        """Exact E_{Y_j|b}[G(b^{j,Y_j})] via the closed form in ``src.voi``."""
        from src.voi import expected_stop_value_gaussian

        return float(expected_stop_value_gaussian(self.mean, self.gain(source), self.var, q_prev, params))

    def summary(self) -> dict[str, float]:
        return {"mean": self.mean, "var": self.var, "n_observed": len(self._obs)}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"GaussianBelief(obs={sorted(self._obs)}, mean={self.mean:.3e}, var={self.var:.3e})"


# ---------------------------------------------------------------------- estimation
def _ledoit_wolf_corr(Z: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage of the second-moment matrix of centered data Z
    toward the scaled identity mu*I. Returns (shrunk matrix, shrinkage intensity).

    The intensity is beta_bar / ||S - mu I||_F^2 with beta_bar = (1/n^2) sum_k
    ||z_k z_k^T - S||_F^2, the same quantity as ``sklearn.covariance.ledoit_wolf_shrinkage``.
    It is invariant to rescaling the columns, so applying it to standardized data
    gives the same intensity sklearn gives on the raw data after its own centering.
    """
    n, p = Z.shape
    S = Z.T @ Z / n
    mu = np.trace(S) / p
    d2 = float(np.sum((S - mu * np.eye(p)) ** 2))
    if d2 <= 0.0:
        return S, 0.0
    outer = np.einsum("ki,kj->kij", Z, Z)
    b2 = float(np.sum((outer - S[None]) ** 2) / n**2)
    b2 = min(b2, d2)
    delta = b2 / d2
    return delta * mu * np.eye(p) + (1.0 - delta) * S, delta


def fit_joint_gaussian(
    data: np.ndarray,
    names: tuple[str, ...] = VAR_NAMES,
    method: str = "ledoit_wolf",
    eig_floor_rel: float = 1e-8,
) -> GaussianJointModel:
    """Estimate N(mu, Sigma) from rows of ``data`` ordered like ``names``.

    method: "sample" (unbiased sample covariance) or "ledoit_wolf" (shrink the
    correlation matrix toward I on standardized data, then rescale, so that the
    very different scales of returns and forecasts do not drive the shrinkage).
    Eigenvalues of the correlation matrix are floored at ``eig_floor_rel`` to keep
    the estimate positive definite.

    Raises ValueError for non-finite data, too few rows, or a constant column
    (spread at rounding level relative to its magnitude, see ``CONSTANT_RTOL``).
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError(f"data must be (n, {len(names)}), got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("data contains non-finite values")
    n = X.shape[0]
    if n < len(names) + 2:
        raise ValueError(f"need at least {len(names) + 2} rows, got {n}")
    mu = X.mean(axis=0)
    Xc = X - mu
    sd = Xc.std(axis=0, ddof=1)
    magnitude = np.max(np.abs(X), axis=0)
    spread = X.max(axis=0) - X.min(axis=0)
    constant = (spread == 0.0) | (sd <= CONSTANT_RTOL * magnitude)
    if np.any(constant):
        bad = [names[i] for i in np.where(constant)[0]]
        raise ValueError(f"constant (zero-variance) column(s) cannot be standardized: {bad}")
    Z = Xc / sd
    if method not in ("sample", "ledoit_wolf"):
        raise ValueError(f"unknown method {method!r}")
    # Exact duplicate columns are collapsed to one representative before estimation and
    # expanded afterwards. Ledoit-Wolf shrinks toward the identity, which would turn an exact
    # duplicate into correlation 1 - delta and create spurious information (gain(B|A) > 0).
    rep, sign, dup_pairs = _duplicate_map(Z, n)
    keep = [i for i in range(len(names)) if rep[i] == i]
    Zr = Z[:, keep]
    if method == "sample":
        Rr = Zr.T @ Zr / (n - 1)
        shrink = 0.0
    else:
        Rr, shrink = _ledoit_wolf_corr(Zr)
        Rr = Rr * (n / (n - 1))  # match ddof=1 scaling of sd
    # Re-normalize the reduced correlation matrix, then expand duplicates exactly.
    dR = np.sqrt(np.diag(Rr))
    Rr = Rr / np.outer(dR, dR)
    pos = {orig: k for k, orig in enumerate(keep)}
    idx = np.array([pos[rep[i]] for i in range(len(names))])
    s = np.asarray(sign, dtype=float)
    R = (s[:, None] * s[None, :]) * Rr[np.ix_(idx, idx)]
    # Floor the eigenvalues of the full correlation matrix to keep it positive definite.
    # Exact duplicates have a zero eigenvalue, so the floor perturbs them by O(eig_floor_rel)
    # only: the conditional gain of a duplicate is then ~1e-9 of Var(r), not exactly zero.
    w, V = np.linalg.eigh(R)
    w_floor = np.maximum(w, eig_floor_rel)
    R = (V * w_floor) @ V.T
    dR = np.sqrt(np.diag(R))
    R = R / np.outer(dR, dR)
    cov = R * np.outer(sd, sd)
    meta = {"method": method, "n": n, "shrinkage": shrink, "eig_min_raw": float(w.min()),
            "n_eig_floored": int(np.sum(w < eig_floor_rel)),
            "duplicates": tuple((names[i], names[j], int(sign[i])) for i, j in dup_pairs)}
    return GaussianJointModel(mean=mu, cov=cov, names=names, meta=meta)


def _duplicate_map(Z: np.ndarray, n: int) -> tuple[list[int], np.ndarray, list[tuple[int, int]]]:
    """Find columns of standardized data Z that are exact duplicates (up to sign).

    Returns (rep, sign, pairs): column i equals sign[i] * column rep[i], where rep[i] is a
    representative (rep[i] == i for representatives). ``pairs`` lists (i, rep[i]) for every
    duplicate. A pair is a duplicate when |sample correlation| > 1 - DUPLICATE_RTOL.
    """
    C = Z.T @ Z / (n - 1)
    p = C.shape[0]
    rep = list(range(p))
    sign = np.ones(p)
    pairs: list[tuple[int, int]] = []
    for i in range(p):
        for j in range(i):
            if rep[j] == j and abs(C[i, j]) > 1.0 - DUPLICATE_RTOL:
                rep[i] = j
                sign[i] = 1.0 if C[i, j] > 0 else -1.0
                pairs.append((i, j))
                break
    return rep, sign, pairs


def fit_from_frame(
    df,
    method: str = "ledoit_wolf",
    last_n: int | None = None,
    cols: Sequence[str] = FRAME_COLS,
) -> GaussianJointModel:
    """Fit the joint model from a pandas frame, mapping ``cols`` to ("r", "y0", "A", "B").

    ``cols`` is matched position-wise: (realized return, free forecast, random-forest
    forecast, gradient-boosting forecast). Rows are used in frame order, which must be
    chronological; if a ``date`` column exists it is checked to be non-decreasing.
    ``last_n`` keeps only the final ``last_n`` rows (the most recent calibration window).

    Raises ValueError on missing values, too few rows, unsorted dates, or a
    ``last_n`` larger than the frame (the caller must not silently fit less data).
    """
    cols = tuple(cols)
    if len(cols) != len(VAR_NAMES):
        raise ValueError(f"cols must have {len(VAR_NAMES)} entries, got {len(cols)}")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns {missing}")
    if "date" in df.columns and not df["date"].is_monotonic_increasing:
        raise ValueError("rows must be in chronological order: the 'date' column is not non-decreasing")
    sub = df.loc[:, list(cols)]
    if last_n is not None:
        last_n = int(last_n)
        if last_n < 1:
            raise ValueError(f"last_n must be >= 1, got {last_n}")
        if last_n > len(sub):
            raise ValueError(f"last_n={last_n} exceeds the {len(sub)} rows available")
        sub = sub.iloc[len(sub) - last_n:]
    n_missing = int(sub.isna().sum().sum())
    if n_missing:
        raise ValueError(f"{n_missing} missing value(s) in {list(cols)}; drop or fill them before fitting")
    need = len(VAR_NAMES) + 2
    if len(sub) < need:
        raise ValueError(f"need at least {need} rows, got {len(sub)}")
    model = fit_joint_gaussian(sub.to_numpy(dtype=float), names=VAR_NAMES, method=method)
    meta = {**dict(model.meta), "cols": cols, "last_n": last_n}
    return GaussianJointModel(model.mean, model.cov, model.names, meta)


def ewma_volatility(returns, halflife: float, min_periods: int = 1) -> np.ndarray:
    """Causal exponentially weighted volatility (bias-corrected, zero-mean squares).

    sigma[t] = sqrt( sum_{s<=t} w_s r_s^2 / sum_{s<=t} w_s ),  w_s = lam^(t-s),
    lam = 0.5^(1/halflife).

    ``returns[t]`` is the return realized at the close of day t, so sigma[t] uses
    returns[0..t] only and is the ex-ante volatility at origin t. The target of that
    origin is the next-day return, which never enters sigma[t]. Entries before
    ``min_periods`` are NaN. Rounding is the only difference from the naive loop.
    """
    x = np.asarray(returns, dtype=float).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("returns contain non-finite values")
    h = float(halflife)
    if not (np.isfinite(h) and h > 0):
        raise ValueError(f"halflife must be finite and > 0, got {halflife}")
    if int(min_periods) < 1:
        raise ValueError(f"min_periods must be >= 1, got {min_periods}")
    lam = 0.5 ** (1.0 / h)
    num = lfilter([1.0], [1.0, -lam], x * x)
    den = lfilter([1.0], [1.0, -lam], np.ones_like(x))
    sigma = np.sqrt(num / den) if x.size else np.empty(0)
    if int(min_periods) > 1:
        sigma[: int(min_periods) - 1] = np.nan
    return sigma


def fit_standardized_model(
    data: np.ndarray,
    sigma: np.ndarray,
    names: tuple[str, ...] = VAR_NAMES,
    method: str = "ledoit_wolf",
) -> GaussianJointModel:
    """Fit the standardized law of x_t / sigma_t from rows of ``data``.

    Row t of ``data`` and entry t of ``sigma`` refer to the same origin, and sigma[t]
    must be known at that origin (e.g. ``ewma_volatility`` output). The result is the
    unit-scale model; the date-t belief is ``scale_model(result, sigma_t)``.
    """
    X = np.asarray(data, dtype=float)
    s = np.asarray(sigma, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != s.shape[0]:
        raise ValueError(f"data rows {X.shape[0] if X.ndim == 2 else X.shape} must match sigma length {s.shape[0]}")
    if not (np.all(np.isfinite(s)) and np.all(s > 0)):
        raise ValueError("sigma must be finite and > 0 at every row (check min_periods / zero returns)")
    model = fit_joint_gaussian(X / s[:, None], names=names, method=method)
    return GaussianJointModel(model.mean, model.cov, model.names, {**dict(model.meta), "standardized": True})


def scale_model(model: GaussianJointModel, s: float) -> GaussianJointModel:
    """Date-t law for a standardized model: N(mu*s, Sigma*s^2) for volatility s > 0."""
    s = float(s)
    if not (np.isfinite(s) and s > 0):
        raise ValueError(f"volatility scale must be finite and > 0, got {s}")
    return GaussianJointModel(
        mean=model.mean * s,
        cov=model.cov * (s * s),
        names=model.names,
        meta={**dict(model.meta), "scale": s},
    )


def conditional_summary(model: GaussianJointModel, observed: Iterable[str] = ()) -> dict[str, float]:
    """Second-order summaries of the belief after observing the names in ``observed``.

    Conditional second moments do not depend on the observed values, so no values are
    needed. Keys (only for unobserved coordinates, r excluded from the sources):
      var_r            Var(r | O)
      rho_<j>          Corr(r, Y_j | O)
      Delta_<j>        Cov(r, Y_j|O)^2 / Var(Y_j|O)
      kappa_<a><b>     Corr(Y_a, Y_b | O) for an unobserved pair
      Delta_<j>|<k>    gain of j after k is also observed (k unobserved)
    A zero conditional variance gives correlation 0.0 (no information to correlate).
    """
    observed = tuple(observed)
    b = model.belief({k: 0.0 for k in observed})
    S = b.conditional_cov
    idx = {name: i for i, name in enumerate(b.unobserved)}
    out: dict[str, float] = {"var_r": b.var}
    sources = [n for n in b.unobserved if n != "r"]

    def corr(a: str, c: str) -> float:
        i, j = idx[a], idx[c]
        den = np.sqrt(S[i, i] * S[j, j])
        return float(S[i, j] / den) if den > 0 else 0.0

    for j in sources:
        out[f"rho_{j}"] = corr("r", j)
        out[f"Delta_{j}"] = b.gain(j)
    for a, c in combinations(sources, 2):
        out[f"kappa_{a}{c}"] = corr(a, c)
    for j in sources:
        for k in sources:
            if k != j:
                out[f"Delta_{j}|{k}"] = model.belief({**{q: 0.0 for q in observed}, k: 0.0}).gain(j)
    return out
