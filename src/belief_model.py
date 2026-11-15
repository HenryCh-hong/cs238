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

This module never sees unqueried forecast values: a belief stores only the
observations passed to it explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import norm

from src.utility import UtilityParams

VAR_NAMES: tuple[str, ...] = ("r", "y0", "A", "B")

# Conditional variances below this fraction of the marginal variance are treated
# as zero (the forecast is fully determined by what was already observed).
DEGENERATE_RTOL = 1e-10


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
            y_o = np.array([self._obs[names[i]] for i in o])
            # Solve via pseudo-inverse of the observed block: exact when S_oo is
            # nonsingular, and the minimum-norm Gaussian conditioning otherwise.
            K = S_uo @ np.linalg.pinv(S_oo, rcond=1e-13, hermitian=True)
            self._mu_u = mu[u] + K @ (y_o - mu[o])
            S_c = S[np.ix_(u, u)] - K @ S_uo.T
        else:
            self._mu_u = mu[u].copy()
            S_c = S[np.ix_(u, u)].copy()
        S_c = 0.5 * (S_c + S_c.T)
        # Clip tiny negative diagonal entries produced by rounding.
        d = np.clip(np.diag(S_c), 0.0, None)
        np.fill_diagonal(S_c, d)
        self._S_u = S_c

    # ------------------------------------------------------------------ protocol
    @property
    def observed(self) -> frozenset[str]:
        return frozenset(self._obs)

    @property
    def observations(self) -> Mapping[str, float]:
        return dict(self._obs)

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
        Clipped to [0, Var_b(r)]."""
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
        """
        n = 201 if n_nodes is None else int(n_nodes)
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
    """Ledoit-Wolf (2004) shrinkage of the correlation matrix of standardized data Z
    toward the identity. Returns (shrunk correlation, shrinkage intensity)."""
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
    if np.any(sd <= 0):
        raise ValueError(f"zero-variance column(s): {[names[i] for i in np.where(sd <= 0)[0]]}")
    Z = Xc / sd
    if method == "sample":
        R = Z.T @ Z / (n - 1)
        shrink = 0.0
    elif method == "ledoit_wolf":
        R, shrink = _ledoit_wolf_corr(Z)
        R = R * (n / (n - 1))  # match ddof=1 scaling of sd
    else:
        raise ValueError(f"unknown method {method!r}")
    # Re-normalize to a correlation matrix and floor its eigenvalues.
    dR = np.sqrt(np.diag(R))
    R = R / np.outer(dR, dR)
    w, V = np.linalg.eigh(R)
    w_floor = np.maximum(w, eig_floor_rel)
    R = (V * w_floor) @ V.T
    dR = np.sqrt(np.diag(R))
    R = R / np.outer(dR, dR)
    cov = R * np.outer(sd, sd)
    meta = {"method": method, "n": n, "shrinkage": shrink, "eig_min_raw": float(w.min()),
            "n_eig_floored": int(np.sum(w < eig_floor_rel))}
    return GaussianJointModel(mean=mu, cov=cov, names=names, meta=meta)
