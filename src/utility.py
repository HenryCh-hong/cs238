"""Trading utility and the terminal (stopping) decision.

Utility of holding position q over one period with realized simple return r:

    u(q, r) = q r - lam q^2 r^2 - tau |q - q_prev|

Returns, utilities, transaction and query costs are all fractions of capital
(1 bp = 1e-4). Under any belief b over r the expected utility depends on b only
through the first two moments m = E_b[r] and s2 = E_b[r^2]:

    E_b[u(q, r)] = q m - lam q^2 s2 - tau |q - q_prev|

so the stop value G(b) = max_q E_b[u(q, r)] is a function of (m, s2, q_prev).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POSITIONS: tuple[int, ...] = (-1, 0, 1)

# Relative tolerance for declaring two expected utilities tied.
TIE_RTOL = 1e-12
TIE_ATOL = 1e-18


@dataclass(frozen=True)
class UtilityParams:
    """lam: penalty on q^2 r^2; tau: proportional transaction cost per unit |dq|."""

    lam: float = 1.0
    tau: float = 1e-4

    def __post_init__(self) -> None:
        if not (np.isfinite(self.lam) and self.lam >= 0):
            raise ValueError(f"lam must be finite and >= 0, got {self.lam}")
        if not (np.isfinite(self.tau) and self.tau >= 0):
            raise ValueError(f"tau must be finite and >= 0, got {self.tau}")


def trading_utility(q: float, r, p: UtilityParams):
    """Gross trading utility q r - lam q^2 r^2 (before transaction cost)."""
    r = np.asarray(r, dtype=float)
    out = q * r - p.lam * q * q * r * r
    return float(out) if out.ndim == 0 else out


def transaction_cost(q: float, q_prev: float, p: UtilityParams) -> float:
    return p.tau * abs(q - q_prev)


def realized_utility(q: float, r, q_prev: float, p: UtilityParams):
    """u(q, r) including the transaction cost (query costs are NOT included)."""
    return trading_utility(q, r, p) - transaction_cost(q, q_prev, p)


def expected_utility(q: float, mean, second_moment, q_prev: float, p: UtilityParams):
    """E_b[u(q, r)] given m = E_b[r] and s2 = E_b[r^2]. Vectorized over m, s2."""
    mean = np.asarray(mean, dtype=float)
    second_moment = np.asarray(second_moment, dtype=float)
    out = q * mean - p.lam * q * q * second_moment - p.tau * abs(q - q_prev)
    return float(out) if out.ndim == 0 else out


def stop_value_array(mean, second_moment, q_prev: float, p: UtilityParams) -> np.ndarray:
    """G = max_q E[u(q, r)], vectorized over arrays of (m, s2). Value only."""
    mean = np.asarray(mean, dtype=float)
    second_moment = np.broadcast_to(np.asarray(second_moment, dtype=float), mean.shape)
    vals = np.stack([expected_utility(q, mean, second_moment, q_prev, p) for q in POSITIONS])
    return vals.max(axis=0)


def _tie_key(q: int, q_prev: float) -> tuple[float, int, int]:
    # Among tied positions prefer the smallest trade, then the smallest exposure, then lowest q.
    return (abs(q - q_prev), abs(q), q)


def choose_position(mean: float, second_moment: float, q_prev: float, p: UtilityParams) -> tuple[float, int]:
    """Terminal decision: returns (G, q*) with q* = argmax_q E[u(q, r)].

    Ties (within TIE_RTOL relative / TIE_ATOL absolute) are broken toward the
    smallest |q - q_prev|, then the smallest |q|, then the lowest q, so the
    decision is deterministic.
    """
    vals = {q: float(expected_utility(q, mean, second_moment, q_prev, p)) for q in POSITIONS}
    if not all(np.isfinite(v) for v in vals.values()):
        raise FloatingPointError(f"non-finite expected utility: {vals}")
    best = max(vals.values())
    tol = TIE_ATOL + TIE_RTOL * max(abs(v) for v in vals.values())
    tied = [q for q, v in vals.items() if v >= best - tol]
    q_star = min(tied, key=lambda q: _tie_key(q, q_prev))
    return vals[q_star], q_star
