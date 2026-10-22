"""CPU-native NumPy accounting for dated allocations and proportional fees."""

from numbers import Real

import numpy as np
import pandas as pd

from .analysis import _check_frame, _check_returns


def _weights(values: dict, columns: pd.Index) -> np.ndarray:
    if not isinstance(values, dict) or set(values) != set(columns):
        raise ValueError("Weights must name every return column exactly once.")
    if any(not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
           for value in values.values()):
        raise ValueError("Weights must be numeric.")
    weights = np.array([values[name] for name in columns], dtype=float)
    if (not np.isfinite(weights).all() or (weights < 0).any()
            or (weights > 1).any() or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("Weights must be finite, nonnegative, and sum to one.")
    return weights


def _retained_value(desired: np.ndarray, held: np.ndarray, rate: float) -> float:
    """Exactly solve v + rate * sum(abs(v * desired - held)) = 1 for v in [0, 1].

    The left side is piecewise linear and strictly increasing because rate < 1
    and desired sums to at most one, so the root lies on the segment between
    the last kink below one and the first kink at or above it.
    """
    moving = desired > 0
    kinks = held[moving] / desired[moving]
    points = np.unique(np.r_[0., kinks[(kinks > 0) & (kinks < 1)], 1.])
    totals = points + rate * np.abs(np.outer(points, desired) - held).sum(axis=1)
    upper = int(np.searchsorted(totals, 1.))
    if totals[upper] == 1:
        return float(points[upper])
    lower = upper - 1
    return float(points[lower] + (1 - totals[lower]) * (points[upper] - points[lower])
                 / (totals[upper] - totals[lower]))


def dynamic_costed_path(returns: pd.DataFrame, targets: pd.DataFrame,
                        cost_bps: float = 0,
                        rebalance_dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Apply dated targets before their return; rebalance daily unless scheduled.

    Fees apply to absolute ETF purchases plus sales; RF is a frictionless cash
    proxy. Entry is charged, exit is not. Fees reduce capital before returns.
    All calculations use NumPy on CPU; this is not a GPU backend.
    """
    _check_returns(returns)
    _check_frame(targets, "Targets")
    if "RF" not in returns:
        raise ValueError("Returns must include the RF cash proxy.")
    if not targets.index.equals(returns.index) or set(targets.columns) != set(returns.columns):
        raise ValueError("Targets must match return dates and name every return column exactly once.")
    desired_weights = targets.reindex(columns=returns.columns).to_numpy(dtype=float)
    if ((desired_weights < 0).any() or (desired_weights > 1).any()
            or not np.allclose(desired_weights.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ValueError("Targets must be nonnegative and sum to one on every date.")
    if (not isinstance(cost_bps, Real) or isinstance(cost_bps, (bool, np.bool_))
            or not np.isfinite(cost_bps) or not 0 <= cost_bps < 10_000):
        raise ValueError("Cost must be finite and between 0 and 10,000 basis points.")
    if rebalance_dates is None:
        rebalance = np.ones(len(returns), dtype=bool)
    else:
        if (not isinstance(rebalance_dates, pd.DatetimeIndex) or rebalance_dates.has_duplicates
                or rebalance_dates.hasnans or not rebalance_dates.isin(returns.index).all()
                or returns.index[0] not in rebalance_dates):
            raise ValueError("Rebalance dates must be unique return dates and include entry.")
        rebalance = returns.index.isin(rebalance_dates)
    risky = np.array(returns.columns != "RF")
    rate = cost_bps / 10_000
    held = np.zeros(returns.shape[1])
    held[returns.columns.get_loc("RF")] = 1
    rows = []
    for day, daily in enumerate(returns.to_numpy(dtype=float)):
        retained, turnover = 1.0, 0.0
        if rebalance[day]:
            desired = desired_weights[day]
            if rate:
                # Both ETF sales and purchases fund the final target weights.
                retained = _retained_value(desired[risky], held[risky], rate)
            turnover = float(np.abs(retained * desired[risky] - held[risky]).sum())
            held = desired.copy()
        gross_return = float(held @ daily)
        net_return = gross_return + (retained - 1) * (1 + gross_return)
        rows.append((net_return, turnover, rate * turnover))
        held *= (1 + daily) / (1 + gross_return)
    return pd.DataFrame(rows, index=returns.index,
                        columns=["return", "traded_fraction", "fee_fraction"])


def monthly_costed_path(returns: pd.DataFrame, targets: dict[str, float],
                        cost_bps: float = 0, initial_weights: dict | None = None
                        ) -> pd.DataFrame:
    """Buy at the seed close, then reset before each new month's first return.

    Optional initial weights preserve pre-existing monthly drift. Between
    scheduled resets holdings drift; repeated dated targets do not force trades.
    """
    _check_returns(returns)
    target = _weights(targets, returns.columns)
    initial = target if initial_weights is None else _weights(initial_weights, returns.columns)
    dated = pd.DataFrame(np.tile(target, (len(returns), 1)),
                         index=returns.index, columns=returns.columns)
    dated.iloc[0] = initial
    months = returns.index.to_period("M")
    first = np.r_[True, months[1:] != months[:-1]]
    return dynamic_costed_path(returns, dated, cost_bps, returns.index[first])
