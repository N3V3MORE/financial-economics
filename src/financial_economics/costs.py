"""CPU-native NumPy accounting for monthly portfolios and proportional fees."""

from numbers import Real

import numpy as np
import pandas as pd

from .analysis import _check_returns


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


def monthly_costed_path(returns: pd.DataFrame, targets: dict[str, float],
                        cost_bps: float = 0, initial_weights: dict | None = None
                        ) -> pd.DataFrame:
    """Buy at the seed close, then reset before each new month's first return.

    Fees apply to absolute ETF purchases plus sales; RF is a frictionless cash
    proxy. Entry is charged, exit is not. Fees reduce capital before returns.
    Optional initial weights preserve a portfolio's pre-existing monthly drift.
    All calculations use NumPy on CPU; this is not a GPU backend.
    """
    _check_returns(returns)
    if "RF" not in returns:
        raise ValueError("Returns must include the RF cash proxy.")
    if (not isinstance(cost_bps, Real) or isinstance(cost_bps, (bool, np.bool_))
            or not np.isfinite(cost_bps) or not 0 <= cost_bps < 10_000):
        raise ValueError("Cost must be finite and between 0 and 10,000 basis points.")
    target = _weights(targets, returns.columns)
    initial = target if initial_weights is None else _weights(initial_weights, returns.columns)
    risky = np.array(returns.columns != "RF")
    rate = cost_bps / 10_000
    held = np.zeros(len(target))
    held[returns.columns.get_loc("RF")] = 1
    months = returns.index.to_period("M")
    rows = []
    for day, daily in enumerate(returns.to_numpy(dtype=float)):
        retained, turnover = 1.0, 0.0
        if day == 0 or months[day] != months[day - 1]:
            desired = initial if day == 0 else target
            if rate:
                # Solve v + fee * sum(abs(v * target - existing)) = 1.
                # Both ETF sales and purchases fund the final target weights.
                low, high = 0.0, 1.0
                for _ in range(55):
                    value = (low + high) / 2
                    volume = np.abs(value * desired[risky] - held[risky]).sum()
                    if value + rate * volume > 1:
                        high = value
                    else:
                        low = value
                retained = (low + high) / 2
            turnover = float(np.abs(retained * desired[risky] - held[risky]).sum())
            held = desired.copy()
        gross_return = float(held @ daily)
        net_return = gross_return + (retained - 1) * (1 + gross_return)
        rows.append((net_return, turnover, rate * turnover))
        held *= (1 + daily) / (1 + gross_return)
    return pd.DataFrame(rows, index=returns.index,
                        columns=["return", "traded_fraction", "fee_fraction"])
