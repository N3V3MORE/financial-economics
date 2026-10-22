"""CPU-native calendar alignment and finite-history EWMA variance forecasts."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_economics.evaluation.metrics import _check_returns


@dataclass(frozen=True)
class ForecastData:
    rows: pd.DataFrame
    inputs: np.ndarray


def prepare_forecasts(spy: pd.Series, calendar: pd.DatetimeIndex,
                      lookback: int = 22) -> ForecastData:
    """Issue at t, trade at t+1 close, and earn the daily return ending at t+2."""
    if not isinstance(spy, pd.Series):
        raise ValueError("SPY returns must be a Series.")
    if not isinstance(lookback, int) or isinstance(lookback, bool) or lookback < 1:
        raise ValueError("Lookback must be a positive integer.")
    _check_returns(spy.to_frame("SPY"), minimum=lookback + 2)
    if not isinstance(calendar, pd.DatetimeIndex) or not spy.index.equals(calendar):
        raise ValueError("SPY returns must match every expected trading-calendar date exactly.")
    returns = spy.to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        squares = returns ** 2
    if not np.isfinite(squares).all():
        raise ValueError("Squared SPY returns must be finite.")
    histories = np.lib.stride_tricks.sliding_window_view(returns, lookback)[:-2]
    inputs = np.stack((histories, histories ** 2), axis=-1)
    rows = pd.DataFrame({"trade_date": calendar[lookback:-1],
                         "return_date": calendar[lookback + 1:],
                         "squared_return": squares[lookback + 1:]},
                        index=calendar[lookback - 1:-2].rename("forecast_date"))
    return ForecastData(rows, inputs)


def qlike(squared_return, prediction) -> np.ndarray:
    """Variance loss log(v) + r-squared/v; zero observed returns remain valid."""
    observed = np.asarray(squared_return, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if (observed.shape != predicted.shape or not observed.size
            or not np.isfinite(observed).all() or (observed < 0).any()
            or not np.isfinite(predicted).all() or (predicted <= 0).any()):
        raise ValueError("QLIKE needs matching finite nonnegative targets and positive variances.")
    return np.log(predicted) + observed / predicted


def ewma_forecasts(prepared: ForecastData, settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose normalized 22-day decay by validation QLIKE, never evaluation loss.

    Earlier rows use the selected setting retrospectively; only evaluation rows
    are eligible for portfolio decisions after validation has ended.
    """
    rows, inputs = prepared.rows, prepared.inputs
    if (inputs.shape != (len(rows), settings.lookback, 2)
            or not np.isfinite(inputs).all()):
        raise ValueError("Forecast inputs must match dates, lookback, and two finite features.")
    validation = ((rows.index >= pd.Timestamp(settings.validation_start))
                  & (rows.return_date <= pd.Timestamp(settings.validation_end)))
    if not validation.any():
        raise ValueError("No date-aligned validation targets are available.")
    targets = rows.squared_return.to_numpy(dtype=float)
    records, forecasts = [], []
    ages = np.arange(settings.lookback - 1, -1, -1)
    for decay in settings.ewma_decays:
        weights = decay ** ages
        weights /= weights.sum()
        prediction = np.maximum(inputs[:, :, 1] @ weights, 1e-12)
        forecasts.append(prediction)
        loss = qlike(targets[validation], prediction[validation])
        records.append({"decay": decay, "qlike": float(loss.mean()),
                        "mse": float(np.mean((targets[validation] - prediction[validation]) ** 2)),
                        "validation_observations": int(validation.sum())})
    tuning = pd.DataFrame(records)
    selected = int(tuning.qlike.argmin())
    tuning["selected"] = np.arange(len(tuning)) == selected
    result = rows.copy()
    result["ewma_variance"] = forecasts[selected]
    return result, tuning
