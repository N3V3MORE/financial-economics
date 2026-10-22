"""Calibrate stock exposure on earlier returns, then compare later portfolios."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_economics.evaluation.metrics import _check_returns, monthly_equal_weight, monthly_fixed_weight


@dataclass(frozen=True)
class ComparisonResult:
    calibration: dict
    training_returns: pd.DataFrame
    evaluation_returns: pd.DataFrame
    exposure: pd.DataFrame


def stock_exposure(
    returns: pd.DataFrame, risk_free: pd.Series, benchmark: str
) -> pd.DataFrame:
    """OLS with an intercept on benchmark excess returns; descriptive coefficients."""
    _check_returns(returns, minimum=3)
    if benchmark not in returns:
        raise ValueError(f"Benchmark {benchmark} is missing from returns.")
    if not isinstance(risk_free, pd.Series):
        raise ValueError("Risk-free returns must be a Series.")
    _check_returns(risk_free.to_frame("RF"))
    if not returns.index.equals(risk_free.index):
        raise ValueError("Risk-free dates must match return dates exactly.")
    excess = returns.sub(risk_free, axis=0)
    market = excess[benchmark].to_numpy(dtype=float)
    roundoff = 4 * np.finfo(float).eps * (
        returns[benchmark].abs().max() + risk_free.abs().max()
    )
    if np.ptp(market) <= roundoff:
        raise ValueError("Stock exposure cannot use constant benchmark excess returns.")
    centered = market - market.mean()
    response = excess.to_numpy(dtype=float)
    means = response.mean(axis=0)
    beta = centered @ (response - means) / (centered @ centered)
    result = pd.DataFrame({
        "intercept_daily": means - beta * market.mean(),
        "beta": beta,
        "n_obs": len(returns),
    }, index=returns.columns)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Stock exposure estimates must be finite.")
    result.index.name = "asset"
    return result


def compare_portfolios(
    returns: pd.DataFrame, risk_free: pd.Series, benchmark: str = "SPY",
    evaluation_start: str = "2018-01-01",
) -> ComparisonResult:
    """Fit using pre-split observations only; preserve the mix's existing holdings."""
    _check_returns(returns)
    control = f"{benchmark} + cash"
    if {"Equal weight", "RF", control}.intersection(returns.columns):
        raise ValueError("Raw asset names conflict with comparison portfolios or cash.")
    if benchmark not in returns:
        raise ValueError(f"Benchmark {benchmark} is missing from returns.")
    if not isinstance(risk_free, pd.Series):
        raise ValueError("Risk-free returns must be a Series.")
    _check_returns(risk_free.to_frame("RF"))
    if not returns.index.equals(risk_free.index):
        raise ValueError("Risk-free dates must match return dates exactly.")
    split = pd.Timestamp(evaluation_start)
    if (pd.isna(split) or split.tz is not None or split != split.normalize()):
        raise ValueError("Evaluation start must be a valid timezone-naive daily date.")

    # Calculate holdings over the entire retained history before either slice.
    combined = returns[[benchmark]].copy()
    combined["Equal weight"] = monthly_equal_weight(returns)
    training = combined.loc[combined.index < split].copy()
    evaluation = combined.loc[combined.index >= split].copy()
    if len(training) < 3 or len(evaluation) < 3:
        raise ValueError("Training and evaluation each need at least three observations.")
    fitted = stock_exposure(training, risk_free.loc[training.index], benchmark)
    raw_beta = float(fitted.loc["Equal weight", "beta"])
    stock_weight = float(np.clip(raw_beta, 0, 1))
    weights = {benchmark: stock_weight, "RF": 1 - stock_weight}

    exposures = []
    for period, frame in [("training", training), ("evaluation", evaluation)]:
        cash_and_stock = pd.DataFrame({benchmark: frame[benchmark],
                                      "RF": risk_free.loc[frame.index]})
        # Training control is an after-fit diagnostic. Evaluation starts funded at targets.
        frame[control] = monthly_fixed_weight(cash_and_stock, weights, control)
        diagnostic = stock_exposure(frame, risk_free.loc[frame.index], benchmark).reset_index()
        diagnostic.insert(0, "period", period)
        exposures.append(diagnostic)

    calibration = {
        "benchmark": benchmark,
        "control_name": control,
        "requested_evaluation_start": str(split.date()),
        "training_start": str(training.index[0].date()),
        "training_end": str(training.index[-1].date()),
        "training_observations": len(training),
        "evaluation_start": str(evaluation.index[0].date()),
        "evaluation_end": str(evaluation.index[-1].date()),
        "evaluation_observations": len(evaluation),
        "evaluation_seed_date": str(training.index[-1].date()),
        "intercept_daily": float(fitted.loc["Equal weight", "intercept_daily"]),
        "raw_beta": raw_beta,
        "stock_weight": stock_weight,
        "cash_weight": 1 - stock_weight,
        "weight_clipped": raw_beta != stock_weight,
    }
    return ComparisonResult(calibration, training, evaluation,
                            pd.concat(exposures, ignore_index=True))
