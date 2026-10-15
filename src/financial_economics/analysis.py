"""Daily return calculations. All inputs and outputs use decimal returns."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _check_frame(data: pd.DataFrame, name: str, minimum: int = 1) -> None:
    if not isinstance(data, pd.DataFrame) or data.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty DataFrame.")
    if len(data) < minimum:
        raise ValueError(f"{name} needs at least {minimum} observations.")
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.hasnans:
        raise ValueError(f"{name} needs a DatetimeIndex without missing dates.")
    if not data.index.is_unique or not data.columns.is_unique:
        raise ValueError(f"{name} dates and column names must be unique.")
    if not data.index.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be increasing.")
    if data.index.tz is not None or not data.index.equals(data.index.normalize()):
        raise ValueError(f"{name} dates must be timezone-naive daily dates.")
    if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
           for dtype in data.dtypes):
        raise ValueError(f"{name} values must be numeric.")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} values must be finite, with no missing observations.")


def _check_returns(returns: pd.DataFrame, minimum: int = 1) -> None:
    _check_frame(returns, "Returns", minimum)
    if (returns <= -1).any().any():
        raise ValueError("Returns must be greater than -100%.")


def prices_to_returns(prices: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Calculate adjusted-price returns for inclusive dates, using a prior seed."""
    _check_frame(prices, "Prices", minimum=2)
    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if (pd.isna(start_date) or pd.isna(end_date)
            or start_date.tz is not None or end_date.tz is not None
            or start_date != start_date.normalize() or end_date != end_date.normalize()
            or start_date > end_date):
        raise ValueError("Choose valid daily start and end dates in increasing order.")
    if not (prices.index < start_date).any():
        raise ValueError("Prices need a pre-start seed observation.")
    returns = prices.pct_change(fill_method=None).loc[start_date:end_date]
    _check_returns(returns)
    return returns


def monthly_equal_weight(returns: pd.DataFrame) -> pd.Series:
    """Reset to equal weights each month; let holdings drift between resets."""
    _check_returns(returns)
    weights = np.full(returns.shape[1], 1 / returns.shape[1])
    months = returns.index.to_period("M")
    portfolio_returns = []
    for position, daily_returns in enumerate(returns.to_numpy(dtype=float)):
        if position == 0 or months[position] != months[position - 1]:
            weights[:] = 1 / len(weights)
        portfolio_return = float(weights @ daily_returns)
        portfolio_returns.append(portfolio_return)
        weights = weights * (1 + daily_returns) / (1 + portfolio_return)
    return pd.Series(portfolio_returns, index=returns.index, name="Equal weight")


def drawdowns(returns: pd.DataFrame) -> pd.DataFrame:
    """Signed declines from prior peaks, including the initial wealth of one."""
    _check_returns(returns)
    wealth = (1 + returns).cumprod()
    if not np.isfinite(wealth.to_numpy()).all() or (wealth <= 0).any().any():
        raise ValueError("Compounded wealth must remain finite and positive.")
    peaks = wealth.cummax().clip(lower=1)
    return wealth / peaks - 1


def performance_summary(
    returns: pd.DataFrame, risk_free: pd.Series, trading_days: int = 252
) -> pd.DataFrame:
    """Historical metrics; CAGR uses 252/T years and sample standard deviations."""
    _check_returns(returns, minimum=2)
    if not isinstance(trading_days, int) or isinstance(trading_days, bool) or trading_days <= 0:
        raise ValueError("trading_days must be a positive integer.")
    if not isinstance(risk_free, pd.Series):
        raise ValueError("Risk-free returns must be a Series.")
    _check_returns(risk_free.to_frame("RF"))
    if not returns.index.equals(risk_free.index):
        raise ValueError("Risk-free dates must match return dates exactly.")
    excess = returns.sub(risk_free, axis=0)
    # Subtraction and mean rounding can leave tiny noise in a constant column.
    roundoff = 4 * np.finfo(float).eps * (returns.abs().max() + risk_free.abs().max())
    varying_excess = excess.max() - excess.min() > roundoff
    excess_volatility = excess.std(ddof=1).where(varying_excess)
    cumulative = (1 + returns).prod() - 1
    result = pd.DataFrame({
        "cumulative_return": cumulative,
        "cagr": (1 + cumulative) ** (trading_days / len(returns)) - 1,
        "annualized_volatility": returns.std(ddof=1) * np.sqrt(trading_days),
        "sharpe": np.sqrt(trading_days) * excess.mean() / excess_volatility.where(excess_volatility > 0),
        "max_drawdown": drawdowns(returns).min(),
        "n_obs": len(returns),
    })
    result.index.name = "asset"
    return result


def capm_summary(
    returns: pd.DataFrame, factors: pd.DataFrame, hac_lags: int = 5
) -> pd.DataFrame:
    """OLS excess returns on Mkt-RF, with an intercept and HAC t inference."""
    _check_returns(returns)
    _check_frame(factors, "Factors")
    if not {"Mkt-RF", "RF"}.issubset(factors.columns):
        raise ValueError("Factors need Mkt-RF and RF columns in decimal units.")
    if not returns.index.equals(factors.index):
        raise ValueError("Factor dates must match return dates exactly.")
    if not isinstance(hac_lags, int) or isinstance(hac_lags, bool) or hac_lags < 0:
        raise ValueError("hac_lags must be a nonnegative integer.")
    if len(returns) <= hac_lags + 2:
        raise ValueError("CAPM needs more observations than hac_lags + 2.")
    if (factors["RF"] <= -1).any():
        raise ValueError("Risk-free returns must be greater than -100%.")
    market = factors["Mkt-RF"]
    if market.max() == market.min():
        raise ValueError("CAPM cannot use a constant market factor.")
    design = sm.add_constant(market, has_constant="add")
    rows = []
    for asset in returns:
        excess = returns[asset] - factors["RF"]
        if excess.max() == excess.min():
            raise ValueError(f"{asset} has constant excess returns; CAPM inference is undefined.")
        result = sm.OLS(excess, design).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_lags, "kernel": "bartlett", "use_correction": True},
            use_t=True,
        )
        precision = np.finfo(float).eps * np.linalg.norm(excess) * len(excess)
        if np.linalg.norm(result.resid) <= precision:
            raise ValueError(f"{asset} has a numerically perfect fit; CAPM inference is undefined.")
        confidence = result.conf_int()
        rows.append({
            "asset": asset,
            "alpha_daily": result.params["const"],
            "alpha_daily_se": result.bse["const"],
            "alpha_daily_ci_low": confidence.loc["const", 0],
            "alpha_daily_ci_high": confidence.loc["const", 1],
            "alpha_pvalue": result.pvalues["const"],
            "beta": result.params["Mkt-RF"],
            "beta_se": result.bse["Mkt-RF"],
            "beta_ci_low": confidence.loc["Mkt-RF", 0],
            "beta_ci_high": confidence.loc["Mkt-RF", 1],
            "beta_pvalue": result.pvalues["Mkt-RF"],
            "r_squared": result.rsquared,
            "n_obs": int(result.nobs),
        })
    return pd.DataFrame(rows).set_index("asset")
