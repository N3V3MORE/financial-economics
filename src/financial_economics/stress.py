"""Retrospective downturn comparisons on shared benchmark closing dates."""

import numpy as np
import pandas as pd

from .analysis import _check_returns


EPISODE_COLUMNS = [
    "episode", "peak_date", "threshold_date", "trough_date", "end_date",
    "benchmark_recovery_date", "benchmark_drawdown", "recovered",
]
COMPARISON_COLUMNS = [
    "episode", "asset", "peak_date", "trough_date", "peak_to_trough_return",
    "window_min_return", "window_min_date", "recovery_date", "recovery_status",
    "recovery_censored", "recovery_trading_days", "recovery_calendar_days",
    "observation_end", "observed_after_trough_trading_days",
]
RECOVERY_TOLERANCE = 1e-12


def _wealth(returns: pd.DataFrame, seed_date) -> pd.DataFrame:
    _check_returns(returns, minimum=1)
    seed = pd.Timestamp(seed_date)
    if (pd.isna(seed) or seed.tz is not None or seed != seed.normalize()
            or seed >= returns.index[0]):
        raise ValueError("Seed must be a daily closing date before the first return.")
    anchor = pd.DataFrame(1.0, index=pd.DatetimeIndex([seed]), columns=returns.columns)
    wealth = pd.concat([anchor, (1 + returns).cumprod()])
    if not np.isfinite(wealth.to_numpy()).all() or (wealth <= 0).any().any():
        raise ValueError("Compounded wealth must remain finite and positive.")
    wealth.index.name = "Date"
    return wealth


def drawdown_episodes(benchmark_returns: pd.Series, seed_date,
                      threshold: float = 0.10) -> pd.DataFrame:
    """All benchmark peak-to-recovery episodes reaching the chosen decline."""
    if not isinstance(benchmark_returns, pd.Series):
        raise ValueError("Benchmark returns must be a Series.")
    if isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Drawdown threshold must be between zero and one.")
    wealth = _wealth(benchmark_returns.to_frame("benchmark"), seed_date)["benchmark"]
    values, dates = wealth.to_numpy(), wealth.index
    peak = trough = 0
    crossing = None
    rows = []

    def record(end, recovered):
        rows.append({
            "episode": len(rows) + 1,
            "peak_date": dates[peak], "threshold_date": dates[crossing],
            "trough_date": dates[trough], "end_date": dates[end],
            "benchmark_recovery_date": dates[end] if recovered else pd.NaT,
            "benchmark_drawdown": values[trough] / values[peak] - 1,
            "recovered": recovered,
        })

    for day in range(1, len(values)):
        if values[day] >= values[peak] * (1 - RECOVERY_TOLERANCE):
            if crossing is not None:
                record(day, True)
            peak = trough = day
            crossing = None
        else:
            if values[day] < values[trough]:
                trough = day
            # Comparing wealth levels avoids subtraction rounding at exactly 10%.
            if crossing is None and values[day] <= values[peak] * (1 - threshold):
                crossing = day
    if crossing is not None:
        record(len(values) - 1, False)
    return pd.DataFrame(rows, columns=EPISODE_COLUMNS)


def compare_episodes(returns: pd.DataFrame, episodes: pd.DataFrame,
                     seed_date) -> pd.DataFrame:
    """Compare existing holdings, preserving monthly weights and shared dates."""
    wealth = _wealth(returns, seed_date)
    required = {"episode", "peak_date", "trough_date"}
    if not required.issubset(episodes.columns) or episodes["episode"].duplicated().any():
        raise ValueError("Episodes require unique identifiers and shared peak/trough dates.")
    rows = []
    for episode in episodes.itertuples(index=False):
        peak, trough = pd.Timestamp(episode.peak_date), pd.Timestamp(episode.trough_date)
        if peak not in wealth.index or trough not in wealth.index or peak >= trough:
            raise ValueError("Episode boundaries must be observed closes in increasing order.")
        for asset in returns:
            baseline = wealth.at[peak, asset]
            path = wealth.loc[peak:trough, asset] / baseline
            after_trough = wealth.loc[trough:, asset]
            hits = after_trough.index[after_trough >= baseline * (1 - RECOVERY_TOLERANCE)]
            recovered = bool(len(hits))
            recovery = hits[0] if recovered else pd.NaT
            status = ("at_or_above_baseline" if recovered and recovery == trough
                      else "recovered" if recovered else "not_recovered")
            rows.append({
                "episode": episode.episode, "asset": asset,
                "peak_date": peak, "trough_date": trough,
                "peak_to_trough_return": path.iloc[-1] - 1,
                "window_min_return": path.min() - 1, "window_min_date": path.idxmin(),
                "recovery_date": recovery, "recovery_status": status,
                "recovery_censored": not recovered,
                "recovery_trading_days": (wealth.index.get_loc(recovery)
                                          - wealth.index.get_loc(trough)) if recovered else np.nan,
                "recovery_calendar_days": (recovery - trough).days if recovered else np.nan,
                "observation_end": wealth.index[-1],
                "observed_after_trough_trading_days": (len(wealth) - 1
                                                       - wealth.index.get_loc(trough)),
            })
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def rolling_correlations(returns: pd.DataFrame, benchmark: str,
                         window: int = 63) -> pd.DataFrame:
    """Trailing daily-return correlations; no filling or future observations."""
    _check_returns(returns, minimum=2)
    if benchmark not in returns:
        raise ValueError("The benchmark must be among the return columns.")
    if (not isinstance(window, int) or isinstance(window, bool)
            or window < 2 or window > len(returns)):
        raise ValueError("Rolling window must be an integer from 2 to the sample length.")
    others = returns.drop(columns=benchmark)
    samples = others.rolling(window, min_periods=window)
    market = returns[benchmark].rolling(window, min_periods=window)
    correlation = samples.corr(returns[benchmark])
    varying = samples.max().sub(samples.min()).gt(0)
    varying_market = market.max().sub(market.min()).gt(0)
    correlation = correlation.where(varying).where(varying_market, axis=0)
    return correlation.where(np.isfinite(correlation)).clip(-1, 1)
