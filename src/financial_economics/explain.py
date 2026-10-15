"""How uncertain are the Phase 3 differences, and which funds drove the mix?"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import _check_returns, monthly_equal_weight_holdings
from .comparison import compare_portfolios
from .stress import drawdown_episodes

MEASURES = ["sharpe", "cagr", "annualized_volatility"]
UNCERTAINTY_COLUMNS = [
    "portfolio", "reference", "measure", "estimate", "ci_low", "ci_high",
    "share_favoring_portfolio", "resamples", "mean_block_length", "seed",
]
CONTRIBUTION_COLUMNS = ["window", "start_date", "end_date", "asset", "contribution"]


def stationary_bootstrap_indices(n: int, resamples: int, mean_block_length: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano rows: random starts, geometric block lengths, wrapping at the end."""
    restart = rng.random((resamples, n)) < 1 / mean_block_length
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(resamples, n))
    steps = np.arange(n)
    # Each row continues from its latest restart, one observation per day.
    latest = np.maximum.accumulate(np.where(restart, steps, 0), axis=1)
    return (np.take_along_axis(starts, latest, axis=1) + steps - latest) % n


def _measures(returns: np.ndarray, risk_free: np.ndarray, trading_days: int) -> dict:
    """Phase 1 formulas along the last axis, for one sample or many resamples."""
    excess = returns - risk_free
    n = returns.shape[-1]
    return {
        "sharpe": np.sqrt(trading_days) * excess.mean(-1) / excess.std(-1, ddof=1),
        "cagr": np.prod(1 + returns, axis=-1) ** (trading_days / n) - 1,
        "annualized_volatility": returns.std(-1, ddof=1) * np.sqrt(trading_days),
    }


def bootstrap_differences(
    returns: pd.DataFrame, risk_free: pd.Series, portfolio: str, references: list[str],
    trading_days: int = 252, resamples: int = 10_000, mean_block_length: float = 21,
    seed: int = 20260101, chunk: int = 500,
) -> pd.DataFrame:
    """Paired percentile intervals for portfolio-minus-reference differences.

    Every resample draws the same dates for all portfolios and the risk-free rate, so
    shared market moves stay paired. Blocks keep short-run dependence and volatility
    clustering; path-dependent measures such as maximum drawdown are not resampled.
    """
    _check_returns(returns, minimum=3)
    names = [portfolio, *references]
    if len(set(names)) != len(names) or not set(names).issubset(returns.columns):
        raise ValueError("Choose a portfolio and distinct references among the return columns.")
    if not isinstance(risk_free, pd.Series) or not returns.index.equals(risk_free.index):
        raise ValueError("Risk-free dates must match return dates exactly.")
    _check_returns(risk_free.to_frame("RF"))
    if (isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100
            or isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1):
        raise ValueError("Use at least 100 resamples and a positive chunk size.")
    if (isinstance(mean_block_length, bool) or not np.isfinite(mean_block_length)
            or not 1 <= mean_block_length <= len(returns)):
        raise ValueError("Mean block length must lie between one and the sample length.")
    values = returns[names].to_numpy(dtype=float).T
    rf = risk_free.to_numpy(dtype=float)
    observed = _measures(values, rf, trading_days)
    rng = np.random.default_rng(seed)
    draws = {measure: [] for measure in MEASURES}
    for size in [chunk] * (resamples // chunk) + [resamples % chunk] * bool(resamples % chunk):
        rows = stationary_bootstrap_indices(len(rf), size, mean_block_length, rng)
        sampled = _measures(values[:, rows], rf[rows], trading_days)
        for measure in MEASURES:
            draws[measure].append(sampled[measure])
    table = []
    for measure in MEASURES:
        stacked = np.concatenate(draws[measure], axis=1)
        if not np.isfinite(stacked).all():
            raise ValueError(f"Resampled {measure} values must be finite.")
        for position, reference in enumerate(references, start=1):
            difference = stacked[0] - stacked[position]
            # Lower volatility counts as favorable; higher Sharpe and CAGR do.
            favorable = difference < 0 if measure == "annualized_volatility" else difference > 0
            low, high = np.percentile(difference, [2.5, 97.5])
            table.append({
                "portfolio": portfolio, "reference": reference, "measure": measure,
                "estimate": observed[measure][0] - observed[measure][position],
                "ci_low": low, "ci_high": high,
                "share_favoring_portfolio": favorable.mean(),
                "resamples": resamples, "mean_block_length": mean_block_length, "seed": seed,
            })
    return pd.DataFrame(table, columns=UNCERTAINTY_COLUMNS)


def fund_contributions(returns: pd.DataFrame, weights: pd.DataFrame,
                       windows: list[tuple[str, object, object]]) -> pd.DataFrame:
    """Each fund's gain or loss per $1 held at the window's starting close.

    A window runs from one closing date to a later one, so its first return is the day
    after the start. A fund's contribution sums its own dollar gains within the window:
    portfolio value before the day times the fund's weight times its return. The
    contributions therefore add up exactly to the portfolio's compounded return.
    """
    _check_returns(returns)
    if not weights.index.equals(returns.index) or list(weights.columns) != list(returns.columns):
        raise ValueError("Weights must have the same dates and funds as returns.")
    if not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-9) or (weights < 0).any().any():
        raise ValueError("Weights must be long-only and sum to one each day.")
    rows = []
    for name, start, end in windows:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        # Contributions are per $1 at the starting close, so that close must exist.
        if start >= end or start not in returns.index or end not in returns.index:
            raise ValueError("Windows need observed start and end closes, in increasing order.")
        days = (returns.index > start) & (returns.index <= end)
        held, daily = weights.loc[days].to_numpy(), returns.loc[days].to_numpy()
        portfolio = (held * daily).sum(axis=1)
        value_before = np.r_[1, np.cumprod(1 + portfolio)[:-1]]
        gains = (value_before[:, None] * held * daily).sum(axis=0)
        rows += [{"window": name, "start_date": start, "end_date": end,
                  "asset": asset, "contribution": gain}
                 for asset, gain in zip(returns.columns, gains)]
    return pd.DataFrame(rows, columns=CONTRIBUTION_COLUMNS)


@dataclass(frozen=True)
class ExplainResult:
    calibration: dict
    uncertainty: pd.DataFrame
    contributions: pd.DataFrame
    episode_windows: list[str]
    correlation: pd.Series


def explain_comparison(
    returns: pd.DataFrame, risk_free: pd.Series, benchmark: str, bond_fund: str,
    evaluation_start: str = "2018-01-01", trading_days: int = 252,
    resamples: int = 10_000, mean_block_length: float = 21, seed: int = 20260101,
) -> ExplainResult:
    """Re-run the Phase 3 split, then add uncertainty ranges and fund contributions."""
    if bond_fund not in returns or bond_fund == benchmark:
        raise ValueError("Choose a bond fund among the study assets, other than the benchmark.")
    result = compare_portfolios(returns, risk_free, benchmark, evaluation_start)
    calibration, evaluation = result.calibration, result.evaluation_returns
    uncertainty = bootstrap_differences(
        evaluation, risk_free.loc[evaluation.index], "Equal weight",
        [calibration["control_name"], benchmark], trading_days, resamples, mean_block_length, seed)
    seed_date = calibration["evaluation_seed_date"]
    episodes = drawdown_episodes(evaluation[benchmark], seed_date, threshold=.10)
    windows = [(f"{pd.Timestamp(row.peak_date):%b %Y} fall", row.peak_date, row.trough_date)
               for row in episodes.itertuples()]
    first, last = evaluation.index[0].year, evaluation.index[-1].year
    whole = (str(first) if first == last else f"{first}–{last}", seed_date, evaluation.index[-1])
    contributions = fund_contributions(returns, monthly_equal_weight_holdings(returns),
                                       [*windows, whole])
    pair = returns[[benchmark, bond_fund]]
    correlation = pair.groupby(pair.index.year).apply(
        lambda year: year[benchmark].corr(year[bond_fund]))
    correlation.index.name, correlation.name = "year", "correlation"
    return ExplainResult(calibration, uncertainty, contributions,
                         [name for name, *_ in windows], correlation)
