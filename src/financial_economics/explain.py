"""Phase 4: could chance explain the Phase 3 comparison, and which funds drove the mix?"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import _check_frame, _check_returns, monthly_equal_weight_holdings
from .comparison import compare_portfolios
from .stress import drawdown_episodes

MEASURES = ["sharpe", "cagr", "annualized_volatility"]
BATCH = 500  # resamples drawn at once, to keep memory use small


@dataclass(frozen=True)
class ExplainResult:
    calibration: dict
    uncertainty: pd.DataFrame
    contributions: pd.DataFrame
    episode_windows: list[str]
    correlation: pd.Series


def resample_rows(n: int, resamples: int, block_length: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Stationary bootstrap (Politis and Romano, 1994): runs of real days from random starts."""
    new_block = rng.random((resamples, n)) < 1 / block_length
    new_block[:, 0] = True
    starts = rng.integers(0, n, size=(resamples, n))
    days = np.arange(n)
    # Each day continues the run that began at the latest new block, wrapping at the end.
    block_began = np.maximum.accumulate(np.where(new_block, days, 0), axis=1)
    return (np.take_along_axis(starts, block_began, axis=1) + days - block_began) % n


def _measures(returns: np.ndarray, risk_free: np.ndarray, trading_days: int) -> dict:
    """Phase 1 formulas, calculated along the last axis."""
    excess = returns - risk_free
    days = returns.shape[-1]
    return {
        "sharpe": np.sqrt(trading_days) * excess.mean(-1) / excess.std(-1, ddof=1),
        "cagr": np.prod(1 + returns, axis=-1) ** (trading_days / days) - 1,
        "annualized_volatility": returns.std(-1, ddof=1) * np.sqrt(trading_days),
    }


def bootstrap_differences(returns: pd.DataFrame, risk_free: pd.Series, portfolio: str,
                          references: list[str], trading_days: int = 252,
                          resamples: int = 10_000, block_length: float = 21,
                          seed: int = 20260101) -> pd.DataFrame:
    """95% ranges for portfolio-minus-reference differences, resampling the same days for all."""
    _check_returns(returns, minimum=2)
    _check_frame(risk_free.to_frame(), "Cash returns", minimum=2)
    if not risk_free.index.equals(returns.index):
        raise ValueError("Cash-return dates must exactly match portfolio dates.")
    if resamples < 100 or not 1 <= block_length <= len(returns):
        raise ValueError("Use at least 100 resamples and a block length within the sample.")
    values = returns[[portfolio, *references]].to_numpy().T
    rf = risk_free.to_numpy()
    observed = _measures(values, rf, trading_days)
    rng = np.random.default_rng(seed)
    batches = []
    for done in range(0, resamples, BATCH):
        rows = resample_rows(len(rf), min(BATCH, resamples - done), block_length, rng)
        batches.append(_measures(values[:, rows], rf[rows], trading_days))

    table = []
    for measure in MEASURES:
        draws = np.concatenate([batch[measure] for batch in batches], axis=1)
        for position, reference in enumerate(references, start=1):
            difference = draws[0] - draws[position]
            # Higher Sharpe and growth are better; lower volatility is better.
            better = difference < 0 if measure == "annualized_volatility" else difference > 0
            low, high = np.percentile(difference, [2.5, 97.5])
            table.append({
                "portfolio": portfolio, "reference": reference, "measure": measure,
                "estimate": observed[measure][0] - observed[measure][position],
                "ci_low": low, "ci_high": high, "share_favoring_portfolio": better.mean(),
                "resamples": resamples, "mean_block_length": block_length, "seed": seed,
            })
    return pd.DataFrame(table)


def fund_contributions(returns: pd.DataFrame, weights: pd.DataFrame,
                       windows: list[tuple]) -> pd.DataFrame:
    """Each fund's gain or loss per $1 in the mix at a window's starting close.

    The amounts add up exactly to the mix's compounded return over the window.
    """
    _check_returns(returns)
    _check_frame(weights, "Contribution weights")
    if not weights.index.equals(returns.index) or set(weights.columns) != set(returns.columns):
        raise ValueError("Contribution weights must match return dates and assets exactly.")
    weights = weights.reindex(index=returns.index, columns=returns.columns)
    if (weights < 0).any().any() or not np.allclose(weights.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("Contribution weights must be nonnegative and sum to one.")
    rows = []
    for name, start, end in windows:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start >= end or start not in returns.index or end not in returns.index:
            raise ValueError("Windows need observed start and end closes, in increasing order.")
        days = (returns.index > start) & (returns.index <= end)
        held, daily = weights.loc[days].to_numpy(), returns.loc[days].to_numpy()
        mix_value = np.cumprod(1 + (held * daily).sum(axis=1))
        value_before = np.r_[1, mix_value[:-1]]
        gains = (value_before[:, None] * held * daily).sum(axis=0)
        for asset, gain in zip(returns.columns, gains):
            rows.append({"window": name, "start_date": start, "end_date": end,
                         "asset": asset, "contribution": gain})
    return pd.DataFrame(rows, columns=["window", "start_date", "end_date", "asset", "contribution"])


def explain_comparison(returns: pd.DataFrame, risk_free: pd.Series, benchmark: str,
                       bond_fund: str, evaluation_start: str = "2018-01-01",
                       trading_days: int = 252, resamples: int = 10_000,
                       block_length: float = 21, seed: int = 20260101) -> ExplainResult:
    """Repeat the Phase 3 split, then add uncertainty ranges and fund contributions."""
    if bond_fund not in returns or bond_fund == benchmark:
        raise ValueError("Choose a bond fund among the study assets, other than the benchmark.")
    result = compare_portfolios(returns, risk_free, benchmark, evaluation_start)
    calibration, evaluation = result.calibration, result.evaluation_returns
    uncertainty = bootstrap_differences(
        evaluation, risk_free.loc[evaluation.index], "Equal weight",
        [calibration["control_name"], benchmark], trading_days, resamples, block_length, seed)

    seed_date = calibration["evaluation_seed_date"]
    episodes = drawdown_episodes(evaluation[benchmark], seed_date, threshold=.10)
    windows = [(f"{row.peak_date:%b %Y} fall", row.peak_date, row.trough_date)
               for row in episodes.itertuples()]
    first, last = evaluation.index[0].year, evaluation.index[-1].year
    whole_period = (str(first) if first == last else f"{first}–{last}", seed_date,
                    evaluation.index[-1])
    contributions = fund_contributions(returns, monthly_equal_weight_holdings(returns),
                                       [*windows, whole_period])

    years = returns.index.year
    correlation = returns[benchmark].groupby(years).corr(returns[bond_fund])
    correlation.index.name, correlation.name = "year", "correlation"
    return ExplainResult(calibration, uncertainty, contributions,
                         [name for name, _, _ in windows], correlation)
