"""Controlled wealth paths protect the common-date downturn comparisons."""

import numpy as np
import pandas as pd
import pytest

from financial_economics.evaluation.metrics import drawdowns, monthly_equal_weight
from financial_economics.portfolio.declines import (
    compare_episodes,
    drawdown_episodes,
    rolling_correlations,
)


SEED = pd.Timestamp("2020-01-01")


def returns_from_wealth(values):
    count = len(next(iter(values.values())))
    wealth = pd.DataFrame(values, index=pd.bdate_range("2020-01-02", periods=count))
    previous = wealth.shift(1)
    previous.iloc[0] = 1.0
    return wealth / previous - 1


def test_first_day_threshold_loss_uses_real_seed_close():
    returns = pd.Series([-0.10], index=pd.DatetimeIndex(["2020-01-02"]), name="SPY")
    result = drawdown_episodes(returns, SEED)
    assert len(result) == 1
    row = result.iloc[0]
    assert row.peak_date == SEED
    assert row.threshold_date == row.trough_date == returns.index[0]
    assert row.benchmark_drawdown == pytest.approx(-0.10)
    assert not row.recovered
    assert pd.isna(row.benchmark_recovery_date)
    assert row.end_date == returns.index[-1]


def test_decline_just_below_threshold_is_not_selected():
    returns = pd.Series([-0.0999, 0.01], index=pd.bdate_range("2020-01-02", periods=2))
    assert drawdown_episodes(returns, SEED).empty


def test_nested_losses_stay_one_episode_until_recovery_and_open_episode_survives():
    returns = returns_from_wealth({"SPY": [1.2, 1.05, 1.1, .96, 1.2, 1.32, 1.17, 1.22]})
    result = drawdown_episodes(returns["SPY"], SEED)
    assert len(result) == 2
    first, second = result.iloc[0], result.iloc[1]
    dates = returns.index
    assert first.peak_date == dates[0]
    assert first.threshold_date == dates[1]
    assert first.trough_date == dates[3]
    assert first.end_date == first.benchmark_recovery_date == dates[4]
    assert first.recovered
    assert first.benchmark_drawdown == pytest.approx(-0.20)
    assert second.peak_date == dates[5]
    assert second.trough_date == dates[6]
    assert second.end_date == dates[7]
    assert not second.recovered
    assert first.episode != second.episode
    comparison = compare_episodes(returns, result, SEED)
    assert comparison.iloc[0].recovery_date == dates[4]


def test_common_start_loss_is_not_the_mix_own_peak_drawdown():
    returns = returns_from_wealth({
        "SPY": [1.1, .99, .88, 1.0, 1.1],
        "Equal weight": [1.0, 1.2, 1.0, 1.05, 1.0],
    })
    episodes = drawdown_episodes(returns["SPY"], SEED)
    result = compare_episodes(returns, episodes, SEED).set_index("asset")
    mix = result.loc["Equal weight"]
    assert mix.peak_date == returns.index[0]
    assert mix.trough_date == returns.index[2]
    assert mix.peak_to_trough_return == pytest.approx(0)
    assert mix.window_min_return == pytest.approx(0)
    assert drawdowns(returns.loc[:returns.index[2]])["Equal weight"].min() == pytest.approx(-1 / 6)
    assert mix.recovery_status == "at_or_above_baseline"
    assert not mix.recovery_censored
    assert mix.recovery_date == returns.index[2]
    assert mix.recovery_trading_days == mix.recovery_calendar_days == 0


def test_window_minimum_and_recovery_use_their_explicit_common_date_definitions():
    returns = returns_from_wealth({
        "SPY": [1.0, .9, .8, .9, 1.0],
        "Equal weight": [1.0, .85, .9, 1.01, 1.02],
    })
    episodes = drawdown_episodes(returns["SPY"], SEED)
    result = compare_episodes(returns, episodes, SEED).set_index("asset")
    mix = result.loc["Equal weight"]
    assert mix.peak_to_trough_return == pytest.approx(-.1)
    assert mix.window_min_return == pytest.approx(-.15)
    assert mix.window_min_date == returns.index[1]
    assert mix.recovery_status == "recovered"
    assert not mix.recovery_censored
    assert mix.recovery_date == returns.index[3]
    assert mix.recovery_trading_days == 1
    assert mix.recovery_calendar_days == (returns.index[3] - returns.index[2]).days
    assert mix.observation_end == returns.index[-1]
    assert mix.observed_after_trough_trading_days == 2


def test_unrecovered_portfolio_is_censored_without_inventing_zero_duration():
    returns = returns_from_wealth({
        "SPY": [1.0, .8, .9, 1.0],
        "Equal weight": [1.0, .9, .95, .99],
    })
    result = compare_episodes(returns, drawdown_episodes(returns["SPY"], SEED), SEED)
    mix = result.set_index("asset").loc["Equal weight"]
    assert mix.recovery_status == "not_recovered"
    assert mix.recovery_censored
    assert pd.isna(mix.recovery_date)
    assert pd.isna(mix.recovery_trading_days)
    assert pd.isna(mix.recovery_calendar_days)
    assert mix.observation_end == returns.index[-1]
    assert mix.observed_after_trough_trading_days == 2


def test_episode_rebasing_preserves_monthly_holdings_instead_of_rebalancing_at_peak():
    dates = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-03",
                              "2020-02-04", "2020-02-05"])
    assets = pd.DataFrame({"SPY": [.1, -.2, -.1, .2, .2],
                           "Other": [1., 0., 0., 0., 0.]}, index=dates)
    combined = pd.DataFrame({"SPY": assets["SPY"],
                             "Equal weight": monthly_equal_weight(assets)})
    episodes = drawdown_episodes(combined["SPY"], pd.Timestamp("2020-01-29"))
    mix = compare_episodes(combined, episodes, pd.Timestamp("2020-01-29"))
    mix = mix.set_index("asset").loc["Equal weight"]
    full_wealth = (1 + combined["Equal weight"]).cumprod()
    expected = full_wealth.loc[dates[2]] / full_wealth.loc[dates[0]] - 1
    reset_at_peak = (1 + monthly_equal_weight(assets.loc[dates[1]:dates[2]])).prod() - 1
    assert mix.peak_to_trough_return == pytest.approx(expected)
    assert not np.isclose(expected, reset_at_peak)


def test_rolling_correlation_warmup_constants_and_no_lookahead():
    returns = pd.DataFrame({"SPY": [.01, -.02, .03, .01, -.01, .02],
                            "Same": [.01, -.02, .03, .01, -.01, .02],
                            "Opposite": [-.01, .02, -.03, -.01, .01, -.02],
                            "Constant": [0.] * 6},
                           index=pd.bdate_range("2020-01-02", periods=6))
    result = rolling_correlations(returns, "SPY", window=3)
    assert list(result.columns) == ["Same", "Opposite", "Constant"]
    assert result.index.equals(returns.index)
    assert result.iloc[:2].isna().all().all()
    np.testing.assert_allclose(result["Same"].iloc[2:], 1)
    np.testing.assert_allclose(result["Opposite"].iloc[2:], -1)
    assert result["Constant"].isna().all()
    prefix = rolling_correlations(returns.iloc[:4], "SPY", window=3)
    pd.testing.assert_frame_equal(result.iloc[:4], prefix)


@pytest.mark.parametrize("threshold", [0, -0.1, 1.0, np.nan, True])
def test_invalid_episode_threshold_is_rejected(threshold):
    returns = pd.Series([-.2, .1], index=pd.bdate_range("2020-01-02", periods=2))
    with pytest.raises((ValueError, TypeError)):
        drawdown_episodes(returns, SEED, threshold=threshold)


def test_seed_must_be_a_valid_preceding_daily_close():
    returns = pd.Series([-.2, .1], index=pd.bdate_range("2020-01-02", periods=2))
    for seed in [returns.index[0], returns.index[-1], pd.NaT, "2020-01-01 12:00"]:
        with pytest.raises((ValueError, TypeError)):
            drawdown_episodes(returns, seed)


@pytest.mark.parametrize("window", [0, 1, -1, 2.5, True])
def test_invalid_correlation_window_is_rejected(window):
    returns = returns_from_wealth({"SPY": [1., .9, 1.1], "Other": [1., 1.1, .95]})
    with pytest.raises((ValueError, TypeError)):
        rolling_correlations(returns, "SPY", window=window)
