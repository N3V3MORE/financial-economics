"""Figure tables: monthly scaling, tie-safe percentiles and decline paths."""

import numpy as np
import pandas as pd
import pytest

from financial_economics.evaluation.replication import DECLINES, MODELS
from financial_economics.reporting.figures import (PUBLISHED_ASSETS, asset_comparison, drawdown_paths,
                                               forecast_paths, quantile_profile)
from financial_economics.datasets.sources.paper import ASSETS


SETTINGS = {"monthly_multiplier": 22}


def inputs():
    dates = pd.bdate_range("2010-01-04", "2020-12-31")
    generator = np.random.default_rng(7)
    returns = pd.DataFrame(generator.normal(0, .01, (len(dates), 1 + len(ASSETS))),
                           index=dates, columns=["EQUITY", *ASSETS])
    bv = generator.uniform(1e-5, 4e-4, len(dates))
    forecasts = pd.DataFrame({"return_date": dates, "realized_bv": bv},
                             index=pd.DatetimeIndex([dates[0] - pd.offsets.BDay(), *dates[:-1]], name="forecast_date"))
    for number, model in enumerate(MODELS):
        forecasts[f"{model}_bv"] = bv * (1 + number / 10)
    return returns, forecasts


def test_forecast_paths_use_the_papers_monthly_scaling():
    _, forecasts = inputs()
    paths = forecast_paths(forecasts, SETTINGS)
    assert paths.realized.iloc[3] == pytest.approx(22 * np.sqrt(forecasts.realized_bv.iloc[3]))
    assert paths.gjr.iloc[3] == pytest.approx(22 * np.sqrt(1.2 * forecasts.realized_bv.iloc[3]))


def test_quantile_groups_stay_equal_count_when_forecasts_are_floored():
    returns, forecasts = inputs()
    forecasts.loc[forecasts.index[:500], "gru_multivar_bv"] = 1e-12
    profile = quantile_profile(forecasts, returns, SETTINGS, bins=10)
    counts = profile.loc[profile.model.eq("gru_multivar"), "count"]
    assert len(counts) == 10 and counts.max() - counts.min() <= 1
    top = forecasts.arfima_bv.rank(method="first") > .9 * len(forecasts)
    expected = returns.EQUITY.loc[forecasts.loc[top, "return_date"]].std()
    assert profile.query("model == 'arfima' and percentile == 10")["std"].item() == pytest.approx(expected)


def test_correlation_check_keeps_published_values_beside_ours():
    returns, _ = inputs()
    correlation, check = asset_comparison(returns)
    assert list(correlation.asset) == ["EQUITY", *ASSETS]
    row = check.set_index("asset").loc["US30"]
    assert row.paper_correlation == PUBLISHED_ASSETS.loc["US30", "paper_correlation"]
    assert row.our_correlation == pytest.approx(returns.US30.corr(returns.EQUITY))


def test_decline_paths_start_at_the_peak_and_compound_to_the_trough():
    returns, _ = inputs()
    daily = pd.DataFrame({"EQUITY": returns.EQUITY, "gru_multivar_clean": returns.EQUITY / 2})
    weights = pd.DataFrame({"return_date": returns.index, "model": "gru_multivar",
                            "stock_target": .6, "defensive_q0.9_cap0.3": .3})
    paths = drawdown_paths(daily, weights)
    label, peak, trough = DECLINES[0]
    covid = paths.loc[paths.decline.eq(label)]
    assert covid.Date.iloc[0] == pd.Timestamp(peak) and covid.Date.iloc[-1] == pd.Timestamp(trough)
    assert covid.index_cumulative.iloc[0] == 0 and covid.day.iloc[0] == 0
    inside = returns.EQUITY.loc[peak:trough].iloc[1:]
    assert covid.index_cumulative.iloc[-1] == pytest.approx((1 + inside).prod() - 1)
    assert covid.strategy_cumulative.iloc[-1] == pytest.approx((1 + inside / 2).prod() - 1)
    assert covid.defensive_weight.eq(.3).all() and covid.stock_weight.eq(.6).all()
