import numpy as np
import pandas as pd
import pytest

from financial_economics.paper_realistic import (delayed_signals, realistic_portfolios,
                                                 sharpe_difference_test, _tests)


SETTINGS = {"calibration_start": "2010-01-01", "evaluation_start": "2011-01-01",
            "evaluation_end": "2011-12-31", "primary_models": ["arfima"], "secondary_models": [],
            "primary_assets": ["TLT"], "secondary_assets": [], "monthly_target": .10,
            "monthly_multiplier": 22, "stress_quantile": .90, "defensive_cap": .20,
            "costs_bps": [0, 10], "main_cost_bps": 10}


def _inputs():
    calendar = pd.bdate_range("2009-12-01", "2012-01-31")
    rng = np.random.default_rng(3)
    daily = pd.DataFrame({"SPY": rng.normal(.0004, .01, len(calendar)),
                          "TLT": rng.normal(.0002, .008, len(calendar)),
                          "RF": .00005}, index=calendar)
    issues = calendar[(calendar >= "2009-12-31") & (calendar <= "2011-12-28")]
    level = 1e-4 * (1.5 + np.sin(np.arange(len(issues)) / 9))
    forecasts = pd.DataFrame({"arfima_bv": level, "arfima_refit_asof": issues}, index=issues)
    return daily, calendar, forecasts


def test_signals_trade_one_session_after_issue_and_earn_the_next():
    calendar = pd.bdate_range("2011-01-03", periods=6)
    forecasts = pd.DataFrame({"x": range(6)}, index=calendar)
    rows, dropped = delayed_signals(forecasts, calendar)
    assert list(rows.trade_date) == list(calendar[1:5]) and list(rows.earn_date) == list(calendar[2:6])
    assert dropped == 2


def test_sharpe_difference_is_zero_for_identical_series_and_signed_otherwise():
    rng = np.random.default_rng(1)
    base = rng.normal(.0005, .01, 3000)
    assert sharpe_difference_test(base, base + 1e-12 * rng.normal(size=3000), 0., 5)["estimate"] == pytest.approx(0, abs=1e-6)
    better = sharpe_difference_test(base + .0005, base, 0., 5)
    assert better["estimate"] > 0 and 0 <= better["p_value"] <= 1


def test_thresholds_and_static_control_use_only_past_forecasts():
    daily, calendar, forecasts = _inputs()
    result = realistic_portfolios(daily, calendar, forecasts, SETTINGS)
    altered = forecasts.copy()
    altered.loc[altered.index >= "2011-01-03", "arfima_bv"] *= 3
    changed = realistic_portfolios(daily, calendar, altered, SETTINGS)
    pd.testing.assert_frame_equal(result["thresholds"], changed["thresholds"])
    pd.testing.assert_frame_equal(result["static_controls"], changed["static_controls"])
    calibration = forecasts.loc["2010-01-01":"2010-12-31", "arfima_bv"]
    expected = np.minimum(.10 / (22 * np.sqrt(calibration)), 1).mean()
    assert result["static_controls"].static_spy_weight.iloc[0] == pytest.approx(expected)


def test_defensive_asset_is_held_only_on_stress_signals_and_cash_otherwise():
    daily, calendar, forecasts = _inputs()
    result = realistic_portfolios(daily, calendar, forecasts, SETTINGS)
    decisions = result["decisions"]
    assert decisions.stress.any() and not decisions.stress.all()
    paths = result["paths"][0]
    calm = decisions.loc[~decisions.stress, "earn_date"]
    # Without fees, both portfolios hold identical targets on calm signal days after a calm reset.
    previous_calm = decisions.stress.shift(fill_value=False).eq(False).to_numpy()
    both = calm[previous_calm[~decisions.stress.to_numpy()]]
    np.testing.assert_allclose(paths.loc[both, "arfima_TLT"], paths.loc[both, "arfima_cash"], atol=1e-12)
    assert (result["accounting"][10]["arfima_cash"].fee_fraction >= 0).all()
    assert result["paths"][10]["arfima_cash"].sum() < result["paths"][0]["arfima_cash"].sum()


def test_realistic_portfolios_accept_a_named_index_benchmark_with_zero_cash():
    calendar = pd.bdate_range("2009-12-01", "2012-01-31")
    rng = np.random.default_rng(5)
    daily = pd.DataFrame({"INDEX": rng.normal(.0003, .012, len(calendar)), "RF": 0.}, index=calendar)
    issues = calendar[(calendar >= "2009-12-31") & (calendar <= "2011-12-28")]
    forecasts = pd.DataFrame({"arfima_bv": 1e-4 * (1.5 + np.sin(np.arange(len(issues)) / 9)),
                              "arfima_refit_asof": issues}, index=issues)
    settings = {"benchmark": "INDEX", "calibration_start": "2010-01-01", "evaluation_start": "2011-01-01",
                "evaluation_end": "2011-12-31", "primary_models": ["arfima"], "secondary_models": [],
                "primary_assets": [], "secondary_assets": [], "monthly_target": .10, "monthly_multiplier": 22,
                "stress_quantile": .90, "defensive_cap": .20, "costs_bps": [0, 10], "main_cost_bps": 10}
    result = realistic_portfolios(daily, calendar, forecasts, settings)
    assert set(result["paths"][10]) == {"INDEX", "arfima_cash", "arfima_static"}
    weight = result["static_controls"].static_spy_weight.iloc[0]
    returns = daily.loc["2011-01-01":"2011-12-31", "INDEX"]
    # With zero cash and no fees, the static mix's first-day return is its fixed weight times the index return.
    assert result["paths"][0]["arfima_static"].iloc[0] == pytest.approx(weight * returns.iloc[0])


@pytest.mark.parametrize("frequency, period, deferred", [("weekly", "W-SUN", "2011-01-18"),
                                                        ("monthly", "M", "2011-02-02")])
def test_cadence_uses_execution_periods_keeps_seed_and_drifts_between_trades(frequency, period, deferred):
    daily, calendar, forecasts = _inputs()
    # A holiday and a missing first signal must postpone a trade, never invent one.
    calendar = calendar.difference(pd.to_datetime(["2011-01-17"]))
    daily = daily.reindex(calendar)
    forecasts = forecasts.loc[forecasts.index.isin(calendar)].drop(pd.Timestamp("2011-01-31"))
    baseline = realistic_portfolios(daily, calendar, forecasts, SETTINGS)
    result = realistic_portfolios(daily, calendar, forecasts, SETTINGS | {"rebalance_frequency": frequency})
    decisions = result["decisions"]
    selected = decisions.loc[decisions.scheduled]
    expected = decisions.groupby(decisions.trade_date.dt.to_period(period), sort=False).head(1)
    pd.testing.assert_frame_equal(selected, expected)
    assert selected.iloc[0].trade_date == pd.Timestamp("2010-12-31")
    assert selected.iloc[1].trade_date == pd.Timestamp("2011-01-03")
    assert pd.Timestamp(deferred) in set(selected.trade_date)
    pd.testing.assert_frame_equal(result["thresholds"], baseline["thresholds"])
    pd.testing.assert_frame_equal(result["static_controls"], baseline["static_controls"])
    for cost in SETTINGS["costs_bps"]:
        pd.testing.assert_series_equal(result["paths"][cost].arfima_static, baseline["paths"][cost].arfima_static)
    accounting = result["accounting"][10]["arfima_cash"]
    assert accounting.index.equals(baseline["paths"][10].index)
    idle = accounting.index.difference(pd.DatetimeIndex(selected.earn_date).union(accounting.index[:1]))
    assert accounting.loc[idle, "traded_fraction"].eq(0).all()
    assert accounting.loc[idle, "fee_fraction"].eq(0).all()
    # Independently evolve stock/cash dollars between two scheduled trades.
    next_trade = selected.earn_date.iloc[1]
    start = accounting.index.get_loc(next_trade)
    following = accounting.index[start + 1]
    stock = selected.stock_target.iloc[1]
    market = result["returns"]
    drifted = stock * (1 + market.loc[next_trade, "SPY"]) / (
        stock * (1 + market.loc[next_trade, "SPY"]) + (1 - stock) * (1 + market.loc[next_trade, "RF"]))
    expected_return = drifted * market.loc[following, "SPY"] + (1 - drifted) * market.loc[following, "RF"]
    assert accounting.loc[following, "return"] == pytest.approx(expected_return)


def test_adding_a_descriptive_asset_leaves_existing_portfolios_and_tests_unchanged():
    daily, calendar, forecasts = _inputs()
    daily["IEF"] = np.random.default_rng(5).normal(.0001, .004, len(daily))
    settings = SETTINGS | {"sharpe_hac_lags": 5, "monthly_hac_lags": 1, "alpha": .05}
    baseline = realistic_portfolios(daily, calendar, forecasts, settings)
    extended = realistic_portfolios(daily, calendar, forecasts, settings | {"secondary_assets": ["IEF"]})
    for cost in SETTINGS["costs_bps"]:
        pd.testing.assert_frame_equal(extended["paths"][cost][baseline["paths"][cost].columns], baseline["paths"][cost])
    assert "arfima_IEF" in extended["paths"][10]
    tests, reference = _tests(extended, calendar, settings), _tests(baseline, calendar, settings)
    added = tests.comparison.eq("IEF_minus_cash")
    pd.testing.assert_frame_equal(tests.loc[~added].reset_index(drop=True), reference)
    assert tests.loc[added, "family"].eq("descriptive").all()


def test_explicit_daily_is_identical_to_default_and_invalid_cadence_fails():
    daily, calendar, forecasts = _inputs()
    baseline = realistic_portfolios(daily, calendar, forecasts, SETTINGS)
    explicit = realistic_portfolios(daily, calendar, forecasts, SETTINGS | {"rebalance_frequency": "daily"})
    for cost in SETTINGS["costs_bps"]:
        pd.testing.assert_frame_equal(explicit["paths"][cost], baseline["paths"][cost])
    pd.testing.assert_frame_equal(explicit["decisions"], baseline["decisions"])
    with pytest.raises(ValueError, match="daily, weekly or monthly"):
        realistic_portfolios(daily, calendar, forecasts, SETTINGS | {"rebalance_frequency": "yearly"})


def test_empty_month_keeps_holdings_and_non_daily_tests_are_descriptive():
    daily, calendar, forecasts = _inputs()
    trades = calendar[calendar.get_indexer(forecasts.index) + 1]
    forecasts = forecasts.loc[trades.month != 3]
    settings = SETTINGS | {"rebalance_frequency": "monthly", "sharpe_hac_lags": 5,
                           "monthly_hac_lags": 1, "alpha": .05}
    result = realistic_portfolios(daily, calendar, forecasts, settings)
    decisions = result["decisions"]
    assert not decisions.loc[decisions.scheduled, "trade_date"].dt.month.eq(3).any()
    march = result["accounting"][10]["arfima_cash"].loc["2011-03"]
    # A February execution may earn March's first return; no later March reset exists.
    assert march.iloc[1:].traded_fraction.eq(0).all()
    assert len(result["returns"]) == len(daily.loc["2011"])
    tests = _tests(result, calendar, settings)
    assert tests.family.eq("descriptive").all() and tests.holm_p_value.isna().all()
