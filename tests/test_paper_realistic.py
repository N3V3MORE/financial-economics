import numpy as np
import pandas as pd
import pytest

from financial_economics.studies.realistic import (_tests, delayed_signals, realistic_portfolios,
                                                 sharpe_difference_test)


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


def test_revised_rules_use_annual_units_common_entry_and_terminal_costs():
    daily, calendar, forecasts = _inputs()
    issues = calendar.get_indexer(forecasts.index)
    forecasts["trade_date"] = calendar[issues + 1]
    forecasts["return_date"] = calendar[issues + 2]
    settings = {**SETTINGS, "risk_mode": "annual", "annual_target": .10,
                "initial_cash": True, "cost_benchmark": True, "liquidate_end": True,
                "require_aligned_horizon": True}
    result = realistic_portfolios(daily, calendar, forecasts, settings)
    decisions = result["decisions"]
    expected = np.minimum(.10 / np.sqrt(252 * forecasts.loc[decisions.issue_date, "arfima_bv"]), 1.)
    np.testing.assert_allclose(decisions.stock_target, expected)
    paths = result["paths"][10]
    np.testing.assert_allclose(paths.iloc[0], result["returns"].RF.iloc[0])
    assert result["accounting"][10]["SPY"].fee_fraction.iloc[1] > 0
    assert result["accounting"][10]["SPY"].fee_fraction.iloc[-1] > 0
    forecasts["return_date"] = forecasts.trade_date
    with pytest.raises(ValueError, match="t\\+2"):
        realistic_portfolios(daily, calendar, forecasts, settings)


def test_terminal_sale_charges_actual_drifted_etf_value():
    from financial_economics.portfolio.execution import dynamic_costed_path

    dates = pd.bdate_range("2020-01-02", periods=2)
    returns = pd.DataFrame({"SPY": [1., 0.], "RF": [0., 0.]}, index=dates)
    targets = pd.DataFrame({"SPY": [.5, .5], "RF": [.5, .5]}, index=dates)
    result = dynamic_costed_path(returns, targets, 100, dates[:1], liquidate_end=True)
    # Entry leaves 1/1.005 wealth. Stock then doubles; sell that drifted value.
    expected_wealth = (1.5 - .01) / 1.005
    assert (1 + result["return"]).prod() == pytest.approx(expected_wealth)
    wealth_before = (1 + result["return"]).cumprod().shift(1, fill_value=1.)
    assert (wealth_before * result.fee_fraction).sum() == pytest.approx(.015 / 1.005)


def test_sharpe_difference_is_zero_for_identical_series_and_signed_otherwise():
    rng = np.random.default_rng(1)
    base = rng.normal(.0005, .01, 3000)
    similar = sharpe_difference_test(base, base + 1e-12 * rng.normal(size=3000), 0., 5)
    assert similar["estimate"] == pytest.approx(0, abs=1e-6)
    assert similar["status"] == "available" and np.isfinite(similar["se"]) and similar["se"] > 0
    better = sharpe_difference_test(base + .0005, base, 0., 5)
    assert better["estimate"] > 0 and 0 <= better["p_value"] <= 1


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("constant_first", [True, False])
def test_sharpe_difference_rejects_constant_excess_returns(constant_first):
    constant = np.full(3000, .001)
    variable = np.random.default_rng(2).normal(.0005, .01, 3000)
    first, second = (constant, variable) if constant_first else (variable, constant)
    test = sharpe_difference_test(first, second, 0., 20)
    assert test["status"] == "unavailable"
    assert all(np.isnan(test[key]) for key in ("estimate", "se", "z", "p_value"))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("invalid", [np.array([]), np.full(25, .001), np.array([np.nan] * 100),
                                    np.array([np.inf] * 100)])
def test_sharpe_difference_handles_insufficient_or_nonfinite_inputs(invalid):
    test = sharpe_difference_test(invalid, invalid, 0., 20)
    assert test["status"] == "unavailable" and np.isnan(test["p_value"])


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("multiple", [1., 2., 3.])
def test_equal_sharpe_positive_rescalings_have_no_inference(multiple):
    base = np.random.default_rng(1).normal(.0005, .01, 3000)
    test = sharpe_difference_test(base, multiple * base, 0., 20)
    assert test["estimate"] == pytest.approx(0., abs=1e-12)
    assert test["status"] == "degenerate_variance"
    assert all(np.isnan(test[key]) for key in ("se", "z", "p_value"))


def test_centered_sharpe_hac_matches_nondegenerate_four_moment_reference():
    rng = np.random.default_rng(902)
    count = 1200
    shocks = rng.normal(size=(count, 3))
    state = shocks.copy()
    for t in range(1, count):
        state[t] = np.array([.6, -.1, .2]) * state[t - 1] + shocks[t]
    risk_free = .00004 + .00001 * np.sin(np.arange(count) / 30)
    first = risk_free + .0004 + .009 * state[:, 0] + .003 * state[:, 1]
    second = risk_free + .0008 + .006 * state[:, 0] + .005 * state[:, 2]
    test = sharpe_difference_test(first, second, risk_free, 20)
    # Independently computed from the original four raw moments and their
    # Bartlett HAC matrix, including the n/(n-4) finite-sample correction.
    expected = {"estimate": -1.3920958064099893, "se": .3639659552112518,
                "z": -3.8247967604607256, "p_value": .00013088001028514645}
    assert test["status"] == "available"
    for key, value in expected.items():
        assert test[key] == pytest.approx(value, rel=1e-12)
    # Common changes of return units must not change the inference.
    scaled = sharpe_difference_test(100 * first, 100 * second, 100 * risk_free, 20)
    for key, value in expected.items():
        assert scaled[key] == pytest.approx(value, rel=1e-12)


@pytest.mark.filterwarnings("error")
def test_identical_realistic_portfolios_have_no_sharpe_inference():
    daily, calendar, forecasts = _inputs()
    # Low forecasts put both timing and the static control entirely in SPY.
    forecasts["arfima_bv"] = 1e-6
    settings = {**SETTINGS, "alpha": .05, "sharpe_hac_lags": 20, "monthly_hac_lags": 6}
    result = realistic_portfolios(daily, calendar, forecasts, settings)
    paths = result["paths"][settings["main_cost_bps"]]
    np.testing.assert_array_equal(paths.arfima_cash, paths.arfima_static)
    tests = _tests(result, calendar, settings)
    timing = tests.loc[tests.comparison.eq("timing_minus_static")].iloc[0]
    assert timing.estimate == 0 and timing.status == "degenerate_variance"
    assert timing[["se", "z", "p_value", "holm_p_value"]].isna().all()
    assert not timing.holm_reject


@pytest.mark.filterwarnings("error")
def test_realistic_inference_handles_no_down_months():
    daily, calendar, forecasts = _inputs()
    daily["SPY"] = .001 + .0008 * np.sin(np.arange(len(daily)) / 9)
    settings = {**SETTINGS, "alpha": .05, "sharpe_hac_lags": 20, "monthly_hac_lags": 6}
    result = realistic_portfolios(daily, calendar, forecasts, settings)
    tests = _tests(result, calendar, settings)
    assert tests.negative_months.eq(0).all()
    defensive = tests.loc[tests.comparison.eq("TLT_minus_cash")].iloc[0]
    assert defensive.status == "missing_condition"
    assert defensive[["estimate", "se", "p_value", "holm_p_value"]].isna().all()
    assert not defensive.holm_reject
    timing = tests.loc[tests.comparison.eq("timing_minus_static")].iloc[0]
    assert timing.status == "available" and np.isfinite(timing.p_value)


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


def _inference_inputs(*, unavailable_asset):
    calendar = pd.bdate_range("2011-01-01", "2020-12-31")
    rng = np.random.default_rng(477)
    market = rng.normal(.0004, .01, len(calendar))
    common = rng.normal(.0004, .006, len(calendar))
    static = common + rng.normal(0, .002, len(calendar))
    # Fixed synthetic effect puts the raw timing p-value near .009: it must
    # not be significant after correcting the six prespecified comparisons.
    cash = common + rng.normal(0, .002, len(calendar)) + .00023380737828166513
    gru_cash = common + rng.normal(0, .002, len(calendar))
    gru_static = common + rng.normal(0, .002, len(calendar))
    paths = pd.DataFrame({"SPY": market, "arfima_cash": cash, "arfima_static": static,
                          "gru_univar_cash": gru_cash, "gru_univar_static": gru_static,
                          "arfima_TLT": cash + rng.normal(0, .0008, len(calendar)),
                          "arfima_GLD": cash + rng.normal(0, .0008, len(calendar)),
                          "gru_univar_TLT": gru_cash + rng.normal(0, .0008, len(calendar)),
                          "gru_univar_GLD": (gru_cash if unavailable_asset else
                                              gru_cash + rng.normal(0, .0008, len(calendar)))},
                         index=calendar)
    settings = {"evaluation_start": "2011-01-01", "evaluation_end": "2020-12-31",
                "primary_models": ["arfima", "gru_univar"], "primary_assets": ["TLT", "GLD"],
                "main_cost_bps": 10, "sharpe_hac_lags": 20, "monthly_hac_lags": 6, "alpha": .05}
    result = {"paths": {10: paths}, "returns": pd.DataFrame({"RF": .00002, "SPY": market}, index=calendar),
              "models": settings["primary_models"].copy(), "assets": settings["primary_assets"].copy()}
    return result, calendar, settings


def test_down_months_use_gross_market_returns_even_when_benchmark_pays_fees():
    result, calendar, settings = _inference_inputs(unavailable_asset=False)
    result["returns"]["SPY"] = .00001
    result["paths"][10]["SPY"] = -.00001
    tests = _tests(result, calendar, settings)
    assert tests.negative_months.eq(0).all()


@pytest.mark.parametrize("unavailable_asset", [False, True], ids=["complete", "one_unavailable"])
def test_primary_holm_keeps_all_six_prespecified_tests(unavailable_asset):
    result, calendar, settings = _inference_inputs(unavailable_asset=unavailable_asset)
    tests = _tests(result, calendar, settings)
    assert len(tests) == 6 and tests.family.eq("primary").all()
    assert tests.p_value.iloc[0] == pytest.approx(.009, abs=1e-7)
    expected = [.054, 1., 1., 1., 1., np.nan if unavailable_asset else 1.]
    np.testing.assert_allclose(tests.holm_p_value, expected, atol=1e-7, rtol=0, equal_nan=True)
    assert not tests.holm_reject.any()
    if unavailable_asset:
        assert tests.status.iloc[-1] == "degenerate_variance"
        assert np.isnan(tests.p_value.iloc[-1])


def test_descriptive_tests_do_not_enter_primary_holm_family():
    result, calendar, settings = _inference_inputs(unavailable_asset=True)
    original = _tests(result, calendar, settings)
    paths = result["paths"][10]
    rng = np.random.default_rng(19)
    result["models"].append("garch")
    result["assets"].append("DBC")
    paths["garch_cash"] = paths.arfima_cash + .002
    paths["garch_static"] = paths.arfima_static
    for model in result["models"]:
        for asset in result["assets"]:
            if f"{model}_{asset}" not in paths:
                paths[f"{model}_{asset}"] = paths[f"{model}_cash"] + .002 + rng.normal(0, .0008, len(calendar))
    tests = _tests(result, calendar, settings)
    primary = tests.loc[tests.family.eq("primary")].reset_index(drop=True)
    pd.testing.assert_frame_equal(primary, original)
    descriptive = tests.loc[tests.family.eq("descriptive")]
    assert len(descriptive) == 6 and descriptive.p_value.lt(.009).all()
    assert descriptive.holm_p_value.isna().all() and not descriptive.holm_reject.any()


def test_all_unavailable_primary_tests_remain_missing_and_never_reject():
    result, calendar, settings = _inference_inputs(unavailable_asset=True)
    paths = result["paths"][10]
    for name in paths.columns.difference(["SPY"]):
        paths[name] = result["returns"].RF
    tests = _tests(result, calendar, settings)
    assert len(tests) == 6
    assert tests.p_value.isna().all() and tests.holm_p_value.isna().all()
    assert not tests.holm_reject.any()
