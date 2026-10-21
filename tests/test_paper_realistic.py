import numpy as np
import pandas as pd
import pytest

from financial_economics.paper_realistic import (delayed_signals, realistic_portfolios,
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
