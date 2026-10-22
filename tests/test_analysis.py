"""Small numerical examples that protect the study's financial calculations."""

import numpy as np
import pandas as pd
import pytest

from financial_economics.evaluation.metrics import (
    capm_summary,
    drawdowns,
    monthly_equal_weight,
    performance_summary,
    prices_to_returns,
)


def frame(values, dates=None):
    dates = dates if dates is not None else pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame(values, index=pd.DatetimeIndex(dates), columns=["A", "B"])


def test_returns_include_first_requested_day_using_prior_close():
    prices = frame([[100, 200], [110, 180], [121, 198]],
                   ["2009-12-31", "2010-01-04", "2010-01-05"])
    result = prices_to_returns(prices, "2010-01-01", "2010-01-05")
    assert list(result.index) == list(prices.index[1:])
    np.testing.assert_allclose(result, [[0.1, -0.1], [0.1, 0.1]])


def test_returns_require_seed_and_do_not_fill_missing_prices():
    prices = frame([[100, 100], [110, 110]])
    with pytest.raises(ValueError, match="seed"):
        prices_to_returns(prices, "2020-01-01", "2020-01-03")
    prices.iloc[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        prices_to_returns(prices, "2020-01-02", "2020-01-03")


def test_monthly_equal_weight_drifts_then_resets():
    returns = frame([[1, 0], [0, 1], [0, 1]],
                    ["2020-01-30", "2020-01-31", "2020-02-03"])
    result = monthly_equal_weight(returns)
    assert result.name == "Equal weight"
    np.testing.assert_allclose(result, [0.5, 1 / 3, 0.5])


def test_metrics_compound_and_count_initial_loss_in_drawdown():
    returns = frame([[-0.2, 0.1], [0.25, -0.1]])
    risk_free = pd.Series([0.01, 0.01], index=returns.index)
    result = performance_summary(returns, risk_free, trading_days=2)
    np.testing.assert_allclose(drawdowns(returns)["A"], [-0.2, 0])
    assert result.loc["A", "cumulative_return"] == pytest.approx(0)
    assert result.loc["A", "cagr"] == pytest.approx(0)
    assert result.loc["A", "annualized_volatility"] == pytest.approx(0.45)
    assert result.loc["A", "sharpe"] == pytest.approx(0.03 / 0.45)
    assert result.loc["A", "max_drawdown"] == pytest.approx(-0.2)
    assert result.loc["A", "n_obs"] == 2
    assert result.loc["B", "cumulative_return"] == pytest.approx(-0.01)
    assert result.loc["B", "max_drawdown"] == pytest.approx(-0.1)


def test_cagr_is_geometric_and_zero_volatility_sharpe_is_undefined():
    returns = frame([[0.1, 0], [0.1, 0]])
    rf = pd.Series(0.0, index=returns.index)
    result = performance_summary(returns, rf, trading_days=4)
    assert result.loc["A", "cagr"] == pytest.approx(1.1 ** 4 - 1)
    assert np.isnan(result.loc["A", "sharpe"])
    assert np.isnan(result.loc["B", "sharpe"])


@pytest.mark.parametrize("varying_rf", [False, True])
def test_numerically_constant_excess_returns_have_undefined_sharpe(varying_rf):
    index = pd.bdate_range("2020-01-01", periods=10)
    rf = pd.Series(np.linspace(0.0001, 0.0002, 10) if varying_rf else 0.0, index=index)
    returns = (rf + 0.001).to_frame("A")
    result = performance_summary(returns, rf)
    assert np.isnan(result.loc["A", "sharpe"])


def test_genuine_variation_in_very_small_returns_keeps_defined_sharpe():
    index = pd.bdate_range("2020-01-01", periods=10)
    returns = pd.DataFrame({"A": 1e-12 + np.linspace(-1e-15, 1e-15, 10)}, index=index)
    rf = pd.Series(0.0, index=index)
    result = performance_summary(returns, rf)
    expected = np.sqrt(252) * returns["A"].mean() / returns["A"].std(ddof=1)
    assert result.loc["A", "sharpe"] == pytest.approx(expected)


def test_capm_recovers_known_daily_alpha_and_beta():
    market = np.linspace(-0.03, 0.03, 40)
    residual = np.sin(np.arange(40)) * 0.002
    design = np.column_stack([np.ones(40), market])
    residual -= design @ np.linalg.lstsq(design, residual, rcond=None)[0]
    factors = pd.DataFrame({"Mkt-RF": market, "RF": 0.0001},
                           index=pd.bdate_range("2020-01-01", periods=40))
    returns = pd.DataFrame({"A": 0.0001 + 0.0003 + 1.2 * market + residual},
                           index=factors.index)
    result = capm_summary(returns, factors)
    assert result.loc["A", "alpha_daily"] == pytest.approx(0.0003)
    assert result.loc["A", "beta"] == pytest.approx(1.2)
    assert 0 < result.loc["A", "alpha_daily_se"] < 0.001
    assert result.loc["A", "beta_ci_low"] < 1.2 < result.loc["A", "beta_ci_high"]
    assert result.loc["A", "beta_pvalue"] < 0.001
    assert result.loc["A", "r_squared"] > 0.99
    assert result.loc["A", "n_obs"] == 40


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0, -1.1])
def test_invalid_returns_are_rejected(bad):
    returns = frame([[0.01, 0], [bad, 0.02]])
    with pytest.raises(ValueError):
        monthly_equal_weight(returns)


def test_duplicate_dates_and_unsorted_dates_are_rejected():
    returns = frame([[0.01, 0], [0, 0.01]])
    with pytest.raises(ValueError, match="increasing"):
        drawdowns(returns.iloc[::-1])
    returns.index = [returns.index[0], returns.index[0]]
    with pytest.raises(ValueError, match="unique"):
        drawdowns(returns)


def test_performance_rejects_missing_or_extra_risk_free_dates():
    returns = frame([[0.01, 0], [0, 0.01]])
    rf = pd.Series([0], index=returns.index[:1])
    with pytest.raises(ValueError, match="exactly"):
        performance_summary(returns, rf)
    rf = pd.Series(0.0, index=pd.bdate_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match="exactly"):
        performance_summary(returns, rf)


def test_capm_rejects_constant_market_and_mismatched_dates():
    returns = frame(np.full((20, 2), 0.01))
    factors = pd.DataFrame({"Mkt-RF": 0.01, "RF": 0}, index=returns.index)
    with pytest.raises(ValueError, match="constant"):
        capm_summary(returns, factors)
    with pytest.raises(ValueError, match="exactly"):
        capm_summary(returns, factors.iloc[:-1])


@pytest.mark.parametrize("slope", [0.0, 1.2])
def test_capm_rejects_degenerate_response_without_inference_warnings(slope):
    index = pd.bdate_range("2020-01-01", periods=20)
    factors = pd.DataFrame({"Mkt-RF": np.linspace(-0.02, 0.02, 20), "RF": 0.0001}, index=index)
    returns = pd.DataFrame({"A": factors["RF"] + 0.0003 + slope * factors["Mkt-RF"]})
    with pytest.raises(ValueError, match="inference"):
        capm_summary(returns, factors)


def test_insufficient_observations_and_invalid_settings_are_rejected():
    returns = frame([[0.01, 0]])
    rf = pd.Series(0.0, index=returns.index)
    with pytest.raises(ValueError, match="at least 2"):
        performance_summary(returns, rf)
    returns = frame([[0.01, 0], [0, 0.01]])
    rf = pd.Series(0.0, index=returns.index)
    with pytest.raises(ValueError, match="trading_days"):
        performance_summary(returns, rf, trading_days=0)
    factors = pd.DataFrame({"Mkt-RF": [0.01, 0.02], "RF": 0}, index=returns.index)
    with pytest.raises(ValueError, match="observations"):
        capm_summary(returns, factors)
