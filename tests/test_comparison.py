"""Chronological calibration must not learn from evaluation returns."""

import numpy as np
import pandas as pd
import pytest

from financial_economics.evaluation.metrics import monthly_equal_weight, monthly_fixed_weight
from financial_economics.portfolio.controls import compare_portfolios, stock_exposure


def sample():
    index = pd.bdate_range("2017-12-18", "2018-01-12")
    excess = np.sin(np.arange(len(index))) * 0.02
    rf = pd.Series(0.0001 + np.arange(len(index)) * 0.00001, index=index, name="RF")
    returns = pd.DataFrame({"SPY": rf + excess, "Bond": rf + 0.0002 - 0.2 * excess})
    return returns, rf


def fitted_exposure(returns, rf, benchmark):
    design = np.column_stack([np.ones(len(returns)), returns[benchmark] - rf])
    return np.linalg.lstsq(design, returns.sub(rf, axis=0), rcond=None)[0]


def test_exposure_recovers_intercept_and_beta_using_variable_risk_free_returns():
    index = pd.bdate_range("2017-01-02", periods=12)
    excess = np.linspace(-0.02, 0.02, len(index))
    rf = pd.Series(0.001 + 0.2 * excess, index=index)
    returns = pd.DataFrame({"SPY": rf + excess, "Mix": rf + 0.002 + 0.6 * excess,
                            "Cash": rf})
    result = stock_exposure(returns, rf, "SPY")
    assert result.loc["Mix", "intercept_daily"] == pytest.approx(0.002)
    assert result.loc["Mix", "beta"] == pytest.approx(0.6)
    assert result.loc["Cash", "beta"] == pytest.approx(0, abs=1e-14)
    assert result.loc["Cash", "intercept_daily"] == pytest.approx(0, abs=1e-14)
    assert result.loc["Mix", "n_obs"] == len(index)


def test_calibration_uses_training_only_and_reports_realized_exposure_in_each_period():
    returns, rf = sample()
    result = compare_portfolios(returns, rf)
    training = returns.index < "2018-01-01"
    full_mix = monthly_equal_weight(returns)
    design = np.column_stack([np.ones(training.sum()), (returns["SPY"] - rf)[training]])
    alpha, beta = np.linalg.lstsq(design, (full_mix - rf)[training], rcond=None)[0]
    assert result.calibration["intercept_daily"] == pytest.approx(alpha)
    assert result.calibration["raw_beta"] == pytest.approx(beta)
    assert result.calibration["stock_weight"] == pytest.approx(beta)
    assert result.calibration["cash_weight"] == pytest.approx(1 - beta)
    assert result.calibration["weight_clipped"] is False
    changed = returns.copy()
    changed.loc[~training] = changed.loc[~training] * 2 + 0.01
    changed_rf = rf.copy()
    changed_rf.loc[~training] += 0.001
    rerun = compare_portfolios(changed, changed_rf)
    assert rerun.calibration == result.calibration
    pd.testing.assert_frame_equal(rerun.training_returns, result.training_returns)
    assert list(result.evaluation_returns) == ["SPY", "Equal weight", "SPY + cash"]
    for period, frame in [("training", result.training_returns),
                          ("evaluation", result.evaluation_returns)]:
        expected = fitted_exposure(frame, rf.loc[frame.index], "SPY")
        observed = result.exposure.loc[result.exposure["period"] == period].set_index("asset")
        np.testing.assert_allclose(observed.loc[frame.columns, "intercept_daily"], expected[0], atol=1e-14)
        np.testing.assert_allclose(observed.loc[frame.columns, "beta"], expected[1], atol=1e-14)
        assert (observed["n_obs"] == len(frame)).all()


def test_weekend_split_uses_next_observation_and_last_training_date_as_seed():
    returns, rf = sample()
    result = compare_portfolios(returns, rf, evaluation_start="2017-12-30")
    metadata = result.calibration
    assert metadata["requested_evaluation_start"] == "2017-12-30"
    assert metadata["training_start"] == "2017-12-18"
    assert metadata["training_end"] == "2017-12-29"
    assert metadata["evaluation_start"] == "2018-01-01"
    assert metadata["evaluation_end"] == "2018-01-12"
    assert metadata["evaluation_seed_date"] == "2017-12-29"
    assert metadata["training_observations"] == len(result.training_returns)
    assert metadata["evaluation_observations"] == len(result.evaluation_returns)
    assert metadata["benchmark"] == "SPY"
    assert metadata["control_name"] == "SPY + cash"
    pd.testing.assert_frame_equal(
        result.evaluation_returns,
        compare_portfolios(returns, rf, evaluation_start="2018-01-01").evaluation_returns,
    )


def test_explicit_benchmark_controls_calibration_and_output_names():
    returns, rf = sample()
    renamed = returns.rename(columns={"SPY": "Equity"})
    result = compare_portfolios(renamed, rf, benchmark="Equity")
    assert result.calibration["benchmark"] == "Equity"
    assert result.calibration["control_name"] == "Equity + cash"
    assert list(result.evaluation_returns) == ["Equity", "Equal weight", "Equity + cash"]
    assert result.calibration["stock_weight"] == pytest.approx(
        compare_portfolios(returns, rf).calibration["stock_weight"]
    )
    with pytest.raises(ValueError):
        compare_portfolios(renamed, rf)


def test_midmonth_split_keeps_mix_holdings_but_initializes_control_at_targets():
    index = pd.bdate_range("2018-01-02", periods=12)
    returns = pd.DataFrame({"SPY": [0.10, -0.03, 0.02, 0.04, -0.02, 0.01,
                                     0.02, -0.01, 0.03, -0.02, 0.01, 0.02],
                            "Bond": [0.0] * 12}, index=index)
    rf = pd.Series(0.0001, index=index)
    result = compare_portfolios(returns, rf, evaluation_start=str(index[7].date()))
    pd.testing.assert_series_equal(
        result.evaluation_returns["Equal weight"], monthly_equal_weight(returns).iloc[7:]
    )
    assert result.evaluation_returns["Equal weight"].iloc[0] != pytest.approx(
        returns.iloc[7].mean()
    )
    control_assets = pd.DataFrame({"SPY": returns["SPY"], "RF": rf}).iloc[7:]
    weights = {"SPY": result.calibration["stock_weight"],
               "RF": result.calibration["cash_weight"]}
    expected = monthly_fixed_weight(control_assets, weights, "SPY + cash")
    pd.testing.assert_series_equal(result.evaluation_returns["SPY + cash"], expected)
    assert expected.iloc[0] == pytest.approx(
        weights["SPY"] * returns["SPY"].iloc[7] + weights["RF"] * rf.iloc[7]
    )


@pytest.mark.parametrize("other_slope, endpoint", [(-4.0, 0.0), (3.0, 1.0)])
def test_exposure_outside_long_only_range_is_reported_and_clipped(other_slope, endpoint):
    returns, rf = sample()
    returns["Bond"] = rf + other_slope * (returns["SPY"] - rf)
    result = compare_portfolios(returns, rf)
    raw = result.calibration["raw_beta"]
    assert raw < 0 if endpoint == 0 else raw > 1
    assert result.calibration["stock_weight"] == endpoint
    assert result.calibration["cash_weight"] == 1 - endpoint
    assert result.calibration["weight_clipped"] is True
    expected = rf.loc[result.evaluation_returns.index] if endpoint == 0 else returns.loc[
        result.evaluation_returns.index, "SPY"]
    np.testing.assert_allclose(result.evaluation_returns["SPY + cash"], expected, atol=1e-14)


@pytest.mark.parametrize("start", ["2017-12-20", "2018-01-11", "2017-01-01", "2019-01-01"])
def test_split_requires_three_observations_in_both_periods(start):
    returns, rf = sample()
    with pytest.raises(ValueError):
        compare_portfolios(returns, rf, evaluation_start=start)


@pytest.mark.parametrize("start", ["not-a-date", "2018-01-01T12:00:00", "2018-01-01T00:00:00Z"])
def test_split_requires_a_valid_timezone_naive_daily_date(start):
    returns, rf = sample()
    with pytest.raises(ValueError):
        compare_portfolios(returns, rf, evaluation_start=start)


def test_constant_training_benchmark_is_rejected_even_if_evaluation_varies():
    returns, rf = sample()
    returns.loc[returns.index < "2018-01-01", "SPY"] = rf + 0.001
    with pytest.raises(ValueError, match="constant"):
        compare_portfolios(returns, rf)


@pytest.mark.parametrize("rf_change", ["missing", "extra"])
def test_risk_free_dates_must_match_exactly(rf_change):
    returns, rf = sample()
    rf = rf.iloc[:-1] if rf_change == "missing" else pd.concat([
        rf, pd.Series([0.001], index=[pd.Timestamp("2018-01-15")])
    ])
    with pytest.raises(ValueError, match="exact"):
        compare_portfolios(returns, rf)


@pytest.mark.parametrize("reserved", ["Equal weight", "RF", "SPY + cash"])
def test_reserved_raw_asset_names_cannot_overwrite_comparison_portfolios(reserved):
    returns, rf = sample()
    returns = returns.rename(columns={"Bond": reserved})
    with pytest.raises(ValueError):
        compare_portfolios(returns, rf)
