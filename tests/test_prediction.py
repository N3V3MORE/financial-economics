"""Small dated-accounting and independent prediction-inference examples."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from financial_economics.config import Study
from financial_economics.costs import dynamic_costed_path, monthly_costed_path
from financial_economics.explain import resample_rows
from financial_economics.forecasting import ewma_forecasts, prepare_forecasts, qlike
from financial_economics.prediction import run_prediction
from financial_economics.prediction_config import PredictionSettings
from financial_economics.prediction_inference import infer_prediction
from financial_economics.return_data import ReturnData


@pytest.mark.parametrize("cost", [0, 10, 25])
def test_dated_monthly_schedule_preserves_existing_engine(cost):
    dates = pd.bdate_range("2020-01-20", periods=30)
    returns = pd.DataFrame({"SPY": np.sin(np.arange(30)) * .02,
                            "TLT": .002, "RF": .0001}, index=dates)
    weights = {"SPY": .5, "TLT": .2, "RF": .3}
    targets = pd.DataFrame(weights, index=dates)
    first = np.r_[True, dates.to_period("M")[1:] != dates.to_period("M")[:-1]]
    pd.testing.assert_frame_equal(dynamic_costed_path(returns, targets, cost, dates[first]),
                                  monthly_costed_path(returns, weights, cost))


def test_daily_switch_charges_both_sides_and_aligns_names_and_dates():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    returns = pd.DataFrame(0., index=dates, columns=["SPY", "TLT", "RF"])
    targets = pd.DataFrame({"SPY": [1., 0.], "TLT": [0., 1.], "RF": 0.}, index=dates)
    path = dynamic_costed_path(returns, targets, 100)
    np.testing.assert_allclose(path["return"], [1 / 1.01 - 1, .99 / 1.01 - 1])
    np.testing.assert_allclose(1 + path["return"] + path.fee_fraction, 1)
    pd.testing.assert_frame_equal(path, dynamic_costed_path(returns, targets[["RF", "TLT", "SPY"]], 100))
    with pytest.raises(ValueError, match="dates"):
        dynamic_costed_path(returns, targets.iloc[1:], 100)
    with pytest.raises(ValueError, match="entry"):
        dynamic_costed_path(returns, targets, 100, dates[1:])


def prediction_fixture(full=False):
    dates = pd.bdate_range("2013-01-01", "2018-12-31").difference(
        pd.to_datetime(["2017-01-02", "2018-01-01"]))
    day = np.arange(len(dates))
    daily = pd.DataFrame({"SPY": .009 * np.sin(day / 7) + .002 * np.sin(day / 41),
                          "TLT": .003 * np.cos(day / 5)}, index=dates)
    factors = pd.DataFrame({"Mkt-RF": daily.SPY - .0001, "RF": .0001}, index=dates)
    data = ReturnData(daily, factors, pd.Timestamp("2012-12-31"), {})
    study = Study("prediction test", "2013-01-01", "2018-12-31", "SPY",
                  {"SPY": "Stocks", "TLT": "Treasuries"})
    settings = PredictionSettings(resamples=100, windows=("2017:2018",))
    forecasts, _ = ewma_forecasts(prepare_forecasts(daily.SPY, dates), settings)
    if full:
        forecasts["gru_variance"] = forecasts.ewma_variance * 1.15
        forecasts.loc[forecasts.index < settings.validation_start, "gru_variance"] = np.nan
    return data, study, settings, forecasts


def test_baseline_dates_allocation_constraints_and_cash_warmup():
    data, study, settings, forecasts = prediction_fixture()
    result = run_prediction(data, study, settings, forecasts)
    assert result.metadata["stage"] == "baseline"
    assert result.metadata["warmup_cash_days"] == 1
    assert result.metadata["freeze_asof"] == "2016-12-30"
    assert "primary_tests" not in result.tables
    first = result.tables["decisions"].iloc[0]
    assert first.forecast_date == pd.Timestamp("2016-12-30")
    assert first.trade_date == pd.Timestamp("2017-01-03")
    assert first.return_date == pd.Timestamp("2017-01-04")
    decisions = result.tables["decisions"]
    crossing = decisions.loc[decisions.forecast_date.eq("2017-12-29")].iloc[0]
    assert crossing.refit_asof == pd.Timestamp("2017-12-29")
    thresholds = result.tables["annual_thresholds"]
    calibration = ((forecasts.index >= settings.validation_start)
                   & (forecasts.index < "2017-12-29"))
    expected = np.quantile(np.sqrt(252 * forecasts.loc[calibration, "ewma_variance"]), .9)
    assert thresholds.loc[thresholds.evaluation_year.eq(2018), "stress_volatility_threshold"].iloc[0] == pytest.approx(expected)
    weights = result.tables["target_weights"].set_index(["portfolio", "Date"])
    pd.testing.assert_series_equal(weights.loc["A"].SPY, weights.loc["C"].SPY)
    assert weights.loc["C"].TLT.between(0, .2).all()
    assert (weights[["SPY", "TLT", "RF"]] >= -1e-14).all().all()
    np.testing.assert_allclose(weights[["SPY", "TLT", "RF"]].sum(axis=1), 1)
    assert weights.loc[("C", pd.Timestamp("2017-01-03")), "RF"] == 1
    assert result.tables["daily_returns"].loc["2017-01-03", "A | cash_10bps"] == pytest.approx(.0001)
    assert result.tables["daily_returns"].loc["2017-01-03", "A | zero_cash_10bps"] == 0
    assert len(result.tables["monthly_returns"]) == 24


def test_future_data_preserves_prior_decisions_and_forecast_holes_fail():
    data, study, settings, forecasts = prediction_fixture(full=True)
    before = run_prediction(data, study, settings, forecasts)
    changed = data.daily.copy()
    changed.loc["2018-07-01":] *= 1.2
    later, _ = ewma_forecasts(prepare_forecasts(changed.SPY, changed.index), settings)
    later["gru_variance"] = later.ewma_variance * 1.15
    later.loc[later.index < settings.validation_start, "gru_variance"] = np.nan
    after = run_prediction(replace(data, daily=changed), study, settings, later)
    for name in ("decisions", "target_weights"):
        left, right = before.tables[name], after.tables[name]
        key = "forecast_date" if name == "decisions" else "Date"
        pd.testing.assert_frame_equal(left.loc[left[key] < "2018-07-01"].reset_index(drop=True),
                                      right.loc[right[key] < "2018-07-01"].reset_index(drop=True))
    assert before.metadata["primary_inference"]
    assert before.tables["primary_tests"].family_size.eq(3).all()
    with pytest.raises(ValueError, match="every eligible return date"):
        run_prediction(data, study, settings, forecasts.drop(pd.Timestamp("2017-03-01")))
    incorrect = forecasts.copy()
    incorrect.loc["2017-03-01", "trade_date"] = pd.Timestamp("2017-03-03")
    with pytest.raises(ValueError, match="trade_date"):
        run_prediction(data, study, settings, incorrect)
    incorrect = forecasts.copy()
    incorrect.loc["2017-03-01", "squared_return"] += .001
    with pytest.raises(ValueError, match="labels"):
        run_prediction(data, study, settings, incorrect)


def test_hac_holm_and_joint_bootstrap_match_independent_reference():
    rng = np.random.default_rng(91)
    monthly = pd.DataFrame(rng.normal(0, .025, (36, 4)),
                           index=pd.date_range("2017-01-31", periods=36, freq="ME"),
                           columns=["SPY", "B", "C", "D"])
    forecasts = pd.DataFrame({"squared_return": rng.uniform(0, .0004, 100),
                              "ewma_variance": rng.uniform(.0001, .0003, 100),
                              "gru_variance": rng.uniform(.0001, .0003, 100)})
    settings = PredictionSettings(resamples=100, block_lengths=(6,))
    tests, intervals = infer_prediction(forecasts, monthly, settings)
    down = monthly.SPY.lt(0).to_numpy()
    loss = qlike(forecasts.squared_return, forecasts.gru_variance) - qlike(forecasts.squared_return, forecasts.ewma_variance)
    references = []
    for values, design, lags in ((loss, np.ones((100, 1)), 20),
                                 (monthly.D - monthly.C, np.column_stack([down, ~down]).astype(float), 6),
                                 (monthly.D - monthly.B, np.column_stack([down, ~down]).astype(float), 6)):
        fit = sm.OLS(values, design).fit(cov_type="HAC", use_t=True,
                                        cov_kwds={"maxlags": lags, "kernel": "bartlett", "use_correction": True})
        references.append((np.asarray(fit.bse)[0], np.asarray(fit.pvalues)[0]))
    np.testing.assert_allclose(tests.hac_se, np.asarray(references)[:, 0])
    np.testing.assert_allclose(tests.p_value, np.asarray(references)[:, 1])
    np.testing.assert_allclose(tests.holm_p_value, multipletests(tests.p_value, method="holm")[1])
    indices = resample_rows(36, 100, 6, np.random.default_rng(settings.seed))
    for position, control in enumerate(("C", "B")):
        draws = []
        for rows in indices:
            sample = monthly.iloc[rows]
            draws.append((sample.D - sample[control]).loc[sample.SPY.lt(0)].mean())
        np.testing.assert_allclose(intervals.iloc[position][["ci_low", "ci_high"]].to_numpy(dtype=float),
                                   np.quantile(draws, [.025, .975]))
        assert intervals.valid_draws.iloc[position] == 100
