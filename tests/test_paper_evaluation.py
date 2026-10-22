"""Matched scoring, funding, missing-signal drift and ratio averaging."""

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financial_economics.paper_evaluation import MODELS, _forecast_scores, _metrics, evaluate_paper
from financial_economics.paper_sources import ASSETS


def fixture():
    with Path("configs/paper-reconstruction.toml").open("rb") as source:
        settings = tomllib.load(source)["reconstruction"]
    dates = pd.bdate_range("2010-01-04", periods=10)
    returns = pd.DataFrame({"EQUITY": [.1, -.08, .04, -.02, .03, -.01, .02, -.03, .01, -.02],
                            **{asset: np.linspace(-.015, .02, 10) for asset in ASSETS}}, index=dates)
    positions = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9])
    issues = pd.DatetimeIndex([pd.Timestamp("2009-12-31"), *dates[positions[1:] - 1]], name="forecast_date")
    bv = np.array([.0001, .0004, .0009, .0004, .0001, .0002, .0003, .0008, .0005])
    forecasts = pd.DataFrame({"return_date": dates[positions], "realized_bv": bv}, index=issues)
    for number, model in enumerate(MODELS):
        forecasts[f"{model}_bv"] = bv * (1 + number * .1)
        forecasts[f"{model}_refit_asof"] = pd.Timestamp("2009-12-31")
    return returns, forecasts, settings


def test_all_variants_funded_from_cash_and_missing_signals_drift():
    returns, forecasts, settings = fixture()
    tables, assumptions = evaluate_paper(returns, forecasts, settings)
    assert assumptions["portfolios"] == 590
    assert tables["daily_returns"].shape == (10, 591)
    stock = .1 / (22 * .01)
    path = tables["daily_returns"].arfima_clean
    assert path.iloc[0] == pytest.approx(stock * .1)
    assert path.iloc[1] == pytest.approx(stock * 1.1 / (1 + stock * .1) * -.08)
    targets = tables["model_target_weights"].query("model == 'arfima'")
    for name in targets.filter(like="defensive_"):
        np.testing.assert_array_less(targets[name].to_numpy(), 1 - targets.stock_target.to_numpy() + 1e-12)
    detail = tables["portfolio_definitions"]
    selected = detail.loc[(detail.model == "arfima") & (detail.asset == "GOLD")]
    assert len(selected) == 9
    expected = selected.paper_inferred_sharpe.mean() - detail.loc["arfima_clean", "paper_inferred_sharpe"]
    assert tables["asset_means"].loc[("arfima", "GOLD"), "paper_inferred_sharpe_minus_clean"] == pytest.approx(expected)


def test_common_forecast_labels_missing_day_preserved_and_future_refit_rejected():
    returns, forecasts, settings = fixture()
    forecasts.loc[forecasts.index[3], "realized_bv"] = np.nan
    tables, _ = evaluate_paper(returns, forecasts, settings)
    assert tables["forecast_scores"].n_obs.tolist() == [8] * 5
    assert tables["forecast_scores"].loc["arfima", "monthly_mse"] == 0
    assert len(tables["forecast_tests"]) == 20
    assert len(tables["daily_returns"]) == len(returns)
    assert (tables["forecast_tests"].loss_column_minus_row >= 0).all()
    forecasts.loc[forecasts.index[0], "gru_multivar_refit_asof"] = pd.Timestamp("2010-01-04")
    with pytest.raises(ValueError, match="future-fitted"):
        evaluate_paper(returns, forecasts, settings)


def test_negative_modified_sharpe_multiplies_volatility():
    paths = pd.DataFrame({"loss": [-.02, .01, -.03, .005]}, index=pd.bdate_range("2020-01-02", periods=4))
    metrics = _metrics(paths, 252).loc["loss"]
    assert metrics.paper_inferred_modified_sharpe == pytest.approx(metrics.cagr * metrics.annualized_volatility)
    assert metrics.sharpe < metrics.paper_inferred_modified_sharpe < 0


def test_dm_statistics_have_paper_sign_and_finite_p_values():
    _, _, settings = fixture()
    t = np.arange(60)
    actual = .0001 * (2 + np.sin(t / 3))
    forecasts = pd.DataFrame({"realized_bv": actual})
    for number, model in enumerate(MODELS):
        forecasts[f"{model}_bv"] = actual * (1 + (number - 2) * .1 * (1 + .3 * np.cos(t)))
    _, tests = _forecast_scores(forecasts, settings)
    assert tests.status.eq("available").all()
    assert np.isfinite(tests.dm_statistic).all()
    assert tests.p_value.between(0, 1).all()
    assert (tests.holm_p_value >= tests.p_value - 1e-15).all()
    assert tests.query("row_model == 'arfima' and column_model == 'gjr'").dm_statistic.lt(0).all()
    assert tests.query("row_model == 'gjr' and column_model == 'gru_multivar'").dm_statistic.gt(0).all()


def test_two_session_forecast_cannot_enter_one_session_evaluation():
    returns, forecasts, settings = fixture()
    changed = forecasts.copy()
    changed.iloc[0, changed.columns.get_loc("return_date")] = returns.index[1]
    with pytest.raises(ValueError, match="first evaluation"):
        evaluate_paper(returns, changed, settings)
    changed = forecasts.copy()
    changed.index = changed.index.where(changed.index != changed.index[1], changed.index[1] - pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="next observed"):
        evaluate_paper(returns, changed, settings)


def test_decline_returns_and_benchmark_score_deltas_use_same_clean_reference():
    returns, forecasts, settings = fixture()
    dates = pd.bdate_range("2020-02-20", periods=len(returns))
    positions = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9])
    returns.index = dates
    forecasts["return_date"] = dates[positions]
    forecasts.index = pd.DatetimeIndex([pd.Timestamp("2020-02-19"), *dates[positions[1:] - 1]])
    tables, _ = evaluate_paper(returns, forecasts, settings)
    decline = tables["decline_asset_means"].loc[("COVID", "gru_multivar")]
    assert decline.loc["EQUITY", "cumulative_return"] == pytest.approx((1 + returns.EQUITY).prod() - 1)
    measure = "paper_inferred_modified_sharpe"
    assert decline.loc["EQUITY", measure + "_minus_clean"] == pytest.approx(
        decline.loc["EQUITY", measure] - decline.loc["cash", measure])
    assert decline.loc["cash", measure + "_minus_clean"] == 0
    assert len(decline) == 15
