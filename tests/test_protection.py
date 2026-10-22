from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from financial_economics.settings.study import Study, load_study
from financial_economics.settings.protection import ProtectionSettings, load_protection
from financial_economics.evaluation.protection import infer_protection, describe_funds
from financial_economics.studies.protection import run_protection
from financial_economics.datasets.returns import ReturnData
from financial_economics.evaluation.attribution import resample_rows
from pathlib import Path


def monthly_fixture():
    rng = np.random.default_rng(31)
    frame = pd.DataFrame(rng.normal(0, .04, (132, 4)),
                         index=pd.date_range("2015-01-31", periods=132, freq="ME"),
                         columns=["SPY", "A", "B", "RF"])
    frame.RF = .001
    frame.B = frame.A  # paired bootstrap must give identical bounds
    return frame


def test_inference_matches_independent_hac_and_is_reproducible():
    monthly = monthly_fixture()
    settings = ProtectionSettings(resamples=300)
    tests, intervals = infer_protection(monthly, settings)
    again, bounds = infer_protection(monthly[["RF", "B", "SPY", "A"]], settings)
    pd.testing.assert_frame_equal(tests.sort_values("fund").reset_index(drop=True),
                                  again.sort_values("fund").reset_index(drop=True))
    pd.testing.assert_frame_equal(intervals.sort_values(["fund", "mean_block_months"]).reset_index(drop=True),
                                  bounds.sort_values(["fund", "mean_block_months"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(intervals.query('fund == "A"').drop(columns="fund").reset_index(drop=True),
                                  intervals.query('fund == "B"').drop(columns="fund").reset_index(drop=True))
    mask = monthly.SPY.lt(0)
    fitted = sm.OLS(monthly.A-monthly.RF, np.column_stack([mask, ~mask]).astype(float)).fit(
        cov_type="HAC", use_t=True, cov_kwds={"maxlags": 6, "use_correction": True, "kernel": "bartlett"})
    assert tests.mean_minus_cash.iloc[0] == pytest.approx((monthly.A-monthly.RF)[mask].mean())
    assert tests.hac_se.iloc[0] == pytest.approx(fitted.bse.iloc[0])
    assert tests.p_value.iloc[0] == pytest.approx(fitted.pvalues.iloc[0])
    np.testing.assert_allclose(tests.holm_p_value, multipletests(tests.p_value, method="holm")[1])


def test_undefined_is_not_zero_and_all_rows_stay_joint():
    frame = monthly_fixture()
    frame.SPY = .01
    tests, intervals = infer_protection(frame, ProtectionSettings(resamples=100))
    assert tests.mean_minus_cash.isna().all() and tests.holm_p_value.isna().all()
    assert intervals.ci_low.isna().all() and intervals.valid_draws.eq(0).all()
    descriptive = describe_funds(frame)
    assert descriptive.query('condition == "negative_spy"').months.eq(0).all()
    indices = resample_rows(132, 1000, 6, np.random.default_rng(123))
    continuation = ((indices[:, 1:] - indices[:, :-1]) % 132 == 1).mean()
    assert .81 < continuation < .86  # geometric blocks preserve adjacency, not iid months


def test_config_freezes_all_thirteen_tests():
    study = load_study(Path("configs/protection.toml"))
    settings = load_protection(Path("configs/protection.toml"), study)
    assert len(study.tickers) == 14 and settings.resamples == 10000
    assert set(settings.fixed_portfolios["original_eight"]) == set(load_study(Path("configs/study.toml")).tickers)


def test_pipeline_preserves_monthly_identity_columns_and_fees(tmp_path):
    dates = pd.bdate_range("2020-01-01", "2021-12-31")
    rng = np.random.default_rng(8)
    returns = pd.DataFrame(rng.normal(.0001, .012, (len(dates), 2)), index=dates, columns=["SPY", "GLD"])
    factors = pd.DataFrame({"Mkt-RF": returns.SPY - .0001, "RF": .0001}, index=dates)
    data = ReturnData(returns, factors, pd.Timestamp("2019-12-31"), {})
    study = Study("test", "2020-01-01", "2021-12-31", "SPY", {"SPY": "Stocks", "GLD": "Gold"})
    settings = ProtectionSettings(allocations=[.1, .4], costs_bps=[0, 10], resamples=100,
                                  training_start="2020-01-01", training_end="2020-12-31",
                                  evaluation_start="2021-01-01", windows=["2020:2021"],
                                  fixed_portfolios={"both": ["SPY", "GLD"]})
    result = run_protection(data, study, settings)
    reordered = run_protection(replace(data, daily=returns[["GLD", "SPY"]]), study, settings)
    for name in result.tables:
        pd.testing.assert_frame_equal(result.tables[name], reordered.tables[name])
    assert result.metadata["zero_cost_allocation_identity"] == "passed"
    fees = result.tables["fees"]
    assert fees.query('cost_bps == 0').fees_per_initial_dollar.eq(0).all()
    assert (fees.query('cost_bps == 10').fees_per_initial_dollar > 0).any()
    assert result.tables["performance"].portfolio.nunique() == 6
    from financial_economics.reporting.protection import write_protection
    output = tmp_path / "results"
    write_protection(result, study, settings, {"sources": {"prices": {"provider": "Synthetic test"}}},
                     output, Path("configs/protection.toml"))
    assert len(list(output.glob("*.png"))) == 3
    figure = pd.read_csv(output / "figure_protection.csv")
    original = result.tables["fund_conditions"].query("condition == 'negative_spy'")
    np.testing.assert_allclose(figure.mean_minus_cash, original.mean_minus_cash)
    with pytest.raises(FileExistsError):
        write_protection(result, study, settings, {}, output, Path("configs/protection.toml"))
    with pytest.raises(ValueError, match="dates"):
        run_protection(replace(data, factors=factors.iloc[1:]), study, settings)
