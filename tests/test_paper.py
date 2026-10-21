"""Small data, horizon, allocation, and offline-command checks for the paper mode."""

from dataclasses import replace
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from financial_economics import cli
from financial_economics.paper import (
    PaperSettings, load_bipower, load_paper_settings, paper_portfolios, prepare_paper_forecasts,
)
from financial_economics.return_data import ReturnData


def write_archive(path, dates, values):
    raw = pd.DataFrame({"Date": [f"{day:%Y-%m-%d} 00:00:00+01:00" for day in dates],
                        "Symbol": ".SPX", "bv": values})
    with ZipFile(path, "w") as archive:
        archive.writestr("oxford.csv", raw.to_csv(index=False))


def test_oxford_dates_keep_local_labels_and_invalid_observations_fail(tmp_path):
    dates = pd.bdate_range("2000-03-27", periods=8)
    path = tmp_path / "valid.zip"
    write_archive(path, dates, np.full(8, .0001))
    series = load_bipower(path, "2000-01-01", "2000-12-31")
    pd.testing.assert_index_equal(series.index, dates.rename("Date"))
    for name, invalid_dates, values in [
        ("duplicate", dates.insert(0, dates[0]), np.full(9, .0001)),
        ("zero", dates, [0., *np.full(7, .0001)]),
        ("nan", dates, [np.nan, *np.full(7, .0001)]),
    ]:
        bad = tmp_path / f"{name}.zip"
        write_archive(bad, invalid_dates, values)
        with pytest.raises(ValueError, match="positive finite"):
            load_bipower(bad, "2000-01-01", "2000-12-31")


def test_next_market_day_target_is_not_next_observed_day_and_future_inputs_are_excluded():
    dates = pd.bdate_range("2009-12-21", periods=14)
    observed = dates.delete(6)
    bv = pd.Series(np.arange(1, len(observed) + 1) * .0001, index=observed)
    prepared = prepare_paper_forecasts(bv, dates, 3)
    row = prepared.rows.loc[dates[5]]
    assert row.return_date == dates[6]
    assert np.isnan(row.realized_bv)
    assert dates[6] not in prepared.rows.index
    earlier = prepared.rows.index <= dates[5]
    changed = bv.copy()
    changed.loc[dates[7]:] *= 20
    mutated = prepare_paper_forecasts(changed, dates, 3)
    np.testing.assert_array_equal(prepared.inputs[earlier], mutated.inputs[earlier])
    with pytest.raises(ValueError, match="known market dates"):
        prepare_paper_forecasts(bv, dates.delete(1), 3)


def portfolio_fixture():
    dates = pd.bdate_range("2010-01-04", periods=6)
    returns = pd.DataFrame({"SPY": [.10, -.08, .04, -.02, .03, -.01],
                            "TLT": [.01, .02, -.01, .01, -.02, .02], "GLD": .005}, index=dates)
    # No new issue on the second day's preceding close: day two must retain drifted holdings.
    positions = [0, 2, 3, 4, 5]
    forecast_dates = pd.DatetimeIndex([pd.Timestamp("2009-12-31"), *dates[[1, 2, 3, 4]]])
    variances = np.array([.0001, .0004, .0009, .0004, .0001])
    forecasts = pd.DataFrame({"return_date": dates[positions], "realized_bv": variances,
                             "ewma_bv": variances}, index=forecast_dates)
    settings = replace(PaperSettings(), evaluation_end="2010-01-31", quantiles=(.9,), caps=(.2,))
    return returns, forecasts, settings


def test_paper_scaling_cash_funding_and_missing_signal_holdings_match_hand_calculation():
    returns, forecasts, settings = portfolio_fixture()
    tables = paper_portfolios(returns, forecasts, settings)
    weights = tables["target_weights"].set_index(["portfolio", "Date"])
    initial_stock = .10 / (22 * np.sqrt(.0001))
    assert weights.loc[("ewma_clean", returns.index[0]), "SPY"] == pytest.approx(initial_stock)
    assert len(tables["daily_returns"]) == len(returns)
    path = tables["daily_returns"].ewma_clean
    assert path.iloc[0] == pytest.approx(initial_stock * .10)
    drifted_stock = initial_stock * 1.10 / (1 + initial_stock * .10)
    assert path.iloc[1] == pytest.approx(drifted_stock * -.08)
    assert not weights.loc[("ewma_clean", returns.index[1]), "rebalance"]
    np.testing.assert_allclose(weights[["SPY", "TLT", "GLD", "RF"]].sum(axis=1), 1)
    assert (weights[["SPY", "TLT", "GLD", "RF"]] >= 0).all().all()
    for asset in ("TLT", "GLD"):
        modified = weights.loc[f"ewma_{asset}_q0.9_cap0.2"]
        pd.testing.assert_series_equal(modified.SPY, weights.loc["ewma_clean"].SPY)
        assert modified[asset].max() == pytest.approx(.2)
    assert tables["forecast_scores"].monthly_mse.iloc[0] == 0
    threshold = tables["thresholds"].threshold.iloc[0]
    assert threshold == pytest.approx(np.quantile(22 * np.sqrt(forecasts.ewma_bv), .9))


def test_invalid_forecast_and_future_refit_fail():
    returns, forecasts, settings = portfolio_fixture()
    invalid = forecasts.copy()
    invalid.loc[invalid.index[2], "ewma_bv"] = np.nan
    with pytest.raises(ValueError, match="invalid ewma"):
        paper_portfolios(returns, invalid, settings)
    invalid = forecasts.assign(gru_bv=forecasts.ewma_bv, gru_refit_asof=pd.Timestamp("2011-01-01"))
    with pytest.raises(ValueError, match="refits must precede"):
        paper_portfolios(returns, invalid, settings)


def test_paper_baseline_command_is_offline_and_refuses_existing_outputs(tmp_path, monkeypatch):
    dates = pd.bdate_range("2009-11-01", "2010-02-26")
    config = tmp_path / "paper.toml"
    config.write_text('''[study]
name = "test"
start = "2009-11-01"
end = "2010-02-28"
benchmark = "SPY"
[assets]
SPY = "Stocks"
TLT = "Treasuries"
GLD = "Gold"
[paper]
training_start = "2009-11-01"
training_end = "2009-12-31"
evaluation_start = "2010-01-01"
evaluation_end = "2010-02-28"
lookback = 3
''')
    raw = tmp_path / "oxford.zip"
    write_archive(raw, dates, .0001 * (2 + np.sin(np.arange(len(dates)))))
    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "french_factors_daily.zip").write_bytes(b"synthetic; parser stubbed")
    returns = pd.DataFrame({"SPY": .01 * np.sin(np.arange(len(dates))),
                            "TLT": .002, "GLD": -.001}, index=dates)
    factors = pd.DataFrame({"RF": .001, "Mkt-RF": returns.SPY}, index=dates)
    monkeypatch.setattr("financial_economics.paper.load_returns", lambda *a, **kw:
                        ReturnData(returns, factors, dates[0], {}))
    monkeypatch.setattr("financial_economics.paper.parse_french_factors", lambda _: factors)
    monkeypatch.setattr("financial_economics.paper_gru.paper_gru_forecasts",
                        lambda *a, **kw: pytest.fail("Baseline must not train"))
    destination = tmp_path / "new-run"
    args = ["--config", str(config), "paper", "--baseline-only", "--data-dir", str(source),
            "--oxford-archive", str(raw), "--output-dir", str(destination)]
    assert cli.main(args) == 0
    metadata = json.loads((destination / "run.json").read_text())
    assert metadata["stage"] == "baseline"
    assert metadata["cash_return"] == 0 and metadata["cost_bps"] == 0
    assert "retrospective" in metadata["thresholds"]
    assert len(metadata["files"]) >= 8
    assert not list(destination.glob("*.md"))
    assert cli.main(args) == 1
    assert load_paper_settings(config).lookback == 3


def test_two_session_horizon_keeps_calendar_delay_and_missing_targets():
    dates = pd.bdate_range("2009-12-21", periods=14)
    observed = dates.delete(6)
    bv = pd.Series(np.arange(1, len(observed) + 1) * .0001, index=observed)
    prepared = prepare_paper_forecasts(bv, dates, lookback=3, horizon=2)
    row = prepared.rows.loc[dates[4]]
    assert row.trade_date == dates[5]
    assert row.return_date == dates[6]
    assert np.isnan(row.realized_bv)
    assert dates[6] not in prepared.rows.index
    np.testing.assert_array_equal(prepared.inputs[2, :, 0], bv.loc[dates[2]:dates[4]])
    original = prepare_paper_forecasts(bv, dates, lookback=3)
    assert "trade_date" not in original.rows
    assert (original.rows.return_date == dates[dates.get_indexer(original.rows.index) + 1]).all()
    for horizon in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            prepare_paper_forecasts(bv, dates, lookback=3, horizon=horizon)
