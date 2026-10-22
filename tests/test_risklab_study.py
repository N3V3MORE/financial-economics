import json

import numpy as np
import pandas as pd
import pytest

from financial_economics.provenance import file_digest
from financial_economics.studies.recent import univariate_inputs
from financial_economics.datasets.volatility import load_risklab_bv


def test_missing_risklab_dates_stay_missing_and_targets_are_actual_t_plus_two():
    calendar = pd.bdate_range("2020-01-01", periods=12)
    bv = pd.Series(np.arange(1, 13, dtype=float), index=calendar).drop(calendar[6])
    prepared = univariate_inputs(bv, calendar, lookback=3)
    rows = prepared.rows
    # A window containing the missing session is skipped, never filled or bridged.
    assert not rows.index.isin(calendar[6:9]).any() and calendar[9:10].isin(rows.index).all()
    for position, issue in enumerate(rows.index):
        assert prepared.inputs[position, :, 0].tolist() == bv.loc[:issue].iloc[-3:].tolist()
        window = calendar[calendar.get_loc(issue) - 2:calendar.get_loc(issue) + 1]
        assert window.isin(bv.index).all()
    # The target is the market session two days after issue, even when that date lacks a measurement.
    assert (rows.return_date == calendar[calendar.get_indexer(rows.index) + 2]).all()
    assert rows.loc[rows.return_date == calendar[6], "realized_bv"].isna().all()
    assert rows.realized_bv.dropna().isin(bv).all()


def test_unknown_market_dates_are_rejected():
    calendar = pd.bdate_range("2020-01-01", periods=6)
    bv = pd.Series(1., index=calendar.append(pd.DatetimeIndex(["2020-02-01"])))
    with pytest.raises(ValueError, match="known market dates"):
        univariate_inputs(bv, calendar, lookback=2)


def test_risklab_volatility_becomes_daily_variance_in_oxford_units(tmp_path):
    rows = ["84398 20200102 0.2 0 0 0.16 0.15 0.2 0 0 0.17 0.16",
            "84398 20200103 0.2 0 0 0 0.15 0.2 0 0 0.17 0.16",
            "84398 20200106 0.2 0 0 0.32 0.15 0.2 0 0 0.17 0.16"]
    source = tmp_path / "risklab-spy.txt"
    source.write_text("\n".join(["SPY", "84398", "SPDR", "3", "20200102", "20200106", *rows]) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {source.name: file_digest(source)}}))
    bv, _ = load_risklab_bv(tmp_path, "2020-01-01", "2020-12-31")
    # Zero is unavailable, not zero risk; annual volatility = sqrt(252 * BV).
    assert bv.index.tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06")]
    np.testing.assert_allclose(np.sqrt(252 * bv), [.16, .32])


def test_history_before_2000_uses_french_dates_then_the_existing_calendar():
    from financial_economics.studies.recent import market_calendar
    observations = pd.DataFrame({"equity_price": 1.}, index=pd.bdate_range("2000-01-03", "2020-12-31"))
    returns = pd.DataFrame(index=pd.bdate_range("2020-06-01", "2025-12-31"))
    french = pd.bdate_range("1995-12-01", "2001-01-31")
    default = market_calendar(observations, returns)
    early = market_calendar(observations, returns, french, "1996-01-02")
    assert default[0] == pd.Timestamp("2000-01-03") and default.is_unique and default.is_monotonic_increasing
    assert early[0] == pd.Timestamp("1996-01-02") and early.is_unique and early.is_monotonic_increasing
    assert early[early >= "2000-01-01"].equals(default)
    with pytest.raises(ValueError, match="French"):
        market_calendar(observations, returns, None, "1996-01-02")


def test_defensive_assets_are_optional_and_default_matches_the_stock_cash_pilot():
    from pathlib import Path

    from financial_economics.studies.proxy import portfolio_settings
    from financial_economics.studies.recent import trial_portfolio_settings
    config = Path("configs/paper-realistic.toml")
    assert trial_portfolio_settings(config, False) == portfolio_settings(config)
    defensive = trial_portfolio_settings(config, True)
    assert defensive["primary_assets"] == ["TLT", "GLD"] and defensive["secondary_assets"] == ["DBC"]
    assert {k: v for k, v in defensive.items() if not k.endswith("_assets")} == {
        k: v for k, v in portfolio_settings(config).items() if not k.endswith("_assets")}
