"""Independent accounting examples and safeguards for the fund comparisons."""

import json

import numpy as np
import pandas as pd
import pytest

from financial_economics.analysis import monthly_equal_weight, monthly_fixed_weight
from financial_economics.comparison import compare_portfolios
from financial_economics.config import Study
from financial_economics.costs import monthly_costed_path
from financial_economics.fund_checks import check_funds, write_fund_checks
from financial_economics import cli


def sample():
    dates = pd.bdate_range("2017-10-02", "2018-04-30")
    t = np.arange(len(dates))
    returns = pd.DataFrame({"SPY": np.sin(t) * .018 + .0003,
                            "Bond": np.cos(t / 2) * .007,
                            "Gold": np.cos(t / 3) * .012 + .0001,
                            "Extra": np.sin(t / 7) * .02}, index=dates)
    return returns, pd.Series(.0001, index=dates)


def run_checks(returns, rf, **kwargs):
    settings = {"windows": ("2018-01-01:2018-04-30",), "costs_bps": (0, 25)}
    settings.update(kwargs)
    return check_funds(returns, rf, ["SPY", "Bond", "Gold"], **settings)


def test_entry_fee_and_rebalance_solve_the_self_financing_identity():
    dates = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03"])
    returns = pd.DataFrame({"Stock": [1.0, 0, 0], "RF": [0., 0., 0.]}, index=dates)
    path = monthly_costed_path(returns, {"Stock": .5, "RF": .5}, 100)
    # Entry: v + .01 * .5v = 1. After stock doubles, it weighs 2/3.
    entry_value = 1 / 1.005
    reset_value = (1 - .01 * (2 / 3)) / (1 - .01 * .5)
    np.testing.assert_allclose(path["return"], [entry_value * 1.5 - 1, 0, reset_value - 1])
    np.testing.assert_allclose(path.fee_fraction, .01 * path.traded_fraction)
    assert path.traded_fraction.iloc[1] == 0
    assert entry_value + path.fee_fraction.iloc[0] == pytest.approx(1)
    assert reset_value + path.fee_fraction.iloc[2] == pytest.approx(1)


def test_both_sides_of_an_etf_switch_are_charged():
    dates = pd.to_datetime(["2020-01-31", "2020-02-03"])
    returns = pd.DataFrame({"A": [0., 0.], "B": [0., 0.], "RF": [0., 0.]}, index=dates)
    result = monthly_costed_path(returns, {"A": 0, "B": 1, "RF": 0}, 100,
                                 {"A": 1, "B": 0, "RF": 0})
    assert result["return"].iloc[0] == pytest.approx(1 / 1.01 - 1)
    assert result["return"].iloc[1] == pytest.approx(.99 / 1.01 - 1)
    assert result.traded_fraction.iloc[1] == pytest.approx(1 + .99 / 1.01)


def test_cash_has_no_fee_and_zero_cost_matches_original_monthly_rule():
    returns, rf = sample()
    inputs = returns[["SPY", "Bond"]].assign(RF=rf)
    cash = monthly_costed_path(inputs, {"SPY": 0, "Bond": 0, "RF": 1}, 25)
    np.testing.assert_array_equal(cash["return"], rf)
    assert cash.fee_fraction.sum() == 0
    weights = {"SPY": .5, "Bond": .3, "RF": .2}
    result = monthly_costed_path(inputs, weights)
    np.testing.assert_allclose(result["return"], monthly_fixed_weight(inputs, weights), atol=1e-15)


@pytest.mark.parametrize("cost", [-1, np.inf, np.nan, True, 10_000])
def test_invalid_costs_are_rejected(cost):
    returns, rf = sample()
    with pytest.raises(ValueError, match="Cost"):
        monthly_costed_path(returns[["SPY"]].assign(RF=rf), {"SPY": 1, "RF": 0}, cost)


@pytest.mark.parametrize("start", ["2018-01-01", "2018-01-16"])
def test_original_results_and_controls_match_including_midmonth_holdings(start):
    returns, rf = sample()
    base = returns[["SPY", "Bond", "Gold"]]
    expected = compare_portfolios(base, rf, evaluation_start=start)
    result = run_checks(returns, rf, windows=(f"{start}:2018-04-30",))
    daily = result.daily_returns.loc[result.daily_returns.cost_bps.eq(0)].set_index("Date")
    np.testing.assert_allclose(daily["Original mix"], expected.evaluation_returns["Equal weight"])
    np.testing.assert_allclose(daily["SPY + cash"], expected.evaluation_returns["SPY + cash"])
    np.testing.assert_allclose(daily["Without SPY"], monthly_equal_weight(base.drop(columns="SPY")).loc[start:])
    assert len(result.portfolios) == 4


def test_future_changes_do_not_change_calibration_or_earlier_returns():
    returns, rf = sample()
    first = run_checks(returns, rf)
    changed = returns.copy()
    changed.loc["2018-03-01":, "Gold"] += .1
    second = run_checks(changed, rf)
    pd.testing.assert_frame_equal(first.calibration, second.calibration)
    pd.testing.assert_frame_equal(first.daily_returns.loc[first.daily_returns.Date < "2018-03-01"],
                                  second.daily_returns.loc[second.daily_returns.Date < "2018-03-01"])


def test_additions_use_the_same_dates_and_do_not_change_the_original_mix():
    returns, rf = sample()
    baseline = run_checks(returns, rf)
    extended = run_checks(returns, rf, additions=("Extra",))
    pd.testing.assert_series_equal(baseline.daily_returns["Original mix"],
                                   extended.daily_returns["Original mix"])
    assert extended.portfolios["With Extra"] == ["SPY", "Bond", "Gold", "Extra"]
    assert set(extended.performance.n_obs) == {len(returns.loc["2018-01-01":])}
    rows = extended.performance
    for name in rows.portfolio.unique():
        selected = rows.loc[rows.portfolio.eq(name)].set_index("cost_bps")
        assert selected.loc[25, "cagr"] < selected.loc[0, "cagr"]


def test_duplicate_periods_and_mismatched_cash_dates_are_rejected():
    returns, rf = sample()
    with pytest.raises(ValueError, match="distinct actual"):
        run_checks(returns, rf, windows=("2018-01-01:2018-04-30",) * 2)
    with pytest.raises(ValueError, match="exactly"):
        run_checks(returns, rf.iloc[:-1])
    with pytest.raises(ValueError, match="three"):
        run_checks(returns, rf, windows=("2018-04-30:2019-04-30",))


def test_writer_retains_sources_and_refuses_to_replace_results(tmp_path):
    returns, rf = sample()
    result = run_checks(returns, rf)
    study = Study("test", "2017-10-02", "2018-04-30", "SPY", {name: name for name in returns})
    output = tmp_path / "checks"
    manifest = {"files": {"prices.csv": {"sha256": "test hash"}}}
    write_fund_checks(result, study, manifest, output, ("2018-01-01:2018-04-30",), (0, 25))
    run = json.loads((output / "run.json").read_text())
    assert run["source_manifest"] == manifest
    assert run["backend"]["device"] == "cpu"
    assert len(pd.read_csv(output / "performance.csv")) == len(result.performance)
    with pytest.raises(FileExistsError):
        write_fund_checks(result, study, manifest, output, (), ())


def test_cli_runs_offline_and_keeps_outputs_outside_the_source(tmp_path, monkeypatch, capsys):
    returns, rf = sample()
    study = Study("test", "2017-10-02", "2018-04-30", "SPY", {name: name for name in returns})
    prices = pd.concat([pd.DataFrame(100., index=pd.to_datetime(["2017-09-29"]),
                                     columns=returns.columns), 100 * (1 + returns).cumprod()])
    factors = pd.DataFrame({"Mkt-RF": returns.SPY, "RF": rf})
    monkeypatch.setattr(cli, "load_study", lambda _: study)
    monkeypatch.setattr(cli, "load_snapshot", lambda *_: (prices, factors, {"files": {}}))
    monkeypatch.setattr(cli, "download_snapshot", lambda *_: pytest.fail("Must run offline"))
    source, output = tmp_path / "data", tmp_path / "output"
    args = ["fund-checks", "--additions", "Extra", "--windows", "2018-01-01:2018-04-30",
            "--costs-bps", "0", "10", "--data-dir", str(source), "--output-dir", str(output)]
    assert cli.main(args) == 0
    assert "summary.md" not in capsys.readouterr().out
    assert set(pd.read_csv(output / "performance.csv").portfolio).issuperset({"Original mix", "With Extra"})
    args[-1] = str(source / "nested")
    assert cli.main(args) == 1
    assert "outside the source" in capsys.readouterr().err
    assert not source.exists()
