"""Phase 4: uncertainty ranges and fund contributions."""

import json

import numpy as np
import pandas as pd
import pytest

from financial_economics import cli
from financial_economics.evaluation.metrics import (monthly_equal_weight, monthly_equal_weight_holdings,
                                          performance_summary)
from financial_economics.evaluation.attribution import bootstrap_differences, fund_contributions


def _returns(days=120):
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2019-12-02", periods=days)
    market = rng.normal(.0004, .01, days)
    returns = pd.DataFrame({"AAA": market, "BBB": .5 * market + rng.normal(0, .004, days),
                            "CCC": -.3 * market + rng.normal(.0002, .006, days)}, index=dates)
    return returns, pd.Series(.0001, index=dates, name="RF")


def test_identical_portfolios_have_zero_ranges():
    returns, rf = _returns()
    returns["Copy"] = returns["BBB"]
    table = bootstrap_differences(returns, rf, "BBB", ["Copy"], resamples=200)
    assert (table[["estimate", "ci_low", "ci_high", "share_favoring_portfolio"]] == 0).all().all()


def test_observed_differences_match_phase1_and_reruns_match():
    returns, rf = _returns()
    first = bootstrap_differences(returns, rf, "BBB", ["AAA", "CCC"], resamples=300)
    pd.testing.assert_frame_equal(first, bootstrap_differences(returns, rf, "BBB", ["AAA", "CCC"],
                                                               resamples=300))
    metrics = performance_summary(returns, rf)
    for row in first.itertuples():
        expected = metrics.at["BBB", row.measure] - metrics.at[row.reference, row.measure]
        assert row.estimate == pytest.approx(expected, rel=1e-10)


def test_contributions_add_up_to_the_mix_return():
    returns, _ = _returns()
    mix = monthly_equal_weight(returns)
    holdings = monthly_equal_weight_holdings(returns)
    # This window crosses a monthly reset.
    start, end = returns.index[10], returns.index[60]
    table = fund_contributions(returns, holdings, [("window", start, end)])
    expected = (1 + mix.loc[(mix.index > start) & (mix.index <= end)]).prod() - 1
    assert table.contribution.sum() == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="observed start"):
        fund_contributions(returns, holdings, [("weekend start", "2020-01-04", end)])


def test_contributions_on_a_hand_checked_example():
    dates = pd.bdate_range("2020-01-01", periods=3)
    returns = pd.DataFrame({"A": [.0, .10, .10], "B": [.0, -.10, .0]}, index=dates)
    table = fund_contributions(returns, monthly_equal_weight_holdings(returns),
                               [("two days", dates[0], dates[2])]).set_index("asset")
    # $0.50 in each: A gains 0.05 then 0.055; B loses 0.05 then nothing.
    assert table.contribution["A"] == pytest.approx(.105)
    assert table.contribution["B"] == pytest.approx(-.05)


def test_explain_writes_its_report_offline(tmp_path, monkeypatch):
    config = tmp_path / "study.toml"
    config.write_text('''[study]
name = "synthetic-explain"
start = "2017-12-01"
end = "2018-03-30"
benchmark = "AAA"
[assets]
AAA = "Benchmark"
BBB = "Other fund"
BND = "Bond fund"
''')
    dates = pd.bdate_range("2017-11-30", "2018-03-30")
    market = np.sin(np.arange(len(dates))) * .005
    market[dates.get_loc("2018-02-05")] = -.15
    daily = np.column_stack([market, .2 * market + .0003, -.3 * market + .0001])
    prices = pd.DataFrame(100 * (1 + daily).cumprod(axis=0), index=dates,
                          columns=["AAA", "BBB", "BND"])
    factors = pd.DataFrame({"Mkt-RF": market, "RF": .0001}, index=dates)
    monkeypatch.setattr(cli, "load_snapshot", lambda *_: (prices, factors, {"files": {}}))
    output = tmp_path / "phase4"
    assert cli.main(["--config", str(config), "explain", "--output-dir", str(output),
                     "--resamples", "200", "--bond-fund", "BND"]) == 0
    assert {path.name for path in output.iterdir()} == {
        "uncertainty.csv", "contributions.csv", "bond_correlation.csv", "run.json",
        "summary.md", "uncertainty.png", "contributions.png",
    }
    contributions = pd.read_csv(output / "contributions.csv")
    assert list(dict.fromkeys(contributions.window)) == ["Jan 2018 fall", "2018"]
    calibration = json.loads((output / "run.json").read_text(encoding="utf-8"))["calibration"]
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert (f"The {calibration['evaluation_start']} to {calibration['evaluation_end']} evaluation "
            "period is one history.") in summary
    assert "eight years" not in summary.lower()
