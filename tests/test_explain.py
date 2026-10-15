"""Phase 4: resampled uncertainty ranges and exact fund contributions."""

import json

import numpy as np
import pandas as pd
import pytest

from financial_economics import cli
from financial_economics.analysis import (monthly_equal_weight, monthly_equal_weight_holdings,
                                          performance_summary)
from financial_economics.explain import (bootstrap_differences, fund_contributions,
                                         stationary_bootstrap_indices)


def _returns(days=120, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-12-02", periods=days)
    market = rng.normal(.0004, .01, days)
    returns = pd.DataFrame({"AAA": market, "BBB": .5 * market + rng.normal(0, .004, days),
                            "CCC": -.3 * market + rng.normal(.0002, .006, days)}, index=dates)
    return returns, pd.Series(.0001, index=dates, name="RF")


def test_bootstrap_rows_are_reproducible_and_form_blocks():
    rows = stationary_bootstrap_indices(50, 400, 10, np.random.default_rng(1))
    again = stationary_bootstrap_indices(50, 400, 10, np.random.default_rng(1))
    assert rows.shape == (400, 50) and rows.min() >= 0 and rows.max() < 50
    np.testing.assert_array_equal(rows, again)
    continued = (np.diff(rows, axis=1) % 50 == 1).mean()
    assert continued == pytest.approx(1 - 1 / 10, abs=.03)
    single = stationary_bootstrap_indices(50, 5, 1, np.random.default_rng(2))
    assert not np.array_equal(single[0], np.arange(50))


def test_identical_portfolios_differ_by_nothing():
    returns, rf = _returns()
    returns["Copy"] = returns["BBB"]
    table = bootstrap_differences(returns, rf, "BBB", ["Copy"], resamples=200, chunk=64)
    assert len(table) == 3
    assert (table[["estimate", "ci_low", "ci_high"]] == 0).all().all()
    assert (table.share_favoring_portfolio == 0).all()


def test_observed_differences_use_the_phase1_formulas_and_seed():
    returns, rf = _returns()
    first = bootstrap_differences(returns, rf, "BBB", ["AAA", "CCC"], resamples=300, seed=9)
    again = bootstrap_differences(returns, rf, "BBB", ["AAA", "CCC"], resamples=300, seed=9)
    other = bootstrap_differences(returns, rf, "BBB", ["AAA", "CCC"], resamples=300, seed=10)
    pd.testing.assert_frame_equal(first, again)
    assert not first.ci_low.equals(other.ci_low)
    metrics = performance_summary(returns, rf)
    for row in first.itertuples():
        expected = metrics.at["BBB", row.measure] - metrics.at[row.reference, row.measure]
        assert row.estimate == pytest.approx(expected, rel=1e-10, abs=1e-12)
        assert row.ci_low <= row.ci_high


@pytest.mark.parametrize("kwargs, message", [
    ({"references": ["BBB"]}, "distinct references"),
    ({"references": ["ZZZ"]}, "distinct references"),
    ({"resamples": 10}, "at least 100"),
    ({"mean_block_length": 0}, "between one"),
])
def test_bootstrap_rejects_unclear_requests(kwargs, message):
    returns, rf = _returns()
    options = {"portfolio": "BBB", "references": ["AAA"], **kwargs}
    with pytest.raises(ValueError, match=message):
        bootstrap_differences(returns, rf, **options)


def test_holdings_reproduce_the_existing_mix():
    returns, _ = _returns()
    holdings = monthly_equal_weight_holdings(returns)
    np.testing.assert_allclose((holdings * returns).sum(axis=1), monthly_equal_weight(returns),
                               rtol=0, atol=1e-15)
    resets = holdings.index.to_period("M") != holdings.index.to_period("M").to_series().shift().to_numpy()
    np.testing.assert_allclose(holdings.loc[resets], 1 / 3)
    assert not np.allclose(holdings.iloc[1], 1 / 3)


def test_contributions_add_up_to_the_compounded_window_return():
    returns, _ = _returns()
    holdings = monthly_equal_weight_holdings(returns)
    mix = monthly_equal_weight(returns)
    # The window crosses a monthly reset and starts after the first observation.
    start, end = returns.index[10], returns.index[60]
    table = fund_contributions(returns, holdings, [("window", start, end)])
    assert list(table.asset) == list(returns.columns)
    expected = (1 + mix.loc[(mix.index > start) & (mix.index <= end)]).prod() - 1
    assert table.contribution.sum() == pytest.approx(expected, rel=1e-12)


def test_contributions_match_dollar_holdings_on_a_simple_example():
    dates = pd.bdate_range("2020-01-01", periods=3)
    returns = pd.DataFrame({"A": [.0, .10, .10], "B": [.0, -.10, .0]}, index=dates)
    holdings = monthly_equal_weight_holdings(returns)
    table = fund_contributions(returns, holdings, [("two days", dates[0], dates[2])]).set_index("asset")
    # $0.50 in each: A gains 0.05 then 0.055; B loses 0.05 then nothing.
    assert table.contribution["A"] == pytest.approx(.105)
    assert table.contribution["B"] == pytest.approx(-.05)


@pytest.mark.parametrize("window", [("reversed", "2020-02-03", "2020-01-02"),
                                    ("weekend end", "2020-01-02", "2020-01-04"),
                                    ("weekend start", "2020-01-04", "2020-01-10"),
                                    ("before history", "2019-11-29", "2020-01-10")])
def test_contributions_reject_unobserved_or_reversed_windows(window):
    returns, _ = _returns()
    with pytest.raises(ValueError, match="observed start and end closes"):
        fund_contributions(returns, monthly_equal_weight_holdings(returns), [window])


@pytest.fixture
def explain_snapshot(tmp_path, monkeypatch):
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
    monkeypatch.setattr(cli, "download_snapshot",
                        lambda *_: pytest.fail("Explain must run offline"))
    return config


def test_explain_writes_complete_offline_report(explain_snapshot, tmp_path):
    output = tmp_path / "phase4"
    assert cli.main(["--config", str(explain_snapshot), "explain", "--output-dir", str(output),
                     "--resamples", "200", "--bond-fund", "BND"]) == 0
    assert {path.name for path in output.iterdir()} == {
        "uncertainty.csv", "contributions.csv", "bond_correlation.csv", "run.json",
        "summary.md", "uncertainty.png", "contributions.png",
    }
    uncertainty = pd.read_csv(output / "uncertainty.csv")
    assert set(uncertainty.reference) == {"AAA + cash", "AAA"}
    contributions = pd.read_csv(output / "contributions.csv")
    assert contributions.window.iloc[-1] == "2018"
    assert contributions.window.nunique() == 2
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["settings"]["bond_fund"] == "BND" and run["portfolio"]["trading_costs"] == 0
    assert "How sure are we?" in (output / "summary.md").read_text(encoding="utf-8")


def test_explain_bond_fund_must_be_a_study_asset(explain_snapshot, tmp_path, capsys):
    output = tmp_path / "phase4"
    assert cli.main(["--config", str(explain_snapshot), "explain", "--output-dir", str(output),
                     "--bond-fund", "TLT"]) == 1
    assert "bond fund" in capsys.readouterr().err
    assert not output.exists()
