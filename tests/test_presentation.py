"""The film must refuse inconsistent saved results rather than display them."""

import json

import numpy as np
import pandas as pd
import pytest

from financial_economics.analysis import performance_summary
from financial_economics.presentation import load_presentation, rebased_window, time_prefix


@pytest.fixture
def saved_results(tmp_path):
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    returns = pd.DataFrame(
        {"SPY": [0.1, -0.2, 0.05], "Equal weight": [0.05, -0.1, 0.02]},
        index=dates,
    )
    returns.index.name = "Date"
    returns.to_csv(tmp_path / "daily_returns.csv")
    performance = performance_summary(returns, pd.Series(0.0, index=dates))
    performance.to_csv(tmp_path / "performance.csv")
    pd.DataFrame({"beta": [1.0, 0.7], "alpha_pvalue": [0.4, 0.3],
                  "n_obs": [3, 3]}, index=returns.columns).to_csv(tmp_path / "capm.csv")
    metadata = {"observations": 3, "actual_start": "2020-01-02",
                "actual_end": "2020-01-06", "study": {"benchmark": "SPY",
                "assets": {"SPY": "Equities"}, "trading_days": 252}}
    (tmp_path / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    return tmp_path


def test_presentation_keeps_initial_capital_and_compounds_all_daily_returns(saved_results):
    result = load_presentation(saved_results)
    np.testing.assert_allclose(result.wealth.iloc[0], [1000, 1000])
    np.testing.assert_allclose(result.wealth.iloc[-1], [924, 963.9])
    assert result.dates[0] == pd.Timestamp("2020-01-02")
    assert result.benchmark == "SPY"


@pytest.mark.parametrize("filename,column,value", [
    ("performance.csv", "cagr", 0.9),
    ("capm.csv", "n_obs", 999),
])
def test_presentation_rejects_results_from_different_runs(saved_results, filename, column, value):
    table = pd.read_csv(saved_results / filename, index_col=0)
    table.loc["SPY", column] = value
    table.to_csv(saved_results / filename)
    with pytest.raises(ValueError, match="inconsistent"):
        load_presentation(saved_results)


def test_window_rebases_existing_wealth_at_the_same_closing_date(saved_results):
    result = load_presentation(saved_results)
    window = rebased_window(result, "2020-01-02", "2020-01-06")
    np.testing.assert_allclose(window.iloc[0], [100, 100])
    np.testing.assert_allclose(window.iloc[-1], [84, 91.8])
    with pytest.raises(ValueError, match="observed"):
        rebased_window(result, "2020-01-01", "2020-01-06")


def test_line_reveal_uses_one_time_for_both_series_and_never_shows_future_points():
    times = np.array([0, 2, 5])
    values = np.array([[100, 100], [90, 110], [150, 80]])
    x, y = time_prefix(times, values, 3)
    np.testing.assert_allclose(x, [0, 2, 3])
    np.testing.assert_allclose(y, [[100, 100], [90, 110], [110, 100]])
    x, y = time_prefix(times, values, 2)
    np.testing.assert_allclose(x, [0, 2])
    np.testing.assert_allclose(y, values[:2])
