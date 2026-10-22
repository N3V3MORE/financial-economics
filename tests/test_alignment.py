import pandas as pd
import pytest
from financial_economics.evaluation.attribution import bootstrap_differences, fund_contributions


def test_alignment_is_explicit():
    dates = pd.bdate_range("2020-01-01", periods=30)
    returns = pd.DataFrame({"A": [.01, -.02, .03] * 10, "B": [.02, .01, -.01] * 10}, index=dates)
    weights = pd.DataFrame({"A": .7, "B": .3}, index=dates)
    window = [("test", dates[0], dates[-1])]
    pd.testing.assert_frame_equal(fund_contributions(returns, weights, window),
                                  fund_contributions(returns, weights[["B", "A"]], window))
    with pytest.raises(ValueError, match="dates and assets"):
        fund_contributions(returns, weights.iloc[1:], window)
    with pytest.raises(ValueError, match="dates"):
        bootstrap_differences(returns, pd.Series(.001, index=dates + pd.Timedelta(days=1)),
                              "A", ["B"], resamples=100)
