"""Monthly holdings examples, including boundary allocations."""

import numpy as np
import pandas as pd
import pytest

from financial_economics.evaluation.metrics import monthly_equal_weight, monthly_fixed_weight


def example_returns():
    return pd.DataFrame(
        {"Stock": [1.0, 0.0, 0.0], "Cash": [0.0, 1.0, 1.0]},
        index=pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03"]),
    )


def test_fixed_targets_drift_inside_month_and_reset_before_next_month_return():
    result = monthly_fixed_weight(example_returns(), {"Stock": 0.25, "Cash": 0.75}, "Mix")
    # After the stock doubles, holdings become 40% stock and 60% cash.
    np.testing.assert_allclose(result, [0.25, 0.60, 0.75])
    assert result.name == "Mix"


@pytest.mark.parametrize("held", ["Stock", "Cash"])
def test_zero_and_one_weights_reproduce_held_asset_exactly(held):
    returns = example_returns()
    weights = {asset: float(asset == held) for asset in returns}
    result = monthly_fixed_weight(returns, weights)
    pd.testing.assert_series_equal(result, returns[held].rename("Portfolio"))


def test_weight_mapping_order_does_not_change_holdings():
    returns = example_returns()
    result = monthly_fixed_weight(returns, {"Cash": 0.75, "Stock": 0.25})
    np.testing.assert_allclose(result, [0.25, 0.60, 0.75])


@pytest.mark.parametrize("weights", [
    {"Stock": 1.0},
    {"Stock": 0.5, "Cash": 0.5, "Other": 0.0},
    {"Stock": 0.5, "Other": 0.5},
    {"Stock": 0.4, "Cash": 0.5},
    {"Stock": 1.1, "Cash": -0.1},
    {"Stock": np.nan, "Cash": 0.5},
    {"Stock": np.inf, "Cash": 0.5},
    {"Stock": True, "Cash": 0.0},
    {"Stock": "0.5", "Cash": 0.5},
])
def test_invalid_targets_are_rejected(weights):
    with pytest.raises(ValueError):
        monthly_fixed_weight(example_returns(), weights)


def test_equal_weight_wrapper_preserves_previous_buy_and_hold_calculation():
    dates = pd.bdate_range("2019-12-20", periods=50)
    returns = pd.DataFrame(
        np.sin(np.arange(150).reshape(50, 3)) * 0.03, index=dates, columns=list("ABC")
    )
    expected = []
    for _, month in returns.groupby(returns.index.to_period("M")):
        sleeve_values = (1 + month).cumprod()
        wealth = sleeve_values.mean(axis=1)
        prior = np.r_[1.0, wealth.to_numpy()[:-1]]
        expected.extend(wealth.to_numpy() / prior - 1)
    result = monthly_equal_weight(returns)
    assert result.name == "Equal weight"
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-15)
    pd.testing.assert_series_equal(
        result, monthly_fixed_weight(returns, dict.fromkeys(returns, 1 / 3), "Equal weight")
    )
