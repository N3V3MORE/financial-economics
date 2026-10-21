import numpy as np
import pandas as pd
import pytest

from financial_economics.paper_fx import _parse_rates, currency_returns


def test_official_rate_parser_keeps_dates_and_skips_missing_quotes():
    raw = b'''<tr><th>31-DEC-09</th><td>1.0359</td></tr>
              <tr><th>1-JAN-10</th><td>ND</td></tr>
              <tr><th>4-JAN-10</th><td>1.0295</td></tr>'''
    parsed = _parse_rates(raw)
    assert parsed.index.tolist() == pd.to_datetime(["2009-12-31", "2010-01-04"]).tolist()
    assert parsed.tolist() == [1.0359, 1.0295]
    with pytest.raises(ValueError, match="positive dated"):
        _parse_rates(b"<html>maintenance</html>")


def test_currency_conversion_uses_seed_and_correct_direction_without_affecting_other_assets():
    dates = pd.to_datetime(["2009-12-31", "2010-01-04", "2010-01-05"])
    observations = pd.DataFrame({"equity_price": [100., 101., 99.], "SWISS10": [100., 100., 105.],
                                 "BUND10": 100.}, index=dates)
    returns = pd.DataFrame({"EQUITY": [.01, -.02], "SWISS10": 0., "BUND10": 0.}, index=dates[1:])
    rates = pd.DataFrame({"usd_chf": [1., .5, .5], "eur_usd": [1., 1.1, 1.1]}, index=dates)
    result, quality = currency_returns(observations, returns, rates)
    np.testing.assert_allclose(result.SWISS10, [1., .05])
    np.testing.assert_allclose(result.BUND10, [.1, 0.])
    pd.testing.assert_series_equal(result.EQUITY, returns.EQUITY)
    assert quality["fx_carries_including_seed"] == {"usd_chf": 0, "eur_usd": 0}
    rates.iloc[-1] *= 3
    changed, _ = currency_returns(observations, returns, rates)
    pd.testing.assert_series_equal(result.iloc[0], changed.iloc[0])
    with pytest.raises(ValueError, match="past observation"):
        currency_returns(observations, returns, rates.iloc[1:])
