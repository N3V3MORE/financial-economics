"""Past-only alignment, local dates, source provenance, and signed features."""

import json

import pandas as pd
import pytest

from financial_economics.paper_sources import _past_values, _yahoo_close


def test_past_levels_preserve_observations_and_reject_future_or_stale_fill():
    source = pd.Series([10., 12.], index=pd.to_datetime(["2020-01-02", "2020-01-06"]))
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    values, carries = _past_values(source, dates, "price")
    assert values.tolist() == [10, 10, 12] and carries == 1
    with pytest.raises(ValueError, match="past observation"):
        _past_values(source, pd.to_datetime(["2020-01-01"]), "price")
    with pytest.raises(ValueError, match="gap longer"):
        _past_values(source, pd.to_datetime(["2020-01-17"]), "price")


def test_yahoo_uses_local_dates_and_rejects_nonpositive_prices():
    timestamps = [int(pd.Timestamp(date, tz="Asia/Kolkata").timestamp()) for date in
                  ("2020-12-31 23:00", "2021-01-01 01:00")]
    raw = {"chart": {"error": None, "result": [{"meta": {"exchangeTimezoneName": "Asia/Kolkata"},
             "timestamp": timestamps,
             "indicators": {"quote": [{"close": [100., 101.]}]}}]}}
    values, _ = _yahoo_close(json.dumps(raw))
    assert values.index.tolist() == [pd.Timestamp("2020-12-31")]
    raw["chart"]["result"][0]["indicators"]["quote"][0]["close"][0] = -1
    with pytest.raises(ValueError, match="positive observed"):
        _yahoo_close(json.dumps(raw))
