from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from financial_economics.paper_international import load_oxford_index
from financial_economics.paper_realistic import realistic_portfolios


def _archive(tmp_path, rows):
    path = tmp_path / "oxford.zip"
    frame = pd.DataFrame(rows, columns=["", "Symbol", "bv", "close_price"])
    with ZipFile(path, "w") as archive:
        archive.writestr("oxford.csv", frame.to_csv(index=False))
    return path


def test_index_loader_keeps_one_symbol_and_drops_unusable_bv_without_filling(tmp_path):
    rows = [("2010-01-04 00:00:00+09:00", ".N225", 1e-4, 100.),
            ("2010-01-05 00:00:00+09:00", ".N225", 0., 101.),
            ("2010-01-05 00:00:00+00:00", ".SPX", 2e-4, 900.),
            ("2010-01-06 00:00:00+09:00", ".N225", 3e-4, 99.)]
    frame, dropped = load_oxford_index(_archive(tmp_path, rows), ".N225", "2010-01-01", "2010-12-31")
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2010-01-04", "2010-01-05", "2010-01-06"]
    assert dropped == 1 and frame.loc["2010-01-05", "bv"] == 0
    np.testing.assert_allclose(frame.close, [100., 101., 99.])


def test_index_loader_rejects_nonpositive_prices(tmp_path):
    rows = [("2010-01-04 00:00:00+00:00", ".NSEI", 1e-4, 0.)]
    with pytest.raises(ValueError, match="close prices"):
        load_oxford_index(_archive(tmp_path, rows), ".NSEI", "2010-01-01", "2010-12-31")


def test_realistic_portfolios_accept_a_named_index_benchmark_with_zero_cash():
    calendar = pd.bdate_range("2009-12-01", "2012-01-31")
    rng = np.random.default_rng(5)
    daily = pd.DataFrame({"INDEX": rng.normal(.0003, .012, len(calendar)), "RF": 0.}, index=calendar)
    issues = calendar[(calendar >= "2009-12-31") & (calendar <= "2011-12-28")]
    forecasts = pd.DataFrame({"arfima_bv": 1e-4 * (1.5 + np.sin(np.arange(len(issues)) / 9)),
                              "arfima_refit_asof": issues}, index=issues)
    settings = {"benchmark": "INDEX", "calibration_start": "2010-01-01", "evaluation_start": "2011-01-01",
                "evaluation_end": "2011-12-31", "primary_models": ["arfima"], "secondary_models": [],
                "primary_assets": [], "secondary_assets": [], "monthly_target": .10, "monthly_multiplier": 22,
                "stress_quantile": .90, "defensive_cap": .20, "costs_bps": [0, 10], "main_cost_bps": 10}
    result = realistic_portfolios(daily, calendar, forecasts, settings)
    assert set(result["paths"][10]) == {"INDEX", "arfima_cash", "arfima_static"}
    weight = result["static_controls"].static_spy_weight.iloc[0]
    returns = daily.loc["2011-01-01":"2011-12-31", "INDEX"]
    # With zero cash and no fees, the static mix's first-day return is its fixed weight times the index return.
    assert result["paths"][0]["arfima_static"].iloc[0] == pytest.approx(weight * returns.iloc[0])
