"""The results page is built from the four saved phase outputs."""

import json

import numpy as np
import pandas as pd
import pytest
from financial_economics import cli


@pytest.mark.parametrize("source, description", [
    ({"provider": "Yahoo Finance via yfinance"}, "Yahoo Finance via yfinance; adjusted closes"),
    ({"provider": "WRDS / CRSP CIZ", "representation": "total_return_index"},
     "WRDS / CRSP CIZ; total-return index reconstructed from daily total returns, not traded prices"),
])
def test_page_shows_all_four_phases(tmp_path, monkeypatch, capsys, source, description):
    config = tmp_path / "study.toml"
    config.write_text('''[study]
name = "synthetic-page"
start = "2017-12-01"
end = "2018-03-30"
benchmark = "SPY"
[assets]
SPY = "Stocks"
OTH = "Other fund"
BND = "Bond fund"
''')
    dates = pd.bdate_range("2017-11-30", "2018-03-30")
    market = np.sin(np.arange(len(dates))) * .005
    market[dates.get_loc("2018-02-05")] = -.15
    daily = np.column_stack([market, .2 * market + .0003, -.3 * market + .0001])
    prices = pd.DataFrame(100 * (1 + daily).cumprod(axis=0), index=dates, columns=["SPY", "OTH", "BND"])
    noise = np.random.default_rng(1).normal(0, .001, len(dates))
    factors = pd.DataFrame({"Mkt-RF": market + noise, "RF": .0001}, index=dates)
    manifest = {"files": {"prices.csv": {"sha256": "0" * 64}}, "sources": {"prices": source}}
    monkeypatch.setattr(cli, "load_snapshot", lambda *_: (prices, factors, manifest))

    for phase, command in enumerate(["analyze", "stress", "compare", "explain"], start=1):
        extra = ["--resamples", "200", "--bond-fund", "BND"] if command == "explain" else []
        assert cli.main(["--config", str(config), command, "--output-dir",
                         str(tmp_path / f"phase{phase}"), *extra]) == 0
    page = tmp_path / "page.html"
    assert cli.main(["--config", str(config), "page", "--outputs-dir", str(tmp_path),
                     "--output", str(page)]) == 0
    html = page.read_text(encoding="utf-8")
    assert html.count("<section>") == 6 and "Jan 2018 fall" in html
    # Wording comes from the recorded settings, not this study's defaults.
    assert "3 funds" in html and "The 2018 test period" in html and "21 days long on average" in html
    assert f"ETF source: {description}." in html
    if source["provider"].startswith("WRDS"):
        assert "Yahoo Finance" not in html

    # Phase 4 must describe the same comparison as Phase 3.
    run4 = tmp_path / "phase4" / "run.json"
    run = json.loads(run4.read_text(encoding="utf-8"))
    run["calibration"]["evaluation_start"] = "2018-02-01"
    run4.write_text(json.dumps(run), encoding="utf-8")
    assert cli.main(["--config", str(config), "page", "--outputs-dir", str(tmp_path),
                     "--output", str(page)]) == 1
    assert "different test periods" in capsys.readouterr().err
