import numpy as np
import pandas as pd

from financial_economics import cli
from financial_economics.config import load_study


def test_analyze_creates_outputs_without_downloading(tmp_path, monkeypatch):
    config = tmp_path / "study.toml"
    config.write_text('''[study]
name = "synthetic-test"
start = "2020-01-02"
end = "2020-04-20"
benchmark = "AAA"
trading_days = 250
[assets]
AAA = "First synthetic asset"
BBB = "Second synthetic asset"
''')
    study = load_study(config)
    dates = pd.bdate_range("2020-01-01", "2020-04-21")
    rng = np.random.default_rng(17)
    market = rng.normal(.0003, .01, len(dates))
    rf = np.full(len(dates), .0001)
    daily = np.column_stack([rf + market + rng.normal(0, .002, len(dates)),
                             rf + .3 * market + rng.normal(0, .006, len(dates))])
    prices = pd.DataFrame(100 * np.cumprod(1 + daily, axis=0), index=dates,
                          columns=study.tickers)
    factors = pd.DataFrame({"Mkt-RF": market, "RF": rf}, index=dates)
    monkeypatch.setattr(cli, "load_snapshot", lambda *_: (prices, factors, {"files": {}}))

    def unexpected_download(*_):
        raise AssertionError("Offline analysis must not download")

    monkeypatch.setattr(cli, "download_snapshot", unexpected_download)
    destination = tmp_path / "outputs"
    result = cli.main(["--config", str(config), "analyze", "--data-dir", str(tmp_path / "data"),
                       "--output-dir", str(destination)])
    assert result == 0
    assert {path.name for path in destination.iterdir()} == {
        "daily_returns.csv", "performance.csv", "capm.csv", "correlation.csv",
        "covariance_annualized.csv", "annual_returns.csv", "run.json", "summary.md",
        "growth.png", "drawdowns.png", "correlations.png", "risk_return.png",
    }
    summary = (destination / "summary.md").read_text(encoding="utf-8")
    assert "Equal weight" in summary
    assert "No optimized strategy" in summary
    assert "250 trading days" in summary and "AAA is the portfolio benchmark" in summary
    performance = pd.read_csv(destination / "performance.csv", index_col=0)
    assert set(performance.index) == {"AAA", "BBB", "Equal weight"}


def test_missing_config_is_a_readable_error(tmp_path, capsys):
    assert cli.main(["--config", str(tmp_path / "absent.toml"), "analyze"]) == 1
    assert "Error:" in capsys.readouterr().err
