import numpy as np
import pandas as pd
import pytest

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


def test_stress_runs_offline_and_preserves_missing_recovery(tmp_path, monkeypatch):
    config = tmp_path / "study.toml"
    config.write_text('''[study]
name = "synthetic-downturn"
start = "2020-01-02"
end = "2020-01-10"
benchmark = "AAA"
[assets]
AAA = "Benchmark"
BBB = "Other fund"
''')
    dates = pd.bdate_range("2020-01-01", "2020-01-10")
    prices = pd.DataFrame({"AAA": [100, 110, 90, 80, 95, 105, 111, 112],
                           "BBB": [100, 100, 75, 65, 70, 75, 78, 79]}, index=dates)
    factors = pd.DataFrame({"Mkt-RF": 0.0, "RF": 0.0}, index=dates)
    monkeypatch.setattr(cli, "load_snapshot", lambda *_: (prices, factors, {"files": {}}))
    monkeypatch.setattr(cli, "download_snapshot", lambda *_: pytest.fail("Stress must be offline"))
    output = tmp_path / "phase2"
    args = ["--config", str(config), "stress", "--data-dir", str(tmp_path / "data"),
            "--output-dir", str(output), "--rolling-window", "3"]
    assert cli.main(args) == 0
    assert {path.name for path in output.iterdir()} == {
        "episodes.csv", "episode_comparison.csv", "rolling_correlations.csv", "run.json",
        "summary.md", "episode_losses.png", "recovery_paths.png", "rolling_correlations.png",
    }
    comparison = pd.read_csv(output / "episode_comparison.csv").set_index("asset")
    assert len(comparison) == 2
    mix = comparison.loc["Equal weight"]
    assert mix.recovery_status == "not_recovered" and mix.recovery_censored
    assert pd.isna(mix.recovery_date) and pd.isna(mix.recovery_trading_days)
    correlations = pd.read_csv(output / "rolling_correlations.csv", index_col=0)
    assert correlations.iloc[:2].isna().all().all()
    import json
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["seed_date"] == "2020-01-01"
    assert metadata["drawdown_threshold"] == .1 and metadata["rolling_window"] == 3
    assert cli.main([*args[:-1], "99"]) == 1


def test_stress_no_qualifying_episode_is_explicit(tmp_path, monkeypatch):
    config = tmp_path / "study.toml"
    config.write_text('''[study]
name = "synthetic-no-downturn"
start = "2020-01-02"
end = "2020-01-10"
benchmark = "AAA"
[assets]
AAA = "Benchmark"
BBB = "Other fund"
''')
    dates = pd.bdate_range("2020-01-01", "2020-01-10")
    prices = pd.DataFrame({"AAA": np.arange(100, 108), "BBB": np.arange(100, 116, 2)}, index=dates)
    factors = pd.DataFrame({"Mkt-RF": 0.0, "RF": 0.0}, index=dates)
    monkeypatch.setattr(cli, "load_snapshot", lambda *_: (prices, factors, {"files": {}}))
    output = tmp_path / "phase2"
    assert cli.main(["--config", str(config), "stress", "--data-dir", str(tmp_path / "data"),
                     "--output-dir", str(output), "--rolling-window", "3"]) == 0
    assert pd.read_csv(output / "episodes.csv").empty
    assert "No benchmark episode" in (output / "summary.md").read_text()
    assert (output / "rolling_correlations.png").exists()
    assert not (output / "episode_losses.png").exists()
