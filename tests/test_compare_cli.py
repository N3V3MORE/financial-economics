"""The complete comparison workflow uses the retained snapshot without downloading."""

import json

import numpy as np
import pandas as pd
import pytest

from financial_economics import cli


@pytest.fixture
def comparison_snapshot(tmp_path, monkeypatch, request):
    config = tmp_path / "study.toml"
    config.write_text('''[study]
name = "synthetic-comparison"
start = "2017-12-01"
end = "2018-01-31"
benchmark = "AAA"
[assets]
AAA = "Benchmark"
BBB = "Other fund"
''')
    dates = pd.bdate_range("2017-11-30", "2018-01-31")
    market = np.sin(np.arange(len(dates))) * .005
    if getattr(request, "param", True):
        market[dates.get_loc("2018-01-04")] = -.15
    daily = np.column_stack([market, .2 * market + .0003])
    prices = pd.DataFrame(100 * (1 + daily).cumprod(axis=0), index=dates,
                          columns=["AAA", "BBB"])
    factors = pd.DataFrame({"Mkt-RF": market, "RF": .0001}, index=dates)
    monkeypatch.setattr(cli, "load_snapshot", lambda *_: (prices, factors, {"files": {}}))
    monkeypatch.setattr(cli, "download_snapshot",
                        lambda *_: pytest.fail("Comparison must run offline"))
    return config


@pytest.mark.parametrize("comparison_snapshot", [True, False], indirect=True)
def test_compare_writes_complete_offline_report(comparison_snapshot, tmp_path, request):
    output = tmp_path / "phase3"
    assert cli.main(["--config", str(comparison_snapshot), "compare", "--data-dir",
                     str(tmp_path / "data"), "--output-dir", str(output)]) == 0
    assert {path.name for path in output.iterdir()} == {
        "calibration.json", "evaluation_returns.csv", "performance.csv",
        "stock_exposure.csv", "episodes.csv", "episode_comparison.csv", "run.json",
        "summary.md", "growth.png", "drawdowns.png", "episode_losses.png",
    }
    calibration = json.loads((output / "calibration.json").read_text())
    assert calibration["training_end"] == "2017-12-29"
    assert calibration["evaluation_start"] == "2018-01-01"
    returns = pd.read_csv(output / "evaluation_returns.csv", index_col=0)
    assert list(returns) == ["AAA", "Equal weight", "AAA + cash"]
    exposure = pd.read_csv(output / "stock_exposure.csv")
    assert set(exposure.period) == {"training", "evaluation"}
    comparisons = pd.read_csv(output / "episode_comparison.csv")
    if request.node.callspec.params["comparison_snapshot"]:
        assert len(comparisons) == 3
        assert set(comparisons.asset) == set(returns)
    else:
        assert comparisons.empty
        assert "No AAA episode" in (output / "summary.md").read_text(encoding="utf-8")


def test_compare_refuses_to_overwrite_source_snapshot(comparison_snapshot, tmp_path, capsys):
    data = tmp_path / "data"
    assert cli.main(["--config", str(comparison_snapshot), "compare", "--data-dir",
                     str(data), "--output-dir", str(data / "nested")]) == 1
    assert "outside the source snapshot" in capsys.readouterr().err
    assert not data.exists()


def test_compare_bad_split_is_a_readable_error(comparison_snapshot, tmp_path, capsys):
    output = tmp_path / "bad-split"
    assert cli.main(["--config", str(comparison_snapshot), "compare", "--output-dir",
                     str(output), "--evaluation-start", "2017-12-02"]) == 1
    assert "three observations" in capsys.readouterr().err
    assert not output.exists()
