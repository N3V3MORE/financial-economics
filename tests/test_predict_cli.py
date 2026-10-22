"""Offline prediction outputs, source integrity, and CPU-only forecast reuse."""

from hashlib import sha256
import json
import sys

import numpy as np
import pandas as pd
import pytest

from financial_economics import cli
from financial_economics.config import load_study
from financial_economics.forecasting import ewma_forecasts, prepare_forecasts
from financial_economics.prediction_config import load_prediction
from financial_economics.return_data import ReturnData


@pytest.fixture
def offline_prediction(tmp_path, monkeypatch):
    config = tmp_path / "prediction.toml"
    config.write_text('''[study]
name = "synthetic prediction"
start = "2013-01-01"
end = "2018-12-31"
benchmark = "SPY"
[assets]
SPY = "Stocks"
TLT = "Treasuries"
GLD = "Gold"
DBC = "Commodities"
[prediction]
resamples = 100
windows = ["2017:2018"]
''', encoding="utf-8")
    dates = pd.bdate_range("2013-01-01", "2018-12-31").difference(
        pd.to_datetime(["2017-01-02", "2018-01-01"]))
    day = np.arange(len(dates))
    returns = pd.DataFrame({"SPY": .009 * np.sin(day / 7) + .002 * np.sin(day / 41),
                            "TLT": .003 * np.cos(day / 5), "GLD": .004 * np.sin(day / 11),
                            "DBC": .005 * np.cos(day / 17)}, index=dates)
    factors = pd.DataFrame({"Mkt-RF": returns.SPY - .0001, "RF": .0001}, index=dates)
    manifest = {"files": {"synthetic_input": "fixture"},
                "sources": {"prices": {"provider": "Synthetic test data"}}}
    data = ReturnData(returns, factors, pd.Timestamp("2012-12-31"), manifest)

    def load_verified(*args, **kwargs):
        assert kwargs == {"verified_monthly": True}
        return data

    monkeypatch.setattr("financial_economics.return_data.load_returns", load_verified)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "financial_economics.gru_forecasting", None)
    study = load_study(config)
    settings = load_prediction(config, study)
    args = ["--config", str(config), "predict", "--data-dir", str(tmp_path / "snapshot")]
    return config, data, study, settings, args


def saved_forecasts(tmp_path, offline_prediction, full=False, change=None):
    _, data, study, settings, _ = offline_prediction
    forecasts, tuning = ewma_forecasts(prepare_forecasts(data.daily.SPY, data.daily.index), settings)
    if full:
        # Deliberately synthetic: this tests saved-model reuse, not neural training.
        forecasts["gru_variance"] = forecasts.ewma_variance * 1.15
        forecasts.loc[forecasts.index < settings.validation_start, "gru_variance"] = np.nan
        forecasts["gru_refit_asof"] = pd.Timestamp("2016-12-30")
    source = tmp_path / "forecast-source"
    source.mkdir()
    forecasts.to_csv(source / "forecasts.csv", index_label="forecast_date", float_format="%.17g", na_rep="NA")
    tuning.to_csv(source / "ewma_tuning.csv", index=False, float_format="%.17g")
    metadata = {"study": study.settings(), "prediction": settings.settings(),
                "source_manifest": data.provenance, "stage": "trained_forecasts" if full else "baseline",
                "files": {path.name: sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}}
    if full:
        metadata["gru"] = {"device": "cuda:0", "test_fixture": True}
    if change == "hash":
        metadata["files"]["forecasts.csv"] = "0" * 64
    elif change == "provenance":
        metadata["source_manifest"] = {"different": "snapshot"}
    elif change == "settings":
        metadata["prediction"] = {**settings.settings(), "volatility_target": .11}
    (source / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    return source


def test_offline_baseline_csv_figures_and_output_guards(offline_prediction, tmp_path, capsys):
    _, data, _, _, args = offline_prediction
    output = tmp_path / "baseline"
    assert cli.main([*args, "--output-dir", str(output), "--ewma-only"]) == 0
    assert "stage: baseline" in capsys.readouterr().out
    assert {path.name for path in output.glob("*.png")} == {
        "forecast_accuracy.png", "growth_drawdowns.png", "protection.png"}
    assert not list(output.glob("*.md")) and not (output / "primary_tests.csv").exists()
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["warmup_cash_days"] == 1 and run["inference"]["primary_family_size"] == 0
    for name, digest in run["files"].items():
        assert sha256((output / name).read_bytes()).hexdigest() == digest
    daily = pd.read_csv(output / "daily_returns.csv", index_col=0, parse_dates=True, float_precision="round_trip")
    figure = pd.read_csv(output / "figure_growth_drawdowns.csv", parse_dates=["Date"], float_precision="round_trip")
    for portfolio in ("A", "C"):
        actual = figure.loc[figure.portfolio.eq(portfolio)].set_index("Date")
        expected = (1 + daily[f"{portfolio} | cash_10bps"]).cumprod()
        np.testing.assert_allclose(actual.loc[daily.index, "wealth"], expected)
        peak = np.maximum.accumulate(np.r_[1., expected.to_numpy()])[1:]
        np.testing.assert_allclose(actual.loc[daily.index, "drawdown"], expected / peak - 1)
    forecasts = pd.read_csv(output / "forecasts.csv", parse_dates=["forecast_date", "return_date"])
    selected = forecasts.loc[forecasts.forecast_date.ge("2016-12-30") & forecasts.return_date.ge("2017-01-01")]
    accuracy = pd.read_csv(output / "figure_forecast_accuracy.csv").iloc[0]
    assert accuracy.qlike == pytest.approx((np.log(selected.ewma_variance) + selected.squared_return / selected.ewma_variance).mean())
    assert accuracy.mse == pytest.approx(((selected.squared_return - selected.ewma_variance) ** 2).mean())
    protection = pd.read_csv(output / "portfolio_conditions.csv")
    expected = protection.loc[protection.scenario.eq("cash_10bps") & protection.condition.eq("negative_spy")]
    pd.testing.assert_frame_equal(pd.read_csv(output / "figure_protection.csv"), expected.reset_index(drop=True))
    inputs = pd.read_csv(output / "evaluation_inputs.csv", index_col=0, parse_dates=True)
    np.testing.assert_allclose(inputs.RF, data.factors.loc[daily.index, "RF"])
    assert cli.main([*args, "--output-dir", str(output), "--ewma-only"]) == 1
    assert cli.main([*args, "--output-dir", str(tmp_path / "snapshot" / "nested"), "--ewma-only"]) == 1
    assert "outside the source snapshot" in capsys.readouterr().err


def test_verified_full_reuse_is_descriptive_for_secondary_asset_without_torch(offline_prediction, tmp_path, capsys):
    *_, args = offline_prediction
    source = saved_forecasts(tmp_path, offline_prediction, full=True)
    output = tmp_path / "gold"
    assert cli.main([*args, "--output-dir", str(output), "--forecast-run", str(source),
                     "--defensive-asset", "GLD"]) == 0
    assert "stage: full; defensive ETF: GLD" in capsys.readouterr().out
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["models"] == ["ewma", "gru"] and run["inference"]["primary_family_size"] == 0
    assert run["forecast_parent"]["forecasts_sha256"] == sha256((source / "forecasts.csv").read_bytes()).hexdigest()
    assert run["gru"]["test_fixture"] and not (output / "primary_tests.csv").exists()
    weights = pd.read_csv(output / "target_weights.csv")
    assert set(weights.portfolio) == set("ABCD") and "GLD" in weights
    assert cli.main([*args, "--output-dir", str(source / "nested"), "--forecast-run", str(source)]) == 1
    assert "outside the saved forecast source" in capsys.readouterr().err


def test_missing_defensive_asset_fails_before_data_or_forecast_work(offline_prediction, tmp_path,
                                                                  monkeypatch, capsys):
    config, _, _, _, args = offline_prediction
    config.write_text(config.read_text(encoding="utf-8").replace('GLD = "Gold"\n', ""), encoding="utf-8")
    monkeypatch.setattr("financial_economics.return_data.load_returns",
                        lambda *_args, **_kwargs: pytest.fail("Must validate before loading data"))
    monkeypatch.setattr("financial_economics.forecasting.prepare_forecasts",
                        lambda *_args: pytest.fail("Must validate before preparing forecasts"))
    output = tmp_path / "missing-asset"
    assert cli.main([*args, "--output-dir", str(output), "--defensive-asset", "GLD"]) == 1
    message = capsys.readouterr().err
    assert "Defensive asset GLD is absent from the study's [assets]" in message
    assert "verified snapshot" in message and not output.exists()


@pytest.mark.parametrize("change, message", [("hash", "hash"), ("provenance", "provenance"),
                                              ("settings", "settings")])
def test_saved_forecast_integrity_failures_are_readable(offline_prediction, tmp_path, capsys, change, message):
    *_, args = offline_prediction
    source = saved_forecasts(tmp_path, offline_prediction, change=change)
    output = tmp_path / "refused"
    assert cli.main([*args, "--output-dir", str(output), "--forecast-run", str(source)]) == 1
    assert message in capsys.readouterr().err and not output.exists()
