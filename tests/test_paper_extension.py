"""Calendar, past-only calibration, funding, costs, and frozen-selection checks."""

from dataclasses import replace
from hashlib import sha256
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from financial_economics import cli
from financial_economics.paper import PaperSettings, prepare_paper_forecasts
from financial_economics.paper_extension import ExtensionSettings, extension_portfolios, load_frozen_selection
from financial_economics.return_data import ReturnData


def extension_fixture():
    calendar = pd.bdate_range("2008-11-03", "2011-01-31")
    bv_dates = calendar.difference(pd.to_datetime(["2010-03-10"]))
    bv = pd.Series(.0001 * (2 + np.sin(np.arange(len(bv_dates)) / 11)), index=bv_dates)
    prepared = prepare_paper_forecasts(bv, calendar, lookback=3, horizon=2)
    rows = prepared.rows.assign(ewma_bv=prepared.inputs[:, :, 0].mean(axis=1))
    rows["gru_bv"] = rows.ewma_bv * 1.1
    rows["gru_refit_asof"] = pd.Timestamp("2008-12-31")
    rows.loc[rows.index < "2008-12-31", ["gru_bv", "gru_refit_asof"]] = [np.nan, pd.NaT]
    n = np.arange(len(calendar))
    daily = pd.DataFrame({"SPY": .01 * np.sin(n / 5), "TLT": .004 * np.cos(n / 7)}, index=calendar)
    factors = pd.DataFrame({"RF": .0001, "Mkt-RF": daily.SPY}, index=calendar)
    data = ReturnData(daily, factors, calendar[0], {})
    paper = replace(PaperSettings(), training_start="2008-11-01", training_end="2008-12-31",
                    evaluation_start="2009-01-01", evaluation_end="2011-01-31", lookback=3,
                    hidden_sizes=(32,))
    settings = replace(ExtensionSettings(), defensive_assets=("TLT",), costs_bps=(0, 10),
                       windows=("2010:2010", "2011:2011"))
    return data, rows, calendar, paper, settings


def test_extension_preserves_calendar_funds_defense_and_charges_entry_correctly():
    data, rows, calendar, paper, settings = extension_fixture()
    tables, counts = extension_portfolios(data, rows, calendar, paper, settings)
    dates = data.daily.loc[settings.evaluation_start:paper.evaluation_end].index
    assert tables["daily_returns"].index.equals(dates)
    assert "SPY" in tables["monthly_returns"]
    assert tables["benchmark_performance"].window.tolist() == ["2010:2011", "2010:2010", "2011:2011"]
    assert counts["initial_cash_days"] == 1 and counts["missing_signal_days"] == 1
    weights = tables["target_weights"].set_index(["risk_mode", "portfolio", "Date"]).sort_index()
    np.testing.assert_allclose(weights[["SPY", "TLT", "RF"]].sum(axis=1), 1)
    assert (weights[["SPY", "TLT", "RF"]] >= 0).all().all()
    for mode in settings.risk_modes:
        for model in ("ewma", "gru"):
            cash = weights.loc[(mode, f"{model}_cash")]
            defense = weights.loc[(mode, f"{model}_TLT")]
            pd.testing.assert_series_equal(cash.SPY, defense.SPY)
            assert defense.TLT.max() <= .2
    path = tables["daily_returns"]["ewma_cash | paper | cash_10bps"]
    w = weights.loc[("paper", "ewma_cash", dates[1]), "SPY"]
    expected = (1 + w * data.daily.loc[dates[1], "SPY"] + (1 - w) * .0001) / (1 + .001 * w) - 1
    assert path.iloc[0] == pytest.approx(.0001)
    assert path.iloc[1] == pytest.approx(expected)
    assert tables["daily_returns"]["ewma_cash | paper | zero_cash_10bps"].iloc[0] == 0
    fees = tables["fees"].loc[lambda x: x.scenario.eq("cash_0bps")]
    assert fees.fee_fraction.eq(0).all()
    missing_signal_return = calendar[calendar.get_loc("2010-03-10") + 2]
    clean = weights.loc[("paper", "ewma_cash")]
    assert not clean.loc[missing_signal_return, "rebalance"]
    previous = calendar[calendar.get_loc(missing_signal_return) - 1]
    prior = clean.loc[previous, ["SPY", "TLT", "RF"]].to_numpy(dtype=float)
    prior_returns = np.r_[data.daily.loc[previous, ["SPY", "TLT"]], .0001]
    held = prior * (1 + prior_returns) / (1 + prior @ prior_returns)
    today = np.r_[data.daily.loc[missing_signal_return, ["SPY", "TLT"]], .0001]
    assert tables["daily_returns"].loc[missing_signal_return, "ewma_cash | paper | cash_0bps"] == pytest.approx(held @ today)


def test_future_forecasts_cannot_change_prior_thresholds_or_allocations():
    data, rows, calendar, paper, settings = extension_fixture()
    original, _ = extension_portfolios(data, rows, calendar, paper, settings)
    changed = rows.copy()
    changed.loc[changed.index >= "2011-01-03", ["ewma_bv", "gru_bv"]] *= 8
    altered, _ = extension_portfolios(data, changed, calendar, paper, settings)
    pd.testing.assert_frame_equal(original["annual_thresholds"], altered["annual_thresholds"])
    pd.testing.assert_frame_equal(original["daily_returns"].loc[:"2010"], altered["daily_returns"].loc[:"2010"])
    thresholds = original["annual_thresholds"]
    assert (thresholds.calibration_last_issue < thresholds["asof"]).all()
    past = rows.loc[(rows.index >= settings.calibration_start) & (rows.index < "2009-12-31"), "ewma_bv"]
    first = thresholds.loc[lambda x: x.model.eq("ewma") & x.risk_mode.eq("paper") & x.evaluation_year.eq(2010)].iloc[0]
    assert first.threshold == pytest.approx(np.quantile(22 * np.sqrt(past), .9))


def test_misaligned_forecast_and_unavailable_refit_fail():
    data, rows, calendar, paper, settings = extension_fixture()
    bad = rows.copy()
    bad.loc[bad.index[-1], "return_date"] = bad.iloc[-1].trade_date
    with pytest.raises(ValueError, match=r"earn t\+2"):
        extension_portfolios(data, bad, calendar, paper, settings)
    bad = rows.copy()
    bad.loc["2010-01-04", "gru_refit_asof"] = pd.Timestamp("2011-01-01")
    with pytest.raises(ValueError, match="refits must be available"):
        extension_portfolios(data, bad, calendar, paper, settings)


def test_frozen_selection_checks_dates_and_tuning_bytes(tmp_path):
    _, _, _, paper, _ = extension_fixture()
    tuning = tmp_path / "gru_tuning.csv"
    pd.DataFrame({"hidden_size": [32], "selected": [True]}).to_csv(tuning, index=False)
    model = {name: json.loads(json.dumps(getattr(paper, name))) for name in
             ("learning_rate", "dropout", "weight_decay", "seeds")}
    model.update(num_layers=2, selected_hidden_size=32, refit_epochs=17,
                 folds=[{"validation_last_target": "2009-12-31"}])
    source = {"stage": "GRU-UNIVAR partial replication", "oxford_archive_sha256": "digest",
              "paper": {"training_end": "2009-12-31", "lookback": 3}, "gru": model,
              "files": {tuning.name: sha256(tuning.read_bytes()).hexdigest()}}
    (tmp_path / "run.json").write_text(json.dumps(source))
    selected, parent = load_frozen_selection(tmp_path, paper, pd.Timestamp("2009-12-31"), "digest")
    assert selected == {"hidden_size": 32, "epochs": 17} and parent["selection_last_target"] == "2009-12-31"
    with pytest.raises(ValueError, match="pre-evaluation"):
        load_frozen_selection(tmp_path, paper, pd.Timestamp("2008-12-31"), "digest")
    with tuning.open("a") as destination:
        destination.write("tampered")
    with pytest.raises(ValueError, match="tuning table hash"):
        load_frozen_selection(tmp_path, paper, pd.Timestamp("2009-12-31"), "digest")


def test_baseline_extension_cli_is_offline_and_outputs_are_not_overwritten(tmp_path, monkeypatch):
    data, _, calendar, paper, settings = extension_fixture()
    config = tmp_path / "extension.toml"
    config.write_text('''[study]
name = "test"
start = "2008-11-01"
end = "2011-01-31"
benchmark = "SPY"
[assets]
SPY = "Stocks"
TLT = "Bonds"
[paper]
training_start = "2008-11-01"
training_end = "2008-12-31"
evaluation_start = "2009-01-01"
evaluation_end = "2011-01-31"
lookback = 3
[extension]
defensive_assets = ["TLT"]
costs_bps = [0, 10]
risk_modes = ["paper"]
windows = ["2010:2010", "2011:2011"]
''')
    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "french_factors_daily.zip").write_bytes(b"synthetic parser stub")
    archive = tmp_path / "bv.zip"
    raw = pd.DataFrame({"Date": calendar.strftime("%Y-%m-%d"), "Symbol": ".SPX", "bv": .0002})
    with ZipFile(archive, "w") as destination:
        destination.writestr("original.csv", raw.to_csv(index=False))
    monkeypatch.setattr("financial_economics.paper_extension.load_returns", lambda *a, **kw: data)
    monkeypatch.setattr("financial_economics.paper_extension.parse_french_factors", lambda _: data.factors)
    output = tmp_path / "new-run"
    args = ["--config", str(config), "paper-extension", "--baseline-only", "--data-dir", str(source),
            "--oxford-archive", str(archive), "--output-dir", str(output)]
    assert cli.main(args) == 0
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["forecast_horizon"] == 2 and metadata["initial_cash_days"] == 1
    assert not list(output.glob("*.md"))
    assert cli.main(args) == 1
    second = tmp_path / "reuse"
    reuse_args = ["--config", str(config), "paper-extension", "--forecast-run", str(output),
                  "--data-dir", str(source), "--oxford-archive", str(archive), "--output-dir", str(second)]
    assert cli.main(reuse_args) == 0
    pd.testing.assert_frame_equal(pd.read_csv(output / "performance.csv"), pd.read_csv(second / "performance.csv"))
