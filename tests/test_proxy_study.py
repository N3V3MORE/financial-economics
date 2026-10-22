"""A failed approximation cannot enter the study; later ETF returns remain complete."""

from argparse import Namespace
import json

import numpy as np
import pandas as pd
import pytest

from financial_economics.studies import proxy as study
from financial_economics.datasets.returns import ReturnData


def args_fixture(tmp_path):
    return Namespace(stage="freeze", output_dir=tmp_path / "run", proxy_run=tmp_path / "proxy",
                     prices_dir=tmp_path / "prices", returns_dir=tmp_path / "returns",
                     portfolio_config=tmp_path / "settings.toml")


def test_failed_proxy_is_rejected_before_any_study_output_is_created(tmp_path):
    args = args_fixture(tmp_path)
    args.proxy_run.mkdir()
    (args.proxy_run / "run.json").write_text(json.dumps({"files": {}, "passes_90_percent_volatility_gate": False}))
    with pytest.raises(ValueError, match="90%"):
        study.run(args)
    assert not args.output_dir.exists()


def test_proxy_stages_keep_all_later_returns_and_estimated_target_labels(tmp_path, monkeypatch):
    args = args_fixture(tmp_path)
    calendar = pd.bdate_range("2000-01-03", "2025-12-31")
    bv = pd.Series(.0001 * (1.2 + .5 * np.sin(np.arange(len(calendar)) / 17)), index=calendar)
    rng = np.random.default_rng(14)
    daily = pd.DataFrame({name: rng.normal(.0003, .008, len(calendar))
                          for name in ("SPY", "TLT", "GLD", "DBC")}, index=calendar)
    factors = pd.DataFrame({"RF": .00005}, index=calendar)
    base = {"stress_quantile": .9, "defensive_cap": .2, "sharpe_hac_lags": 5,
            "monthly_hac_lags": 1, "alpha": .05}
    monkeypatch.setattr(study, "load_realistic_settings", lambda _: base)
    monkeypatch.setattr(study, "load_accepted_proxy", lambda _: (
        bv, {"final_test_missing_dates": [], "calibration_last_label": "2013-12-31"}))
    monkeypatch.setattr(study, "load_daily_measures", lambda _: (pd.DataFrame(index=calendar), {}))
    monkeypatch.setattr(study, "identities", lambda _: {"fixed": "inputs"})
    monkeypatch.setattr(study, "package_code_digests", lambda: {"fixed": "code"})
    monkeypatch.setattr(study, "load_study", lambda _: None)
    monkeypatch.setattr(study, "load_returns", lambda *a, **k: ReturnData(daily, factors, calendar[0], {}))

    def forecast(prepared, *_):
        frame = prepared.rows.copy()
        assert prepared.inputs.shape[1:] == (22, 1)
        frame["gru_univar_log_bv"] = prepared.inputs[:, -1, 0] * 1.05
        frame["gru_univar_log_refit_asof"] = frame.index
        return frame, pd.DataFrame({"selected": [True]}), {"forecast_horizon": 2}

    monkeypatch.setattr(study, "matched_gru", forecast)
    study.run(args)
    for stage in ("forecast", "evaluate"):
        args.stage = stage
        study.run(args)
    output = args.output_dir / "evaluate"
    paths = pd.read_csv(output / "daily_returns_10bps.csv", index_col=0, parse_dates=True)
    assert paths.index.equals(calendar[(calendar >= "2022-01-01") & (calendar <= "2025-12-31")])
    np.testing.assert_allclose(paths.iloc[0], factors.RF.iloc[0])
    tests = pd.read_csv(output / "tests.csv")
    assert len(tests) == 3 and tests.family.eq("proxy_exploratory").all()
    frame = pd.read_csv(output / "forecasts.csv")
    assert frame.target_is_estimated.all()
    assert frame.input_observations.eq(22).all()
    assert pd.read_csv(output / "performance.csv").cost_bps.nunique() == 6
    with pytest.raises(ValueError, match="never overwritten"):
        study.run(args)


def test_trading_histories_count_sessions_across_weekends_and_holidays():
    calendar = pd.bdate_range("2023-12-01", "2024-02-01").difference(pd.to_datetime(["2023-12-25", "2024-01-01"]))
    bv = pd.Series(np.arange(len(calendar)) / 100000 + .0001, index=calendar)
    prepared = study.trading_histories(bv, calendar)
    assert prepared.inputs.shape == (len(calendar) - 23, 22, 1)
    np.testing.assert_array_equal(prepared.inputs[0, :, 0], bv.iloc[:22])
    assert prepared.rows.trade_date.iloc[0] == calendar[22]
    assert prepared.rows.return_date.iloc[0] == calendar[23]
    assert prepared.rows.realized_bv.iloc[0] == bv.iloc[23]
    assert (prepared.rows.history_last - prepared.rows.history_first).dt.days.max() > 29
    with pytest.raises(ValueError, match="every ordered market date"):
        study.trading_histories(bv.drop(calendar[5]), calendar)
