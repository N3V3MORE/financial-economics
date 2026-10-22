"""Matched inputs, honest selection, correct horizon and immutable revised runs."""

from argparse import Namespace
import json

import numpy as np
import pandas as pd
import pytest

from financial_economics import paper_revised as revised


def inputs_fixture():
    calendar = pd.bdate_range("2006-01-02", "2011-01-12")
    t = np.arange(len(calendar))
    bv = pd.Series(.0001 * (1.3 + np.sin(t / 13)), index=calendar)
    observations = pd.DataFrame({"GOLD": 100 + t / 10 + np.cos(t / 11),
                                 "OIL": 60 + np.sin(t / 12), "gov_yield": 3 + np.cos(t / 100),
                                 "corp_yield": 5 + np.sin(t / 55)}, index=calendar)
    return observations, bv, calendar


def settings_fixture():
    return {"training_end": "2009-12-31", "evaluation_start": "2010-01-01", "folds": 2,
            "profiles": [[2, 3, 256, .1, .0001]], "seeds": [11], "max_epochs": 1,
            "patience": 1, "learning_rate": .001, "lookback": 3,
            "window_start": "2006-01-01", "window_end": "2009-12-31", "window": "rolling"}


def test_matched_histories_have_same_variance_channel_and_true_execution_targets():
    observations, bv, calendar = inputs_fixture()
    one, five, quality = revised.matched_inputs(observations, bv, calendar, 3)
    pd.testing.assert_frame_equal(one.rows, five.rows)
    np.testing.assert_array_equal(one.inputs, five.inputs[:, :, :1])
    positions = calendar.get_indexer(one.rows.index)
    assert pd.DatetimeIndex(one.rows.trade_date).equals(calendar[positions + 1])
    assert pd.DatetimeIndex(one.rows.return_date).equals(calendar[positions + 2])
    assert quality["external_feature_lag_sessions"] == 1
    day = one.rows.index[150]
    changed = observations.copy()
    changed.loc[day:, "GOLD"] *= 2
    _, altered, _ = revised.matched_inputs(changed, bv, calendar, 3)
    # Even the same-date gold change is unavailable to that day's forecast.
    np.testing.assert_array_equal(five.inputs[five.rows.index <= day],
                                  altered.inputs[altered.rows.index <= day])
    following = calendar[calendar.get_loc(day) + 1]
    assert not np.array_equal(five.inputs[five.rows.index == following],
                              altered.inputs[altered.rows.index == following])


def test_missing_target_is_not_filled_and_does_not_change_input_histories():
    observations, bv, calendar = inputs_fixture()
    original, _, _ = revised.matched_inputs(observations, bv, calendar, 3)
    missing = calendar[-2]
    sparse, _, _ = revised.matched_inputs(observations, bv.drop(missing), calendar, 3)
    assert sparse.rows.loc[sparse.rows.return_date.eq(missing), "realized_bv"].isna().all()
    through = calendar[-4]
    np.testing.assert_array_equal(original.inputs[original.rows.index <= through],
                                  sparse.inputs[sparse.rows.index <= through])


def test_stopping_and_selection_blocks_are_disjoint_and_labels_are_purged():
    observations, bv, calendar = inputs_fixture()
    prepared, _, _ = revised.matched_inputs(observations, bv, calendar, 3)
    for training, stopping, scoring in revised.selection_folds(prepared.rows, "2009-12-31", 3):
        assert not ((training & stopping) | (training & scoring) | (stopping & scoring)).any()
        assert prepared.rows.loc[training, "return_date"].max() < prepared.rows.index[stopping][0]
        assert prepared.rows.loc[stopping, "return_date"].max() < prepared.rows.index[scoring][0]
        assert prepared.rows.loc[scoring, "return_date"].max() <= pd.Timestamp("2009-12-31")
    changed = prepared.rows.copy()
    changed.loc[changed.return_date > "2009-12-31", "realized_bv"] = np.nan
    first = revised.selection_folds(prepared.rows, "2009-12-31", 3)
    second = revised.selection_folds(changed, "2009-12-31", 3)
    for original, altered in zip(first, second, strict=True):
        for a, b in zip(original, altered, strict=True):
            np.testing.assert_array_equal(a, b)


def test_garch_second_step_matches_independent_arch_horizon_forecast():
    from arch import arch_model

    observations, bv, calendar = inputs_fixture()
    prepared, _, _ = revised.matched_inputs(observations, bv, calendar, 3)
    rows = prepared.rows.loc["2010-01-05":"2010-01-05"]
    rng = np.random.default_rng(8)
    values = np.empty(len(calendar))
    variance = .0001
    for i in range(len(values)):
        values[i] = rng.normal() * np.sqrt(variance)
        variance = .000002 + .15 * values[i] ** 2 + .8 * variance
    returns = pd.Series(values, index=calendar)
    result, fits = revised.two_step_classical(rows, bv, returns, settings_fixture(), progress=lambda *a, **k: None)
    history = returns.loc["2006-01-01":rows.index[0]]
    size = len(returns.loc["2006":"2009"])
    for name, asymmetric in (("garch", False), ("gjr", True)):
        model = arch_model(100 * history.iloc[-size:].to_numpy(), mean="Constant", vol="GARCH", p=1,
                           o=int(asymmetric), q=1, power=2, dist="normal", rescale=False)
        fitted = model.fit(disp="off", show_warning=False, options={"maxiter": 1000, "ftol": 1e-8})
        reference = fitted.forecast(horizon=2, method="analytic", reindex=False).residual_variance.iloc[-1, 1] / 10000
        assert result[f"{name}_bv"].iloc[0] == pytest.approx(reference, rel=1e-9)
    assert fits.horizon.eq(2).all()


def test_matched_default_histories_use_exactly_22_trading_observations():
    observations, bv, calendar = inputs_fixture()
    one, five, quality = revised.matched_inputs(observations, bv, calendar)
    pd.testing.assert_frame_equal(one.rows, five.rows)
    assert one.inputs.shape[1:] == (22, 1) and five.inputs.shape[1:] == (22, 5)
    np.testing.assert_array_equal(one.inputs, five.inputs[:, :, :1])
    np.testing.assert_array_equal(one.inputs[0, :, 0], bv.loc[:one.rows.index[0]].iloc[-22:])
    assert quality["history_trading_days"] == 22


def test_tiny_cuda_matched_models_use_identical_folds_and_available_labels(tmp_path):
    try:
        torch, _ = revised._require_cuda()
    except RuntimeError as error:
        if "no compatible GPU" in str(error) or "prediction extra" in str(error):
            pytest.skip(str(error))
        raise
    observations, bv, calendar = inputs_fixture()
    one, five, _ = revised.matched_inputs(observations, bv, calendar, 3)
    metadata = []
    for name, prepared in (("gru_univar_log", one), ("gru_multivar_log", five)):
        frame, tuning, model = revised.matched_gru(prepared, calendar, tmp_path / name,
            settings_fixture(), name, progress=lambda *a, **k: None)
        assert frame[f"{name}_bv"].dropna().gt(0).all()
        assert np.isfinite(tuning.selection_log_mse).all()
        metadata.append(model)
        assert model["device"] == "cuda:0" and model["forecast_horizon"] == 2
        for annual in model["annual_refits"]:
            assert annual["training_last_target"] <= annual["refit_asof"]
            saved = torch.load(tmp_path / name / "2010" / "seed_11.pt", weights_only=True)
            assert saved["target_horizon_sessions"] == 2 and saved["smearing"] > 0
    assert metadata[0]["folds"] == metadata[1]["folds"]


def test_saved_stage_checks_protocol_and_file_integrity(tmp_path):
    stage = tmp_path / "classical"
    stage.mkdir()
    frame = pd.DataFrame({"ewma_bv": [.0001]}, index=pd.to_datetime(["2010-01-04"]))
    frame.to_csv(stage / "forecasts.csv")
    meta = {"protocol_sha256": "frozen", "files": {"forecasts.csv": revised.file_digest(stage / "forecasts.csv")}}
    (stage / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    revised._read_stage(tmp_path, "classical", "frozen")
    with pytest.raises(ValueError, match="exact frozen protocol"):
        revised._read_stage(tmp_path, "classical", "different")
    with (stage / "forecasts.csv").open("a") as destination:
        destination.write("modified")
    with pytest.raises(ValueError, match="saved hash"):
        revised._read_stage(tmp_path, "classical", "frozen")


def test_freeze_never_overwrites_and_changed_code_requires_new_run(tmp_path, monkeypatch):
    config = tmp_path / "settings.toml"
    config.write_text('[reconstruction]\nlookback=3\n', encoding="utf-8")
    monkeypatch.setattr(revised, "_input_identity", lambda _: {"source": "fixed"})
    monkeypatch.setattr(revised, "_portfolio_settings", lambda _: {"risk_mode": "annual"})
    monkeypatch.setattr(revised, "package_code_digests", lambda: {"model": "original"})
    args = Namespace(stage="freeze", config=config, output_dir=tmp_path / "run", data_dir=tmp_path / "source",
                     returns_dir=tmp_path / "returns", risklab_dir=tmp_path / "risklab")
    revised.run(args)
    frozen = json.loads((args.output_dir / "protocol.json").read_text())
    assert frozen["forecast_settings"]["lookback"] == 3
    assert "history_calendar_days" not in frozen["forecast_settings"]
    with pytest.raises(ValueError, match="never overwritten"):
        revised.run(args)
    monkeypatch.setattr(revised, "package_code_digests", lambda: {"model": "changed"})
    args.stage = "classical"
    with pytest.raises(ValueError, match="changed after freezing"):
        revised.run(args)


def test_staged_revision_evaluates_complete_returns_and_joint_exploratory_tests(tmp_path, monkeypatch):
    from financial_economics.return_data import ReturnData

    observations, bv, _ = inputs_fixture()
    # A complete calendar year is required for the conditional monthly tests.
    calendar = pd.bdate_range("2006-01-02", "2011-12-30")
    observations = observations.reindex(calendar).ffill()
    observations["equity_price"] = 100 * np.exp(np.arange(len(calendar)) * .0002)
    bv = pd.Series(.0001 * (1.3 + np.sin(np.arange(len(calendar)) / 13)), index=calendar)
    rng = np.random.default_rng(81)
    daily = pd.DataFrame({name: rng.normal(.0003, .008, len(calendar))
                          for name in ("SPY", "TLT", "GLD", "DBC")}, index=calendar)
    factors = pd.DataFrame({"RF": .00003}, index=calendar)
    portfolio = {"calibration_start": "2010-01-01", "evaluation_start": "2011-01-01",
                 "evaluation_end": "2011-12-31", "primary_models": list(revised.MODELS),
                 "secondary_models": [], "primary_assets": ["TLT", "GLD"], "secondary_assets": ["DBC"],
                 "risk_mode": "annual", "annual_target": .10, "stress_quantile": .9, "defensive_cap": .2,
                 "costs_bps": list(revised.COSTS_BPS), "main_cost_bps": 10, "windows": ["2011:2011"],
                 "initial_cash": True, "liquidate_end": True, "cost_benchmark": True,
                 "require_aligned_horizon": True, "sharpe_hac_lags": 5, "monthly_hac_lags": 1, "alpha": .05}
    config = tmp_path / "config.toml"
    config.write_text('[reconstruction]\nlookback=3\ntraining_end="2009-12-31"\n'
                      'evaluation_start="2010-01-01"\nevaluation_end="2011-12-31"\ndm_hac_lags=5\n',
                      encoding="utf-8")
    monkeypatch.setattr(revised, "_input_identity", lambda _: {"source": "verified"})
    monkeypatch.setattr(revised, "package_code_digests", lambda: {"code": "frozen"})
    monkeypatch.setattr(revised, "_portfolio_settings", lambda _: portfolio)
    monkeypatch.setattr(revised, "load_sources", lambda _: (observations, {}))
    monkeypatch.setattr(revised, "load_risklab_bv", lambda *a: (bv, {}))
    monkeypatch.setattr(revised, "load_study", lambda _: None)
    monkeypatch.setattr(revised, "load_returns", lambda *a, **kw: ReturnData(daily, factors, calendar[0], {}))

    def classical(rows, *_):
        frame = rows.copy()
        for i, name in enumerate(revised.MODELS[:4]):
            frame[f"{name}_bv"] = bv.reindex(frame.index) * (1 + i / 10)
            frame[f"{name}_refit_asof"] = frame.index
        return frame, pd.DataFrame({"horizon": [2]})

    def neural(prepared, calendar, checkpoints, settings, name):
        frame = prepared.rows.copy()
        positions = getattr(prepared, "lengths", np.full(len(frame), prepared.inputs.shape[1])) - 1
        frame[f"{name}_bv"] = prepared.inputs[np.arange(len(frame)), positions, 0] * (
            1.05 if name == "gru_univar_log" else 1.15)
        frame[f"{name}_refit_asof"] = pd.Timestamp("2009-12-31")
        return frame, pd.DataFrame({"selected": [True]}), {"folds": [{"same_dates": True}]}

    monkeypatch.setattr(revised, "two_step_classical", classical)
    monkeypatch.setattr(revised, "matched_gru", neural)
    args = Namespace(stage="freeze", config=config, portfolio_config=config,
                     output_dir=tmp_path / "run", data_dir=tmp_path / "source",
                     returns_dir=tmp_path / "returns", risklab_dir=tmp_path / "risklab")
    revised.run(args)
    for stage in revised.STAGES:
        args.stage = stage
        revised.run(args)
    output = args.output_dir / "evaluate"
    tests = pd.read_csv(output / "tests.csv")
    assert len(tests) == 27 and tests.family.eq("revised_exploratory").all()
    metrics = pd.read_csv(output / "performance.csv")
    assert set(metrics.cost_bps) == set(revised.COSTS_BPS)
    paths = pd.read_csv(output / "daily_returns_10bps.csv", index_col=0)
    assert len(paths) == len(daily.loc["2011"])
    np.testing.assert_allclose(paths.iloc[0], factors.RF.iloc[0])
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["model"]["forecast_horizon"] == 2
    with pytest.raises(ValueError, match="never overwritten"):
        revised.run(args)


def test_issue_dates_whose_history_spans_a_missing_session_are_skipped():
    observations, bv, calendar = inputs_fixture()
    gap = calendar[1300]
    one, five, quality = revised.matched_inputs(observations, bv.drop(gap), calendar, 3)
    pd.testing.assert_frame_equal(one.rows, five.rows)
    position = calendar.get_loc(gap)
    assert not one.rows.index.isin(calendar[position:position + 3]).any()
    assert calendar[[position - 1, position + 3]].isin(one.rows.index).all()
    for row, issue in enumerate(one.rows.index):
        end = calendar.get_loc(issue)
        np.testing.assert_array_equal(one.inputs[row, :, 0], bv.loc[calendar[end - 2:end + 1]])
    assert quality["issues_skipped_for_incomplete_history"] >= 3


def test_segmented_arfima_matches_the_ungapped_estimator_and_never_bridges_gaps():
    from financial_economics.paper_models import _arfima_forecast, _fractional_prediction

    _, bv, calendar = inputs_fixture()
    window = bv.iloc[:400]
    _, details = _arfima_forecast(window.to_numpy())
    reference = _fractional_prediction(np.r_[window, details["raw_forecast_bv"]], details["d"], details["mean_bv"])
    raw, segmented = revised.segmented_arfima(revised.consecutive_segments(window, calendar))
    assert segmented["segments"] == 1 and segmented["d"] == pytest.approx(details["d"], abs=1e-9)
    assert raw == pytest.approx(max(reference, 1e-12), rel=1e-9)
    gapped = window.drop(window.index[[100, 101, 250]])
    segments = revised.consecutive_segments(gapped, calendar)
    assert [len(segment) for segment in segments] == [100, 148, 149]
    # Values before the final gap move the estimate, but never the final run's lag weights.
    changed = gapped.copy()
    changed.iloc[:5] *= 3
    _, a = revised.segmented_arfima(revised.consecutive_segments(gapped, calendar))
    _, b = revised.segmented_arfima(revised.consecutive_segments(changed, calendar))
    assert a["final_segment_length"] == b["final_segment_length"] == 149
    with pytest.raises(ValueError, match="market sessions"):
        revised.consecutive_segments(pd.Series(1., index=[pd.Timestamp("2006-01-07")]), calendar)


def test_hansen_lunde_scale_uses_only_pre_evaluation_dates_and_aligns_targets():
    calendar = pd.bdate_range("2008-01-01", "2011-12-30")
    rng = np.random.default_rng(3)
    bv = pd.Series(rng.uniform(.5, 1.5, len(calendar)) * 1e-4, index=calendar)
    returns = pd.Series(rng.normal(0, 1, len(calendar)) * np.sqrt(1.4 * bv), index=calendar)
    factor = revised.hansen_lunde_factor(bv, returns, "2009-12-31")
    early = calendar[calendar <= "2009-12-31"]
    assert factor == pytest.approx((returns[early] ** 2).sum() / bv[early].sum())
    later = returns.copy()
    later.loc["2010":] *= 10
    assert revised.hansen_lunde_factor(bv, later, "2009-12-31") == factor
    issues = calendar[300:-2]
    frame = pd.DataFrame({"return_date": calendar[302:], "realized_bv": bv.loc[calendar[302:]].to_numpy()},
                         index=issues)
    for model in revised.MODELS:
        frame[f"{model}_bv"] = frame.realized_bv * (factor if model in ("garch", "gjr") else 1.)
    scores, tests = revised.robust_forecast_scores(frame, returns, factor, 5)
    intraday = scores[(scores.target == "intraday") & (scores.loss == "variance_mse")]
    # Rescaled close-to-close forecasts equal the intraday ones on the intraday target.
    assert intraday.mean_loss.max() == pytest.approx(0., abs=1e-30)
    assert set(scores.loss) == {"variance_mse", "qlike", "volatility_mse_reference"}
    # Identical forecasts leave no loss difference to test, so no p-value is invented.
    assert set(tests.target) == {"intraday", "close_to_close"} and "holm_p_value" in tests
