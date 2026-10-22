"""Independent holdout, units, missingness and byte-integrity checks for a proxy."""

from argparse import Namespace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from financial_economics import volatility_proxy as proxy


def test_percentage_accuracy_is_not_correlation_and_missing_dates_count():
    dates = pd.bdate_range("2020-01-01", periods=4)
    actual = pd.Series([.1, .2, .3, .4], index=dates)
    estimated = (actual * 1.1).iloc[:3]
    result = proxy.accuracy(actual, estimated)
    assert result["accuracy_score"] == pytest.approx(.9)
    assert result["median_percentage_error"] == pytest.approx(.1)
    assert result["bv_mean_percentage_error"] == pytest.approx(.21)
    assert result["correlation"] == pytest.approx(1)
    assert result["coverage"] == .75 and result["missing_days"] == 1
    assert result["share_within_10_percent"] == .75
    assert result["minimum_daily_accuracy"] == pytest.approx(.9)
    assert result["days_below_85_percent_accuracy"] == 1
    assert not proxy.passes_requested_gate(result)
    unavailable = proxy.accuracy(actual, estimated.iloc[:0])
    assert unavailable["accuracy_score"] is None and unavailable["scored_days"] == 0


def test_calibration_never_uses_later_labels_and_export_remains_positive():
    dates = pd.bdate_range("2008-01-01", "2011-01-01")
    x = pd.Series(.1 + np.arange(len(dates)) / 10000, index=dates)
    features = pd.DataFrame({"trade_rv5": x})
    actual = x * .9
    fitted = proxy.fit_proxy(features, actual.loc[:"2009-12-31"], "scale", ("trade_rv5",))
    altered = actual.copy()
    altered.loc["2010":] *= 100
    same = proxy.fit_proxy(features, altered.loc[:"2009-12-31"], "scale", ("trade_rv5",))
    assert fitted == same and fitted["scale"] == pytest.approx(.9)
    estimated = proxy.estimate_proxy(features, fitted)
    np.testing.assert_allclose(estimated, actual)
    assert estimated.gt(0).all() and fitted["calibration_last"] == "2009-12-31"


def test_log_mapping_matches_known_power_relation():
    dates = pd.bdate_range("2000-01-01", periods=100)
    x = pd.DataFrame({"trade_rv5": np.linspace(.05, .5, len(dates))}, index=dates)
    actual = .8 * x.trade_rv5 ** 1.1
    fitted = proxy.fit_proxy(x, actual, "log_linear", ("trade_rv5",))
    np.testing.assert_allclose(proxy.estimate_proxy(x, fitted), actual, rtol=1e-12)
    with pytest.raises(ValueError, match="60"):
        proxy.fit_proxy(x.iloc[:4], actual, "scale", ("trade_rv5",))


def test_missing_estimator_uses_prior_ranking_but_never_fills_absent_days():
    dates = pd.bdate_range("2020-01-01", periods=4)
    first = pd.Series([.2, .3], index=dates[:2])
    second = pd.Series([.8, .7], index=dates[1:3])
    estimated, chosen = proxy.available_estimates([first, second], [0, 1])
    np.testing.assert_allclose(estimated, [.2, .3, .7])
    assert chosen.to_list() == [0, 0, 1]
    assert dates[3] not in estimated.index


def test_source_rejects_tampering_and_does_not_treat_zero_as_a_risk_observation(tmp_path):
    raw = (b"SPY\n84398\nSPDR S & P 500 E T F TRUST\n2\n20200102\n20200103\n"
           b"84398 20200102 .1 1 .01 .12 .13 .11 1 .01 .12 .14\n"
           b"84398 20200103 .1 1 .01 0 .13 .11 1 .01 .12 .14\n")
    (tmp_path / "risklab-spy.txt").write_bytes(raw)
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {
        "risklab-spy.txt": hashlib.sha256(raw).hexdigest()}}))
    frame, _ = proxy.load_risklab(tmp_path)
    assert frame.trade_rv5.isna().sum() == 1
    assert frame.index[0] == pd.Timestamp("2020-01-02")
    (tmp_path / "risklab-spy.txt").write_bytes(raw + b"changed")
    with pytest.raises(ValueError, match="hash"):
        proxy.load_risklab(tmp_path)


def test_existing_outputs_are_never_replaced(tmp_path):
    args = Namespace(data_dir=tmp_path / "source", oxford_archive=tmp_path / "archive.zip", output_dir=tmp_path)
    with pytest.raises(ValueError, match="new output"):
        proxy.run(args)


def test_daily_price_estimates_use_real_ranges_and_only_past_context(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=80)
    timestamp = ((dates.tz_localize("America/New_York") + pd.Timedelta(hours=16)).asi8 // 10**9).tolist()
    result = {"meta": {"symbol": "^GSPC", "currency": "USD", "exchangeTimezoneName": "America/New_York"},
              "timestamp": timestamp, "indicators": {"quote": [{
                  "open": [100.] * len(dates), "high": [102.] * len(dates),
                  "low": [98.] * len(dates), "close": [101.] * len(dates)}]}}
    raw = json.dumps({"chart": {"result": [result]}}).encode()
    (tmp_path / "equity-prices.json").write_bytes(raw)
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {
        "equity-prices.json": hashlib.sha256(raw).hexdigest()}}))
    measures, _ = proxy.load_daily_measures(tmp_path)
    expected_gk = .5 * np.log(102 / 98) ** 2 - (2 * np.log(2) - 1) * np.log(101 / 100) ** 2
    assert measures.daily_gk.iloc[0] == pytest.approx(np.sqrt(252 * expected_gk))
    assert measures.daily_gk_lag1.iloc[0] != measures.daily_gk_lag1.iloc[0]
    assert len(measures) == len(dates) and not measures.index.dayofweek.isin([5, 6]).any()
    assert "daily_gk_past30" not in measures
    assert measures.daily_gk_past22.iloc[23] == pytest.approx(np.sqrt(252 * expected_gk))
    result["indicators"]["quote"][0].update(open=[100.] * len(dates), high=[101.] * len(dates),
                                            low=[100.] * len(dates), close=[101.] * len(dates))
    raw = json.dumps({"chart": {"result": [result]}}).encode()
    (tmp_path / "equity-prices.json").write_bytes(raw)
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {
        "equity-prices.json": hashlib.sha256(raw).hexdigest()}}))
    monotonic, _ = proxy.load_daily_measures(tmp_path)
    assert monotonic.daily_rs_ratio.eq(1.).all()


def test_rejected_proxy_cannot_be_used_for_later_forecasting(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"files": {}, "passes_90_percent_volatility_gate": False}))
    with pytest.raises(ValueError, match="90%"):
        proxy.load_accepted_proxy(tmp_path)


@pytest.mark.parametrize(("passed", "exit_code"), [(True, 0), (False, 2)])
def test_command_exit_status_distinguishes_a_failed_accuracy_gate(monkeypatch, passed, exit_code):
    monkeypatch.setattr(proxy, "run", lambda _: passed)
    assert proxy.main(["--data-dir", "source", "--oxford-archive", "archive.zip",
                       "--output-dir", "new-run"]) == exit_code


def test_good_average_does_not_override_an_individual_day_failure():
    dates = pd.bdate_range("2021-01-01", periods=10)
    actual = pd.Series(.1, index=dates)
    estimate = actual.copy()
    estimate.iloc[-1] *= 1.2
    score = proxy.accuracy(actual, estimate)
    assert score["accuracy_score"] == pytest.approx(.98)
    assert score["minimum_daily_accuracy"] == pytest.approx(.8)
    assert score["days_below_85_percent_accuracy"] == 1
    assert not proxy.passes_requested_gate(score)
    estimate.iloc[-1] = .115
    assert proxy.passes_requested_gate(proxy.accuracy(actual, estimate))


def test_individual_floor_also_covers_history_used_for_training():
    periods = ["calibration", "selection", "final_test"]
    scores = pd.DataFrame({"period": periods, "scope": "all", "coverage": 1.,
                           "accuracy_score": .95, "minimum_daily_accuracy": .86,
                           "days_below_85_percent_accuracy": 0})
    assert proxy.passes_historical_gate(scores)
    scores.loc[0, "minimum_daily_accuracy"] = .5
    scores.loc[0, "days_below_85_percent_accuracy"] = 1
    assert not proxy.passes_historical_gate(scores)
    assert not proxy.passes_historical_gate(scores.iloc[1:])


@pytest.mark.parametrize("method", ["log_ridge", "log_quadratic"])
def test_richer_mapping_uses_training_only_scaling_and_serializable_parameters(method):
    dates = pd.bdate_range("2008-01-01", "2011-12-30")
    x = pd.DataFrame({"trade_rv5": np.linspace(.05, .5, len(dates)),
                      "daily_rs": .03 + np.abs(np.sin(np.arange(len(dates)) / 13)) / 10}, index=dates)
    actual = .8 * x.trade_rv5 ** .9
    training = actual.loc[:"2009"]
    fitted = proxy.fit_proxy(x, training, method, tuple(x))
    changed = x.copy()
    changed.loc["2010":] *= 10
    assert fitted == proxy.fit_proxy(changed, training, method, tuple(x))
    restored = json.loads(json.dumps(fitted))
    np.testing.assert_allclose(proxy.estimate_proxy(x, fitted), proxy.estimate_proxy(x, restored))
    assert np.isfinite(proxy.estimate_proxy(changed, fitted)).all()


def test_selection_prioritizes_the_worst_day_after_requiring_a_good_average():
    frame = pd.DataFrame({"candidate": [0, 1, 2], "coverage": [1., 1., 1.],
                          "accuracy_score": [.95, .91, .89],
                          "maximum_percentage_error": [.3, .2, .15],
                          "days_below_85_percent_accuracy": [4, 2, 0],
                          "mean_percentage_error": [.05, .09, .11]})
    assert proxy._ranked_selection(frame).candidate.to_list() == [1, 0]
    frame["calibration_accuracy_score"] = [.92, .5, .94]
    frame["calibration_maximum_percentage_error"] = [.4, 20., .3]
    assert proxy._ranked_selection(frame).candidate.to_list() == [0]


@pytest.mark.parametrize("bad_day", [False, True])
def test_complete_run_applies_both_gates_before_loading_for_a_study(tmp_path, monkeypatch, bad_day):
    dates = pd.bdate_range("2021-01-01", "2022-03-31")
    x = pd.Series(np.linspace(.1, .2, len(dates)), index=dates)
    bv = (.8 * x) ** 2 / 252
    if bad_day:
        bv.loc["2021-12-27"] *= 2.25
    periods = {"calibration": ("2021-01-01", "2021-06-30"),
               "selection": ("2021-07-01", "2021-09-30"),
               "final_test": ("2021-10-01", "2022-02-25")}
    monkeypatch.setattr(proxy, "PERIODS", periods)
    monkeypatch.setattr(proxy, "load_risklab", lambda _: (pd.DataFrame({"trade_rv5": x, "quote_rv5": x}), {}))
    monkeypatch.setattr(proxy, "load_bipower", lambda *a: bv.loc[:"2022-02-25"])
    monkeypatch.setattr(proxy, "protocol", lambda *a: {
        "candidates": [("scale", ("trade_rv5",))], "accuracy_threshold": .90,
        "individual_accuracy_min": .85, "coverage_min": 1.})
    args = Namespace(data_dir=tmp_path / "source", oxford_archive=tmp_path / "oxford.zip",
                     output_dir=tmp_path / "run", prices_dir=None)
    proxy.run(args)
    metadata = json.loads((args.output_dir / "run.json").read_text())
    assert metadata["passes_90_percent_volatility_gate"]
    assert metadata["passes_requested_accuracy_gate"] is not bad_day
    if bad_day:
        with pytest.raises(ValueError, match="85% every-day"):
            proxy.load_accepted_proxy(args.output_dir)
        failures = pd.read_csv(args.output_dir / "failures.csv")
        assert len(failures) == 1 and failures.Date.iloc[0] == "2021-12-27"
    else:
        values, _ = proxy.load_accepted_proxy(args.output_dir)
        np.testing.assert_allclose(values, (.8 * x) ** 2 / 252)


def test_trade_quote_disagreement_is_flagged_without_removing_a_date():
    dates = pd.bdate_range("2020-01-01", periods=3)
    inputs = pd.DataFrame({"trade_rv5": [.1, .3, .1], "quote_rv5": [.1, .1, .11]}, index=dates)
    clean, flagged = proxy.reliable_measures(inputs)
    assert clean.index.equals(dates) and flagged == 1
    assert clean.iloc[1].isna().all() and clean.iloc[0].notna().all()
    pd.testing.assert_frame_equal(inputs, pd.DataFrame({"trade_rv5": [.1, .3, .1], "quote_rv5": [.1, .1, .11]}, index=dates))


def test_convex_mapping_cannot_amplify_disagreeing_inputs_with_negative_weights():
    dates = pd.bdate_range("2000-01-01", periods=100)
    x = pd.DataFrame({"a": np.linspace(.05, .5, len(dates)),
                      "b": np.linspace(.06, .6, len(dates))}, index=dates)
    actual = .8 * np.sqrt(x.a * x.b)
    fitted = proxy.fit_proxy(x, actual, "log_convex", tuple(x))
    weights = np.array(fitted["coefficients"][1:])
    assert weights.min() >= 0 and weights.sum() == pytest.approx(1.)
    np.testing.assert_allclose(proxy.estimate_proxy(x, fitted), actual, rtol=1e-6)
    changed = x.copy()
    changed["a"] *= 4
    ratio = proxy.estimate_proxy(changed, fitted) / proxy.estimate_proxy(x, fitted)
    assert ratio.between(1, 4).all()
