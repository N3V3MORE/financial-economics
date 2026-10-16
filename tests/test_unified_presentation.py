"""The unified film must not silently combine stale or edited phase outputs."""

import json

import numpy as np
import pandas as pd
import pytest

from financial_economics import (comparison_report, explain_report, report, stress_report,
                                 unified_presentation)
from financial_economics.analysis import monthly_equal_weight, performance_summary
from financial_economics.comparison import compare_portfolios
from financial_economics.config import load_study
from financial_economics.explain import explain_comparison
from financial_economics.stress import compare_episodes, drawdown_episodes, rolling_correlations
from financial_economics.unified_presentation import load_story


@pytest.fixture
def story_repository(tmp_path, monkeypatch):
    (tmp_path / "study.toml").write_text('''[study]
name = "synthetic film"
start = "2017-08-01"
end = "2018-04-30"
benchmark = "SPY"
trading_days = 252
hac_lags = 5
[assets]
SPY = "Stocks"
GLD = "Gold"
''', encoding="utf-8")
    study = load_study(tmp_path / "study.toml")
    dates = pd.bdate_range(study.start, study.end, name="Date")
    rng = np.random.default_rng(31)
    returns = pd.DataFrame({"SPY": rng.normal(.0005, .012, len(dates)),
                            "GLD": rng.normal(.0003, .009, len(dates))}, index=dates)
    # Include a clear drawdown and recovery in each side of the split.
    for start in [20, 120]:
        returns.iloc[start:start + 6, 0] = -.035
        returns.iloc[start + 6:start + 14, 0] = .04
    factors = pd.DataFrame({"Mkt-RF": returns.SPY * .8 + rng.normal(0, .002, len(dates)),
                            "RF": rng.uniform(0, .0001, len(dates))}, index=dates)
    seed = pd.Timestamp("2017-07-31")
    prices = pd.concat([pd.DataFrame(100., index=[seed], columns=returns.columns),
                        100 * (1 + returns).cumprod()]).rename_axis("Date")
    snapshot = tmp_path / "data/snapshot"
    snapshot.mkdir(parents=True)
    files = {name: {"sha256": str(number) * 64} for number, name in enumerate(
        ["adjusted_close.csv", "factors.csv", "french_factors_daily.zip"], start=1)}
    manifest = {"files": files}
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in files:
        (snapshot / name).write_bytes(b"synthetic boundary fixture")

    def snapshot_boundary(actual_study, directory):
        assert actual_study == study
        assert directory == snapshot
        return prices, factors, manifest

    monkeypatch.setattr(unified_presentation, "load_snapshot", snapshot_boundary)
    monkeypatch.setattr(report, "_write_figures", lambda *args: None)
    monkeypatch.setattr(stress_report, "_episode_figures", lambda *args: None)
    monkeypatch.setattr(stress_report, "_correlation_figure", lambda *args: None)
    monkeypatch.setattr(comparison_report, "_figures", lambda *args: None)
    monkeypatch.setattr(explain_report, "_write_figures", lambda *args: None)
    report.write_outputs(study, returns, factors, manifest, tmp_path / "outputs/phase1")
    combined = returns.assign(**{"Equal weight": monthly_equal_weight(returns)})
    episodes2 = drawdown_episodes(returns.SPY, seed)
    stress_report.write_stress_outputs(
        study, combined, episodes2,
        compare_episodes(combined[["SPY", "Equal weight"]], episodes2, seed),
        rolling_correlations(combined, "SPY", 63), manifest, tmp_path / "outputs/phase2",
        drawdown_threshold=.10, rolling_window=63, seed_date=seed)
    result = compare_portfolios(returns, factors.RF)
    evaluation = result.evaluation_returns
    episodes3 = drawdown_episodes(evaluation.SPY, result.calibration["evaluation_seed_date"])
    comparison_report.write_comparison_outputs(
        study, result, performance_summary(evaluation, factors.RF.loc[evaluation.index]),
        episodes3, compare_episodes(evaluation, episodes3, result.calibration["evaluation_seed_date"]),
        manifest, tmp_path / "outputs/phase3")
    explained = explain_comparison(returns, factors.RF, "SPY", "GLD", resamples=200)
    explain_report.write_explain_outputs(study, explained, "GLD", manifest,
                                         tmp_path / "outputs/phase4")
    return tmp_path


def test_load_story_checks_three_serialized_phases_against_snapshot(story_repository):
    result = load_story(story_repository)
    assert list(result.phase3_returns) == ["SPY", "Equal weight", "SPY + cash"]
    assert result.phase3_episodes.peak_date.dtype == "datetime64[ns]"
    assert len(result.phase1_returns) > len(result.phase3_returns)
    assert result.evidence["stock_weight"] == result.calibration["stock_weight"]
    assert result.evidence["phase2_episodes"] >= 2
    assert result.evidence["phase3_episodes"] >= 1
    assert len(result.evidence["output_sha256"]) == 18
    assert result.bond_fund == "GLD"
    assert set(result.uncertainty.reference) == {"SPY + cash", "SPY"}
    assert result.contributions.window.iloc[-1] == "2018"
    assert len(result.evidence["input_sha256"]) == 5


@pytest.mark.parametrize("phase,filename,column,increment", [
    (1, "capm.csv", "beta", .2),
    (1, "performance.csv", "sharpe", .2),
    (2, "episodes.csv", "benchmark_drawdown", .01),
    (2, "episode_comparison.csv", "recovery_trading_days", 1),
    (3, "evaluation_returns.csv", "SPY + cash", .01),
    (3, "performance.csv", "sharpe", .2),
    (3, "stock_exposure.csv", "beta", .2),
    (3, "episode_comparison.csv", "recovery_trading_days", 1),
    (4, "contributions.csv", "contribution", .01),
])
def test_load_story_rejects_edited_numerical_evidence(
    story_repository, phase, filename, column, increment
):
    path = story_repository / "outputs" / f"phase{phase}" / filename
    table = pd.read_csv(path)
    table.loc[0, column] += increment
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="inconsistent"):
        load_story(story_repository)


@pytest.mark.parametrize("phase", [1, 2, 3])
def test_load_story_rejects_outputs_from_another_snapshot(story_repository, phase):
    path = story_repository / "outputs" / f"phase{phase}" / "run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    run["snapshot_files"]["factors.csv"]["sha256"] = "other snapshot"
    path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        load_story(story_repository)


@pytest.mark.parametrize("filename", ["calibration.json", "run.json"])
def test_load_story_rejects_a_changed_frozen_target(story_repository, filename):
    path = story_repository / "outputs/phase3" / filename
    saved = json.loads(path.read_text(encoding="utf-8"))
    calibration = saved if filename == "calibration.json" else saved["calibration"]
    calibration["stock_weight"] += .01
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration stock_weight"):
        load_story(story_repository)

def test_load_story_rejects_changed_recovery_dates(story_repository):
    path = story_repository / "outputs/phase3/episode_comparison.csv"
    table = pd.read_csv(path)
    table.loc[0, "recovery_date"] = "2018-04-30"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="episode_comparison.csv is inconsistent"):
        load_story(story_repository)
