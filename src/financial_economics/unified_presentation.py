"""Verify all four saved phases before displaying their findings in one film."""

from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (capm_summary, monthly_equal_weight, performance_summary,
                       prices_to_returns)
from .comparison import compare_portfolios
from .config import load_study
from .data import align_sample, load_snapshot
from .explain import explain_comparison
from .presentation import PresentationData, load_presentation
from .stress import compare_episodes, drawdown_episodes


@dataclass
class StoryData:
    phase1: PresentationData
    phase1_returns: pd.DataFrame
    phase2_episodes: pd.DataFrame
    phase2_comparison: pd.DataFrame
    phase3_returns: pd.DataFrame
    phase3_performance: pd.DataFrame
    phase3_episodes: pd.DataFrame
    phase3_comparison: pd.DataFrame
    exposure: pd.DataFrame
    calibration: dict
    evidence: dict
    uncertainty: pd.DataFrame
    contributions: pd.DataFrame
    bond_correlation: pd.Series
    bond_fund: str


def _check_table(path: Path, expected: pd.DataFrame, *, indexed: bool = False) -> None:
    """Compare saved cells, including dates/statuses and missing recovery values."""
    options = {"index_col": 0} if indexed else {}
    saved = pd.read_csv(path, **options)
    # Use the report's CSV convention so dates and nullable values compare directly.
    serialized = expected.to_csv(index=indexed, date_format="%Y-%m-%d", na_rep="NA")
    calculated = pd.read_csv(StringIO(serialized), **options)
    try:
        pd.testing.assert_frame_equal(saved, calculated, check_dtype=False,
                                      rtol=1e-8, atol=1e-10)
    except AssertionError as error:
        raise ValueError(f"Saved {path.name} is inconsistent; rerun its phase.") from error


def _check_calibration(saved: dict, expected: dict) -> None:
    if saved.keys() != expected.keys():
        raise ValueError("Saved calibration is inconsistent; rerun compare.")
    for name, value in expected.items():
        actual = saved[name]
        if isinstance(value, float):
            valid = (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                     and np.isclose(actual, value, rtol=1e-8, atol=1e-10))
        else:
            valid = type(actual) is type(value) and actual == value
        if not valid:
            raise ValueError(f"Saved calibration {name} is inconsistent; rerun compare.")


def load_story(root: Path) -> StoryData:
    """Recompute displayed evidence offline from the retained, hash-checked snapshot."""
    root = Path(root)
    study = load_study(root / "study.toml")
    prices, factors, manifest = load_snapshot(study, root / "data/snapshot")
    returns = prices_to_returns(prices, study.start, study.end)
    returns, factors = align_sample(returns, factors, study)
    combined = returns.copy()
    combined["Equal weight"] = monthly_equal_weight(returns)
    seed = prices.index[prices.index < returns.index[0]][-1]
    folders = [root / "outputs" / f"phase{phase}" for phase in [1, 2, 3, 4]]
    metadata = [json.loads((folder / "run.json").read_text(encoding="utf-8"))
                for folder in folders]
    for phase, run in enumerate(metadata, start=1):
        if (run.get("study") != study.settings()
                or run.get("snapshot_files") != manifest["files"]
                or run.get("portfolio", {}).get("trading_costs") != 0
                or run.get("portfolio", {}).get("taxes") != 0):
            raise ValueError(f"Phase {phase} provenance is inconsistent; rerun its phase.")

    phase1 = load_presentation(folders[0])
    _check_table(folders[0] / "daily_returns.csv", combined, indexed=True)
    _check_table(folders[0] / "performance.csv",
                 performance_summary(combined, factors["RF"], study.trading_days), indexed=True)
    _check_table(folders[0] / "capm.csv",
                 capm_summary(combined, factors, study.hac_lags), indexed=True)

    episodes2 = drawdown_episodes(combined[study.benchmark], seed, threshold=.10)
    comparison2 = compare_episodes(combined[[study.benchmark, "Equal weight"]], episodes2, seed)
    run2 = metadata[1]
    if (run2.get("drawdown_threshold") != .10 or run2.get("episodes") != len(episodes2)
            or run2.get("observations") != len(returns)
            or run2.get("seed_date") != str(seed.date())
            or run2.get("actual_start") != str(returns.index[0].date())
            or run2.get("actual_end") != str(returns.index[-1].date())):
        raise ValueError("Phase 2 settings are inconsistent; rerun stress with the 10% rule.")
    _check_table(folders[1] / "episodes.csv", episodes2)
    _check_table(folders[1] / "episode_comparison.csv", comparison2)

    result = compare_portfolios(returns, factors["RF"], study.benchmark, "2018-01-01")
    calibration = json.loads((folders[2] / "calibration.json").read_text(encoding="utf-8"))
    _check_calibration(calibration, result.calibration)
    _check_calibration(metadata[2].get("calibration", {}), result.calibration)
    evaluation = result.evaluation_returns
    performance = performance_summary(evaluation, factors.loc[evaluation.index, "RF"],
                                      study.trading_days)
    evaluation_seed = result.calibration["evaluation_seed_date"]
    episodes3 = drawdown_episodes(evaluation[study.benchmark], evaluation_seed, threshold=.10)
    comparison3 = compare_episodes(evaluation, episodes3, evaluation_seed)
    for filename, table, indexed in [
        ("evaluation_returns.csv", evaluation, True),
        ("performance.csv", performance, True),
        ("stock_exposure.csv", result.exposure, False),
        ("episodes.csv", episodes3, False),
        ("episode_comparison.csv", comparison3, False),
    ]:
        _check_table(folders[2] / filename, table, indexed=indexed)

    settings = metadata[3].get("settings", {})
    if (settings.get("evaluation_start") != "2018-01-01"
            or not isinstance(settings.get("resamples"), int)
            or not isinstance(settings.get("seed"), int)):
        raise ValueError("Phase 4 settings are inconsistent; rerun explain.")
    _check_calibration(metadata[3].get("calibration", {}), result.calibration)
    explained = explain_comparison(
        returns, factors["RF"], study.benchmark, settings.get("bond_fund"), "2018-01-01",
        study.trading_days, settings["resamples"], settings.get("mean_block_length"),
        settings["seed"])
    for filename, table, indexed in [
        ("uncertainty.csv", explained.uncertainty, False),
        ("contributions.csv", explained.contributions, False),
        ("bond_correlation.csv", explained.correlation.to_frame(), True),
    ]:
        _check_table(folders[3] / filename, table, indexed=indexed)

    inputs =[root / "study.toml", root / "data/snapshot/manifest.json",
              *[root / "data/snapshot" / name for name in manifest["files"]]]
    filenames = [
        ["daily_returns.csv", "performance.csv", "capm.csv", "run.json"],
        ["episodes.csv", "episode_comparison.csv", "run.json"],
        ["calibration.json", "evaluation_returns.csv", "performance.csv",
         "stock_exposure.csv", "episodes.csv", "episode_comparison.csv", "run.json"],
        ["uncertainty.csv", "contributions.csv", "bond_correlation.csv", "run.json"],
    ]
    outputs = [folder / name for folder, names in zip(folders, filenames) for name in names]
    def hashes(paths):
        return {str(path.relative_to(root)).replace("\\", "/"): sha256(path.read_bytes()).hexdigest()
                for path in paths}

    evidence = {
        "snapshot_files": manifest["files"],
        "input_sha256": hashes(inputs), "output_sha256": hashes(outputs),
        "stock_weight": calibration["stock_weight"], "cash_weight": calibration["cash_weight"],
        "full_observations": len(combined), "evaluation_observations": len(evaluation),
        "full_seed_date": str(seed.date()),
        "phase2_episodes": len(episodes2), "phase3_episodes": len(episodes3),
    }
    return StoryData(phase1, combined, episodes2, comparison2, evaluation, performance,
                     episodes3, comparison3, result.exposure, calibration, evidence,
                     explained.uncertainty, explained.contributions, explained.correlation,
                     settings["bond_fund"])
