"""Trial 2022-2025 SPY/cash test trained and scored on Risk Lab volatility alone.

Every date, from 2000 training history to the 2025 evaluation, uses the same Risk Lab
SPY 5-minute trade measure; missing dates stay missing, and a forecast is issued only
when its whole history is consecutive market sessions. Settings match the earlier
proxy pilot so the two can be compared.
Optional: add the configured defensive ETFs in stress periods, and start the
training history before 2000 on French's daily market calendar.
Neural fitting requires cuda:0; EWMA/accounting are CPU-native.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from financial_economics.settings.study import load_study
from financial_economics.datasets.public import parse_french_factors
from financial_economics.models.variance import ForecastData
from financial_economics.studies.realistic import (_performance, _tests, load_realistic_settings, realistic_portfolios,
                              sharpe_difference_test)
from financial_economics.studies.historical import (EWMA_DEFINITION, _read_stage, hansen_lunde_factor, matched_gru,
                            robust_forecast_scores)
from financial_economics.datasets.sources.paper import load_sources
from financial_economics.provenance import directory_digests, file_digest, package_code_digests, utc_now
from financial_economics.studies.proxy import FORECAST_SETTINGS, portfolio_settings
from financial_economics.datasets.returns import load_returns
from financial_economics.datasets.volatility import load_risklab_bv


DEFAULT_HISTORY_START = "2000-01-01"
FORECAST_HAC_LAGS = 20


def market_calendar(observations, daily_returns, french_dates=None, history_start=DEFAULT_HISTORY_START):
    """French sessions before 2000, public S&P 500 sessions through 2020, then CRSP through 2025."""
    early = observations.equity_price.dropna().loc["2000-01-01":"2020-12-31"].index
    late = daily_returns.loc["2021-01-01":FORECAST_SETTINGS["evaluation_end"]].index
    start = pd.Timestamp(history_start)
    if start >= pd.Timestamp(DEFAULT_HISTORY_START):
        return early[early >= start].append(late)
    if french_dates is None:
        raise ValueError("History before 2000 needs French's daily market calendar.")
    return french_dates[(french_dates >= start) & (french_dates < early[0])].append(early).append(late)


def french_calendar(returns_dir):
    """Daily market dates from the verified French archive in the returns snapshot."""
    manifest = json.loads((returns_dir / "manifest.json").read_text(encoding="utf-8"))
    archive = returns_dir / "french_factors_daily.zip"
    if manifest["files"][archive.name]["sha256"] != file_digest(archive):
        raise ValueError("French archive does not match the returns snapshot manifest.")
    return parse_french_factors(archive.read_bytes()).index


def trial_portfolio_settings(config, defensive):
    """The pilot's stock/cash rules, optionally with the configured stress-period defensive ETFs."""
    settings = portfolio_settings(config)
    if defensive:
        original = load_realistic_settings(config)
        settings = {**settings, "primary_assets": original["primary_assets"],
                    "secondary_assets": original["secondary_assets"]}
    return settings


def univariate_inputs(bv, calendar, lookback):
    """The last consecutive market sessions as history; the actual session t+2 as target.

    Issue dates whose window contains a missing variance are skipped, never bridged.
    """
    if not bv.index.isin(calendar).all() or not np.isfinite(bv).all() or (bv <= 0).any():
        raise ValueError("Risk Lab variance must be positive on known market dates.")
    values = bv.reindex(calendar).to_numpy()
    complete = np.lib.stride_tricks.sliding_window_view(np.isfinite(values), lookback).all(axis=1)
    issues = calendar[lookback - 1:][complete]
    positions = calendar.get_indexer(issues)
    keep = positions + 2 < len(calendar)
    histories = np.lib.stride_tricks.sliding_window_view(values, lookback)[complete][keep]
    positions = positions[keep]
    rows = pd.DataFrame({"trade_date": calendar[positions + 1], "return_date": calendar[positions + 2],
                         "realized_bv": bv.reindex(calendar[positions + 2]).to_numpy()},
                        index=issues[keep].rename("forecast_date"))
    return ForecastData(rows, histories[:, :, None].copy())


def identities(args):
    return {"risklab_manifest": file_digest(args.risklab_dir / "manifest.json"),
            "source_manifest": file_digest(args.data_dir / "manifest.json"),
            "returns_manifest": file_digest(args.returns_dir / "manifest.json"),
            "portfolio_config": file_digest(args.portfolio_config)}


def run(args):
    root = args.output_dir.resolve()
    protected = (args.risklab_dir.resolve(), args.data_dir.resolve(), args.returns_dir.resolve())
    if any(root.is_relative_to(path) or path.is_relative_to(root) for path in protected):
        raise ValueError("Risk Lab study outputs must be separate from all source directories.")
    settings = trial_portfolio_settings(args.portfolio_config, args.defensive_assets)
    inputs = identities(args)
    if args.stage == "freeze":
        if root.exists():
            raise ValueError("Freeze requires a new directory; outputs are never overwritten.")
        frozen = {"created_at_utc": utc_now(), "inputs": inputs, "code": package_code_digests(),
                  "forecast_settings": FORECAST_SETTINGS, "portfolio_settings": settings,
                  "history_start": args.history_start,
                  "target": "Risk Lab SPY 5-minute trade realized variance on every date; no Oxford labels",
                  "missing_dates_rule": "issue only after the last lookback sessions are all observed; "
                                        "portfolios hold their last target",
                  "ewma_definition": EWMA_DEFINITION,
                  "status": "trial; exploratory; all study history already inspected",
                  "limitations": ["SPY ETF trades, not the S&P 500 index; intraday only, no overnight term",
                                  "Risk Lab dates absent from the market calendar stay missing, never filled",
                                  "Taxes excluded; execution fees are scenarios"]}
        root.mkdir(parents=True)
        (root / "protocol.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        print(f"Frozen Risk Lab study: {root}", flush=True)
        return
    protocol_path = root / "protocol.json"
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (frozen["inputs"] != inputs or frozen["code"] != package_code_digests()
            or frozen["forecast_settings"] != FORECAST_SETTINGS or frozen["portfolio_settings"] != settings
            or frozen.get("history_start", DEFAULT_HISTORY_START) != args.history_start):
        raise ValueError("Risk Lab study inputs, settings or calculation code changed after freezing.")
    destination = root / args.stage
    if destination.exists():
        raise ValueError("Saved Risk Lab study stages are never overwritten.")
    observations, _ = load_sources(args.data_dir)
    data = load_returns(load_study(args.portfolio_config), args.returns_dir, verified_monthly=True)
    french = (french_calendar(args.returns_dir)
              if pd.Timestamp(args.history_start) < pd.Timestamp(DEFAULT_HISTORY_START) else None)
    calendar = market_calendar(observations, data.daily, french, args.history_start)
    bv, _ = load_risklab_bv(args.risklab_dir, args.history_start, FORECAST_SETTINGS["evaluation_end"])
    bv = bv.loc[bv.index.isin(calendar)]
    prepared = univariate_inputs(bv, calendar, FORECAST_SETTINGS["lookback"])
    protocol_digest = file_digest(protocol_path)
    with TemporaryDirectory(prefix=".risklab-study-", dir=root) as temporary:
        output = Path(temporary) / "stage"
        output.mkdir()
        if args.stage == "forecast":
            frame, tuning, metadata = matched_gru(
                prepared, calendar, output / "checkpoints", FORECAST_SETTINGS, "gru_univar_log")
            first = calendar[calendar < FORECAST_SETTINGS["evaluation_start"]][-1]
            frame = frame.loc[frame.index >= first].copy()
            for issue in frame.index:
                history = bv.loc[:issue].iloc[-FORECAST_SETTINGS["lookback"]:]
                if calendar.get_loc(history.index[-1]) - calendar.get_loc(history.index[0]) != len(history) - 1:
                    raise ValueError("EWMA histories must be consecutive market sessions.")
                weights = .94 ** np.arange(len(history) - 1, -1, -1)
                frame.loc[issue, "ewma_bv"] = float(history @ (weights / weights.sum()))
            frame["ewma_refit_asof"] = frame.index
            tables = {"forecasts": frame, "tuning": tuning}
            metadata["missing_risklab_market_dates"] = int((~calendar.isin(bv.index)).sum())
        else:
            frame, _ = _read_stage(root, "forecast", protocol_digest)
            result = realistic_portfolios(data.daily.assign(RF=data.factors.RF), calendar, frame, settings)
            tests = _tests(result, calendar, settings)
            paths = result["paths"][settings["main_cost_bps"]]
            difference = {"model": "gru_univar_log", "comparison": "gru_minus_ewma",
                          "measure": "annualized Sharpe difference (Ledoit-Wolf HAC)",
                          **sharpe_difference_test(paths.gru_univar_log_cash, paths.ewma_cash,
                                                  result["returns"].RF.to_numpy(), settings["sharpe_hac_lags"])}
            tests = pd.concat([tests, pd.DataFrame([difference])], ignore_index=True)
            tests["family"] = "risklab_trial"
            tests["holm_p_value"] = np.where(tests.p_value.notna(),
                multipletests(tests.p_value.fillna(1), method="holm")[1], np.nan)
            tests["holm_reject"] = tests.holm_p_value < settings["alpha"]
            performance = _performance(result, settings)
            performance["annual_equity_budget"] = .10
            spy = np.log1p(data.daily.SPY.dropna())
            factor = hansen_lunde_factor(bv, spy, FORECAST_SETTINGS["training_end"])
            scored = frame.loc[frame.return_date.between(settings["evaluation_start"], settings["evaluation_end"])]
            robust_scores, robust_tests = robust_forecast_scores(
                scored, spy, factor, FORECAST_HAC_LAGS, models=settings["primary_models"])
            tables = {"forecasts": frame, "performance": performance, "tests": tests,
                      "forecast_robust_scores": robust_scores, "forecast_robust_tests": robust_tests,
                      "decisions": result["decisions"], "thresholds": result["thresholds"],
                      "static_controls": result["static_controls"]}
            for cost, cost_paths in result["paths"].items():
                tables[f"daily_returns_{cost:g}bps"] = cost_paths
            metadata = {"portfolio_days": len(result["returns"]), "forecast_horizon": 2,
                        "models": settings["primary_models"], "history_start": args.history_start,
                        "defensive_assets": [*settings["primary_assets"], *settings["secondary_assets"]],
                        "hansen_lunde_factor": factor, "close_to_close_returns": "CRSP SPY log total returns",
                        "actual_return_source": "verified WRDS/CRSP ETF total returns"}
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=not isinstance(table.index, pd.RangeIndex))
        record = {"created_at_utc": utc_now(), "stage": args.stage, "protocol_sha256": protocol_digest,
                  "model": metadata, "files": directory_digests(output)}
        (output / "run.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        output.rename(destination)
    print(f"Saved Risk Lab study stage: {destination}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "forecast", "evaluate"))
    parser.add_argument("--risklab-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Public sources for the 2000-2020 calendar")
    parser.add_argument("--returns-dir", type=Path, required=True)
    parser.add_argument("--portfolio-config", type=Path, default=Path("configs/paper-realistic.toml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--defensive-assets", action="store_true",
                        help="Add the portfolio config's defensive ETFs during stress periods")
    parser.add_argument("--history-start", default=DEFAULT_HISTORY_START,
                        help="First training-history date (YYYY-MM-DD); before 2000 uses French's calendar")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(1, f"Risk Lab study: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
