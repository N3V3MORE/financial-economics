"""Trial 2022-2025 SPY/cash test trained and scored on Risk Lab volatility alone.

Every date, from 2000 training history to the 2025 evaluation, uses the same Risk Lab
SPY 5-minute trade measure; no Oxford labels are spliced in and missing dates stay
missing. Settings match the earlier proxy pilot so the two can be compared.
Neural fitting requires cuda:0; EWMA/accounting are CPU-native.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .config import load_study
from .forecasting import ForecastData
from .paper_realistic import _performance, _tests, realistic_portfolios, sharpe_difference_test
from .paper_revised import _read_stage, matched_gru
from .paper_sources import load_sources
from .provenance import directory_digests, file_digest, package_code_digests, utc_now
from .proxy_study import FORECAST_SETTINGS, portfolio_settings
from .return_data import load_returns
from .volatility_proxy import load_risklab_bv


def market_calendar(observations, daily_returns):
    """Public S&P 500 sessions through 2020, then verified CRSP sessions through 2025."""
    early = observations.equity_price.dropna().loc["2000-01-01":"2020-12-31"].index
    late = daily_returns.loc["2021-01-01":FORECAST_SETTINGS["evaluation_end"]].index
    return early.append(late)


def univariate_inputs(bv, calendar, lookback):
    """The last observed sessions as history; the actual session t+2 as target."""
    if not bv.index.isin(calendar).all() or not np.isfinite(bv).all() or (bv <= 0).any():
        raise ValueError("Risk Lab variance must be positive on known market dates.")
    issues = bv.index[lookback - 1:]
    positions = calendar.get_indexer(issues)
    keep = positions + 2 < len(calendar)
    histories = np.lib.stride_tricks.sliding_window_view(bv.to_numpy(), lookback)[keep]
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
    settings = portfolio_settings(args.portfolio_config)
    inputs = identities(args)
    if args.stage == "freeze":
        if root.exists():
            raise ValueError("Freeze requires a new directory; outputs are never overwritten.")
        frozen = {"created_at_utc": utc_now(), "inputs": inputs, "code": package_code_digests(),
                  "forecast_settings": FORECAST_SETTINGS, "portfolio_settings": settings,
                  "target": "Risk Lab SPY 5-minute trade realized variance on every date; no Oxford labels",
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
            or frozen["forecast_settings"] != FORECAST_SETTINGS or frozen["portfolio_settings"] != settings):
        raise ValueError("Risk Lab study inputs, settings or calculation code changed after freezing.")
    destination = root / args.stage
    if destination.exists():
        raise ValueError("Saved Risk Lab study stages are never overwritten.")
    observations, _ = load_sources(args.data_dir)
    data = load_returns(load_study(args.portfolio_config), args.returns_dir, verified_monthly=True)
    calendar = market_calendar(observations, data.daily)
    bv, _ = load_risklab_bv(args.risklab_dir, "2000-01-01", FORECAST_SETTINGS["evaluation_end"])
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
            tables = {"forecasts": frame, "performance": performance, "tests": tests,
                      "decisions": result["decisions"], "thresholds": result["thresholds"],
                      "static_controls": result["static_controls"]}
            for cost, cost_paths in result["paths"].items():
                tables[f"daily_returns_{cost:g}bps"] = cost_paths
            metadata = {"portfolio_days": len(result["returns"]), "forecast_horizon": 2,
                        "models": settings["primary_models"],
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
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(1, f"Risk Lab study: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
