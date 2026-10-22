"""Exploratory 2022-2025 SPY/cash test using approved Oxford-like estimates.

The full historical target series is estimated consistently, without splicing
Oxford labels into later proxy labels. Actual verified CRSP returns score the
portfolios. Neural fitting requires cuda:0; EWMA/accounting are CPU-native.
"""

import argparse
from financial_economics._layout import code_matches
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from financial_economics.settings.study import load_study
from financial_economics.models.variance import ForecastData
from financial_economics.studies.realistic import _performance, _tests, load_realistic_settings, realistic_portfolios, sharpe_difference_test
from financial_economics.studies.historical import COSTS_BPS, _read_stage, matched_gru
from financial_economics.provenance import directory_digests, file_digest, package_code_digests, utc_now
from financial_economics.datasets.returns import load_returns
from financial_economics.datasets.volatility import load_accepted_proxy, load_daily_measures


FORECAST_SETTINGS = {"training_end": "2020-12-31", "evaluation_start": "2021-01-01",
                     "evaluation_end": "2025-12-31", "lookback": 22,
                     "folds": 5, "profiles": [[32, 32, 32, 0., .0001]],
                     "seeds": [11, 22, 33], "max_epochs": 12, "patience": 8,
                     "learning_rate": .001}


def trading_histories(bv, calendar, lookback=22):
    """Use exactly the last 22 observed trading sessions, with actual t+2 targets."""
    if (type(lookback) is not int or lookback < 2 or len(calendar) < lookback + 2
            or not calendar.is_unique or not calendar.is_monotonic_increasing
            or not bv.index.equals(calendar) or not np.isfinite(bv).all() or (bv <= 0).any()):
        raise ValueError("Trading histories require positive estimates for every ordered market date.")
    histories = np.lib.stride_tricks.sliding_window_view(bv.to_numpy(), lookback)[:-2]
    rows = pd.DataFrame({"trade_date": calendar[lookback:-1], "return_date": calendar[lookback + 1:],
                         "realized_bv": bv.iloc[lookback + 1:].to_numpy(),
                         "history_first": calendar[:len(histories)],
                         "history_last": calendar[lookback - 1:-2], "input_observations": lookback},
                        index=calendar[lookback - 1:-2].rename("forecast_date"))
    return ForecastData(rows, histories[:, :, None].copy())


def portfolio_settings(config):
    original = load_realistic_settings(config)
    return {**original, "calibration_start": "2021-01-01", "evaluation_start": "2022-01-01",
            "evaluation_end": "2025-12-31", "primary_models": ["ewma", "gru_univar_log"],
            "secondary_models": [], "primary_assets": [], "secondary_assets": [],
            "risk_mode": "annual", "annual_target": .10, "initial_cash": True,
            "cost_benchmark": True, "liquidate_end": True, "require_aligned_horizon": True,
            "costs_bps": list(COSTS_BPS), "main_cost_bps": 10,
            "windows": ["2022:2025", "2023:2025", "2022:2022"]}


def identities(args):
    return {"proxy_run": file_digest(args.proxy_run / "run.json"),
            "proxy_protocol": file_digest(args.proxy_run / "protocol.json"),
            "prices_manifest": file_digest(args.prices_dir / "manifest.json"),
            "returns_manifest": file_digest(args.returns_dir / "manifest.json"),
            "portfolio_config": file_digest(args.portfolio_config)}


def run(args):
    root = args.output_dir.resolve()
    protected = (args.proxy_run.resolve(), args.prices_dir.resolve(), args.returns_dir.resolve())
    if any(root.is_relative_to(path) or path.is_relative_to(root) for path in protected):
        raise ValueError("Proxy study outputs must be separate from all source directories.")
    bv, proxy_metadata = load_accepted_proxy(args.proxy_run)
    settings = portfolio_settings(args.portfolio_config)
    inputs = identities(args)
    if args.stage == "freeze":
        if root.exists():
            raise ValueError("Freeze requires a new directory; outputs are never overwritten.")
        frozen = {"created_at_utc": utc_now(), "inputs": inputs, "code": package_code_digests(),
                  "forecast_settings": FORECAST_SETTINGS, "portfolio_settings": settings,
                  "model_choice": "single existing 32/32 architecture; no new width search; epochs selected on pre-2021 stopping blocks",
                  "target": "all dates are estimated BV, never presented as newly observed Oxford data",
                  "proxy_selected_through": "2016-12-31",
                  "proxy_coefficient_labels_end": proxy_metadata["calibration_last_label"],
                  "status": "exploratory; source accuracy checked through February 2022; all study history already inspected",
                  "limitations": ["90% average / 85% daily gate concerns historical same-day measurements, not future forecasts or profitable trades",
                                  "Daily price-range fallbacks and ETF/index differences remain measurement limitations",
                                  "Raw variance percentage errors exceed volatility percentage errors",
                                  "Overnight risk and exact release-time availability remain unverified",
                                  "Taxes excluded; execution fees are scenarios"]}
        root.mkdir(parents=True)
        (root / "protocol.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        print(f"Frozen proxy study: {root}", flush=True)
        return
    protocol_path = root / "protocol.json"
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (frozen["inputs"] != inputs or not code_matches(frozen["code"], package_code_digests())
            or frozen["forecast_settings"] != FORECAST_SETTINGS or frozen["portfolio_settings"] != settings):
        raise ValueError("Proxy inputs, settings or calculation code changed after freezing.")
    destination = root / args.stage
    if destination.exists():
        raise ValueError("Saved proxy study stages are never overwritten.")
    measures, _ = load_daily_measures(args.prices_dir)
    calendar = measures.loc["2000-01-01":"2025-12-31"].index
    bv = bv.reindex(calendar)
    prepared = trading_histories(bv, calendar, FORECAST_SETTINGS["lookback"])
    protocol_digest = file_digest(protocol_path)
    with TemporaryDirectory(prefix=".proxy-study-", dir=root) as temporary:
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
            frame["target_is_estimated"] = True
            tables = {"forecasts": frame, "tuning": tuning}
            metadata["target_is_estimated"] = True
        else:
            frame, _ = _read_stage(root, "forecast", protocol_digest)
            data = load_returns(load_study(args.portfolio_config), args.returns_dir, verified_monthly=True)
            result = realistic_portfolios(data.daily.assign(RF=data.factors.RF), calendar, frame, settings)
            tests = _tests(result, calendar, settings)
            paths = result["paths"][settings["main_cost_bps"]]
            difference = {"model": "gru_univar_log", "comparison": "gru_minus_ewma",
                          "measure": "annualized Sharpe difference (Ledoit-Wolf HAC)",
                          **sharpe_difference_test(paths.gru_univar_log_cash, paths.ewma_cash,
                                                  result["returns"].RF.to_numpy(), settings["sharpe_hac_lags"])}
            tests = pd.concat([tests, pd.DataFrame([difference])], ignore_index=True)
            tests["family"] = "proxy_exploratory"
            tests["holm_p_value"] = np.where(tests.p_value.notna(),
                multipletests(tests.p_value.fillna(1), method="holm")[1], np.nan)
            tests["holm_reject"] = tests.holm_p_value < settings["alpha"]
            performance = _performance(result, settings)
            performance["annual_equity_budget"] = .10
            performance["realized_vol_to_equity_budget"] = performance.annualized_volatility / .10
            tables = {"forecasts": frame, "performance": performance, "tests": tests,
                      "decisions": result["decisions"], "thresholds": result["thresholds"],
                      "static_controls": result["static_controls"]}
            for cost, paths in result["paths"].items():
                tables[f"daily_returns_{cost:g}bps"] = paths
            metadata = {"portfolio_days": len(result["returns"]), "forecast_horizon": 2,
                        "history_trading_days": FORECAST_SETTINGS["lookback"], "models": settings["primary_models"],
                        "targets_are_estimates": True, "actual_return_source": "verified WRDS/CRSP ETF total returns"}
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=not isinstance(table.index, pd.RangeIndex))
        record = {"created_at_utc": utc_now(), "stage": args.stage, "protocol_sha256": protocol_digest,
                  "model": metadata, "proxy_missing_dates_in_validation": proxy_metadata["final_test_missing_dates"],
                  "files": directory_digests(output)}
        (output / "run.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        output.rename(destination)
    print(f"Saved proxy study stage: {destination}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "forecast", "evaluate"))
    parser.add_argument("--proxy-run", type=Path, required=True)
    parser.add_argument("--prices-dir", type=Path, required=True)
    parser.add_argument("--returns-dir", type=Path, required=True)
    parser.add_argument("--portfolio-config", type=Path, default=Path("configs/paper-realistic.toml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(1, f"Proxy study: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
