"""Download once, then analyze the saved snapshot without network access."""

import argparse
from importlib import import_module
from pathlib import Path
import sys

from .evaluation.metrics import monthly_equal_weight, performance_summary, prices_to_returns
from .settings.study import load_study
from .datasets.public import align_sample, download_snapshot, load_snapshot
from .reporting.baseline import write_outputs
from .portfolio.declines import compare_episodes, drawdown_episodes, rolling_correlations

# Staged studies and source tools parse their own options; fecon passes everything
# after the command name through to the module's main().
MODULE_COMMANDS = {
    "paper-reconstruction": ("studies.replication", "Run full paper reconstruction stages"),
    "paper-realistic": ("studies.realistic", "Run realistic trading tests on saved reconstruction forecasts"),
    "paper-revised": ("studies.historical", "Run the matched historical methodology revision stages"),
    "proxy-study": ("studies.proxy", "Run the later-year study on an accepted estimated proxy"),
    "risklab-study": ("studies.recent", "Run the later-year Risk Lab study stages"),
    "paper-figures": ("reporting.figures", "Rebuild paper figures from a saved reconstruction run"),
    "paper-macro-sources": ("datasets.sources.macro", "Save WRDS/FRED yields and EIA Brent snapshots"),
    "volatility-proxy": ("datasets.volatility", "Estimate the diagnostic Oxford-like volatility proxy"),
    "wrds-credentials": ("datasets.credentials", "Save your WRDS login with the native Windows dialog"),
}


def _predict(args, study):
    from tempfile import TemporaryDirectory

    from .models.variance import ewma_forecasts, prepare_forecasts
    from .portfolio.prediction import run_prediction
    from .settings.prediction import load_prediction
    from .reporting.prediction import load_forecast_run, write_prediction
    from .datasets.returns import load_returns

    if args.defensive_asset not in study.tickers:
        raise ValueError(f"Defensive asset {args.defensive_asset} is absent from the study's [assets]. "
                         "Choose a configured ETF or use a config and verified snapshot containing it.")
    destination = args.output_dir.resolve()
    if destination.exists() or destination.is_relative_to(args.data_dir.resolve()):
        raise ValueError("Use a new output directory outside the source snapshot.")
    if args.forecast_run and destination.is_relative_to(args.forecast_run.resolve()):
        raise ValueError("Keep prediction outputs outside the saved forecast source.")
    settings = load_prediction(args.config, study)
    data = load_returns(study, args.data_dir, verified_monthly=True)
    extra_tables, model_metadata = {}, {}

    def finish(forecasts, checkpoints=None):
        result = run_prediction(data, study, settings, forecasts, args.defensive_asset)
        dates = result.tables["daily_returns"].index
        extra_tables["evaluation_inputs"] = data.daily.loc[dates, ["SPY", args.defensive_asset]].assign(
            RF=data.factors.loc[dates, "RF"])
        write_prediction(result, study, settings, data.provenance, destination, args.config,
                         extra_tables, model_metadata, checkpoints)
        print(f"Prediction stage: {result.metadata['stage']}; defensive ETF: {args.defensive_asset}; "
              f"{result.metadata['forecast_observations']} forecasts; "
              f"{result.metadata['warmup_cash_days']} initial cash-only trading day(s).")
        print(f"Results: {destination}")

    if args.forecast_run:
        forecasts, extra_tables, model_metadata = load_forecast_run(
            args.forecast_run, study, settings, data.provenance)
        checkpoints = args.forecast_run / "checkpoints"
        finish(forecasts, checkpoints if args.defensive_asset == "TLT" and checkpoints.exists() else None)
    else:
        prepared = prepare_forecasts(data.daily.SPY, data.daily.index, settings.lookback)
        forecasts, tuning = ewma_forecasts(prepared, settings)
        extra_tables["ewma_tuning"] = tuning
        if args.ewma_only:
            finish(forecasts)
        else:
            from .models.gpu import gru_forecasts

            destination.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".prediction-models-", dir=destination.parent) as temporary:
                checkpoints = Path(temporary) / "checkpoints"
                gru, tuning, metadata = gru_forecasts(prepared, settings, data.daily.index, checkpoints)
                forecasts["gru_variance"] = gru.gru_variance
                forecasts["gru_refit_asof"] = gru.gru_refit_asof
                extra_tables["gru_tuning"] = tuning
                model_metadata["gru"] = metadata
                finish(forecasts, checkpoints)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in MODULE_COMMANDS:
        module = import_module(f".{MODULE_COMMANDS[argv[0]][0]}", __package__)
        return module.main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/study.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Save a new source snapshot")
    download.add_argument("--data-dir", type=Path, default=Path("data/snapshot"))
    download.add_argument("--source", choices=["public", "wrds"], default="public")
    download.add_argument("--verify-monthly", action="store_true", help="Require CRSP daily/monthly reconciliation")
    analyze = commands.add_parser("analyze", help="Analyze a verified snapshot offline")
    analyze.add_argument("--data-dir", type=Path, default=Path("data/snapshot"))
    analyze.add_argument("--output-dir", type=Path, default=Path("outputs/phase1"))
    stress = commands.add_parser("stress", help="Compare benchmark downturns offline")
    stress.add_argument("--data-dir", type=Path, default=Path("data/snapshot"))
    stress.add_argument("--output-dir", type=Path, default=Path("outputs/phase2"))
    stress.add_argument("--drawdown-threshold", type=float, default=0.10)
    stress.add_argument("--rolling-window", type=int, default=63)
    compare = commands.add_parser("compare", help="Compare a train-calibrated stock/cash control offline")
    compare.add_argument("--data-dir", type=Path, default=Path("data/snapshot"))
    compare.add_argument("--output-dir", type=Path, default=Path("outputs/phase3"))
    compare.add_argument("--evaluation-start", default="2018-01-01")
    explain = commands.add_parser("explain", help="Add uncertainty ranges and fund contributions offline")
    explain.add_argument("--data-dir", type=Path, default=Path("data/snapshot"))
    explain.add_argument("--output-dir", type=Path, default=Path("outputs/phase4"))
    explain.add_argument("--evaluation-start", default="2018-01-01")
    explain.add_argument("--resamples", type=int, default=10_000)
    explain.add_argument("--block-length", type=float, default=21)
    explain.add_argument("--seed", type=int, default=20260101)
    explain.add_argument("--bond-fund", default="TLT")
    checks = commands.add_parser("fund-checks", help="Check fund removals, additions, periods, and costs offline")
    checks.add_argument("--data-dir", type=Path, default=Path("data/snapshot"))
    checks.add_argument("--output-dir", type=Path, default=Path("outputs/fund-checks"))
    checks.add_argument("--additions", nargs="*", default=[])
    checks.add_argument("--windows", nargs="+")
    checks.add_argument("--costs-bps", nargs="+", type=float, default=[0, 10, 25])
    protection = commands.add_parser("protection", help="Run the frozen ETF-versus-cash study offline")
    protection.add_argument("--data-dir", type=Path, required=True)
    protection.add_argument("--output-dir", type=Path, required=True)
    predict = commands.add_parser("predict", help="Evaluate daily volatility forecasts and ETF/cash protection")
    predict.add_argument("--data-dir", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--defensive-asset", choices=["TLT", "GLD", "DBC"], default="TLT")
    prediction_source = predict.add_mutually_exclusive_group()
    prediction_source.add_argument("--ewma-only", action="store_true", help="Run the CPU-only EWMA baseline")
    prediction_source.add_argument("--forecast-run", type=Path, help="Reuse verified saved forecasts without training")
    paper = commands.add_parser("paper", help="Run a closer GRU-UNIVAR paper replication with ETF substitutes")
    paper.add_argument("--data-dir", type=Path, required=True)
    paper.add_argument("--oxford-archive", type=Path, required=True)
    paper.add_argument("--output-dir", type=Path, required=True)
    paper.add_argument("--baseline-only", action="store_true", help="Run only the CPU-native BV/EWMA baseline")
    extension = commands.add_parser("paper-extension", help="Add realistic trading assumptions to original BV forecasts")
    extension.add_argument("--data-dir", type=Path, required=True)
    extension.add_argument("--oxford-archive", type=Path, required=True)
    extension.add_argument("--output-dir", type=Path, required=True)
    extension.add_argument("--rebalance-frequency", choices=["daily", "weekly"],
                           help="Override the configured trading schedule (default: daily)")
    extension.add_argument("--cash-only", action="store_true", help="Compare SPY/cash without defensive ETFs")
    extension.add_argument("--min-allocation-change", type=float,
                           help="Minimum change from last traded stock target; 0.05 = five percentage points (SPY/cash only)")
    extension_source = extension.add_mutually_exclusive_group(required=True)
    extension_source.add_argument("--baseline-only", action="store_true")
    extension_source.add_argument("--selection-run", type=Path, help="Reuse pre-evaluation model settings; fit new delayed models")
    extension_source.add_argument("--forecast-run", type=Path, help="Reuse verified extension forecasts without training")
    page = commands.add_parser("page", help="Write one HTML page from the saved phase outputs")
    page.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    page.add_argument("--output", type=Path, default=Path("outputs/page/index.html"))
    for name, (_, help_text) in MODULE_COMMANDS.items():
        commands.add_parser(name, help=f"{help_text} (options: fecon {name} --help)", add_help=False)
    args = parser.parse_args(argv)
    if args.command in MODULE_COMMANDS:
        parser.error(f"{args.command} takes its own options after the command name, "
                     f"not the shared --config; see fecon {args.command} --help.")

    try:
        study = load_study(args.config)
        if args.command == "page":
            from .reporting.page import write_page

            write_page(args.outputs_dir, args.output)
            print(f"Open: {args.output.resolve()}")
        elif args.command == "download":
            if args.verify_monthly and args.source != "wrds":
                raise ValueError("--verify-monthly requires --source wrds for CRSP daily/monthly "
                                 "reconciliation; remove it to download public prices.")
            if args.source == "wrds":
                from .datasets.sources.crsp import download_wrds_snapshot

                download_wrds_snapshot(study, args.data_dir, verify_monthly=args.verify_monthly)
            else:
                download_snapshot(study, args.data_dir)
            print(f"Saved source snapshot: {args.data_dir.resolve()}")
        elif args.command == "paper-extension":
            from .studies.extension import run_extension_command

            run_extension_command(args, study)
        elif args.command == "paper":
            from .studies.paper import run_paper_command

            run_paper_command(args, study)
        elif args.command == "predict":
            _predict(args, study)
        elif args.command == "protection":
            from .studies.protection import run_protection
            from .settings.protection import load_protection
            from .reporting.protection import write_protection
            from .datasets.returns import load_returns

            if args.output_dir.exists() or args.output_dir.resolve().is_relative_to(args.data_dir.resolve()):
                raise ValueError("Use a new output directory outside the source snapshot.")
            settings = load_protection(args.config, study)
            data = load_returns(study, args.data_dir, verified_monthly=True)
            result = run_protection(data, study, settings)
            write_protection(result, study, settings, data.provenance, args.output_dir, args.config)
            print(f"Validated {result.metadata['monthly_observations']} months, "
                  f"{result.metadata['candidate_count']} candidates, and "
                  f"{result.metadata['portfolio_count']} fixed portfolio definitions.")
            print(f"Results: {args.output_dir.resolve()}")
        else:
            prices, factors, manifest = load_snapshot(study, args.data_dir)
            returns = prices_to_returns(prices, study.start, study.end)
            returns, factors = align_sample(returns, factors, study)
            if args.output_dir.resolve().is_relative_to(args.data_dir.resolve()):
                raise ValueError("Keep generated outputs outside the source snapshot.")
            if args.command == "fund-checks":
                from .portfolio.fund_checks import DEFAULT_WINDOWS, check_funds, write_fund_checks

                windows = tuple(args.windows) if args.windows else DEFAULT_WINDOWS
                costs = tuple(args.costs_bps)
                base_assets = [fund for fund in study.tickers if fund not in args.additions]
                result = check_funds(returns, factors["RF"], base_assets, tuple(args.additions),
                                     study.benchmark, windows, costs, study.trading_days)
                write_fund_checks(result, study, manifest, args.output_dir, windows, costs)
                print(f"Checked {len(result.portfolios)} mixes across {len(windows)} windows "
                      f"and {len(costs)} cost scenarios.")
                print(f"Results: {args.output_dir.resolve()}")
                return 0
            elif args.command == "explain":
                from .evaluation.attribution import explain_comparison
                from .reporting.attribution import write_explain_outputs

                result = explain_comparison(
                    returns, factors["RF"], study.benchmark, args.bond_fund,
                    args.evaluation_start, study.trading_days, args.resamples,
                    args.block_length, args.seed)
                write_explain_outputs(study, result, args.bond_fund, manifest, args.output_dir)
                sharpe = result.uncertainty.iloc[0]
                print(f"Sharpe difference, mix minus {sharpe.reference}: {sharpe.estimate:+.3f} "
                      f"(95% range {sharpe.ci_low:+.3f} to {sharpe.ci_high:+.3f}).")
            elif args.command == "compare":
                from .portfolio.controls import compare_portfolios
                from .reporting.comparison import write_comparison_outputs

                result = compare_portfolios(returns, factors["RF"], study.benchmark,
                                            args.evaluation_start)
                evaluation = result.evaluation_returns
                metrics = performance_summary(evaluation, factors.loc[evaluation.index, "RF"],
                                              study.trading_days)
                seed = result.calibration["evaluation_seed_date"]
                episodes = drawdown_episodes(evaluation[study.benchmark], seed, threshold=.10)
                comparisons = compare_episodes(evaluation, episodes, seed)
                write_comparison_outputs(study, result, metrics, episodes, comparisons,
                                         manifest, args.output_dir)
                print(f"Calibrated on {len(result.training_returns):,} days; "
                      f"compared three portfolios on {len(evaluation):,} later days.")
                print(f"Frozen targets: {result.calibration['stock_weight']:.2%} "
                      f"{study.benchmark}, {result.calibration['cash_weight']:.2%} cash.")
            elif args.command == "stress":
                from .reporting.stress import write_stress_outputs

                seed = prices.index[prices.index < returns.index[0]][-1]
                combined = returns.copy()
                combined["Equal weight"] = monthly_equal_weight(returns)
                episodes = drawdown_episodes(combined[study.benchmark], seed,
                                             args.drawdown_threshold)
                comparison = compare_episodes(combined[[study.benchmark, "Equal weight"]],
                                              episodes, seed)
                correlations = rolling_correlations(combined, study.benchmark, args.rolling_window)
                write_stress_outputs(
                    study, combined, episodes, comparison, correlations, manifest, args.output_dir,
                    drawdown_threshold=args.drawdown_threshold, rolling_window=args.rolling_window,
                    seed_date=seed,
                )
                print(f"Compared {len(episodes)} downturns across {len(returns):,} daily observations.")
            else:
                write_outputs(study, returns, factors, manifest, args.output_dir)
                print(f"Analyzed {len(returns):,} days and {len(returns.columns)} funds.")
            print(f"Open: {(args.output_dir / 'summary.md').resolve()}")
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
