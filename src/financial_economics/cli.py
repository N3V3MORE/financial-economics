"""Download once, then analyze the saved snapshot without network access."""

import argparse
from pathlib import Path
import sys

from .analysis import monthly_equal_weight, performance_summary, prices_to_returns
from .config import load_study
from .data import align_sample, download_snapshot, load_snapshot
from .report import write_outputs
from .stress import compare_episodes, drawdown_episodes, rolling_correlations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("study.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Save a new source snapshot")
    download.add_argument("--data-dir", type=Path, default=Path("data/snapshot"))
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
    page = commands.add_parser("page", help="Write one HTML page from the saved phase outputs")
    page.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    page.add_argument("--output", type=Path, default=Path("outputs/page/index.html"))
    args = parser.parse_args(argv)

    try:
        study = load_study(args.config)
        if args.command == "page":
            from .page import write_page

            write_page(args.outputs_dir, args.output)
            print(f"Open: {args.output.resolve()}")
        elif args.command == "download":
            download_snapshot(study, args.data_dir)
            print(f"Saved source snapshot: {args.data_dir.resolve()}")
        else:
            prices, factors, manifest = load_snapshot(study, args.data_dir)
            returns = prices_to_returns(prices, study.start, study.end)
            returns, factors = align_sample(returns, factors, study)
            if args.output_dir.resolve().is_relative_to(args.data_dir.resolve()):
                raise ValueError("Keep generated outputs outside the source snapshot.")
            if args.command == "fund-checks":
                from .fund_checks import DEFAULT_WINDOWS, check_funds, write_fund_checks

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
                from .explain import explain_comparison
                from .explain_report import write_explain_outputs

                result = explain_comparison(
                    returns, factors["RF"], study.benchmark, args.bond_fund,
                    args.evaluation_start, study.trading_days, args.resamples,
                    args.block_length, args.seed)
                write_explain_outputs(study, result, args.bond_fund, manifest, args.output_dir)
                sharpe = result.uncertainty.iloc[0]
                print(f"Sharpe difference, mix minus {sharpe.reference}: {sharpe.estimate:+.3f} "
                      f"(95% range {sharpe.ci_low:+.3f} to {sharpe.ci_high:+.3f}).")
            elif args.command == "compare":
                from .comparison import compare_portfolios
                from .comparison_report import write_comparison_outputs

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
                from .stress_report import write_stress_outputs

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
