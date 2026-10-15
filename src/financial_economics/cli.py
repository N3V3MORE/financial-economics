"""Download once, then analyze the saved snapshot without network access."""

import argparse
from pathlib import Path
import sys

from .analysis import prices_to_returns
from .config import load_study
from .data import align_sample, download_snapshot, load_snapshot
from .report import write_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("study.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Save a new source snapshot")
    download.add_argument("--data-dir", type=Path, default=Path("data/snapshot"))
    analyze = commands.add_parser("analyze", help="Analyze a verified snapshot offline")
    analyze.add_argument("--data-dir", type=Path, default=Path("data/snapshot"))
    analyze.add_argument("--output-dir", type=Path, default=Path("outputs/phase1"))
    args = parser.parse_args(argv)

    try:
        study = load_study(args.config)
        if args.command == "download":
            download_snapshot(study, args.data_dir)
            print(f"Saved source snapshot: {args.data_dir.resolve()}")
        else:
            prices, factors, manifest = load_snapshot(study, args.data_dir)
            returns = prices_to_returns(prices, study.start, study.end)
            returns, factors = align_sample(returns, factors, study)
            if args.output_dir.resolve().is_relative_to(args.data_dir.resolve()):
                raise ValueError("Keep generated outputs outside the source snapshot.")
            write_outputs(study, returns, factors, manifest, args.output_dir)
            print(f"Analyzed {len(returns):,} days and {len(returns.columns)} funds.")
            print(f"Open: {(args.output_dir / 'summary.md').resolve()}")
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
