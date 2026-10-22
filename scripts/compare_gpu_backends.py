"""Compare GRU variance forecasts across GPU RNN backends (cuDNN, MIOpen, native) on fixed synthetic data.

Record a reference on one machine, then compare another machine's backend against it:

    python scripts/compare_gpu_backends.py record scripts/gpu-backend-reference/miopen
    python scripts/compare_gpu_backends.py compare scripts/gpu-backend-reference/miopen --output outputs/backend-cudnn

The returns are a seeded GARCH(1,1)-like series, so every machine with the locked NumPy
builds the same inputs; their digest is checked before any forecasts are compared.
Set FINANCIAL_ECONOMICS_NATIVE_RNN=1 to record or compare PyTorch's native GPU RNN instead.
"""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd

from financial_economics.forecasting import prepare_forecasts, qlike
from financial_economics.gru_forecasting import gru_forecasts
from financial_economics.prediction_config import PredictionSettings


def synthetic_returns():
    """Daily returns with clustered volatility, like equity index data."""
    dates = pd.bdate_range("2008-01-01", "2020-12-31")
    rng = np.random.default_rng(20261022)
    variance, returns = 1e-4, []
    for _ in dates:
        value = np.sqrt(variance) * rng.standard_t(6) / np.sqrt(1.5)
        returns.append(value)
        variance = 2e-6 + .08 * value ** 2 + .9 * variance
    return pd.Series(returns, index=dates, name="return")


def fit():
    returns = synthetic_returns()
    settings = replace(PredictionSettings(), seeds=(11, 22))
    prepared = prepare_forecasts(returns, returns.index, lookback=settings.lookback)
    start = time.perf_counter()
    with TemporaryDirectory() as checkpoints:
        forecasts, tuning, metadata = gru_forecasts(prepared, settings, returns.index, Path(checkpoints) / "gru",
                                                    progress=lambda message: None)
    seconds = time.perf_counter() - start
    issued = forecasts.gru_variance.notna()
    realized = prepared.rows.squared_return.reindex(forecasts.index)
    summary = {"input_sha256": sha256(returns.to_numpy().tobytes()).hexdigest(),
               "seconds": round(seconds, 1), "forecasts": int(issued.sum()),
               "qlike": float(qlike(realized[issued], forecasts.gru_variance[issued]).mean()),
               **{key: metadata[key] for key in ("gpu", "torch_version", "cuda_version", "hip_version",
                                                 "cudnn_version", "miopen_version", "rnn_backend",
                                                 "selected_hidden_size", "selected_learning_rate",
                                                 "refit_epochs", "selected_best_epochs")}}
    return forecasts.loc[issued, ["gru_variance"]], tuning, summary


def save(directory, forecasts, tuning, summary):
    directory.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(directory / "forecasts.csv", float_format="%.17g")
    tuning.to_csv(directory / "tuning.csv", index=False, float_format="%.17g")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")


def compare(reference_dir, forecasts, tuning, summary):
    reference = json.loads((reference_dir / "summary.json").read_text(encoding="utf-8"))
    if reference["input_sha256"] != summary["input_sha256"]:
        raise SystemExit("Synthetic inputs differ from the reference; check that the locked NumPy is installed.")
    expected = pd.read_csv(reference_dir / "forecasts.csv", index_col=0, parse_dates=True,
                           float_precision="round_trip").gru_variance
    actual = forecasts.gru_variance
    if not actual.index.equals(expected.index):
        raise SystemExit("Forecast dates differ from the reference.")
    gap = (actual - expected).abs() / expected
    expected_tuning = pd.read_csv(reference_dir / "tuning.csv", float_precision="round_trip")
    print(f"Reference: {reference['rnn_backend']} on {reference['gpu']} ({reference['seconds']} s)")
    print(f"This run:  {summary['rnn_backend']} on {summary['gpu']} ({summary['seconds']} s)")
    print(f"Identical inputs: yes ({summary['forecasts']} forecasts)")
    for key in ("selected_hidden_size", "selected_learning_rate", "refit_epochs", "selected_best_epochs"):
        same = "same" if reference[key] == summary[key] else "DIFFERENT"
        print(f"{key}: {reference[key]} vs {summary[key]} ({same})")
    print(f"Validation QLIKE max difference across the grid: "
          f"{float((tuning.qlike - expected_tuning.qlike).abs().max()):.3g}")
    print(f"Forecast relative gap: max {gap.max():.3g}, median {gap.median():.3g}, "
          f"99th percentile {gap.quantile(.99):.3g}")
    print(f"Forecast correlation: {np.corrcoef(actual, expected)[0, 1]:.10f}")
    print(f"Evaluation QLIKE: {reference['qlike']:.10f} vs {summary['qlike']:.10f}")
    print(f"Identical forecasts: {'yes' if actual.equals(expected) else 'no'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="fit on this GPU and save a reference")
    record.add_argument("output", type=Path)
    check = commands.add_parser("compare", help="fit on this GPU and compare with a saved reference")
    check.add_argument("reference", type=Path)
    check.add_argument("--output", type=Path, help="also save this run's forecasts here")
    args = parser.parse_args(argv)
    forecasts, tuning, summary = fit()
    if args.command == "record":
        if args.output.exists():
            raise SystemExit(f"{args.output} already exists; references are never overwritten.")
        save(args.output, forecasts, tuning, summary)
        print(f"Saved {summary['rnn_backend']} reference: {args.output}")
    else:
        if args.output:
            save(args.output, forecasts, tuning, summary)
        compare(args.reference, forecasts, tuning, summary)


if __name__ == "__main__":
    main()
