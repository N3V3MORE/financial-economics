"""The paper's international robustness markets: does volatility timing beat a static mix?

Example: python -m financial_economics.paper_international forecasts --symbol .N225 --help
Each index uses its own Oxford-Man bipower variation and local close price, so
returns are price-only in local currency and cash earns zero (no local rates).
Classical fits run on CPU; GRU fits require CUDA:0. No safe-haven assets are tested.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .paper import prepare_paper_forecasts


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_oxford_index(archive, symbol, start, end):
    """Return dated BV and close price for one index; nonpositive BV rows are dropped, never filled."""
    with ZipFile(archive) as source:
        names = [name for name in source.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Oxford archive must contain exactly one CSV.")
        raw = pd.read_csv(source.open(names[0]))
    selected = raw.loc[raw.Symbol.eq(symbol)]
    dates = pd.DatetimeIndex(pd.to_datetime(selected.iloc[:, 0].astype(str).str[:10], format="%Y-%m-%d"), name="Date")
    frame = pd.DataFrame({"bv": selected.bv.to_numpy(dtype=float),
                          "close": selected.close_price.to_numpy(dtype=float)}, index=dates).loc[start:end]
    if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{symbol} needs sorted, unique Oxford dates.")
    if not np.isfinite(frame.close).all() or (frame.close <= 0).any():
        raise ValueError(f"{symbol} close prices must be positive and finite.")
    valid = np.isfinite(frame.bv) & (frame.bv > 0)
    return frame, int((~valid).sum())


def _forecasts(args, settings):
    frame, dropped = load_oxford_index(args.oxford_archive, args.symbol, settings["start"], settings["end"])
    calendar = frame.index
    bv = frame.bv.loc[np.isfinite(frame.bv) & (frame.bv > 0)]
    prepared = prepare_paper_forecasts(bv, calendar, settings["lookback"])
    # Each market's last pre-forecast session ends both the classical window and GRU tuning labels.
    freeze = calendar[calendar < settings["forecast_start"]][-1]
    eligible = prepared.rows.loc[prepared.rows.index >= freeze]
    window_end = min(pd.Timestamp(settings["window_end"]), eligible.index[0])
    training_end = min(pd.Timestamp(settings["training_end"]), freeze)
    price_returns = np.log(frame.close).diff().dropna()
    from .paper_models import classical_forecasts

    classical, fits, classical_model = classical_forecasts(
        eligible, bv, price_returns, window="rolling",
        window_start=settings["window_start"], window_end=str(window_end.date()))
    from .paper_networks import reconstruct_gru

    with args.reconstruction_config.open("rb") as source:
        gru_settings = tomllib.load(source)["reconstruction"]
    destination = args.output_dir / args.symbol.lstrip(".")
    if destination.exists():
        raise ValueError(f"Existing results are never overwritten: {destination}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".international-", dir=args.output_dir) as temporary:
        output = Path(temporary) / "run"
        output.mkdir()
        gru, tuning, gru_model = reconstruct_gru(
            prepared, calendar, output / "checkpoints", model_name="gru_univar",
            training_end=str(training_end.date()), evaluation_start=settings["forecast_start"],
            **{name: gru_settings[name] for name in ("profiles", "seeds", "max_epochs",
                                                    "patience", "learning_rate", "folds")})
        forecasts = classical.join(gru.loc[eligible.index, ["gru_univar_bv", "gru_univar_refit_asof"]])
        forecasts.to_csv(output / "forecasts.csv", index_label="forecast_date")
        fits.to_csv(output / "daily_fits.csv", index=False)
        tuning.to_csv(output / "tuning.csv", index=False)
        frame.to_csv(output / "index_inputs.csv")
        metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "symbol": args.symbol,
                    "inputs": {"oxford_archive": _digest(args.oxford_archive), "config": _digest(args.config),
                               "reconstruction_config": _digest(args.reconstruction_config)},
                    "settings": settings, "dropped_nonpositive_bv_rows": dropped,
                    "market_dates": {"classical_window_end": str(window_end.date()),
                                     "gru_training_end": str(training_end.date())},
                    "calendar": "dates with an Oxford close price for this index; BV targets never filled",
                    "classical": classical_model, "gru": gru_model,
                    "code": {name: _digest(Path(__file__).with_name(name))
                             for name in ("paper_international.py", "paper.py", "paper_models.py",
                                          "paper_networks.py", "paper_gru.py", "gru_forecasting.py")}}
        metadata["files"] = {str(path.relative_to(output)): _digest(path)
                             for path in sorted(output.rglob("*")) if path.is_file()}
        (output / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        output.rename(destination)
    print(f"Saved {args.symbol} forecasts: {destination}", flush=True)


def _evaluate(args, settings):
    from .paper_realistic import (_intervals, _performance, _tests, load_realistic_settings,
                                  realistic_portfolios)

    realistic = load_realistic_settings(args.config)
    destination = args.output_dir / "evaluation"
    if destination.exists():
        raise ValueError(f"Existing results are never overwritten: {destination}")
    tables, provenance = {"performance": [], "tests": [], "bootstrap_intervals": [], "static_controls": []}, {}
    for symbol, name in zip(settings["symbols"], settings["names"]):
        directory = args.output_dir / symbol.lstrip(".")
        metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        for file in ("forecasts.csv", "index_inputs.csv"):
            if _digest(directory / file) != metadata["files"][file]:
                raise ValueError(f"Saved {symbol} {file} does not match its provenance hash.")
        frame = pd.read_csv(directory / "index_inputs.csv", index_col=0, parse_dates=True, float_precision="round_trip")
        forecasts = pd.read_csv(directory / "forecasts.csv", index_col=0, parse_dates=True, float_precision="round_trip")
        for model in (*realistic["primary_models"], *realistic["secondary_models"]):
            forecasts[f"{model}_refit_asof"] = pd.to_datetime(forecasts[f"{model}_refit_asof"])
        calendar = frame.index
        daily = pd.DataFrame({realistic["benchmark"]: frame.close.pct_change(fill_method=None), "RF": 0.},
                             index=calendar).iloc[1:]
        result = realistic_portfolios(daily, calendar[1:], forecasts, realistic)
        for key, table in (("performance", _performance(result, realistic)),
                           ("tests", _tests(result, calendar[1:], realistic)),
                           ("bootstrap_intervals", _intervals(result, realistic)),
                           ("static_controls", result["static_controls"])):
            tables[key].append(table.assign(symbol=symbol, market=name))
        provenance[symbol] = {"run_sha256": _digest(directory / "run.json"),
                              "missing_signal_days": result["missing_signal_days"],
                              "dropped_issue_dates": result["dropped_issue_dates"]}
    tests = pd.concat(tables["tests"], ignore_index=True).drop(columns=["holm_p_value", "holm_reject"])
    from statsmodels.stats.multitest import multipletests

    primary = tests.family.eq("primary") & tests.p_value.notna()
    tests["holm_p_value"] = np.nan
    tests.loc[primary, "holm_p_value"] = multipletests(tests.loc[primary, "p_value"], method="holm")[1]
    tests["holm_reject"] = tests.holm_p_value < realistic["alpha"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".international-evaluation-", dir=args.output_dir) as temporary:
        output = Path(temporary) / "run"
        output.mkdir()
        tests.to_csv(output / "tests.csv", index=False)
        for key in ("performance", "bootstrap_intervals", "static_controls"):
            pd.concat(tables[key], ignore_index=True).to_csv(output / f"{key}.csv", index=False)
        metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "study": "International timing-versus-static robustness check",
                    "settings": settings, "realistic": realistic, "config_sha256": _digest(args.config),
                    "forecast_runs": provenance,
                    "assumptions": {"returns": "Oxford-Man local close-price returns; no dividends; local currency",
                                    "cash": "zero cash income; no local risk-free series supplied",
                                    "execution": "issue at close t, trade at close t+1, earn t+2 on each index calendar",
                                    "primary_family": "timing minus static Sharpe for ARFIMA and one-input GRU in three markets; one Holm adjustment",
                                    "scope": "no safe-haven assets; descriptive comparison with the U.S. result"},
                    "code": {name: _digest(Path(__file__).with_name(name))
                             for name in ("paper_international.py", "paper_realistic.py", "costs.py", "analysis.py")}}
        metadata["files"] = {path.name: _digest(path) for path in sorted(output.iterdir())}
        (output / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        output.rename(destination)
    print(f"Saved international evaluation: {destination}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("forecasts", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path("paper-international.toml"))
    parser.add_argument("--reconstruction-config", type=Path, default=Path("paper-reconstruction.toml"))
    parser.add_argument("--oxford-archive", type=Path, required=True)
    parser.add_argument("--symbol", help="Oxford symbol for the forecasts stage, e.g. .N225")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.config.open("rb") as source:
        raw = tomllib.load(source)
    settings = raw["international"]
    if args.stage == "forecasts":
        if args.symbol not in settings["symbols"]:
            raise ValueError("Choose a symbol listed in the international config.")
        _forecasts(args, settings)
    else:
        _evaluate(args, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
