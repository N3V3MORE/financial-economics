"""Matched, positive, two-session forecasts and annual-risk portfolio diagnostics.

Run ``python -m financial_economics.paper_revised --help``. Freeze a new run,
then run classical, gru_univar_log, gru_multivar_log and evaluate sequentially.
Historical results are exploratory. Original reconstruction caches stay frozen.
Classical fitting/accounting are CPU-native; neural fitting requires cuda:0.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .config import load_study
from .forecasting import ForecastData
from .gru_forecasting import _annual_splits, _gpu_inputs, _gpu_metadata, _require_cuda, _uses_native_rnn
from .paper import load_bipower
from .paper_evaluation import _forecast_scores
from .paper_gru import _cv_folds
from .paper_models import _arfima_forecast, _fractional_prediction, _garch_forecast
from .paper_networks import _profiles
from .paper_networks_log import _fit, _log_inputs, _log_predictions, _log_scale
from .paper_realistic import (_declines, _performance, _tests, load_realistic_settings,
                              realistic_portfolios, sharpe_difference_test)
from .paper_sources import _past_values, load_sources
from .provenance import directory_digests, file_digest, git_dirty, git_head, package_code_digests, utc_now, verify_files
from .return_data import load_returns


MODELS = ("ewma", "arfima", "garch", "gjr", "gru_univar_log", "gru_multivar_log")
STAGES = ("classical", "gru_univar_log", "gru_multivar_log", "evaluate")
COSTS_BPS = (0, 1, 2, 5, 10, 25)


def matched_inputs(observations, bv, calendar, lookback=22):
    """Both GRUs receive identical dated histories; extra inputs lag one session.

    There is no artificial 2002 cutoff: start when all substitute features exist.
    Same-date BV availability and historical macro vintages remain assumptions.
    Targets are the actual market session t+2; missing labels remain missing.
    """
    if type(lookback) is not int or lookback < 2:
        raise ValueError("Matched histories require a lookback of at least two.")
    if (not isinstance(calendar, pd.DatetimeIndex) or calendar.hasnans
            or not calendar.is_unique or not calendar.is_monotonic_increasing
            or calendar.tz is not None or not calendar.equals(calendar.normalize())):
        raise ValueError("Matched forecasts require a unique ordered daily calendar.")
    if (not bv.index.is_unique or not bv.index.is_monotonic_increasing
            or not bv.index.isin(calendar).all() or not np.isfinite(bv).all() or (bv <= 0).any()):
        raise ValueError("BV must contain positive observations on known market dates.")
    names = ("GOLD", "OIL", "gov_yield", "corp_yield")
    first = max(observations[name].first_valid_index() for name in names)
    dates = calendar[calendar >= first]
    levels, carries = {}, {}
    for name in names:
        levels[name], carries[name] = _past_values(observations[name], dates, name)
    extra = pd.DataFrame({"gold_log": np.log(levels["GOLD"]).diff(),
                          "oil_log": np.log(levels["OIL"]).diff(),
                          "gov_change": levels["gov_yield"].diff(),
                          "corp_change": levels["corp_yield"].diff()}, index=dates).shift(1)
    features = pd.concat([bv.reindex(dates).rename("bv"), extra], axis=1).dropna()
    if len(features) < lookback + 2 or not np.isfinite(features).all().all():
        raise ValueError("Insufficient finite common features for matched forecasting.")
    issues = features.index[lookback - 1:]
    positions = calendar.get_indexer(issues)
    keep = positions + 2 < len(calendar)
    inputs = np.lib.stride_tricks.sliding_window_view(
        features.to_numpy(), lookback, axis=0).transpose(0, 2, 1)[keep].copy()
    positions = positions[keep]
    rows = pd.DataFrame({"trade_date": calendar[positions + 1], "return_date": calendar[positions + 2],
                         "realized_bv": bv.reindex(calendar[positions + 2]).to_numpy()},
                        index=issues[keep].rename("forecast_date"))
    return (ForecastData(rows.copy(), inputs[:, :, :1].copy()), ForecastData(rows.copy(), inputs),
            {"first_common_issue": str(rows.index[0].date()), "common_observations": len(rows),
             "input_channels": list(features), "external_feature_lag_sessions": 1,
             "feature_carries": carries, "target_horizon_sessions": 2, "history_trading_days": lookback})


def selection_folds(rows, training_end, folds):
    """Disjoint training, early-stop and selection blocks, with label purging."""
    available = rows.realized_bv.notna().to_numpy()
    eligible = (rows.return_date <= pd.Timestamp(training_end)).to_numpy() & available
    result = []
    for training, validation in _cv_folds(rows, eligible, folds):
        positions = np.flatnonzero(validation)
        if len(positions) < 6:
            raise ValueError("Each fold needs separate nonempty stopping and selection blocks.")
        stop_positions, score_positions = np.array_split(positions, 2)
        stopping, scoring = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
        stopping[stop_positions], scoring[score_positions] = True, True
        training &= (rows.return_date < rows.index[stop_positions[0]]).to_numpy()
        stopping &= (rows.return_date < rows.index[score_positions[0]]).to_numpy()
        if not training.any() or not stopping.any():
            raise ValueError("Purging left no available training or stopping labels.")
        result.append((training, stopping, scoring))
    return result


def matched_gru(prepared, calendar, checkpoint_dir, settings, model_name, progress=print):
    """Select on data never used to stop a fit; refit annually for t+2 targets."""
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists():
        raise ValueError("Use new checkpoints; fitted models are never replaced.")
    folds = selection_folds(prepared.rows, settings["training_end"], settings["folds"])
    candidates = _profiles(settings["profiles"])
    seeds = settings["seeds"]
    torch, device = _require_cuda()
    rows, inputs = prepared.rows, prepared.inputs
    def gpu_inputs(selected, scaling):
        return _gpu_inputs(torch, _log_inputs(inputs[selected]), scaling, device)
    targets = rows.realized_bv.to_numpy()
    records, fold_records = [], []
    for number, (training, stopping, scoring) in enumerate(folds, 1):
        fold_records.append({"fold": number, "training_observations": int(training.sum()),
                             "training_last_target": str(rows.loc[training, "return_date"].max().date()),
                             "stopping_first_issue": str(rows.index[stopping][0].date()),
                             "stopping_last_target": str(rows.loc[stopping, "return_date"].max().date()),
                             "selection_first_issue": str(rows.index[scoring][0].date()),
                             "selection_last_target": str(rows.loc[scoring, "return_date"].max().date()),
                             "stopping_observations": int(stopping.sum()),
                             "selection_observations": int(scoring.sum())})
    for candidate, profile in enumerate(candidates):
        errors, epochs_saved = [], []
        for number, (training, stopping, scoring) in enumerate(folds, 1):
            scaling = _log_scale(prepared, training)
            score_inputs = gpu_inputs(scoring, scaling)
            predictions = []
            for seed in seeds:
                progress(f"{model_name}: profile={candidate}, fold={number}, seed={seed}", flush=True)
                model, epoch, smearing = _fit(
                    torch, prepared, training, stopping, scaling, profile, seed,
                    settings["max_epochs"], settings["patience"], settings["learning_rate"])
                predictions.append(np.exp(_log_predictions(torch, model, score_inputs, scaling)) * smearing)
                epochs_saved.append(epoch)
            if not np.isfinite(predictions).all() or (np.asarray(predictions) <= 0).any():
                raise RuntimeError("Selection forecasts must be finite and positive.")
            errors.extend((np.log(np.mean(predictions, axis=0)) - np.log(targets[scoring])).tolist())
        records.append({"profile": candidate, **profile, "selection_log_mse": float(np.mean(np.square(errors))),
                        "selection_observations": len(errors), "best_epochs": epochs_saved})
    tuning = pd.DataFrame(records)
    selected = int(tuning.selection_log_mse.argmin())
    tuning["selected"] = np.arange(len(tuning)) == selected
    profile = candidates[selected]
    epochs = max(1, int(np.median(tuning.iloc[selected].best_epochs)))
    result = rows.copy()
    result[f"{model_name}_bv"], result[f"{model_name}_refit_asof"] = np.nan, pd.NaT
    checkpoint_dir.mkdir(parents=True)
    annual, nodes = [], set()
    for year, asof, training, issued in _annual_splits(rows, calendar, settings["evaluation_start"]):
        training &= np.isfinite(targets)
        scaling = _log_scale(prepared, training)
        issued_inputs = gpu_inputs(issued, scaling)
        predictions, paths = [], []
        for seed in seeds:
            progress(f"{model_name}: annual refit={year}, seed={seed}, epochs={epochs}", flush=True)
            model, _, smearing = _fit(torch, prepared, training, None, scaling, profile, seed,
                                      epochs, settings["patience"], settings["learning_rate"])
            nodes.update(getattr(model, "_rnn_autograd_nodes", ()))
            predictions.append(np.exp(_log_predictions(torch, model, issued_inputs, scaling)) * smearing)
            path = checkpoint_dir / str(year) / f"seed_{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                       **scaling, **profile, "smearing": smearing, "seed": seed, "epochs": epochs,
                       "input_channels": inputs.shape[2], "lookback": inputs.shape[1],
                       "target_horizon_sessions": 2, "model_name": model_name,
                       "refit_asof": str(asof.date()),
                       "training_last_target": str(rows.loc[training, "return_date"].max().date()),
                       **_gpu_metadata(torch, native_rnn=_uses_native_rnn(torch),
                                       rnn_autograd_nodes=getattr(model, "_rnn_autograd_nodes", ()))}
            with path.open("xb") as destination:
                torch.save(payload, destination)
            paths.append(path.relative_to(checkpoint_dir).as_posix())
        predicted = np.mean(predictions, axis=0)
        if not np.isfinite(predicted).all() or (predicted <= 0).any():
            raise RuntimeError("Revised log-variance forecasts must be finite and positive.")
        result.loc[issued, f"{model_name}_bv"] = predicted
        result.loc[issued, f"{model_name}_refit_asof"] = asof
        annual.append({"year": year, "refit_asof": str(asof.date()),
                       "training_last_target": payload["training_last_target"],
                       "training_observations": int(training.sum()), "checkpoints": paths})
    return result, tuning, {"model_name": model_name, "selected_profile": profile, "refit_epochs": epochs,
                            "seeds": seeds, "folds": fold_records, "annual_refits": annual,
                            "selection_rule": "log MSE on blocks distinct from early stopping",
                            "epoch_rule": "median selected-profile stopping epoch; no annual retuning",
                            "forecast_horizon": 2, "positive_transform": "exp(log BV) times train-only smearing",
                            "history_trading_days": inputs.shape[1],
                            **_gpu_metadata(torch, native_rnn=_uses_native_rnn(torch), rnn_autograd_nodes=nodes)}


def two_step_classical(rows, bv, price_returns, settings, progress=print):
    """Analytic h=2 GARCH/GJR and recursive h=2 ARFIMA, fitted sequentially."""
    result, records, starts = rows.copy(), [], {}
    weights = .94 ** np.arange(settings["lookback"] - 1, -1, -1)
    lengths = {"arfima": len(bv.loc[settings["window_start"]:settings["window_end"]]),
               "garch": len(price_returns.loc[settings["window_start"]:settings["window_end"]])}
    lengths["gjr"] = lengths["garch"]
    if min(lengths.values()) < 60:
        raise ValueError("Classical models require at least 60 initial observations.")
    for name in MODELS[:4]:
        result[f"{name}_bv"], result[f"{name}_refit_asof"] = np.nan, pd.NaT
    previous_month = None
    for issue in rows.index:
        if issue.month != previous_month:
            progress(f"Classical h=2 fits: {issue.date()}", flush=True)
        previous_month = issue.month
        history = bv.loc[:issue]
        result.loc[issue, "ewma_bv"] = float(history.iloc[-len(weights):] @ (weights / weights.sum()))
        result.loc[issue, "ewma_refit_asof"] = issue
        for name, source in (("arfima", bv), ("garch", price_returns), ("gjr", price_returns)):
            observed = source.loc[settings["window_start"]:issue]
            if settings["window"] == "rolling":
                observed = observed.iloc[-lengths[name]:]
            if name == "arfima":
                _, details = _arfima_forecast(observed.to_numpy())
                raw = _fractional_prediction(np.r_[observed, details["raw_forecast_bv"]],
                                              details["d"], details["mean_bv"])
            else:
                one_step, details, starts[name] = _garch_forecast(
                    observed.to_numpy(), asymmetric=name == "gjr", starting_values=starts.get(name))
                raw = details["omega_bv"] + (details["alpha"] + details["beta"] + .5 * details["gamma"]) * one_step
            if not np.isfinite(raw):
                raise RuntimeError(f"{name} produced a nonfinite h=2 forecast at {issue}.")
            result.loc[issue, f"{name}_bv"] = max(float(raw), 1e-12)
            result.loc[issue, f"{name}_refit_asof"] = issue
            records.append({"issue_date": issue, "model": name, "training_first": observed.index[0],
                            "training_last": observed.index[-1], "training_observations": len(observed),
                            "horizon": 2, "raw_forecast_bv": raw, "forecast_floored": raw < 1e-12})
    return result, pd.DataFrame(records)


def _input_identity(args):
    return {name: file_digest(path) for name, path in {
        "reconstruction_config": args.config, "portfolio_config": args.portfolio_config,
        "source_manifest": args.data_dir / "manifest.json", "oxford_archive": args.oxford_archive,
        "returns_manifest": args.returns_dir / "manifest.json"}.items()}


def _portfolio_settings(args):
    settings = load_realistic_settings(args.portfolio_config)
    return {**settings, "primary_models": list(MODELS), "secondary_models": [],
            "risk_mode": "annual", "annual_target": .10, "costs_bps": list(COSTS_BPS),
            "main_cost_bps": 10, "initial_cash": True, "cost_benchmark": True,
            "liquidate_end": True, "require_aligned_horizon": True}


def _read_stage(root, stage, protocol_digest):
    directory = root / stage
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if metadata.get("protocol_sha256") != protocol_digest:
        raise ValueError("Revised stages must share the exact frozen protocol.")
    verify_files(directory, metadata["files"], "Revised stage file does not match its saved hash.")
    frame = pd.read_csv(directory / "forecasts.csv", index_col=0, parse_dates=True, float_precision="round_trip")
    for name in frame:
        if name.endswith("_asof") or name in ("trade_date", "return_date", "history_first", "history_last"):
            frame[name] = pd.to_datetime(frame[name])
    return frame, metadata


def run(args):
    root = args.output_dir.resolve()
    protected = [args.data_dir.resolve(), args.returns_dir.resolve(), args.oxford_archive.resolve()]
    if any(root.is_relative_to(path) or path.is_relative_to(root) for path in protected):
        raise ValueError("Revised outputs must be separate from every source input.")
    with args.config.open("rb") as source:
        settings = tomllib.load(source)["reconstruction"]
    identity = _input_identity(args)
    portfolio_settings = _portfolio_settings(args)
    if args.stage == "freeze":
        if root.exists():
            raise ValueError("Freeze requires a new output directory; runs are never overwritten.")
        protocol = {"created_at_utc": utc_now(), "inputs": identity, "code": package_code_digests(),
                    "forecast_settings": settings, "portfolio_settings": portfolio_settings,
                    "forecast_horizon": 2, "external_feature_lag_sessions": 1,
                    "evaluation_status": "exploratory revision of previously inspected historical data",
                    "subperiods": "slices of continuously held paths; liquidation charged only at the final evaluation close",
                    "selection": "matched observations; distinct purged stopping and scoring blocks",
                    "comparison_family": "all six timing/static tests; all configured defensive additions; both GRUs versus EWMA; richer versus univariate GRU; one joint Holm adjustment",
                    "limitations": ["Oxford ends in February 2022; substitute inputs here end in 2020",
                                    "Macro data are revised historical vintages; one-session lag is an assumption",
                                    "Same-close BV availability not verified; intraday BV omits overnight risk",
                                    "10% is an equity risk budget; defensive additions can change total risk",
                                    "Taxes are investor-specific and excluded; costs are scenarios, not estimated execution prices"],
                    "git_commit": git_head(), "git_dirty": git_dirty()}
        root.mkdir(parents=True)
        (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        print(f"Frozen revised protocol: {root}", flush=True)
        return
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (protocol["inputs"] != identity or protocol["code"] != package_code_digests()
            or protocol["forecast_settings"] != settings):
        raise ValueError("Inputs or calculation code changed after freezing; create a separate revised run.")
    protocol_digest = file_digest(protocol_path)
    destination = root / args.stage
    if destination.exists():
        raise ValueError("Use a new stage directory; saved results are never overwritten.")
    observations, _ = load_sources(args.data_dir)
    calendar = observations.equity_price.dropna().loc[:settings["evaluation_end"]].index
    bv = load_bipower(args.oxford_archive, "2000-01-01", settings["evaluation_end"])
    univar, multivar, quality = matched_inputs(observations, bv, calendar, settings["lookback"])
    first_refit = calendar[calendar < settings["evaluation_start"]][-1]
    eligible = univar.rows.loc[univar.rows.index >= first_refit]
    with TemporaryDirectory(prefix=".revised-", dir=root) as temporary:
        output = Path(temporary) / "stage"
        output.mkdir()
        model_metadata = {}
        if args.stage == "classical":
            prices = np.log(observations.equity_price.dropna()).diff().dropna()
            forecasts, fits = two_step_classical(eligible, bv, prices, settings)
            tables = {"forecasts": forecasts, "daily_fits": fits}
            model_metadata = {"backend": "CPU-native SciPy/arch", "forecast_horizon": 2,
                              "ewma_decay": .94, "ewma_horizon_rule": "conditional level held without future observations"}
        elif args.stage in ("gru_univar_log", "gru_multivar_log"):
            prepared = univar if args.stage == "gru_univar_log" else multivar
            forecasts, tuning, model_metadata = matched_gru(
                prepared, calendar, output / "checkpoints", settings, args.stage)
            tables = {"forecasts": forecasts.loc[eligible.index], "tuning": tuning}
        else:
            forecasts, _ = _read_stage(root, "classical", protocol_digest)
            fold_dates = None
            for stage in STAGES[1:3]:
                frame, metadata = _read_stage(root, stage, protocol_digest)
                if (not frame.index.equals(forecasts.index)
                        or not frame[["trade_date", "return_date", "realized_bv"]].equals(
                            forecasts[["trade_date", "return_date", "realized_bv"]])):
                    raise ValueError("All revised models must forecast identical dated targets.")
                if fold_dates is not None and metadata["model"]["folds"] != fold_dates:
                    raise ValueError("Both GRUs must share training, stopping and scoring dates.")
                fold_dates = metadata["model"]["folds"]
                forecasts[[f"{stage}_bv", f"{stage}_refit_asof"]] = frame[[f"{stage}_bv", f"{stage}_refit_asof"]]
            data = load_returns(load_study(args.portfolio_config), args.returns_dir, verified_monthly=True)
            daily = data.daily.assign(RF=data.factors.RF)
            result = realistic_portfolios(daily, calendar, forecasts, portfolio_settings)
            metrics = _performance(result, portfolio_settings)
            metrics["annual_equity_budget"] = .10
            metrics["realized_vol_to_equity_budget"] = metrics.annualized_volatility / .10
            tests = _tests(result, calendar, portfolio_settings)
            extra_tests = []
            paths = result["paths"][10]
            for first, second in (("gru_univar_log", "ewma"), ("gru_multivar_log", "ewma"),
                                  ("gru_multivar_log", "gru_univar_log")):
                extra_tests.append({"model": first, "comparison": f"{first}_minus_{second}",
                                    "measure": "annualized Sharpe difference (Ledoit-Wolf HAC)",
                                    **sharpe_difference_test(paths[f"{first}_cash"], paths[f"{second}_cash"],
                                                            result["returns"].RF.to_numpy(),
                                                            portfolio_settings["sharpe_hac_lags"])})
            tests = pd.concat([tests, pd.DataFrame(extra_tests)], ignore_index=True)
            tests["family"] = "revised_exploratory"
            tests["holm_p_value"] = np.where(tests.p_value.notna(),
                multipletests(tests.p_value.fillna(1), method="holm")[1], np.nan)
            tests["holm_reject"] = tests.holm_p_value < portfolio_settings["alpha"]
            scored = forecasts.loc[forecasts.return_date.between(
                portfolio_settings["evaluation_start"], portfolio_settings["evaluation_end"])]
            scores, forecast_tests = _forecast_scores(scored, {**settings, "monthly_multiplier": np.sqrt(252)}, MODELS)
            scores = scores.rename(columns={"monthly_mse": "annual_volatility_mse", "monthly_mape": "annual_volatility_mape"})
            forecast_tests["measure"] = forecast_tests.measure.str.replace("monthly", "annual_volatility", regex=False)
            tables = {"forecasts": forecasts, "performance": metrics, "tests": tests,
                      "forecast_scores": scores, "forecast_tests": forecast_tests,
                      "declines": _declines(result, portfolio_settings), "thresholds": result["thresholds"],
                      "static_controls": result["static_controls"], "decisions": result["decisions"]}
            for cost, paths in result["paths"].items():
                tables[f"daily_returns_{cost:g}bps"] = paths
            model_metadata = {"portfolio_days": len(result["returns"]), "models": list(MODELS),
                              "forecast_horizon": 2, "inference": "exploratory; previously inspected history"}
        for name, frame in tables.items():
            frame.to_csv(output / f"{name}.csv", index=not isinstance(frame.index, pd.RangeIndex))
        metadata = {"created_at_utc": utc_now(), "stage": args.stage, "protocol_sha256": protocol_digest,
                    "quality": quality, "model": model_metadata,
                    "files": directory_digests(output)}
        (output / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        output.rename(destination)
    print(f"Saved revised stage: {destination}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", *STAGES))
    parser.add_argument("--config", type=Path, default=Path("configs/paper-reconstruction.toml"))
    parser.add_argument("--portfolio-config", type=Path, default=Path("configs/paper-realistic.toml"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--returns-dir", type=Path, required=True)
    parser.add_argument("--oxford-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(1, f"Revised study: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
