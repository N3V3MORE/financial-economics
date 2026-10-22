"""CPU-native realistic portfolios for frozen, delayed Oxford-BV forecasts."""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib

import numpy as np
import pandas as pd

from .analysis import _check_returns, performance_summary
from .costs import dynamic_costed_path
from .data import parse_french_factors
from .paper import load_bipower, load_paper_settings, prepare_paper_forecasts
from .paper_realistic import _stress_thresholds
from .provenance import file_digest, git_dirty, git_head, utc_now
from .return_data import compound_months, load_returns


@dataclass(frozen=True)
class ExtensionSettings:
    evaluation_start: str = "2010-01-01"
    calibration_start: str = "2009-01-01"
    defensive_assets: tuple[str, ...] = ("TLT", "GLD", "DBC")
    costs_bps: tuple[float, ...] = (0, 10, 25)
    stress_quantile: float = .90
    defensive_cap: float = .20
    annual_target: float = .10
    risk_modes: tuple[str, ...] = ("paper", "annual10")
    windows: tuple[str, ...] = ("2010:2020", "2021:2021", "2017:2021")
    rebalance_frequency: str = "daily"
    min_allocation_change: float = 0.0

    def validate(self, paper):
        start, calibration, end = map(pd.Timestamp, (self.evaluation_start, self.calibration_start,
                                                      paper.evaluation_end))
        if (pd.isna(start) or pd.isna(calibration) or start.tz is not None or calibration.tz is not None
                or start != start.normalize() or calibration != calibration.normalize()
                or not pd.Timestamp(paper.evaluation_start) <= calibration < start < end
                or start.day != 1 or end != end + pd.offsets.MonthEnd(0)):
            raise ValueError("Extension needs past calibration and complete evaluation months.")
        if (len(set(self.defensive_assets)) != len(self.defensive_assets)
                or not set(self.defensive_assets) <= {"TLT", "GLD", "DBC"}):
            raise ValueError("Choose unique TLT/GLD/DBC defensive substitutes.")
        if self.rebalance_frequency not in ("daily", "weekly"):
            raise ValueError("Rebalance frequency must be daily or weekly.")
        if (isinstance(self.min_allocation_change, bool) or not np.isfinite(self.min_allocation_change)
                or not 0 <= self.min_allocation_change <= 1):
            raise ValueError("Minimum allocation change must be a finite fraction from 0 to 1.")
        if self.min_allocation_change > 0 and self.defensive_assets:
            raise ValueError("Minimum allocation changes currently require SPY/cash only.")
        if (not self.risk_modes or len(set(self.risk_modes)) != len(self.risk_modes)
                or not set(self.risk_modes) <= {"paper", "annual10"}):
            raise ValueError("Risk modes must be unique paper/annual10 values.")
        if (not self.costs_bps or len(set(self.costs_bps)) != len(self.costs_bps)
                or 0 not in self.costs_bps or 10 not in self.costs_bps
                or any(isinstance(value, bool) or not np.isfinite(value) or not 0 <= value < 10000
                       for value in self.costs_bps)):
            raise ValueError("Costs must be finite, nonnegative, unique, and include 0/10 bps.")
        for value in (self.stress_quantile, self.defensive_cap, self.annual_target):
            if isinstance(value, bool) or not np.isfinite(value) or not 0 < value < 1:
                raise ValueError("Invalid extension quantile, cap, or annual target.")
        for window in self.windows:
            first, last = window.split(":")
            if not self.evaluation_start[:4] <= first <= last <= paper.evaluation_end[:4]:
                raise ValueError("Extension windows must lie inside evaluation years.")


def load_extension_settings(path, paper):
    with Path(path).open("rb") as source:
        raw = tomllib.load(source)
    if "extension" not in raw:
        raise ValueError("Choose a config containing [extension] settings.")
    values = raw["extension"]
    for name in ("defensive_assets", "costs_bps", "risk_modes", "windows"):
        if name in values:
            values[name] = tuple(values[name])
    settings = ExtensionSettings(**values)
    settings.validate(paper)
    return settings


def load_frozen_selection(directory, paper, freeze, archive_digest):
    """Reuse hyperparameters selected before evaluation, never original weights."""
    directory = Path(directory)
    raw = (directory / "run.json").read_bytes()
    source = json.loads(raw)
    model = source.get("gru", {})
    if (source.get("oxford_archive_sha256") != archive_digest
            or source.get("stage") != "GRU-UNIVAR partial replication"
            or pd.Timestamp(source["paper"]["training_end"]) > freeze
            or not model.get("folds")
            or any(pd.Timestamp(fold["validation_last_target"]) > freeze for fold in model["folds"])):
        raise ValueError("Frozen model selection must use this BV source and pre-evaluation labels only.")
    for name in ("learning_rate", "dropout", "weight_decay", "seeds"):
        if model.get(name) != json.loads(json.dumps(getattr(paper, name))):
            raise ValueError(f"Frozen model {name} does not match the extension config.")
    if source["paper"]["lookback"] != paper.lookback or model.get("num_layers") != 2:
        raise ValueError("Frozen model architecture does not match the extension.")
    path = directory / "gru_tuning.csv"
    if file_digest(path) != source.get("files", {}).get(path.name):
        raise ValueError("Frozen tuning table hash does not match its provenance.")
    tuning = pd.read_csv(path)
    selected = tuning.loc[tuning.selected.eq(True)]
    hidden, epochs = model["selected_hidden_size"], model["refit_epochs"]
    if len(selected) != 1 or int(selected.iloc[0].hidden_size) != hidden:
        raise ValueError("Frozen tuning evidence disagrees with the selected hidden size.")
    return {"hidden_size": hidden, "epochs": epochs}, {
        "directory": str(directory.resolve()), "run_sha256": sha256(raw).hexdigest(),
        "selection_last_target": max(fold["validation_last_target"] for fold in model["folds"])}


def _allocation_change_signals(stock, scheduled, minimum_change):
    """Filter target changes against the last accepted target, never live holdings."""
    if minimum_change == 0:
        return scheduled.copy()
    accepted = np.zeros(len(stock), dtype=bool)
    previous = None
    for position in np.flatnonzero(scheduled):
        current = stock[position]
        if previous is None or abs(current - previous) >= minimum_change - 1e-12:
            accepted[position] = True
            previous = current
    return accepted


def extension_portfolios(data, forecasts, calendar, paper, settings):
    """Keep complete calendars; do not rebalance without a new delayed signal."""
    settings.validate(paper)
    assets = ["SPY", *settings.defensive_assets, "RF"]
    daily = data.daily.loc[settings.evaluation_start:paper.evaluation_end, assets[:-1]].copy()
    daily["RF"] = data.factors.RF.reindex(daily.index)
    _check_returns(daily, minimum=2)
    expected = calendar[(calendar >= settings.evaluation_start) & (calendar <= paper.evaluation_end)]
    if not daily.index.equals(expected):
        raise ValueError("Extension returns must retain every market-calendar evaluation date.")
    rows = forecasts.copy()
    positions = calendar.get_indexer(rows.index)
    if (rows.index.has_duplicates or not rows.index.is_monotonic_increasing
            or (positions < 0).any() or (positions + 2 >= len(calendar)).any()
            or not pd.DatetimeIndex(rows.trade_date).equals(calendar[positions + 1])
            or not pd.DatetimeIndex(rows.return_date).equals(calendar[positions + 2])):
        raise ValueError("Extension forecasts must issue at t, trade t+1, and earn t+2 returns.")
    freeze = calendar[calendar < pd.Timestamp(settings.evaluation_start)][-1]
    eligible = rows.loc[(rows.index >= freeze) & rows.return_date.isin(daily.index)].copy()
    if eligible.empty or eligible.return_date.duplicated().any():
        raise ValueError("Extension evaluation needs unique delayed forecast return dates.")
    scheduled = np.ones(len(eligible), dtype=bool)
    if settings.rebalance_frequency == "weekly":
        # Schedule by execution close, not by the following session's return.
        scheduled = ~eligible.trade_date.dt.to_period("W-SUN").duplicated().to_numpy()
    models = [name for name in ("ewma", "gru") if f"{name}_bv" in rows]
    if "ewma" not in models:
        raise ValueError("Extension requires an EWMA baseline.")
    for model in models:
        history = rows.loc[rows.index >= settings.calibration_start, f"{model}_bv"]
        if not np.isfinite(history).all() or (history <= 0).any():
            raise ValueError(f"Missing or invalid {model} forecasts in calibration/evaluation history.")
    if "gru" in models and ("gru_refit_asof" not in rows
                            or rows.loc[rows.index >= settings.calibration_start, "gru_refit_asof"].isna().any()
                            or (rows.gru_refit_asof.dropna() > rows.index[rows.gru_refit_asof.notna()]).any()):
        raise ValueError("GRU refits must be available by their issue dates.")
    full = f"{settings.evaluation_start[:4]}:{paper.evaluation_end[:4]}"
    windows = list(dict.fromkeys([full, *settings.windows]))
    benchmark = []
    for window in windows:
        first, last = window.split(":")
        selected = daily.loc[first:last]
        if len(selected) >= 2:
            summary = performance_summary(selected[["SPY"]], selected.RF).reset_index()
            summary["calmar"] = summary.cagr / (-summary.max_drawdown).replace(0, np.nan)
            benchmark.append(summary.assign(window=window, representation="gross buy-and-hold"))
    scenarios = [(f"cash_{cost:g}bps", cost, True) for cost in settings.costs_bps]
    scenarios.append(("zero_cash_10bps", 10, False))
    paths_saved, months_saved = {"SPY": daily.SPY}, {}
    rebalance_counts = {}
    metrics, decisions, thresholds, targets_saved, fees_saved, conditions, scores = [], [], [], [], [], [], []
    for first_last in windows:
        first, last = first_last.split(":")
        scored = eligible.loc[eligible.return_date.dt.year.between(int(first), int(last))]
        scored = scored.loc[scored.realized_bv.notna()]
        for model in models:
            actual = paper.monthly_multiplier * np.sqrt(scored.realized_bv)
            predicted = paper.monthly_multiplier * np.sqrt(scored[f"{model}_bv"])
            scores.append({"window": first_last, "model": model, "n_obs": len(scored),
                           "monthly_mse": float(((actual - predicted) ** 2).mean()),
                           "monthly_mape": float(((actual - predicted).abs() / actual).mean())})
    for mode in settings.risk_modes:
        multiplier = paper.monthly_multiplier if mode == "paper" else np.sqrt(252)
        target_level = paper.monthly_target if mode == "paper" else settings.annual_target
        definitions = {}
        for model in models:
            predicted = multiplier * np.sqrt(eligible[f"{model}_bv"].to_numpy())
            stock = np.minimum(target_level / predicted, 1.)
            accepted = _allocation_change_signals(stock, scheduled, settings.min_allocation_change)
            rebalance = pd.DatetimeIndex([daily.index[0], *eligible.loc[accepted, "return_date"]]).unique().sort_values()
            rebalance_counts[f"{model} | {mode}"] = int(accepted.sum())
            annual, records = _stress_thresholds(
                multiplier * np.sqrt(rows[f"{model}_bv"]), calendar,
                {"calibration_start": settings.calibration_start, "stress_quantile": settings.stress_quantile},
                sorted(daily.index.year.unique()))
            cutoffs = eligible.return_date.dt.year.map(annual).to_numpy()
            thresholds.extend({"risk_mode": mode, "model": model, **record} for record in records)
            stressed = predicted > cutoffs
            decisions.append(eligible[["trade_date", "return_date"]].reset_index().assign(
                model=model, risk_mode=mode, volatility=predicted, threshold=cutoffs,
                stock_target=stock, stress=stressed, scheduled=scheduled, rebalance=accepted))
            for asset in ("cash", *settings.defensive_assets):
                name = f"{model}_{asset}"
                targets = pd.DataFrame(np.nan, index=daily.index, columns=assets)
                targets.iloc[0] = [0.] * (len(assets) - 1) + [1.]
                weights = pd.DataFrame(0., index=pd.DatetimeIndex(eligible.return_date), columns=assets)
                weights["SPY"] = stock
                if asset != "cash":
                    weights[asset] = np.where(stressed, np.minimum(1 - stock, settings.defensive_cap), 0.)
                weights["RF"] = 1 - weights.drop(columns="RF").sum(axis=1)
                weights = weights.loc[weights.index.isin(rebalance)]
                targets.loc[weights.index] = weights.to_numpy()
                targets = targets.ffill()
                definitions[name] = (targets, rebalance)
                targets_saved.append(targets.reset_index(names="Date").assign(
                    portfolio=name, risk_mode=mode, rebalance=targets.index.isin(rebalance)))
        for scenario, cost, cash_income in scenarios:
            inputs = daily.copy()
            if not cash_income:
                inputs["RF"] = 0.
            paths, accounting = {}, {}
            for name, (targets, rebalance) in definitions.items():
                path = dynamic_costed_path(inputs, targets, cost, rebalance)
                paths[name] = path["return"]
                before = (1 + path["return"]).cumprod().shift(1, fill_value=1.)
                accounting[name] = path.assign(fees_per_initial_dollar=before * path.fee_fraction)
                fees_saved.append(accounting[name].reset_index(names="Date").assign(
                    portfolio=name, risk_mode=mode, scenario=scenario))
            paths = pd.DataFrame(paths)
            monthly = compound_months(paths.assign(SPY=daily.SPY), calendar,
                                      settings.evaluation_start, paper.evaluation_end)
            months_saved.setdefault("SPY", monthly.SPY)
            for window in windows:
                first, last = window.split(":")
                selected = paths.loc[first:last]
                if len(selected) < 2:
                    continue
                summary = performance_summary(selected, daily.RF.loc[selected.index]).reset_index(names="portfolio")
                summary["calmar"] = summary.cagr / (-summary.max_drawdown).replace(0, np.nan)
                summary["turnover"] = [accounting[name].loc[first:last, "traded_fraction"].sum() for name in summary.portfolio]
                summary["fees_per_initial_dollar"] = [accounting[name].loc[first:last, "fees_per_initial_dollar"].sum()
                                                      for name in summary.portfolio]
                metrics.append(summary.assign(window=window, risk_mode=mode, scenario=scenario,
                                               cost_bps=cost, cash_income=cash_income))
                down = monthly.loc[first:last, "SPY"] < 0
                for model in models:
                    for asset in settings.defensive_assets:
                        gap = (monthly[f"{model}_{asset}"] - monthly[f"{model}_cash"]).loc[first:last]
                        conditions.append({"window": window, "risk_mode": mode, "scenario": scenario,
                                           "model": model, "asset": asset, "negative_months": int(down.sum()),
                                           "mean_down_month_advantage": gap.loc[down].mean()})
            suffix = f" | {mode} | {scenario}"
            paths_saved.update({name + suffix: paths[name] for name in paths})
            months_saved.update({name + suffix: monthly[name] for name in paths})
    return {"forecasts": rows, "forecast_scores": pd.DataFrame(scores), "performance": pd.concat(metrics, ignore_index=True),
            "benchmark_performance": pd.concat(benchmark, ignore_index=True),
            "decisions": pd.concat(decisions, ignore_index=True), "annual_thresholds": pd.DataFrame(thresholds),
            "target_weights": pd.concat(targets_saved, ignore_index=True), "fees": pd.concat(fees_saved, ignore_index=True),
            "daily_returns": pd.DataFrame(paths_saved), "monthly_returns": pd.DataFrame(months_saved),
            "portfolio_conditions": pd.DataFrame(conditions, columns=[
                "window", "risk_mode", "scenario", "model", "asset", "negative_months",
                "mean_down_month_advantage"])}, {
                "freeze_asof": str(freeze.date()), "portfolio_days": len(daily),
                "signal_days": len(eligible), "initial_cash_days": int((daily.index < eligible.return_date.min()).sum()),
                "cadence_signal_days": int(scheduled.sum()),
                "rebalance_signal_days": int(scheduled.sum()) if settings.min_allocation_change == 0 else None,
                "rebalance_signal_days_by_strategy": rebalance_counts,
                "missing_signal_days": int((daily.index >= eligible.return_date.min()).sum() - len(eligible))}


def run_extension_command(args, study):
    """Run the original BV measure through 2021 with frozen delayed models."""
    paper = load_paper_settings(args.config)
    settings = load_extension_settings(args.config, paper)
    if args.rebalance_frequency:
        settings = replace(settings, rebalance_frequency=args.rebalance_frequency)
    if args.cash_only:
        settings = replace(settings, defensive_assets=())
    if args.min_allocation_change is not None:
        settings = replace(settings, min_allocation_change=args.min_allocation_change)
    settings.validate(paper)
    destination = args.output_dir.resolve()
    protected = [args.data_dir.resolve(), args.oxford_archive.resolve()]
    protected += [path.resolve() for path in (args.selection_run, args.forecast_run) if path]
    if destination.exists() or any(destination.is_relative_to(path) or path.is_relative_to(destination) for path in protected):
        raise ValueError("Use a new extension output directory separate from all source inputs/runs.")
    data = load_returns(study, args.data_dir, verified_monthly=True)
    if not study.start <= settings.evaluation_start < paper.evaluation_end <= study.end:
        raise ValueError("Extension evaluation must fit inside the ETF snapshot.")
    calendar = parse_french_factors((args.data_dir / "french_factors_daily.zip").read_bytes()).loc[
        paper.training_start:paper.evaluation_end].index
    bv = load_bipower(args.oxford_archive, paper.training_start, paper.evaluation_end)
    if bv.empty or bv.index[0] != calendar[0] or bv.index[-1] != calendar[-1]:
        raise ValueError("Original BV input must cover the full extension span; sources are never spliced.")
    archive_digest = file_digest(args.oxford_archive)
    prepared = prepare_paper_forecasts(bv, calendar, paper.lookback, horizon=2)
    forecasts = prepared.rows.copy()
    weights = paper.ewma_decay ** np.arange(paper.lookback - 1, -1, -1)
    forecasts["ewma_bv"] = prepared.inputs[:, :, 0] @ (weights / weights.sum())
    freeze = calendar[calendar < settings.evaluation_start][-1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".paper-extension-", dir=destination.parent) as temporary:
        output = Path(temporary) / "run"
        output.mkdir()
        model, parent = {}, {}
        if args.forecast_run:
            source_raw = (args.forecast_run / "run.json").read_bytes()
            source = json.loads(source_raw)
            if (source.get("forecast_horizon") != 2 or source.get("paper") != json.loads(json.dumps(asdict(paper)))
                    or source.get("source_manifest") != data.provenance
                    or source.get("oxford_archive_sha256") != archive_digest):
                raise ValueError("Reused extension forecasts must match horizon, model settings, and source data.")
            path = args.forecast_run / "forecasts.csv"
            if file_digest(path) != source.get("files", {}).get(path.name):
                raise ValueError("Reused extension forecast hash does not match provenance.")
            forecasts = pd.read_csv(path, index_col="forecast_date", float_precision="round_trip")
            forecasts.index = pd.to_datetime(forecasts.index)
            for name in ("trade_date", "return_date", "gru_refit_asof"):
                if name in forecasts:
                    forecasts[name] = pd.to_datetime(forecasts[name])
            if not forecasts.index.equals(prepared.rows.index):
                raise ValueError("Reused forecast dates do not match the prepared BV histories.")
            model, parent = source.get("gru", {}), {"forecast_run": str(args.forecast_run.resolve()),
                                                    "run_sha256": sha256(source_raw).hexdigest()}
        elif not args.baseline_only:
            from .paper_gru import paper_gru_forecasts

            frozen, parent = load_frozen_selection(args.selection_run, paper, freeze, archive_digest)
            gru, _, model = paper_gru_forecasts(prepared, paper, calendar, output / "checkpoints",
                                               frozen_selection=frozen)
            forecasts[["gru_bv", "gru_refit_asof"]] = gru[["gru_bv", "gru_refit_asof"]]
        tables, counts = extension_portfolios(data, forecasts, calendar, paper, settings)
        tables["evaluation_inputs"] = data.daily.loc[settings.evaluation_start:paper.evaluation_end,
                                                     ["SPY", *settings.defensive_assets]].assign(RF=data.factors.RF)
        for name, frame in tables.items():
            frame.to_csv(output / f"{name}.csv", index=not isinstance(frame.index, pd.RangeIndex))
        metadata = {"created_at_utc": utc_now(), "stage": "realistic original-BV extension",
                    "study": study.settings(), "paper": asdict(paper), "extension": asdict(settings),
                    "source_manifest": data.provenance, "oxford_archive_sha256": archive_digest,
                    "gru": model, "model_parent": parent, "forecast_horizon": 2, **counts,
                    "execution": "issue t close; trade t+1 close; earn daily return ending t+2",
                    "rebalance_schedule": ("every eligible execution close" if settings.rebalance_frequency == "daily"
                                           else "first eligible execution close per Monday-Sunday week; holdings drift between trades"),
                    "allocation_change_rule": "0 disables filtering; otherwise first scheduled entry always trades, then absolute SPY target change from last accepted target must meet the minimum (1e-12 tolerance); full target reset; actual holdings drift does not trigger trades",
                    "thresholds": "assigned by earned-return year; model forecasts since calibration_start issued strictly before the preceding annual close; shared with realistic portfolios",
                    "calibration": "2009 forecasts reconstructed with pre-2009 fitted weights and architecture selected through 2009; all calibration known before 2010 evaluation",
                    "missing_data": "missing BV labels unscored; missing signals retain holdings and drift; no return day dropped",
                    "main_scenario": "paper risk scaling; actual cash income; 10 bps per ETF dollar bought or sold",
                    "fees": "entry and purchases plus sales; exit and taxes excluded",
                    "subperiod_fees": "slices of continuously held paths; fees_per_initial_dollar uses the full run's original capital",
                    "sharpe": "observed RF remains the opportunity-cost benchmark in every scenario",
                    "inference": "descriptive historical comparisons; no new significance or unseen-performance claim",
                    "limitations": ["ETF substitutes for paper indices", "Ends in 2021; no full 2022-2025 test",
                                    "Architecture chosen on previously inspected histories; 2021 is an additional historical year"],
                    "git_commit": git_head(),
                    "git_dirty": git_dirty(),
                    "config_sha256": file_digest(args.config),
                    "code_sha256": {path.name: file_digest(path)
                                    for path in sorted(Path(__file__).parent.glob("*.py"))},
                    "files": {path.relative_to(output).as_posix(): file_digest(path)
                              for path in sorted(output.rglob("*")) if path.is_file()}}
        (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError("Extension output already exists.")
        output.rename(destination)
    main = tables["performance"].loc[lambda frame: frame.window.eq(
        f"{settings.evaluation_start[:4]}:{paper.evaluation_end[:4]}") & frame.risk_mode.eq("paper")
        & frame.scenario.eq("cash_10bps") & frame.portfolio.str.endswith("_cash")]
    print(main[["portfolio", "cagr", "annualized_volatility", "sharpe", "max_drawdown", "turnover"]].to_string(index=False))
    print(f"Results: {destination}; {counts['portfolio_days']} market days; original BV only; descriptive comparisons.")
