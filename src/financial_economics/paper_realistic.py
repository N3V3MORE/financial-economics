"""Realistic trading test of saved reconstruction forecasts on one ETF snapshot.

Example: python -m financial_economics.paper_realistic --help
A forecast issued at close t trades at close t+1 and earns the t+2 return.
Fees, observed cash income and past-only stress thresholds apply. Forecasts are
never refitted. Portfolio accounting, HAC tests and the bootstrap run on CPU.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from .analysis import _check_returns, performance_summary
from .config import load_study
from .costs import dynamic_costed_path
from .data import parse_french_factors
from .explain import bootstrap_differences
from .paper_evaluation import DECLINES
from .prediction_inference import _hac_test
from .provenance import file_digest, git_branch, git_head, utc_now
from .return_data import compound_months, load_returns


FORECAST_STAGES = {"evaluation": ("arfima", "garch", "gjr", "gru_univar", "gru_multivar"),
                   "gru_univar_log": ("gru_univar_log",), "gru_multivar_log": ("gru_multivar_log",)}
REBALANCE_PERIODS = {"daily": None, "weekly": "W-SUN", "monthly": "M"}


def load_realistic_settings(path):
    with Path(path).open("rb") as source:
        settings = tomllib.load(source)["realistic"]
    start, evaluation, end = map(pd.Timestamp, (settings["calibration_start"], settings["evaluation_start"],
                                                 settings["evaluation_end"]))
    if not (start < evaluation < end and evaluation.day == 1 and end == end + pd.offsets.MonthEnd(0)):
        raise ValueError("Realistic test needs a calibration year before complete evaluation months.")
    models = [*settings["primary_models"], *settings["secondary_models"]]
    assets = [*settings["primary_assets"], *settings["secondary_assets"]]
    if len(set(models)) != len(models) or len(set(assets)) != len(assets) or not settings["primary_models"]:
        raise ValueError("Models and defensive assets must be unique, with at least one primary model.")
    if settings["main_cost_bps"] not in settings["costs_bps"]:
        raise ValueError("The main cost scenario must be one of the cost scenarios.")
    for name in ("monthly_target", "stress_quantile", "defensive_cap"):
        if not 0 < settings[name] < 1:
            raise ValueError(f"Invalid realistic {name}.")
    return settings


def delayed_signals(forecasts, calendar):
    """Issue at close t, trade at close t+1, earn the t+2 return on the ETF calendar."""
    positions = calendar.get_indexer(forecasts.index)
    keep = (positions >= 0) & (positions + 2 < len(calendar))
    rows = forecasts.loc[keep].copy()
    rows["trade_date"] = calendar[positions[keep] + 1]
    rows["earn_date"] = calendar[positions[keep] + 2]
    return rows, int((~keep).sum())


def sharpe_difference_test(first, second, risk_free, lags, trading_days=252):
    """Ledoit and Wolf (2008) HAC delta-method test of first minus second Sharpe ratio."""
    excess = np.column_stack([np.asarray(first) - risk_free, np.asarray(second) - risk_free])
    unavailable = {"estimate": np.nan, "se": np.nan, "z": np.nan, "p_value": np.nan, "status": "unavailable"}
    count = len(excess)
    # A constant column can have a tiny positive computed variance because its
    # floating-point mean is rounded. Check its range before any division.
    if count <= lags + 5 or not np.isfinite(excess).all() or (np.ptp(excess, axis=0) == 0).any():
        return unavailable
    mean = excess.mean(axis=0)
    centered = excess - mean
    variance = (centered ** 2).mean(axis=0)
    if not np.isfinite(variance).all() or (variance <= 0).any():
        return unavailable
    deviation = np.sqrt(variance)
    sharpe = mean / deviation
    difference = sharpe[0] - sharpe[1]
    # Centered influence functions give the same four-moment delta method,
    # without subtracting nearly equal covariance terms for similar portfolios.
    standardized = centered / deviation
    influence = standardized - .5 * sharpe * (standardized ** 2 - 1)
    gap = influence[:, 0] - influence[:, 1]
    gap -= gap.mean()
    # Positive rescalings have the same Sharpe and influence function; avoid
    # interpreting their floating-point roundoff as testable variation.
    roundoff = 32 * np.finfo(float).eps * np.max(np.abs(influence))
    long_run_variance = gap @ gap / count
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        long_run_variance += 2 * weight * (gap[lag:] @ gap[:-lag]) / count
    sampling_variance = long_run_variance / (count - 4)
    scale = np.sqrt(trading_days)
    if not np.isfinite(difference):
        return unavailable
    if (not np.isfinite(sampling_variance) or sampling_variance <= 0
            or np.max(np.abs(gap)) <= roundoff):
        return {**unavailable, "estimate": float(difference * scale), "status": "degenerate_variance"}
    se = float(np.sqrt(sampling_variance))
    z = difference / se
    return {"estimate": float(difference * scale), "se": se * scale, "z": float(z),
            "p_value": float(2 * norm.sf(abs(z))), "status": "available"}


def _stress_thresholds(volatility, calendar, settings, years):
    """Each year's cutoff uses only forecasts issued before that year's first trading session."""
    cutoffs, records = {}, []
    for year in years:
        asof = calendar[calendar < pd.Timestamp(year=year, month=1, day=1)][-1]
        past = volatility.loc[(volatility.index >= settings["calibration_start"]) & (volatility.index < asof)]
        if past.empty:
            raise ValueError("Past-only thresholds need calibration forecasts.")
        cutoffs[year] = float(np.quantile(past, settings["stress_quantile"]))
        records.append({"evaluation_year": year, "asof": asof, "calibration_last_issue": past.index[-1],
                        "calibration_forecasts": len(past), "threshold": cutoffs[year]})
    return cutoffs, records


def risk_budget(settings):
    """Return target and daily-volatility multiplier with explicit risk units."""
    mode = settings.get("risk_mode", "paper")
    if mode == "annual":
        target, multiplier = settings.get("annual_target", .10), np.sqrt(252)
    elif mode == "paper":
        target, multiplier = settings["monthly_target"], settings["monthly_multiplier"]
    else:
        raise ValueError("Risk mode must be paper or annual.")
    if (isinstance(target, bool) or not np.isfinite(target) or not 0 < target < 1
            or isinstance(multiplier, bool) or not np.isfinite(multiplier) or multiplier <= 0):
        raise ValueError("Risk target and volatility multiplier must be finite and positive.")
    return target, multiplier


def realistic_portfolios(daily, calendar, forecasts, settings):
    """Model-timed stock/cash, matched static stock/cash, and stress-period defensive additions.

    The stock column is settings["benchmark"] (SPY by default)."""
    assets = [*settings["primary_assets"], *settings["secondary_assets"]]
    models = [*settings["primary_models"], *settings["secondary_models"]]
    frequency = settings.get("rebalance_frequency", "daily")
    if frequency not in REBALANCE_PERIODS:
        raise ValueError("Rebalance frequency must be daily, weekly or monthly.")
    benchmark = settings.get("benchmark", "SPY")
    columns = [benchmark, *assets, "RF"]
    first, last = settings["evaluation_start"], settings["evaluation_end"]
    returns = daily.loc[first:last, columns].copy()
    _check_returns(returns, minimum=2)
    expected = calendar[(calendar >= first) & (calendar <= last)]
    if not returns.index.equals(expected):
        raise ValueError("Realistic returns must keep every ETF-calendar evaluation date.")
    rows, dropped = delayed_signals(forecasts, calendar)
    if settings.get("require_aligned_horizon", False):
        if ("return_date" not in rows or "trade_date" not in forecasts
                or not pd.DatetimeIndex(rows.return_date).equals(pd.DatetimeIndex(rows.earn_date))
                or not pd.DatetimeIndex(forecasts.loc[rows.index, "trade_date"]).equals(
                    pd.DatetimeIndex(rows.trade_date))):
            raise ValueError("Aligned forecasts must target the t+2 return earned after execution.")
    signals = rows.loc[rows.earn_date.isin(returns.index)]
    initial_cash = settings.get("initial_cash", False)
    if initial_cash:
        freeze = calendar[calendar < returns.index[0]][-1]
        signals = signals.loc[(signals.index >= freeze) & (signals.earn_date > returns.index[0])]
    if signals.empty or signals.earn_date.duplicated().any():
        raise ValueError("Realistic evaluation needs unique delayed signal dates.")
    period = REBALANCE_PERIODS[frequency]
    scheduled = (np.ones(len(signals), dtype=bool) if period is None else
                 ~signals.trade_date.dt.to_period(period).duplicated().to_numpy())
    signal_dates = pd.DatetimeIndex(signals.earn_date)
    rebalance = signal_dates[scheduled]
    years = sorted(returns.index.year.unique())
    month_starts = returns.index[~returns.index.to_period("M").duplicated()]
    target_level, multiplier = risk_budget(settings)
    definitions, thresholds, statics, decisions = {}, [], [], []
    for model in models:
        values = rows[f"{model}_bv"]
        asof = rows[f"{model}_refit_asof"]
        if (not np.isfinite(values).all() or (values <= 0).any()
                or asof.isna().any() or (asof > rows.index).any()):
            raise ValueError(f"Invalid or future-fitted {model} forecasts.")
        volatility = multiplier * np.sqrt(values)
        stock_all = np.minimum(target_level / volatility, 1.)
        calibration = stock_all.loc[(stock_all.index >= settings["calibration_start"])
                                    & (stock_all.index < pd.Timestamp(first))]
        static_weight = float(calibration.mean())
        statics.append({"model": model, "static_spy_weight": static_weight,
                        "calibration_issues": len(calibration),
                        "rule": "mean model stock target over calibration-year issues; fixed before evaluation"})
        cutoffs, records = _stress_thresholds(volatility, calendar, settings, years)
        thresholds += [{"model": model, **record} for record in records]
        signal_volatility = volatility.loc[signals.index].to_numpy()
        stock = stock_all.loc[signals.index].to_numpy()
        cutoff = signals.earn_date.dt.year.map(cutoffs).to_numpy()
        stressed = signal_volatility > cutoff
        decisions.append(pd.DataFrame({"issue_date": signals.index, "trade_date": signals.trade_date.to_numpy(),
                                       "earn_date": signal_dates, "model": model, "volatility": signal_volatility,
                                       "threshold": cutoff, "stock_target": stock, "stress": stressed,
                                       "scheduled": scheduled}))
        for asset in ("cash", *assets):
            targets = pd.DataFrame(np.nan, index=returns.index, columns=columns)
            targets.iloc[0] = [0.] * (len(columns) - 1) + [1.]
            weights = pd.DataFrame(0., index=signal_dates, columns=columns)
            weights[benchmark] = stock
            if asset != "cash":
                weights[asset] = np.where(stressed, np.minimum(1 - stock, settings["defensive_cap"]), 0.)
            weights["RF"] = 1 - weights.drop(columns="RF").sum(axis=1)
            targets.loc[rebalance] = weights.loc[rebalance].to_numpy()
            definitions[f"{model}_{asset}"] = (targets.ffill(), rebalance.union(returns.index[:1]))
        static = pd.DataFrame(0., index=returns.index, columns=columns)
        static[benchmark], static["RF"] = static_weight, 1 - static_weight
        static_rebalance = month_starts
        if initial_cash:
            static.iloc[0] = [0.] * (len(columns) - 1) + [1.]
            static_rebalance = month_starts.union(returns.index[1:2])
        definitions[f"{model}_static"] = (static, static_rebalance)
    if settings.get("cost_benchmark", False):
        benchmark_targets = pd.DataFrame(0., index=returns.index, columns=columns)
        benchmark_targets[benchmark] = 1.
        benchmark_entry = returns.index[:1]
        if initial_cash:
            benchmark_targets.iloc[0] = [0.] * (len(columns) - 1) + [1.]
            benchmark_entry = returns.index[:2]
        definitions[benchmark] = (benchmark_targets, benchmark_entry)
    paths_by_cost, accounting_by_cost = {}, {}
    for cost in settings["costs_bps"]:
        paths, accounting = {benchmark: returns[benchmark]}, {}
        for name, (targets, rebalance) in definitions.items():
            path = dynamic_costed_path(returns, targets, cost, rebalance,
                                       liquidate_end=settings.get("liquidate_end", False))
            paths[name] = path["return"]
            before = (1 + path["return"]).cumprod().shift(1, fill_value=1.)
            accounting[name] = path.assign(fees_per_initial_dollar=before * path.fee_fraction)
        paths_by_cost[cost], accounting_by_cost[cost] = pd.DataFrame(paths), accounting
    return {"paths": paths_by_cost, "accounting": accounting_by_cost, "returns": returns,
            "thresholds": pd.DataFrame(thresholds), "static_controls": pd.DataFrame(statics),
            "decisions": pd.concat(decisions, ignore_index=True), "models": models, "assets": assets,
            "rebalance_frequency": frequency, "scheduled_signal_days": int(scheduled.sum()),
            "dropped_issue_dates": dropped,
            "missing_signal_days": int(len(returns) - signals.earn_date.isin(returns.index).sum())}


def _performance(result, settings):
    tables = []
    for cost, paths in result["paths"].items():
        for window in settings["windows"]:
            first, last = window.split(":")
            selected = paths.loc[first:last]
            summary = performance_summary(selected, result["returns"].RF.loc[selected.index]).reset_index(names="portfolio")
            summary["calmar"] = summary.cagr / (-summary.max_drawdown).replace(0, np.nan)
            accounting = result["accounting"][cost]
            summary["turnover"] = [accounting[name].loc[first:last, "traded_fraction"].sum() if name in accounting else 0.
                                   for name in summary.portfolio]
            summary["fees_per_initial_dollar"] = [accounting[name].loc[first:last, "fees_per_initial_dollar"].sum()
                                                  if name in accounting else 0. for name in summary.portfolio]
            tables.append(summary.assign(window=window, cost_bps=cost))
    return pd.concat(tables, ignore_index=True)


def _declines(result, settings):
    paths = result["paths"][settings["main_cost_bps"]]
    rows = []
    for label, peak, trough in DECLINES:
        window = paths.loc[(paths.index > peak) & (paths.index <= trough)]
        if len(window) < 2:
            continue
        loss = (1 + window).prod() - 1
        rows += [{"decline": label, "peak": peak, "trough": trough, "portfolio": name, "return": float(value)}
                 for name, value in loss.items()]
    return pd.DataFrame(rows)


def _tests(result, calendar, settings):
    """Primary Holm family: timing vs static and each primary asset vs cash, per primary model."""
    paths = result["paths"][settings["main_cost_bps"]]
    rf = result["returns"].RF.to_numpy()
    monthly = compound_months(paths, calendar, settings["evaluation_start"], settings["evaluation_end"])
    benchmark = settings.get("benchmark", "SPY")
    market_months = compound_months(result["returns"][[benchmark]], calendar,
                                    settings["evaluation_start"], settings["evaluation_end"])
    down = (market_months[benchmark] < 0).to_numpy(dtype=float)
    design = np.column_stack([down, 1 - down])
    rows = []
    for model in result["models"]:
        family = ("primary" if model in settings["primary_models"]
                  and result.get("rebalance_frequency", "daily") == "daily" else "descriptive")
        test = sharpe_difference_test(paths[f"{model}_cash"], paths[f"{model}_static"], rf,
                                      settings["sharpe_hac_lags"])
        rows.append({"family": family, "model": model, "comparison": "timing_minus_static",
                     "measure": "annualized Sharpe difference (Ledoit-Wolf HAC)", "n_obs": len(rf),
                     "negative_months": int(down.sum()), **test})
        for asset in result["assets"]:
            difference = (monthly[f"{model}_{asset}"] - monthly[f"{model}_cash"]).to_numpy()
            inference = _hac_test(difference, design, 0, settings["monthly_hac_lags"], settings["alpha"])
            rows.append({"family": family if asset in settings["primary_assets"] else "descriptive",
                         "model": model, "comparison": f"{asset}_minus_cash",
                         "measure": "mean monthly return difference in negative-SPY months (HAC)",
                         "n_obs": len(difference), "negative_months": int(down.sum()),
                         "estimate": float(difference[down == 1].mean()) if down.any() else np.nan,
                         "se": inference["hac_se"],
                         "z": np.nan, "p_value": inference["p_value"], "status": inference["status"]})
    tests = pd.DataFrame(rows)
    primary = tests.family.eq("primary")
    primary_p_values = tests.loc[primary, "p_value"]
    # Unavailable tests keep their place in the prespecified family, as in
    # prediction/protection inference, but are never reported as significant.
    # Weekly and monthly cadence runs have no primary family to adjust.
    tests["holm_p_value"] = np.nan
    if primary.any():
        adjusted = multipletests(primary_p_values.fillna(1), method="holm")[1]
        tests.loc[primary, "holm_p_value"] = np.where(primary_p_values.notna(), adjusted, np.nan)
    tests["holm_reject"] = tests.holm_p_value < settings["alpha"]
    return tests


def _intervals(result, settings):
    paths = result["paths"][settings["main_cost_bps"]]
    rf = result["returns"].RF
    tables = []
    for model in settings["primary_models"]:
        tables.append(bootstrap_differences(paths, rf, f"{model}_cash", [f"{model}_static", settings.get("benchmark", "SPY")],
                                            resamples=settings["resamples"],
                                            block_length=settings["block_length"], seed=settings["seed"]))
        for asset in result["assets"]:
            tables.append(bootstrap_differences(paths, rf, f"{model}_{asset}", [f"{model}_cash"],
                                                resamples=settings["resamples"],
                                                block_length=settings["block_length"], seed=settings["seed"]))
    return pd.concat(tables, ignore_index=True)


def _load_forecasts(run_dir, models):
    """Require verified forecast bytes and one shared reconstruction input identity."""
    frames, provenance = [], {}
    identity_fields = ("source_manifest", "oxford_archive", "config")
    canonical_inputs = None
    for stage, names in FORECAST_STAGES.items():
        wanted = [name for name in names if name in models]
        if not wanted:
            continue
        directory = run_dir / stage
        metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        inputs = metadata.get("inputs")
        if (not isinstance(inputs, dict)
                or any(not isinstance(inputs.get(name), str) or len(inputs[name]) != 64
                       or any(character not in "0123456789abcdefABCDEF" for character in inputs[name])
                       for name in identity_fields)):
            raise ValueError(f"Saved {stage} forecasts require valid SHA-256 input identities for "
                             "source_manifest, oxford_archive and config.")
        identity = {name: inputs[name].lower() for name in identity_fields}
        if canonical_inputs is None:
            canonical_inputs = identity
        elif identity != canonical_inputs:
            changed = ", ".join(name for name in identity_fields if identity[name] != canonical_inputs[name])
            raise ValueError(f"Saved forecast stages have different input identities ({changed}); "
                             "use stages from one reconstruction run.")
        if file_digest(directory / "forecasts.csv") != metadata["files"]["forecasts.csv"]:
            raise ValueError(f"Saved {stage} forecasts do not match their provenance hash.")
        frame = pd.read_csv(directory / "forecasts.csv", index_col=0, parse_dates=True, float_precision="round_trip")
        keep = [column for name in wanted for column in (f"{name}_bv", f"{name}_refit_asof")]
        frame = frame[keep]
        for name in wanted:
            frame[f"{name}_refit_asof"] = pd.to_datetime(frame[f"{name}_refit_asof"])
        frames.append(frame)
        provenance[stage] = {"run_sha256": file_digest(directory / "run.json"),
                             "forecasts_sha256": metadata["files"]["forecasts.csv"],
                             "inputs": identity}
    combined = pd.concat(frames, axis=1, join="inner")
    if any(len(frame) != len(combined) for frame in frames):
        raise ValueError("Saved forecast stages must cover identical issue dates.")
    missing = [name for name in models if f"{name}_bv" not in combined]
    if missing:
        raise ValueError(f"Missing saved forecasts for: {', '.join(missing)}.")
    return combined, provenance


def run(args):
    settings = load_realistic_settings(args.config)
    if getattr(args, "rebalance_frequency", None) is not None:
        settings["rebalance_frequency"] = args.rebalance_frequency
    bootstrap_enabled = not getattr(args, "skip_bootstrap", False)
    study = load_study(args.config)
    destination = args.output_dir.resolve()
    protected = [args.data_dir.resolve(), args.forecast_run.resolve()]
    if destination.exists() or any(destination.is_relative_to(path) for path in protected):
        raise ValueError("Use a new realistic output directory outside the source inputs and forecast run.")
    data = load_returns(study, args.data_dir, verified_monthly=True)
    calendar = parse_french_factors((args.data_dir / "french_factors_daily.zip").read_bytes()).index
    calendar = calendar[(calendar >= study.start) & (calendar <= study.end)]
    daily = data.daily.copy()
    daily["RF"] = data.factors.RF.reindex(daily.index)
    models = [*settings["primary_models"], *settings["secondary_models"]]
    forecasts, forecast_provenance = _load_forecasts(args.forecast_run, models)
    result = realistic_portfolios(daily, calendar, forecasts, settings)
    tables = {"performance": _performance(result, settings), "tests": _tests(result, calendar, settings),
              "declines": _declines(result, settings),
              "thresholds": result["thresholds"], "static_controls": result["static_controls"],
              "decisions": result["decisions"],
              "daily_returns": result["paths"][settings["main_cost_bps"]]}
    if bootstrap_enabled:
        tables["bootstrap_intervals"] = _intervals(result, settings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".paper-realistic-", dir=destination.parent) as temporary:
        output = Path(temporary) / "run"
        output.mkdir()
        for name, frame in tables.items():
            frame.to_csv(output / f"{name}.csv", index=name == "daily_returns")
        code = {name: file_digest(Path(__file__).with_name(name))
                for name in ("paper_realistic.py", "costs.py", "analysis.py", "explain.py",
                             "prediction_inference.py", "return_data.py", "paper_evaluation.py")}
        metadata = {"created_at_utc": utc_now(),
                    "study": "Realistic trading test of saved public-data reconstruction forecasts",
                    "settings": settings, "config_sha256": file_digest(args.config),
                    "source_manifest": data.provenance, "forecasts": forecast_provenance,
                    "calculation_versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "statsmodels")},
                    "branch": git_branch(),
                    "head": git_head(),
                    "code": code,
                    "bootstrap_enabled": bootstrap_enabled,
                    "signals": {"dropped_issue_dates_not_on_etf_calendar": result["dropped_issue_dates"],
                                "evaluation_days_without_new_signal": result["missing_signal_days"],
                                "scheduled_signal_days": result["scheduled_signal_days"],
                                "rebalance_frequency": result["rebalance_frequency"]},
                    "assumptions": {
                        "execution": "issue at close t, trade at close t+1, earn t+2; one-step forecasts are one session stale",
                        "schedule": "daily or first eligible execution close of each Monday-Sunday week/calendar month; seed entry retained; daily holdings drift between trades",
                        "allocation": "stock=min(target/(22*sqrt(BV)),1); defensive=stress*min(1-stock,cap); residual cash",
                        "thresholds": "each year's 90th percentile of the model's own forecasts issued since calibration start and before that year",
                        "static_control": "SPY/cash at the model's mean calibration-year stock target, reset monthly",
                        "returns": "WRDS/CRSP total returns for SPY, TLT, GLD and DBC; French daily RF earns cash income",
                        "fees": "per ETF dollar bought or sold, entry included, exit and taxes excluded; RF frictionless",
                        "primary_family": ("per primary model: timing minus static Sharpe, TLT and GLD minus cash in negative-SPY months; one Holm adjustment"
                                           if result["rebalance_frequency"] == "daily" else
                                           "none; weekly/monthly cadence comparisons are exploratory"),
                        "descriptive": "secondary models, DBC, subperiods, declines, bootstrap intervals and all weekly/monthly tests",
                        "post_hoc": "log-BV GRUs were added after raw-BV results were inspected",
                        "cadence_comparisons": "weekly/monthly runs are exploratory; daily forecast allocations held longer, not longer-horizon forecasts; no adjusted primary tests for these runs",
                        "history": "2011-2020 was inspected in earlier runs; this is retrospective, not untouched data"}}
        metadata["files"] = {path.name: file_digest(path) for path in sorted(output.iterdir())}
        (output / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        output.rename(destination)
    print(f"Saved realistic test: {destination}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper-realistic.toml"))
    parser.add_argument("--data-dir", type=Path, required=True, help="Verified WRDS/CRSP ETF snapshot")
    parser.add_argument("--forecast-run", type=Path, required=True, help="Reconstruction output directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebalance-frequency", choices=tuple(REBALANCE_PERIODS),
                        help="Override trading cadence; daily remains the default")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip descriptive bootstrap intervals for saved-forecast sensitivity runs")
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
