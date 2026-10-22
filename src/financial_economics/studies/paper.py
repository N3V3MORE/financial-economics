"""Small GRU-UNIVAR paper replication with explicit ETF and execution substitutes.

Calendar preparation, EWMA, and portfolio accounting are CPU-native. Neural
training requires CUDA or ROCm. The paper's full-test quantiles are retrospective.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import tomllib
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from financial_economics.evaluation.metrics import _check_returns, performance_summary
from financial_economics.portfolio.execution import dynamic_costed_path
from financial_economics.provenance import package_code_digests
from financial_economics.datasets.public import parse_french_factors
from financial_economics.models.variance import ForecastData
from financial_economics.datasets.returns import load_returns


OXFORD_URL = ("https://web.archive.org/web/20220301022212id_/https://realized.oxford-man.ox.ac.uk/"
              "images/oxfordmanrealizedvolatilityindices.zip")
OXFORD_SHA256 = "f620ad530e35e4e28c667cfd53a9b68e8cac42133bdf3e734614fffc58f87be1"
PAPER_URL = "https://doi.org/10.1016/j.ribaf.2021.101610"


@dataclass(frozen=True)
class PaperSettings:
    training_start: str = "2000-01-01"
    training_end: str = "2009-12-31"
    evaluation_start: str = "2010-01-01"
    evaluation_end: str = "2020-12-31"
    lookback: int = 22
    hidden_sizes: tuple[int, ...] = (16, 32)
    seeds: tuple[int, ...] = (11, 22, 33)
    folds: int = 5
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 8
    learning_rate: float = .001
    dropout: float = .10
    weight_decay: float = .0001
    ewma_decay: float = .94
    monthly_target: float = .10
    monthly_multiplier: float = 22
    quantiles: tuple[float, ...] = (.90, .95, .99)
    caps: tuple[float, ...] = (.10, .20, .30)
    defensive_assets: tuple[str, ...] = ("TLT", "GLD")

    def validate(self):
        dates = pd.DatetimeIndex([self.training_start, self.training_end,
                                  self.evaluation_start, self.evaluation_end])
        if (dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize())
                or not dates[0] < dates[1] < dates[2] < dates[3]):
            raise ValueError("Paper training and evaluation dates must be ordered daily dates.")
        for name in ("lookback", "folds", "batch_size", "max_epochs", "patience"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"Invalid paper {name}.")
        if self.folds < 2 or self.patience > self.max_epochs:
            raise ValueError("Paper folds must be at least two; patience cannot exceed epochs.")
        for name, lower in (("hidden_sizes", 1), ("seeds", 0)):
            values = getattr(self, name)
            if (not values or len(values) != len(set(values))
                    or any(type(value) is not int or value < lower for value in values)):
                raise ValueError(f"Invalid paper {name}.")
        for name, lower, upper in (("learning_rate", 0, 1), ("ewma_decay", 0, 1),
                                    ("monthly_target", 0, 1), ("monthly_multiplier", 0, np.inf)):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not lower < value < upper:
                raise ValueError(f"Invalid paper {name}.")
        if (not np.isfinite(self.dropout) or not 0 <= self.dropout < 1
                or not np.isfinite(self.weight_decay) or self.weight_decay < 0):
            raise ValueError("Invalid paper regularization.")
        for name in ("quantiles", "caps"):
            values = getattr(self, name)
            if (not values or len(values) != len(set(values))
                    or any(isinstance(value, bool) or not np.isfinite(value) or not 0 < value < 1
                           for value in values)):
                raise ValueError(f"Invalid paper {name}.")
        if (not self.defensive_assets or len(set(self.defensive_assets)) != len(self.defensive_assets)
                or not set(self.defensive_assets) <= {"TLT", "GLD"}):
            raise ValueError("Paper defensive substitutes must be unique TLT/GLD tickers.")


def load_paper_settings(path):
    with Path(path).open("rb") as source:
        raw = tomllib.load(source)
    if "paper" not in raw:
        raise ValueError("Choose a config containing [paper] settings, such as paper.toml.")
    values = raw["paper"]
    for name in ("hidden_sizes", "seeds", "quantiles", "caps", "defensive_assets"):
        if name in values:
            values[name] = tuple(values[name])
    settings = PaperSettings(**values)
    settings.validate()
    return settings


def load_bipower(archive, start, end):
    """Read original .SPX BV; dates are local labels, not UTC-converted instants."""
    try:
        with ZipFile(archive) as source:
            names = [name for name in source.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError("Oxford archive must contain exactly one CSV.")
            raw = pd.read_csv(source.open(names[0]))
    except BadZipFile as error:
        raise ValueError("Oxford input must be a CSV ZIP archive.") from error
    if not {"Symbol", "bv"} <= set(raw):
        raise ValueError("Oxford input needs Symbol and bv columns; squared returns are not a substitute.")
    selected = raw.loc[raw.Symbol.eq(".SPX")]
    dates = pd.to_datetime(selected.iloc[:, 0].astype(str).str[:10], format="%Y-%m-%d")
    result = pd.Series(selected.bv.to_numpy(dtype=float), index=pd.DatetimeIndex(dates), name="bv")
    result.index.name = "Date"
    if (result.empty or result.index.has_duplicates or not result.index.is_monotonic_increasing
            or result.index.hasnans or not np.isfinite(result).all() or (result <= 0).any()):
        raise ValueError("SPX bipower variation needs sorted unique dates and positive finite values.")
    return result.loc[start:end]


def prepare_paper_forecasts(bipower, calendar, lookback=22, horizon=1):
    """Use latest observed BVs; target the requested actual market session.

    Missing BV targets stay missing. No return days or target values are filled.
    Missing issue dates produce no new signal; portfolio holdings then drift.
    """
    if (not isinstance(calendar, pd.DatetimeIndex) or calendar.hasnans
            or not calendar.is_unique or not calendar.is_monotonic_increasing
            or calendar.tz is not None or not calendar.equals(calendar.normalize())):
        raise ValueError("Paper forecasts need a sorted unique daily market calendar.")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Paper forecast horizon must be a positive integer.")
    if (type(lookback) is not int or lookback < 1 or len(bipower) <= lookback
            or not isinstance(bipower.index, pd.DatetimeIndex)
            or bipower.index.has_duplicates or not bipower.index.is_monotonic_increasing
            or not np.isfinite(bipower).all() or (bipower <= 0).any()
            or len(bipower.index.difference(calendar))):
        raise ValueError("Paper inputs require positive observed BV on known market dates.")
    issues = bipower.index[lookback - 1:]
    positions = calendar.get_indexer(issues)
    usable = positions + horizon < len(calendar)
    returns = calendar[positions[usable] + horizon]
    inputs = np.lib.stride_tricks.sliding_window_view(bipower.to_numpy(), lookback)[usable, :, None]
    rows = pd.DataFrame({"return_date": returns, "realized_bv": bipower.reindex(returns).to_numpy()},
                        index=issues[usable].rename("forecast_date"))
    if horizon > 1:
        rows["trade_date"] = calendar[positions[usable] + 1]
    return ForecastData(rows, inputs)


def paper_portfolios(returns, forecasts, settings, trading_days=252):
    """Zero cash/fees; idealized preceding-close trading and full-test quantiles."""
    settings.validate()
    _check_returns(returns, minimum=2)
    if not {"SPY", *settings.defensive_assets} <= set(returns):
        raise ValueError("Paper returns must include SPY and the selected defensive substitutes.")
    required = {"return_date", "realized_bv", "ewma_bv"}
    if (not required <= set(forecasts) or forecasts.index.has_duplicates
            or not forecasts.index.is_monotonic_increasing
            or forecasts.return_date.duplicated().any()
            or not (forecasts.index < pd.DatetimeIndex(forecasts.return_date)).all()):
        raise ValueError("Paper forecasts need unique ordered issue/next-return dates and BV columns.")
    daily = returns.loc[settings.evaluation_start:settings.evaluation_end].copy()
    if daily.empty:
        raise ValueError("Paper evaluation returns are empty.")
    daily["RF"] = 0.
    eligible = forecasts.loc[forecasts.return_date.isin(daily.index)].copy()
    models = [name for name in ("ewma", "gru") if f"{name}_bv" in eligible]
    paths, allocations, thresholds, scores, descriptions = {"SPY": daily.SPY}, [], [], [], []
    for model in models:
        values = eligible[f"{model}_bv"].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Missing or invalid {model} BV forecasts in evaluation.")
        if model == "gru" and ("gru_refit_asof" not in eligible
                               or eligible.gru_refit_asof.isna().any()
                               or (eligible.gru_refit_asof > eligible.index).any()):
            raise ValueError("GRU refits must precede their forecast issue dates.")
        monthly = pd.Series(settings.monthly_multiplier * np.sqrt(values),
                            index=pd.DatetimeIndex(eligible.return_date))
        if monthly.index[0] != daily.index[0]:
            raise ValueError("Paper allocation requires a forecast for the first evaluation return.")
        stock = (settings.monthly_target / monthly).clip(upper=1)
        labeled = np.isfinite(eligible.realized_bv.to_numpy(dtype=float))
        actual = eligible.realized_bv.to_numpy(dtype=float)[labeled]
        if not len(actual) or (actual <= 0).any():
            raise ValueError("Paper scoring needs positive observed BV labels.")
        realized_monthly = settings.monthly_multiplier * np.sqrt(actual)
        scores.append({"model": model, "n_obs": len(actual),
                       "bv_mse": float(np.mean((actual - values[labeled]) ** 2)),
                       "monthly_mse": float(np.mean((realized_monthly - monthly.to_numpy()[labeled]) ** 2)),
                       "monthly_mape": float(np.mean(np.abs(realized_monthly - monthly.to_numpy()[labeled])
                                                    / realized_monthly))})
        definitions = [(f"{model}_clean", None, None, 0., np.nan)]
        for quantile in settings.quantiles:
            threshold = float(monthly.quantile(quantile))
            thresholds.append({"model": model, "quantile": quantile, "threshold": threshold,
                               "calibration": "full testing period; retrospective"})
            for asset in settings.defensive_assets:
                for cap in settings.caps:
                    definitions.append((f"{model}_{asset}_q{quantile:g}_cap{cap:g}", asset, threshold, cap, quantile))
        for name, asset, threshold, cap, quantile in definitions:
            weights = pd.DataFrame(0., index=stock.index, columns=daily.columns)
            weights["SPY"] = stock
            if asset is not None:
                weights[asset] = np.minimum(1 - stock, cap).where(monthly >= threshold, 0.)
            weights["RF"] = 1 - weights.drop(columns="RF").sum(axis=1)
            dated = weights.reindex(daily.index).ffill()
            path = dynamic_costed_path(daily, dated, 0, rebalance_dates=stock.index)
            paths[name] = path["return"]
            allocations.append(dated.assign(portfolio=name, rebalance=dated.index.isin(stock.index)))
            descriptions.append({"portfolio": name, "model": model, "asset": asset or "cash",
                                 "quantile": quantile,
                                 "cap": cap})
    paths = pd.DataFrame(paths, index=daily.index)
    metrics = performance_summary(paths, daily.RF, trading_days)
    metrics["calmar"] = metrics.cagr / (-metrics.max_drawdown).replace(0, np.nan)
    details = pd.DataFrame(descriptions).set_index("portfolio").join(metrics)
    additions = details.loc[details.asset.ne("cash")].copy()
    for metric in ("sharpe", "calmar"):
        additions[f"{metric}_minus_clean"] = [row[metric] - metrics.loc[f"{row.model}_clean", metric]
                                               for _, row in additions.iterrows()]
    summary = additions.groupby(["model", "asset", "quantile"])[
        ["sharpe", "calmar", "sharpe_minus_clean", "calmar_minus_clean"]].mean().reset_index()
    return {"daily_returns": paths, "performance": metrics, "portfolios": details,
            "target_weights": pd.concat(allocations).rename_axis("Date").reset_index(),
            "thresholds": pd.DataFrame(thresholds), "forecast_scores": pd.DataFrame(scores),
            "protection_by_quantile": summary}


def run_paper_command(args, study):
    """Validate retained inputs, optionally train, and save one new local run."""
    settings = load_paper_settings(args.config)
    destination = args.output_dir.resolve()
    if (destination.exists() or destination.is_relative_to(args.data_dir.resolve())
            or destination == args.oxford_archive.resolve()
            or args.oxford_archive.resolve().is_relative_to(destination)):
        raise ValueError("Use a new paper output directory outside the source inputs.")
    if not study.start <= settings.evaluation_start < settings.evaluation_end <= study.end:
        raise ValueError("Paper evaluation must fit inside the retained ETF snapshot.")
    data = load_returns(study, args.data_dir, verified_monthly=True)
    calendar = parse_french_factors((args.data_dir / "french_factors_daily.zip").read_bytes()).loc[
        settings.training_start:settings.evaluation_end].index
    bipower = load_bipower(args.oxford_archive, settings.training_start, settings.evaluation_end)
    if bipower.empty or bipower.index[0] != calendar[0] or bipower.index[-1] != calendar[-1]:
        raise ValueError("Oxford archive must cover the full requested training/evaluation span.")
    prepared = prepare_paper_forecasts(bipower, calendar, settings.lookback)
    forecasts = prepared.rows.copy()
    weights = settings.ewma_decay ** np.arange(settings.lookback - 1, -1, -1)
    weights /= weights.sum()
    forecasts["ewma_bv"] = prepared.inputs[:, :, 0] @ weights
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".paper-", dir=destination.parent) as temporary:
        output = Path(temporary) / "run"
        output.mkdir()
        model_metadata, extra = {}, {}
        if not args.baseline_only:
            from financial_economics.models.paper_gru import paper_gru_forecasts

            gru, tuning, model_metadata = paper_gru_forecasts(
                prepared, settings, calendar, output / "checkpoints")
            forecasts[["gru_bv", "gru_refit_asof"]] = gru[["gru_bv", "gru_refit_asof"]]
            extra["gru_tuning"] = tuning
        tables = paper_portfolios(data.daily, forecasts, settings, study.trading_days)
        tables.update(extra, forecasts=forecasts, observed_bipower=bipower.to_frame(),
                      evaluation_inputs=data.daily.loc[settings.evaluation_start:settings.evaluation_end])
        transformed = settings.monthly_multiplier * np.sqrt(bipower)
        checks = {"observations": len(bipower), "scaled_mean": float(transformed.mean()),
                  "scaled_std": float(transformed.std()),
                  "missing_market_dates": calendar.difference(bipower.index).strftime("%Y-%m-%d").tolist()}
        scores = tables["forecast_scores"].set_index("model")
        if "gru" in scores.index:
            tables["paper_comparison"] = pd.DataFrame({
                "metric": ["monthly_mse", "monthly_mape", "clean_sharpe", "clean_calmar"],
                "paper_GRU_UNIVAR": [.0043, .3046, .95, .70],
                "our_ETF_substitute": [scores.loc["gru", "monthly_mse"], scores.loc["gru", "monthly_mape"],
                                       tables["performance"].loc["gru_clean", "sharpe"],
                                       tables["performance"].loc["gru_clean", "calmar"]]})
        for name, frame in tables.items():
            frame.to_csv(output / f"{name}.csv", index=not isinstance(frame.index, pd.RangeIndex))
        digest = sha256(args.oxford_archive.read_bytes()).hexdigest()
        metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "paper_url": PAPER_URL,
                    "stage": "baseline" if args.baseline_only else "GRU-UNIVAR partial replication",
                    "study": study.settings(), "paper": asdict(settings), "gru": model_metadata,
                    "source_manifest": data.provenance, "oxford_archive_sha256": digest,
                    "oxford_archive_source": OXFORD_URL if digest == OXFORD_SHA256 else "supplied archive; origin unverified",
                    "data_checks": checks, "monthly_scaling": "22 * sqrt(predicted raw daily BV); Table 1 matches",
                    "execution": "idealized issue-close allocation for next market close return; no trading delay",
                    "thresholds": "full-test forecast quantiles; retrospective, not executable out-of-sample thresholds",
                    "missing_data": "22 latest observed BV inputs; missing targets unscored; no new signal => holdings drift",
                    "cash_return": 0, "cost_bps": 0,
                    "limitations": ["SPY/TLT/GLD ETF returns substitute for Datastream indices",
                                    "Only GRU-UNIVAR; no original multivariate, ARFIMA or GARCH replication",
                                    "Small explicit tuning grid and three seeds; original exact settings/code unavailable",
                                    "Daily sqrt(BV) MSE training inferred from Table 1 scaling; original code unavailable",
                                    "Historical, idealized comparison; no claim of executable or unseen performance"],
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
                    "config_sha256": sha256(args.config.read_bytes()).hexdigest(),
                    "code_sha256": package_code_digests(),
                    "files": {path.relative_to(output).as_posix(): sha256(path.read_bytes()).hexdigest()
                              for path in sorted(output.rglob("*")) if path.is_file()}}
        (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"Paper output already exists: {destination}")
        output.rename(destination)
    print(f"Paper stage: {metadata['stage']}; {len(tables['daily_returns'])} portfolio days; zero cash income and fees.")
    print(tables["performance"].loc[[name for name in ("SPY", "ewma_clean", "gru_clean")
                                    if name in tables["performance"].index],
                                   ["cagr", "annualized_volatility", "sharpe", "max_drawdown", "calmar"]].to_string())
    print(f"Results: {destination}")
