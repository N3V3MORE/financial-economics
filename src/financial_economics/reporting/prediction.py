"""Saved forecast reuse, CSV-backed research figures, and atomic result output."""

from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from financial_economics.models.variance import qlike
from financial_economics.provenance import directory_digests, file_digest, git_state, package_code_digests, utc_now


def load_forecast_run(directory, study, settings, manifest):
    """Verify saved settings, provenance, and bytes before reusing forecasts.

    This path does not import PyTorch or load executable model checkpoints.
    """
    directory = Path(directory).resolve()
    run_path = directory / "run.json"
    source = json.loads(run_path.read_text(encoding="utf-8"))
    normalized = json.loads(json.dumps(settings.settings()))
    if source.get("prediction") != normalized:
        raise ValueError("Saved forecast prediction settings do not match this run.")
    if source.get("source_manifest") != manifest:
        raise ValueError("Saved forecast source provenance does not match this snapshot.")
    if "study" in source and source["study"] != json.loads(json.dumps(study.settings())):
        raise ValueError("Saved forecast study settings do not match this run.")
    files = source.get("files", {})
    forecast_path = directory / "forecasts.csv"
    digest = file_digest(forecast_path)
    if files.get("forecasts.csv") != digest:
        raise ValueError("Saved forecasts.csv hash does not match its run metadata.")
    forecasts = pd.read_csv(forecast_path, index_col="forecast_date", float_precision="round_trip")
    forecasts.index = pd.to_datetime(forecasts.index)
    for column in ("trade_date", "return_date", "gru_refit_asof"):
        if column in forecasts:
            forecasts[column] = pd.to_datetime(forecasts[column])
    tables = {}
    for name in ("ewma_tuning", "gru_tuning"):
        path = directory / f"{name}.csv"
        if path.exists():
            if files.get(path.name) != file_digest(path):
                raise ValueError(f"Saved {path.name} hash does not match its run metadata.")
            tables[name] = pd.read_csv(path, float_precision="round_trip")
    metadata = {"forecast_parent": {"directory": str(directory),
                                    "run_sha256": file_digest(run_path),
                                    "forecasts_sha256": digest, "stage": source.get("stage")}}
    if "gru" in source:
        metadata["gru"] = source["gru"]
    return forecasts, tables, metadata


def _figure_tables(tables, result, study, settings):
    full = f"{settings.evaluation_start[:4]}:{study.end[:4]}"
    windows = list(dict.fromkeys([full, *settings.windows]))
    forecasts = tables["forecasts"]
    eligible = forecasts.loc[(forecasts.index >= pd.Timestamp(result.metadata["freeze_asof"]))
                             & (forecasts.return_date >= pd.Timestamp(settings.evaluation_start))]
    scores = []
    for window in windows:
        first, last = window.split(":")
        selected = eligible.loc[eligible.return_date.dt.year.between(int(first), int(last))]
        for model in result.metadata["models"]:
            if not len(selected):
                continue
            observed = selected.squared_return.to_numpy(dtype=float)
            predicted = selected[f"{model}_variance"].to_numpy(dtype=float)
            scores.append({"window": window, "model": model, "n_obs": len(selected),
                           "qlike": float(qlike(observed, predicted).mean()),
                           "mse": float(np.mean((observed - predicted) ** 2))})
    scenarios = tables["scenarios"]
    actual_cash = scenarios.loc[scenarios.cash_income]
    main = actual_cash.loc[actual_cash.cost_bps.eq(10)]
    main = (main if len(main) else actual_cash).iloc[0]
    paths = tables["daily_returns"]
    suffix = f" | {main.scenario}"
    paths = paths[[name for name in paths if name.endswith(suffix)]].copy()
    paths.columns = [name.removesuffix(suffix) for name in paths]
    if "evaluation_inputs" in tables:
        paths["SPY"] = tables["evaluation_inputs"].SPY
    wealth = (1 + paths).cumprod()
    seed = pd.DataFrame(1., index=pd.DatetimeIndex([pd.Timestamp(result.metadata["freeze_asof"])]),
                         columns=wealth.columns)
    wealth = pd.concat([seed, wealth])
    drawdown = wealth / wealth.cummax() - 1
    growth = pd.concat([pd.DataFrame({"Date": wealth.index, "portfolio": name,
                                      "wealth": wealth[name].to_numpy(),
                                      "drawdown": drawdown[name].to_numpy(),
                                      "scenario": main.scenario, "cost_bps": main.cost_bps})
                        for name in wealth], ignore_index=True)
    protection = tables["portfolio_conditions"].loc[
        tables["portfolio_conditions"].scenario.eq(main.scenario)
        & tables["portfolio_conditions"].condition.eq("negative_spy")].copy()
    return {"figure_forecast_accuracy": pd.DataFrame(scores),
            "figure_growth_drawdowns": growth, "figure_protection": protection}


def _figures(output, result, source):
    """Render only the saved figure CSVs, with no hidden plotting calculations."""
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#222222", "axes.labelcolor": "#222222"})
    note = (f"{result.metadata['evaluation_start']} to {result.metadata['evaluation_end']} | {source}"
            f" | {result.metadata['stage']} | defensive ETF: {result.metadata['defensive_asset']}")

    def save(fig, name):
        fig.savefig(output / name, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    scores = pd.read_csv(output / "figure_forecast_accuracy.csv")
    windows = scores.window.drop_duplicates().tolist()
    colors = {"ewma": "#31688e", "gru": "#a66f00"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, measure, label in zip(axes, ("qlike", "mse"), ("Mean QLIKE", "Mean squared error"), strict=True):
        for model, group in scores.groupby("model", sort=False):
            ordered = group.set_index("window").reindex(windows)
            axis.plot(range(len(windows)), ordered[measure], marker="o", color=colors[model],
                      linestyle="-" if model == "ewma" else "--", label=model.upper())
        axis.set_xticks(range(len(windows)), [value.replace(":", "–") for value in windows])
        axis.set_ylabel(label)
        axis.set_title(label + " (lower is better)", loc="left")
        axis.grid(axis="y", color="#eeeeee")
    axes[0].legend(frameon=False)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    counts = "; ".join(f"{row.window}: n={row.n_obs}" for row in scores.drop_duplicates("window").itertuples())
    fig.suptitle("Daily squared-return forecast accuracy", x=.04, ha="left", fontsize=15)
    fig.text(.04, .015, note + "\n" + counts + "; overlapping periods are descriptive.", fontsize=8)
    fig.tight_layout(rect=(0, .1, 1, .93))
    save(fig, "forecast_accuracy.png")

    growth = pd.read_csv(output / "figure_growth_drawdowns.csv", parse_dates=["Date"])
    palette = {"A": "#31688e", "B": "#a66f00", "C": "#8c6bb1", "D": "#b45435", "SPY": "#444444"}
    labels = {"A": "A: EWMA / cash", "B": "B: GRU / cash",
              "C": "C: EWMA / ETF + cash", "D": "D: GRU / ETF + cash", "SPY": "SPY (before fees)"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, group in growth.groupby("portfolio", sort=False):
        for axis, measure in zip(axes, ("wealth", "drawdown"), strict=True):
            axis.plot(group.Date, group[measure], color=palette[name],
                      linestyle="--" if name in ("B", "D") else "-", label=labels[name], linewidth=1.5)
    axes[0].set_ylabel("Value of initial $1")
    axes[1].set_ylabel("Decline from prior peak")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0)
    for axis in axes:
        axis.grid(axis="y", color="#eeeeee")
    dates = growth.Date.drop_duplicates().sort_values()
    ticks = [dates.iloc[0], *[pd.Timestamp(year=year, month=1, day=1)
                              for year in range(dates.iloc[0].year + 2, dates.iloc[-1].year, 2)], dates.iloc[-1]]
    axes[1].set_xticks(ticks, [value.strftime("%Y-%m") for value in ticks])
    fig.suptitle("Portfolio growth and drawdowns", x=.07, ha="left", fontsize=16)
    cost = growth.cost_bps.iloc[0]
    fig.text(.07, .015, note + f"\nA–D: actual cash income; {cost:g} bps per ETF dollar bought or sold; entry included, exit/taxes excluded. SPY excludes fees."
             f"\n{result.metadata['warmup_cash_days']} initial evaluation trading day(s) held in cash before the first executable frozen-settings forecast.", fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, .95))
    save(fig, "growth_drawdowns.png")

    protection = pd.read_csv(output / "figure_protection.csv")
    windows = protection.window.drop_duplicates().tolist()
    pairs = protection[["portfolio", "control"]].drop_duplicates().itertuples(index=False, name=None)
    pairs = list(pairs)
    fig, axis = plt.subplots(figsize=(11, 6))
    offsets = np.linspace(-.22, .22, len(pairs)) if len(pairs) > 1 else [0]
    for offset, (portfolio, control) in zip(offsets, pairs, strict=True):
        group = protection.loc[protection.portfolio.eq(portfolio) & protection.control.eq(control)]
        group = group.set_index("window").reindex(windows)
        axis.scatter(group.mean_advantage, np.arange(len(windows)) + offset,
                     color=palette[portfolio], marker="s" if control == "B" else "o",
                     label=f"{portfolio} minus {control}", s=45)
    counts = protection.drop_duplicates("window").set_index("window").reindex(windows).months
    axis.set_yticks(range(len(windows)), [f"{value.replace(':', '–')}  (n={count})"
                                          for value, count in zip(windows, counts, strict=True)])
    axis.invert_yaxis()
    axis.axvline(0, color="#666666", linewidth=1)
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.set_xlabel("Mean paired monthly return advantage in negative-SPY months")
    axis.set_title("Protection across historical periods", loc="left", fontsize=16, pad=15)
    axis.grid(axis="x", color="#eeeeee")
    axis.legend(frameon=False, loc="best")
    fig.text(.03, .015, note + "\nDescriptive comparisons; overlapping periods are not independent confirmations."
             "\nPositive values favor the first portfolio; uncertainty and the prespecified test family are reported separately when available.", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, 1))
    save(fig, "protection.png")


def write_prediction(result, study, settings, manifest, directory, config_path,
                     extra_tables=None, model_metadata=None, checkpoints=None):
    """Write a new result directory atomically; never replace prior outputs."""
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(f"Prediction output already exists: {directory}")
    model_metadata = model_metadata or {}
    parent = model_metadata.get("forecast_parent", {}).get("directory")
    if parent and directory.is_relative_to(Path(parent).resolve()):
        raise ValueError("Keep prediction outputs outside the saved forecast source.")
    tables = dict(result.tables)
    extra_tables = extra_tables or {}
    if set(tables) & set(extra_tables):
        raise ValueError("Extra tables must not replace prediction result tables.")
    tables.update(extra_tables)
    tables.update(_figure_tables(tables, result, study, settings))
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{directory.name}-", dir=directory.parent) as temporary:
        output = Path(temporary)
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=isinstance(table.index, pd.DatetimeIndex),
                         index_label="forecast_date" if name == "forecasts" else None,
                         float_format="%.17g", na_rep="NA")
        if checkpoints is not None:
            shutil.copytree(Path(checkpoints), output / "checkpoints")
        provider = manifest.get("sources", {}).get("prices", {}).get("provider", "Unspecified provider")
        _figures(output, result, provider)
        commit, dirty = git_state()
        metadata = {"created_at_utc": utc_now(),
                    "study": study.settings(), "prediction": settings.settings(),
                    "source_manifest": manifest, **result.metadata, **model_metadata,
                    "git_commit": commit, "git_dirty": dirty,
                    "config_sha256": file_digest(Path(config_path)),
                    "code_sha256": package_code_digests(),
                    "versions": {"python": platform.python_version(), **{name: version(name) for name in
                                 ("numpy", "pandas", "statsmodels", "matplotlib", "financial-economics")}},
                    "inference": {"primary_family_size": 3 if result.metadata["primary_inference"] else 0,
                                  "tests": "paired QLIKE HAC20; down/other monthly HAC6; Bartlett, small-sample correction, Student t, two-sided Holm",
                                  "bootstrap": "paired whole monthly rows; seed and blocks in prediction settings; pointwise percentile intervals conditional on fitted models"},
                    "undefined_values": "NA; never replaced with zero",
                    "figure_sources": {"forecast_accuracy.png": "figure_forecast_accuracy.csv",
                                       "growth_drawdowns.png": "figure_growth_drawdowns.csv",
                                       "protection.png": "figure_protection.csv"},
                    "files": directory_digests(output)}
        (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if directory.exists():
            raise FileExistsError(f"Prediction output already exists: {directory}")
        output.rename(directory)
