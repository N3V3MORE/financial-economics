"""Redraw the paper's Figures 3-7 from a saved U.S. reconstruction run.

Example: python -m financial_economics.paper_figures --help
Reads verified outputs only; nothing is refitted. Every PNG is drawn from its
saved figure_*.csv table, with no hidden plotting calculations.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter, PercentFormatter
import numpy as np
import pandas as pd

from .paper_evaluation import DECLINES, MODELS
from .paper_sources import ASSETS, load_sources, prepare_sources


LABELS = {"arfima": "ARFIMA", "garch": "GARCH", "gjr": "GJR-GARCH",
          "gru_univar": "GRU-UNIVAR", "gru_multivar": "GRU-MULTIVAR"}
# Figure 6 first row and Table A.2 standard deviations (2010-2020), as printed.
PUBLISHED_ASSETS = pd.DataFrame({
    "paper_correlation": (.04, .2, .2, .3, .2, .1, .09, .1, .07, -.4, -.5, -.2, -.3),
    "paper_daily_std": (.0098, .0191, .0134, .0134, .0674, .0147, .0225, .0133, .0233,
                        .0043, .0097, .0026, .0035)}, index=pd.Index(ASSETS, name="asset"))
PAPER_TEST_DAYS = 2869
BLUE, ORANGE, RED, GRAY, LIGHT, INK, MUTED = ("#2a78d6", "#eb6834", "#e34948", "#52514e",
                                              "#bdbcb6", "#0b0b0b", "#52514e")
FIGURE_MODEL = "gru_multivar"
FIGURE_STRESS = (.9, .3)  # Figure 7's shown scenario: 90th percentile, 30% cap


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _eligible(forecasts, returns):
    return forecasts.loc[forecasts.return_date.isin(returns.index)]


def forecast_paths(forecasts, settings, models=MODELS):
    """Figure 3: realized and predicted monthly volatility on each return date."""
    multiplier = settings["monthly_multiplier"]
    table = pd.DataFrame({"return_date": forecasts.return_date.to_numpy(),
                          "realized": multiplier * np.sqrt(forecasts.realized_bv.to_numpy())})
    for model in models:
        table[model] = multiplier * np.sqrt(forecasts[f"{model}_bv"].to_numpy())
    return table


def strategy_path(daily_returns, weights, model=FIGURE_MODEL):
    """Figure 4: growth of $1 in the index and the clean strategy, plus equity weight."""
    stock = weights.loc[weights.model.eq(model)].set_index("return_date").stock_target
    table = pd.DataFrame({"index_value": (1 + daily_returns.EQUITY).cumprod(),
                          "strategy_value": (1 + daily_returns[f"{model}_clean"]).cumprod(),
                          "stock_weight": stock.reindex(daily_returns.index).ffill()})
    return table.rename_axis("Date").reset_index()


def quantile_profile(forecasts, returns, settings, models=MODELS, bins=100):
    """Figure 5: next-day index return statistics within each predicted-volatility percentile."""
    eligible = _eligible(forecasts, returns)
    following = returns.EQUITY.reindex(eligible.return_date).to_numpy()
    rows = []
    for model in models:
        volatility = pd.Series(settings["monthly_multiplier"] * np.sqrt(eligible[f"{model}_bv"].to_numpy()))
        # Ranking first keeps floored, tied forecasts from collapsing percentile edges.
        bucket = pd.qcut(volatility.rank(method="first"), bins, labels=False) + 1
        stats = pd.Series(following).groupby(bucket.to_numpy()).agg(["min", "max", "median", "std", "count"])
        rows.append(stats.rename_axis("percentile").reset_index().assign(model=model))
    return pd.concat(rows, ignore_index=True)


def asset_comparison(returns):
    """Figure 6 and its check against the paper's printed correlations and volatilities."""
    columns = ["EQUITY", *ASSETS]
    correlation = returns[columns].corr()
    check = PUBLISHED_ASSETS.copy()
    check["our_correlation"] = correlation.loc[list(ASSETS), "EQUITY"]
    check["our_daily_std"] = returns[list(ASSETS)].std()
    check["our_min"], check["our_max"] = returns[list(ASSETS)].min(), returns[list(ASSETS)].max()
    return correlation.rename_axis("asset").reset_index(), check.reset_index()


def drawdown_paths(daily_returns, weights, model=FIGURE_MODEL, stress=FIGURE_STRESS):
    """Figure 7: cumulative returns and weights from each published peak to its trough."""
    quantile, cap = stress
    chosen = weights.loc[weights.model.eq(model)].set_index("return_date")
    chosen = chosen.reindex(daily_returns.index).ffill()
    rows = []
    for label, peak, trough in DECLINES:
        dates = daily_returns.loc[peak:trough].index
        inside = daily_returns.loc[dates[1:]]
        rows.append(pd.DataFrame({
            "decline": label, "day": np.arange(len(dates)), "Date": dates,
            "index_cumulative": np.r_[0., (1 + inside.EQUITY).cumprod() - 1],
            "strategy_cumulative": np.r_[0., (1 + inside[f"{model}_clean"]).cumprod() - 1],
            "stock_weight": chosen.loc[dates, "stock_target"].to_numpy(),
            "defensive_weight": chosen.loc[dates, f"defensive_q{quantile:g}_cap{cap:g}"].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def figure_tables(forecasts, returns, daily_returns, weights, settings):
    eligible = _eligible(forecasts, returns)
    correlation, check = asset_comparison(returns)
    return {"figure3_forecasts": forecast_paths(eligible, settings),
            "figure4_strategy": strategy_path(daily_returns, weights),
            "figure5_quantiles": quantile_profile(forecasts, returns, settings),
            "figure6_correlation": correlation, "figure6_paper_check": check,
            "figure7_drawdowns": drawdown_paths(daily_returns, weights)}


def _figures(output, note, local_bonds=False):
    bonds = "Swiss/German in CHF/EUR" if local_bonds else "Swiss/German in USD"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": LIGHT})

    def save(fig, name, text):
        fig.text(.02, .005, note + "\n" + text, fontsize=7.5, color=MUTED, va="bottom")
        fig.savefig(output / name, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def grid(axis, which="y"):
        axis.grid(axis=which, color="#eeeeec", linewidth=.8)

    paths = pd.read_csv(output / "figure3_forecasts.csv", parse_dates=["return_date"])
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(11, 12), sharex=True, sharey=True)
    for axis, model in zip(axes, MODELS):
        axis.plot(paths.return_date, paths.realized, color=LIGHT, linewidth=.8, label="Realized (Oxford-Man BV)")
        axis.plot(paths.return_date, paths[model], color=BLUE, linewidth=.9, label=f"Predicted ({LABELS[model]})")
        axis.set_title(LABELS[model], loc="left", fontsize=11)
        axis.set_ylabel("Monthly volatility")
        grid(axis)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Figure 3 (ours): predicted vs realized S&P 500 volatility, 2010–2020",
                 x=.02, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, .04, 1, .97))
    save(fig, "figure3_forecasts.png", "Monthly volatility = 22 × √(daily bipower variation), the paper's Equation 24."
         " Raw GRU-MULTIVAR forecasts below zero are floored, so they show as near-zero spikes in 2017–2018.")

    path = pd.read_csv(output / "figure4_strategy.csv", parse_dates=["Date"])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, height_ratios=(2, 1))
    top.plot(path.Date, path.index_value, color=GRAY, linewidth=1.4, label="S&P 500 total return")
    top.plot(path.Date, path.strategy_value, color=BLUE, linewidth=1.4, label="Target-volatility strategy")
    for column, color in (("index_value", GRAY), ("strategy_value", BLUE)):
        top.annotate(f"{path[column].iloc[-1]:.2f}", (path.Date.iloc[-1], path[column].iloc[-1]),
                     xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    top.set_ylabel("Value of $1")
    top.legend(frameon=False, loc="upper left")
    bottom.plot(path.Date, path.stock_weight, color=BLUE, linewidth=.6)
    bottom.set_ylim(0, 1.05)
    bottom.yaxis.set_major_formatter(PercentFormatter(1))
    bottom.set_ylabel("Stock weight")
    for axis in (top, bottom):
        grid(axis)
    fig.suptitle(f"Figure 4 (ours): {LABELS[FIGURE_MODEL]} strategy, 10% monthly volatility target",
                 x=.02, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, .05, 1, .96))
    save(fig, "figure4_strategy.png", "Stock weight = min(10% / predicted monthly volatility, 100%); the rest earns zero cash,"
         " no fees and same-close trading, as in the paper.")

    profile = pd.read_csv(output / "figure5_quantiles.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for axis, model in zip(axes.flat, MODELS):
        group = profile.loc[profile.model.eq(model)]
        for column, color, width, label in (("min", LIGHT, .8, "Lowest / highest return"),
                                            ("max", LIGHT, .8, "_nolegend_"), ("median", GRAY, .8, "Median return"),
                                            ("std", BLUE, 2, "Standard deviation")):
            axis.plot(group.percentile, group[column], color=color, linewidth=width, label=label)
        axis.axvline(90, color=ORANGE, linewidth=1, linestyle="--", label="90th percentile (stress line)")
        axis.set_title(LABELS[model], loc="left", fontsize=11)
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        grid(axis)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), frameon=False, loc="center")
    for axis in axes[-1]:
        axis.set_xlabel("Predicted-volatility percentile")
    for axis in axes[:, 0]:
        axis.set_ylabel("Next-day S&P 500 return")
    fig.suptitle("Figure 5 (ours): index returns by predicted-volatility percentile, 2010–2020",
                 x=.02, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, .05, 1, .96))
    save(fig, "figure5_quantiles.png", "Each model's forecasts are split into 100 equal-count groups (about 28 days each);"
         " lines show the next day's index returns within each group.")

    correlation = pd.read_csv(output / "figure6_correlation.csv", index_col="asset")
    names = ["S&P 500" if name == "EQUITY" else name for name in correlation.index]
    values = correlation.to_numpy()
    # Upper triangle as in the paper, so the S&P 500 row comes first; the empty row and column are dropped.
    shown = np.where(np.triu(np.ones_like(values, dtype=bool), 1), values, np.nan)[:-1, 1:]
    colormap = LinearSegmentedColormap.from_list("diverging", [BLUE, "#f0efec", RED])
    fig, axis = plt.subplots(figsize=(10, 8.5))
    image = axis.imshow(shown, cmap=colormap, vmin=-1, vmax=1)
    for row, column in zip(*np.where(~np.isnan(shown))):
        axis.text(column, row, f"{shown[row, column]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    axis.set_xticks(range(len(names) - 1), names[1:], rotation=60, ha="right")
    axis.set_yticks(range(len(names) - 1), names[:-1])
    axis.spines[:].set_visible(False)
    axis.tick_params(length=0)
    fig.colorbar(image, ax=axis, shrink=.7, label="Correlation of daily returns")
    axis.set_title("Figure 6 (ours): correlation of daily returns, 2010–2020", loc="left", fontsize=14)
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, "figure6_correlation.png", "Public proxies: futures for commodities, Brent spot for oil, raw bond-ETF prices"
         f" ({bonds}). The paper used Datastream spot and benchmark bond indices.")

    check = pd.read_csv(output / "figure6_paper_check.csv").iloc[::-1]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
    rows = np.arange(len(check))
    for axis, ours, paper, label in ((left, "our_correlation", "paper_correlation", "Correlation with S&P 500"),
                                     (right, "our_daily_std", "paper_daily_std", "Daily return volatility (log scale)")):
        axis.hlines(rows, check[ours], check[paper], color=LIGHT, linewidth=2)
        axis.scatter(check[paper], rows, color=ORANGE, marker="D", s=45, label="Paper (Datastream)", zorder=3)
        axis.scatter(check[ours], rows, color=BLUE, s=55, label="Ours (public proxies)", zorder=3)
        axis.set_xlabel(label)
        grid(axis, "x")
    left.axvline(0, color=GRAY, linewidth=1)
    left.set_yticks(rows, check.asset)
    left.legend(frameon=False, loc="lower right")
    right.set_xscale("log")
    right.set_xticks([.002, .005, .01, .02, .05])
    right.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    right.xaxis.set_minor_formatter(NullFormatter())
    position = dict(zip(check.asset, rows))
    oil = check.set_index("asset").loc["OIL"]
    right.annotate("Paper's WTI series includes the\n−306% day when oil went negative",
                   (oil.paper_daily_std, position["OIL"]), xytext=(-10, 16), textcoords="offset points",
                   ha="right", fontsize=8, color=INK)
    right.annotate("Ours now in local currency, as the\npaper's statistics imply" if local_bonds else
                   "Ours are converted to US dollars;\nthe paper's match local-currency returns",
                   (check.set_index("asset").loc["SWISS10", "our_daily_std"], position["SWISS10"]),
                   xytext=(14, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color=INK)
    fig.suptitle("Do our proxies behave like the paper's data? (2010–2020)", x=.02, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, .05, 1, .95))
    save(fig, "figure6_paper_check.png", "Paper values are read from its Figure 6 (one decimal) and Table A.2."
         f" Paper test sample: {PAPER_TEST_DAYS} weekdays; ours uses observed S&P 500 trading days only.")

    drawdowns = pd.read_csv(output / "figure7_drawdowns.csv", parse_dates=["Date"])
    quantile, cap = FIGURE_STRESS
    fig, axes = plt.subplots(len(DECLINES), 2, figsize=(12, 14))
    for (left, right), (label, peak, trough) in zip(axes, DECLINES):
        group = drawdowns.loc[drawdowns.decline.eq(label)]
        left.plot(group.day, group.index_cumulative, color=GRAY, linewidth=1.5, label="S&P 500 total return")
        left.plot(group.day, group.strategy_cumulative, color=BLUE, linewidth=1.5, label="Strategy, no safe haven")
        left.yaxis.set_major_formatter(PercentFormatter(1))
        left.set_title(f"{peak} to {trough}", loc="left", fontsize=11)
        right.step(group.day, group.stock_weight, where="post", color=BLUE, linewidth=1.5, label="Stock weight")
        right.step(group.day, group.defensive_weight, where="post", color=ORANGE, linewidth=1.5,
                   label="Safe-haven weight")
        right.set_ylim(0, 1.05)
        right.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        for axis in (left, right):
            grid(axis)
    axes[0, 0].legend(frameon=False, loc="lower left")
    axes[0, 1].legend(frameon=False, loc="upper right")
    for axis in axes[-1]:
        axis.set_xlabel("Trading days since the index peak")
    axes[0, 0].set_ylabel("Cumulative return")
    axes[0, 1].set_ylabel("Portfolio weight")
    fig.suptitle(f"Figure 7 (ours): the five worst S&P 500 falls, {LABELS[FIGURE_MODEL]} strategy",
                 x=.02, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, .03, 1, .97))
    save(fig, "figure7_drawdowns.png", f"Weights show the {quantile:.0%} stress line and {cap:.0%} safe-haven cap,"
         " the scenario the paper shows. The weight is the same whichever safe-haven asset is held.")


def run(args):
    local_bonds = getattr(args, "local_bonds", False)
    evaluation = args.run_dir / ("evaluation_local" if local_bonds else "evaluation")
    metadata = json.loads((evaluation / "run.json").read_text(encoding="utf-8"))
    inputs = metadata["inputs"]
    if (_digest(args.data_dir / "manifest.json") != inputs["source_manifest"]
            or _digest(args.oxford_archive) != inputs["oxford_archive"]
            or (not local_bonds and (args.fx_dir is None or _digest(args.fx_dir / "manifest.json")
                                     != metadata["evaluation_fx"]["manifest_sha256"]))):
        raise ValueError("Figure inputs must be the exact snapshots used by the evaluation run.")
    used = ("forecasts.csv", "daily_returns.csv", "model_target_weights.csv")
    for name in used:
        if _digest(evaluation / name) != metadata["files"][name]:
            raise ValueError(f"Saved evaluation file changed after its run: {name}.")
    destination = args.output_dir.resolve()
    if destination.exists():
        raise ValueError(f"Existing results are never overwritten: {destination}")
    from .paper_fx import currency_returns, local_currency_returns

    observations, _ = load_sources(args.data_dir)
    returns = prepare_sources(observations, args.oxford_archive, metadata["settings"]["lookback"])[4]
    returns, _ = (local_currency_returns(observations, returns) if local_bonds else
                  currency_returns(observations, returns, load_sources(args.fx_dir)[0]))
    settings = metadata["settings"]
    returns = returns.loc[settings["evaluation_start"]:settings["evaluation_end"]]
    forecasts = pd.read_csv(evaluation / "forecasts.csv", index_col="forecast_date",
                            parse_dates=["forecast_date", "return_date"])
    daily_returns = pd.read_csv(evaluation / "daily_returns.csv", index_col="Date", parse_dates=True,
                                float_precision="round_trip")
    weights = pd.read_csv(evaluation / "model_target_weights.csv", index_col=0, parse_dates=["return_date"])
    if (not daily_returns.index.equals(returns.index)
            or not np.allclose(daily_returns.EQUITY, returns.EQUITY, rtol=0, atol=1e-12)):
        raise ValueError("Rebuilt index returns differ from the evaluation run.")
    tables = figure_tables(forecasts, returns, daily_returns, weights, settings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".paper-figures-", dir=destination.parent) as temporary:
        output = Path(temporary) / "run"
        output.mkdir()
        for name, frame in tables.items():
            frame.to_csv(output / f"{name}.csv", index=False)
        _figures(output, f"Public-data reconstruction of Kaczmarek et al. (2022) | {args.run_dir.name}/{evaluation.name}"
                 " | descriptive", local_bonds)
        record = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "study": "Paper-style figures from the saved public-data U.S. reconstruction; nothing refitted",
                  "evaluation_run": {"path": str(evaluation), "manifest_sha256": _digest(evaluation / "run.json"),
                                     "files": {name: metadata["files"][name] for name in used}},
                  "inputs": inputs | ({"bond_currency": "local CHF/EUR, post hoc"} if local_bonds else
                                      {"fx_manifest": metadata["evaluation_fx"]["manifest_sha256"]}),
                  "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                  "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "code": {"paper_figures.py": _digest(__file__)},
                  "notes": {"figure3": "22 * sqrt(BV) monthly scaling; raw-BV forecasts, floors included",
                            "figure4_and_7": f"{FIGURE_MODEL} clean strategy; zero fees, zero cash income, same-close trading",
                            "figure5": "100 equal-count forecast groups; next-day S&P 500 total returns",
                            "figure6": "Pearson correlation of daily USD returns on S&P 500 trading days",
                            "figure6_paper_check": "paper correlations read from Figure 6 at printed precision; std from Table A.2",
                            "figure7": f"published decline windows; weights at {FIGURE_STRESS[0]:g} quantile, {FIGURE_STRESS[1]:g} cap"}}
        record["files"] = {path.name: _digest(path) for path in sorted(output.iterdir())}
        (output / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        output.rename(destination)
    print(f"Saved paper figures: {destination}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Reconstruction output with an evaluation stage")
    parser.add_argument("--data-dir", type=Path, required=True, help="Public snapshot used by that run")
    parser.add_argument("--fx-dir", type=Path, help="H.10 FX snapshot used by that run; not needed with --local-bonds")
    parser.add_argument("--oxford-archive", type=Path, required=True)
    parser.add_argument("--local-bonds", action="store_true",
                        help="Draw from evaluation_local, with Swiss/German bonds in local currency")
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
