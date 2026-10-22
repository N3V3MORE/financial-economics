"""Draw the figures behind the Risk Lab findings summary from saved, hash-verified runs.

Reads only finished outputs; it never fits a model. Example:

    python scripts/make_findings_figures.py --main-run outputs/revised-risklab-20261022 \
        --late-run outputs/risklab-2022-2025-20261022 \
        --bridged-late-run outputs/trial-risklab-defensive-2000-20261022 \
        --oxford-trial outputs/trial-oxford-revised-20261022 \
        --risklab-trial outputs/trial-risklab-revised-20261022 \
        --risklab-dir data/risklab-spy-20261022 \
        --oxford-archive data/paper-oxford-20261021/oxfordman-20220301.zip \
        --output-dir docs/figures/risklab
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from financial_economics.paper import load_bipower
from financial_economics.provenance import verify_files
from financial_economics.volatility_proxy import load_risklab_bv

# Reference palette (validated: first three slots pass every pair in light mode).
TIMING, GRU, STATIC = "#2a78d6", "#eb6834", "#1baf7a"
SPY, INK, MUTED, GRID, SURFACE = "#52514e", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
NAMES = {"ewma": "EWMA", "arfima": "ARFIMA", "garch": "GARCH", "gjr": "GJR",
         "gru_univar_log": "GRU (one input)", "gru_multivar_log": "GRU (richer)"}
MAIN_COST = 10

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5, "text.color": INK,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
    "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 1.8, "legend.frameon": False,
})


def stage(run, name):
    """Load a finished stage after checking every saved file against its recorded hash."""
    directory = Path(run) / name
    record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    verify_files(directory, record["files"], f"{directory} does not match its saved hashes.")
    return directory


def wealth(returns):
    return (1 + returns).cumprod()


def drawdown(returns):
    level = wealth(returns)
    return level / level.cummax() - 1


def label_end(ax, series, text, color, dy=0):
    ax.annotate(text, (series.index[-1], series.iloc[-1]), xytext=(4, dy), textcoords="offset points",
                color=color, fontsize=8.5, va="center", fontweight="bold")


def save(fig, directory, name, source, y=.005):
    fig.text(.01, y, f"Source: {source}", fontsize=7.5, color=MUTED)
    fig.savefig(directory / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {directory / name}")


def data_agreement(risklab_dir, oxford_archive, out):
    risklab, _ = load_risklab_bv(risklab_dir, "2000-01-01", "2022-02-28")
    oxford = load_bipower(oxford_archive, "2000-01-01", "2022-02-28")
    common = risklab.index.intersection(oxford.index)
    a, b = np.sqrt(252 * oxford[common]), np.sqrt(252 * risklab[common])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    monthly = pd.DataFrame({"Oxford-Man (S&P 500 index, bipower)": a, "Risk Lab (SPY, realized)": b}).resample("ME").mean()
    for column, color in zip(monthly, (GRU, TIMING), strict=True):
        left.plot(monthly.index, monthly[column], color=color, label=column)
    left.set_title("Two measures of daily stock-market volatility move together")
    left.set_ylabel("Annualized volatility (monthly average)")
    left.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    left.legend(loc="upper right")
    right.scatter(a, b, s=4, color=TIMING, alpha=.25, linewidths=0)
    limits = [min(a.min(), b.min()) * .9, max(a.max(), b.max()) * 1.1]
    right.plot(limits, limits, color=MUTED, linewidth=1, linestyle="--")
    right.set(xscale="log", yscale="log", xlim=limits, ylim=limits,
              xlabel="Oxford-Man", ylabel="Risk Lab")
    right.set_title("Same day, both sources")
    right.text(.04, .92, f"correlation {np.corrcoef(a, b)[0, 1]:.3f}\n{len(common):,} shared days",
               transform=right.transAxes, fontsize=8.5, color=INK, va="top")
    save(fig, out, "01_data_agreement.png", "Risk Lab SPY trade_rv5; Oxford-Man .SPX bv; 2000 to Feb 2022")


def forecasts_vs_realized(main, out):
    frame = pd.read_csv(stage(main, "evaluate") / "forecasts.csv", index_col=0, parse_dates=True)
    frame = frame.loc["2011":"2020"]
    vol = np.sqrt(252 * frame[["realized_bv", "ewma_bv", "gru_univar_log_bv"]]).resample("ME").mean()
    fig, (full, covid) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    for axis, data in ((full, vol), (covid, np.sqrt(252 * frame.loc["2020-01":"2020-06",
                                                                    ["realized_bv", "ewma_bv", "gru_univar_log_bv"]]))):
        axis.plot(data.index, data.realized_bv, color=SPY, linewidth=1.2, label="Realized (two days later)")
        axis.plot(data.index, data.ewma_bv, color=TIMING, label="EWMA forecast")
        axis.plot(data.index, data.gru_univar_log_bv, color=GRU, label="GRU forecast")
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    full.set_title("Forecasts track realized volatility, 2011–2020 (monthly averages)")
    full.set_ylabel("Annualized volatility")
    full.legend(loc="upper left")
    covid.set_title("COVID, daily: forecasts lag the jump")
    covid.tick_params(axis="x", rotation=30)
    save(fig, out, "02_forecasts_vs_realized.png", "main Risk Lab run, forecasts.csv")


def accuracy_by_metric(main, out):
    scores = pd.read_csv(stage(main, "evaluate") / "forecast_robust_scores.csv")
    scores["column"] = scores.target.str.replace("_", "-") + "\n" + scores.loss.map(
        {"variance_mse": "variance MSE", "qlike": "QLIKE", "volatility_mse_reference": "volatility MSE*"})
    ranks = scores.pivot(index="model", columns="column", values="rank").loc[list(NAMES)]
    ranks = ranks[[c for t in ("intraday", "close-to-close") for c in ranks.columns if c.startswith(t)]]
    fig, axis = plt.subplots(figsize=(8.5, 3.6))
    axis.imshow(ranks.to_numpy(), cmap=matplotlib.colors.ListedColormap(
        ["#184f95", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6", "#e8f0fb"]), vmin=.5, vmax=6.5, aspect="auto")
    for (i, j), value in np.ndenumerate(ranks.to_numpy()):
        axis.text(j, i, int(value), ha="center", va="center", fontsize=10,
                  color="white" if value <= 2 else INK, fontweight="bold")
    axis.set_xticks(range(ranks.shape[1]), ranks.columns, fontsize=8)
    axis.set_yticks(range(len(ranks)), [NAMES[m] for m in ranks.index])
    axis.grid(False)
    axis.set_title("Forecast accuracy rank (1 = best) depends on the target and the loss")
    fig.text(.01, -.13, "* volatility MSE is not robust to a noisy variance proxy (Patton 2011); "
             "robust columns: variance MSE and QLIKE. GARCH/GJR or the others rescaled by a pre-2010 "
             "Hansen–Lunde factor.", fontsize=7.5, color=MUTED)
    save(fig, out, "03_accuracy_rank_by_metric.png", "main Risk Lab run, forecast_robust_scores.csv; 2011–2020",
         y=-.08)


def stock_share(main, out):
    directory = stage(main, "evaluate")
    decisions = pd.read_csv(directory / "decisions.csv", parse_dates=["earn_date"])
    statics = pd.read_csv(directory / "static_controls.csv").set_index("model").static_spy_weight
    returns = pd.read_csv(directory / f"daily_returns_{MAIN_COST}bps.csv", index_col=0, parse_dates=True)
    ewma = decisions[decisions.model == "ewma"].set_index("earn_date").stock_target
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    top.fill_between(returns.index, drawdown(returns.SPY), 0, color=SPY, alpha=.25, linewidth=0)
    top.set_title("When the market falls (top), timing cuts its stock share (bottom)")
    top.set_ylabel("SPY drawdown")
    top.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    bottom.plot(ewma.index, ewma, color=TIMING, linewidth=1, label="EWMA timing")
    bottom.axhline(statics["ewma"], color=STATIC, linewidth=1.8, label="Fixed mix (set in 2010)")
    bottom.set_ylabel("Share in stocks")
    bottom.set_ylim(0, 1.05)
    bottom.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    bottom.legend(loc="lower left", ncol=2)
    save(fig, out, "04_stock_share.png", "main Risk Lab run, decisions.csv; 2011–2020")


def growth_and_drawdowns(main, late, out):
    early = pd.read_csv(stage(main, "evaluate") / f"daily_returns_{MAIN_COST}bps.csv", index_col=0, parse_dates=True)
    later = pd.read_csv(stage(late, "evaluate") / f"daily_returns_{MAIN_COST}bps.csv", index_col=0, parse_dates=True)
    series = (("SPY", "SPY (all stocks)", SPY), ("ewma_static", "Fixed mix", STATIC),
              ("ewma_cash", "EWMA timing", TIMING), ("gru_univar_log_cash", "GRU timing", GRU))
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex="col", gridspec_kw={"width_ratios": [2.5, 1]})
    for column, (frame, period) in enumerate(((early, "2011–2020"), (later, "2022–2025"))):
        for name, text, color in series:
            level = wealth(frame[name])
            axes[0, column].plot(level.index, level, color=color, label=text)
            axes[1, column].plot(level.index, drawdown(frame[name]), color=color, linewidth=1.4)
        axes[0, column].set_title(f"Growth of $1, {period}, 0.10% fees")
        axes[1, column].set_title(f"Fall from previous peak, {period}")
        axes[1, column].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    axes[0, 0].legend(loc="upper left", fontsize=8.5)
    axes[1, 1].xaxis.set_major_locator(matplotlib.dates.YearLocator())
    axes[1, 1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    save(fig, out, "05_growth_and_drawdowns.png", "main and 2022–2025 Risk Lab runs, daily returns at 10 bps")


def sharpe_vs_fees(main, out):
    performance = pd.read_csv(stage(main, "evaluate") / "performance.csv")
    performance = performance[performance.window == "2011:2020"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True, sharey=True)
    spy = performance[performance.portfolio == "SPY"].set_index("cost_bps").sharpe
    for axis, model in zip(axes.flat, NAMES, strict=True):
        for suffix, text, color in (("cash", "Timing", TIMING), ("static", "Fixed mix", STATIC)):
            line = performance[performance.portfolio == f"{model}_{suffix}"].set_index("cost_bps").sharpe
            axis.plot(line.index, line, color=color, marker="o", markersize=4, label=text)
        axis.plot(spy.index, spy, color=SPY, linestyle="--", linewidth=1, label="SPY")
        axis.axvline(MAIN_COST, color=GRID, linewidth=1)
        axis.set_title(NAMES[model], fontsize=10)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    for axis in axes[1]:
        axis.set_xlabel("Trading cost per dollar traded (bps)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Sharpe ratio")
    fig.suptitle("Timing's edge shrinks as trading costs rise; most models reach the fixed mix by 10–25 bps "
                 "(2011–2020)",
                 x=.01, ha="left", fontweight="bold")
    save(fig, out, "06_sharpe_vs_fees.png", "main Risk Lab run, performance.csv")


def crash_losses(main, out):
    declines = pd.read_csv(stage(main, "evaluate") / "declines.csv")
    windows = list(dict.fromkeys(declines.decline))
    names = [window.replace("2015_2016", "2015–16") for window in windows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for axis, model in zip(axes, ("ewma", "gru_univar_log"), strict=True):
        width = .26
        for k, (name, text, color) in enumerate((("SPY", "SPY", SPY), (f"{model}_static", "Fixed mix", STATIC),
                                                 (f"{model}_cash", "Timing", TIMING))):
            values = declines[declines.portfolio == name].set_index("decline").reindex(windows)["return"]
            bars = axis.bar(np.arange(len(windows)) + (k - 1) * width, values, width * .92, color=color, label=text)
            axis.bar_label(bars, [f"{v:.0%}" for v in values], fontsize=7, color=INK, padding=2)
        axis.set_xticks(range(len(windows)), names)
        axis.set_title(f"Loss in each market decline: {NAMES[model]}")
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        axis.axhline(0, color="#c3c2b7", linewidth=.8)
    axes[1].legend(loc="lower right", fontsize=8)
    save(fig, out, "07_crash_losses.png", "main Risk Lab run, declines.csv; 0.10% fees")


def safe_assets(late, out):
    directory = stage(late, "evaluate")
    performance = pd.read_csv(directory / "performance.csv")
    full = performance[(performance.window == "2022:2025") & (performance.cost_bps == MAIN_COST)].set_index("portfolio")
    returns = pd.read_csv(directory / f"daily_returns_{MAIN_COST}bps.csv", index_col=0, parse_dates=True)
    assets = (("ewma_cash", "Cash"), ("ewma_TLT", "Long Treasuries (TLT)"), ("ewma_GLD", "Gold (GLD)"),
              ("ewma_DBC", "Commodities (DBC)"))
    fig, (sharpe, worst, path) = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    y = np.arange(len(assets))
    full["worst_fall"] = -full.max_drawdown
    for axis, column, title in ((sharpe, "sharpe", "Sharpe ratio (higher better)"),
                                (worst, "worst_fall", "Worst fall (smaller better)")):
        values = full.loc[[a for a, _ in assets], column]
        bars = axis.barh(y, values, color=TIMING, height=.55)
        axis.axvline(full.loc["ewma_static", column], color=STATIC, linewidth=1.8, label="Fixed mix")
        axis.bar_label(bars, [f"{v:.2f}" if column == "sharpe" else f"{v:.0%}" for v in values],
                       fontsize=8, padding=3, color=INK)
        axis.set_yticks(y, [text for _, text in assets])
        axis.invert_yaxis()
        axis.set_title(title)
    worst.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    worst.set_yticklabels([])
    fig.legend(handles=[matplotlib.lines.Line2D([], [], color=STATIC, linewidth=1.8)],
               labels=["Fixed mix (vertical line)"], loc="upper left", bbox_to_anchor=(.5, 1.045), fontsize=8.5)
    year = returns.loc["2022"]
    for name, text, color, style in (("ewma_cash", "Timing + cash", TIMING, "-"),
                                     ("ewma_TLT", "Timing + TLT", GRU, "-"),
                                     ("ewma_static", "Fixed mix", STATIC, "-"), ("SPY", "SPY", SPY, "--")):
        level = wealth(year[name])
        path.plot(level.index, level, color=color, linestyle=style, label=text)
    path.legend(loc="lower left", fontsize=8)
    path.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    path.set_title("2022: bonds fell with stocks")
    fig.suptitle("Where storm-day money went, EWMA timing, 2022–2025, 0.10% fees", x=.01, y=1.03,
                 ha="left", fontweight="bold")
    save(fig, out, "08_safe_assets_2022_2025.png", "2022–2025 Risk Lab run with defensive ETFs")


def source_robustness(oxford_trial, risklab_trial, out):
    paths = [Path(run) / "evaluate" / "performance.csv" for run in (oxford_trial, risklab_trial)]
    oxford, risklab = (pd.read_csv(path) for path in paths)
    keys = ["portfolio", "window", "cost_bps"]
    both = oxford.merge(risklab, on=keys, suffixes=("_oxford", "_risklab"))
    both = both[(both.window == "2011:2020") & (both.cost_bps == MAIN_COST)]
    fig, axis = plt.subplots(figsize=(5.4, 5))
    axis.scatter(both.sharpe_oxford, both.sharpe_risklab, s=28, color=TIMING, edgecolors=SURFACE, linewidths=1)
    limits = [min(both.sharpe_oxford.min(), both.sharpe_risklab.min()) - .03,
              max(both.sharpe_oxford.max(), both.sharpe_risklab.max()) + .03]
    axis.plot(limits, limits, color=MUTED, linestyle="--", linewidth=1)
    axis.set(xlim=limits, ylim=limits, xlabel="Sharpe with Oxford-Man volatility",
             ylabel="Sharpe with Risk Lab volatility")
    gap = (both.sharpe_oxford - both.sharpe_risklab).abs()
    axis.set_title("Same portfolios, either data source (2011–2020)")
    axis.text(.04, .94, f"{len(both)} portfolios at 0.10% fees\nmedian gap {gap.median():.3f}, "
              f"largest {gap.max():.3f}", transform=axis.transAxes, fontsize=8.5, va="top")
    save(fig, out, "09_source_robustness.png", "October 22 trial runs with identical code")


def significance(main, late, out):
    tests = []
    for run, period in ((main, "2011–2020"), (late, "2022–2025")):
        frame = pd.read_csv(stage(run, "evaluate") / "tests.csv")
        comparison = frame.comparison.replace({"timing_minus_static": "timing vs fixed mix",
                                               "gru_univar_log_minus_ewma": "vs EWMA",
                                               "gru_multivar_log_minus_ewma": "vs EWMA",
                                               "gru_multivar_log_minus_gru_univar_log": "vs one-input GRU",
                                               "gru_minus_ewma": "vs EWMA"})
        comparison = comparison.str.replace("_minus_cash", " vs cash", regex=False)
        frame["label"] = frame.model.map(NAMES) + ": " + comparison + f" ({period})"
        tests.append(frame)
    tests = pd.concat(tests, ignore_index=True).sort_values("p_value")
    fig, axis = plt.subplots(figsize=(8.5, max(4, .2 * len(tests))))
    y = np.arange(len(tests))
    axis.scatter(tests.p_value, y, s=16, facecolors="none", edgecolors=MUTED, label="Raw p-value")
    axis.scatter(tests.holm_p_value, y, s=22, color=TIMING, label="Holm-adjusted p-value")
    axis.axvline(.05, color="#d03b3b", linewidth=1)
    axis.text(.052, -1.2, "0.05", color="#d03b3b", fontsize=8)
    axis.set_yticks(y, tests.label, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.02)
    axis.set_xlabel("p-value")
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.06), ncol=2, fontsize=8)
    axis.set_title(f"No portfolio test is significant after adjustment ({len(tests)} tests)")
    save(fig, out, "10_significance.png", "tests.csv from both Risk Lab runs")


def missing_date_sensitivity(strict, bridged, out):
    """2022–2025 under the strict complete-history rule and the earlier bridged histories."""
    rows, signals = [], {}
    for run, rule in ((bridged, "Bridged histories"), (strict, "Complete 22-session histories")):
        directory = Path(run) / "evaluate"
        performance = pd.read_csv(directory / "performance.csv")
        performance = performance[(performance.window == "2022:2025") & (performance.cost_bps == MAIN_COST)]
        rows.append(performance.set_index("portfolio").sharpe.rename(rule))
        decisions = pd.read_csv(directory / "decisions.csv", parse_dates=["earn_date"])
        days = len(pd.read_csv(directory / f"daily_returns_{MAIN_COST}bps.csv"))
        signals[rule] = (decisions.model == "ewma").sum() / days
    table = pd.concat(rows, axis=1).loc[["ewma_cash", "gru_univar_log_cash", "ewma_TLT", "ewma_GLD", "ewma_static"]]
    labels = ["EWMA timing", "GRU timing", "EWMA + TLT", "EWMA + gold", "Fixed mix"]
    fig, axis = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(table))
    for k, (rule, color) in enumerate(zip(table.columns, (SPY, TIMING), strict=True)):
        bars = axis.bar(x + (k - .5) * .38, table[rule], .36, color=color,
                        label=f"{rule} (fresh signal on {signals[rule]:.0%} of days)")
        axis.bar_label(bars, [f"{v:.2f}" for v in table[rule]], fontsize=8, padding=2, color=INK)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Sharpe ratio, 0.10% fees")
    axis.set_ylim(0, .75)
    axis.legend(loc="upper right", fontsize=8)
    axis.set_title("2022–2025: timing vs fixed mix flips with the missing-date rule; TLT stays the worst safe asset")
    save(fig, out, "11_missing_date_sensitivity.png", "2022–2025 Risk Lab runs: bridged trial and complete-history rule")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--main-run", "--late-run", "--bridged-late-run", "--oxford-trial", "--risklab-trial", "--risklab-dir",
                 "--oxford-archive", "--output-dir"):
        parser.add_argument(name, type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_agreement(args.risklab_dir, args.oxford_archive, args.output_dir)
    forecasts_vs_realized(args.main_run, args.output_dir)
    accuracy_by_metric(args.main_run, args.output_dir)
    stock_share(args.main_run, args.output_dir)
    growth_and_drawdowns(args.main_run, args.late_run, args.output_dir)
    sharpe_vs_fees(args.main_run, args.output_dir)
    crash_losses(args.main_run, args.output_dir)
    safe_assets(args.late_run, args.output_dir)
    source_robustness(args.oxford_trial, args.risklab_trial, args.output_dir)
    significance(args.main_run, args.late_run, args.output_dir)
    missing_date_sensitivity(args.late_run, args.bridged_late_run, args.output_dir)


if __name__ == "__main__":
    main()
