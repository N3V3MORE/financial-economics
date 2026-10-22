"""Save the chronological SPY/cash comparison and its calibration evidence."""

import json
from importlib.metadata import version
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd

from .analysis import drawdowns
from .comparison import ComparisonResult
from .config import Study
from .data import price_source_description
from .report import _save_figure
from .stress_report import _date


def _number(value, spec=".3f") -> str:
    return "undefined" if pd.isna(value) else format(value, spec)


def _figures(result: ComparisonResult, episodes: pd.DataFrame,
             comparisons: pd.DataFrame, output: Path) -> None:
    returns = result.evaluation_returns
    calibration = result.calibration
    benchmark, control = calibration["benchmark"], calibration["control_name"]
    names = [benchmark, "Equal weight", control]
    labels = {benchmark: benchmark, "Equal weight": "Fund mix", control: control}
    colors = {benchmark: "#B45232", "Equal weight": "#285E89", control: "#30766B"}
    styles = {benchmark: "-", "Equal weight": "-", control: "--"}
    seed = pd.Timestamp(calibration["evaluation_seed_date"])
    anchor = pd.DataFrame(1.0, index=[seed], columns=returns.columns)
    wealth = pd.concat([anchor, (1 + returns).cumprod()])
    period = f"{_date(returns.index[0])} to {_date(returns.index[-1])}"

    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.titleweight": "bold"}):
        for filename, title, subtitle, values, ylabel in [
            ("growth.png", "Growth over the same evaluation dates",
             f"{period} · $1,000 at the preceding close · monthly target resets",
             wealth * 1000, "Portfolio value"),
            ("drawdowns.png", "Declines from each portfolio's own prior peak",
             f"{period} · initial capital included in the peak calculation",
             pd.concat([anchor - 1, drawdowns(returns)]), "Drawdown"),
        ]:
            figure, axis = plt.subplots(figsize=(11.2, 5.4))
            for name in names:
                label = labels[name]
                if filename == "growth.png":
                    label += f" (${values[name].iloc[-1]:,.0f} at end)"
                axis.plot(values.index.to_pydatetime(), values[name].to_numpy(),
                          label=label, color=colors[name], linestyle=styles[name],
                          linewidth=1.7 if name == "Equal weight" else 1.4)
            axis.set_title(title, loc="left", pad=33)
            axis.text(0, 1.025, subtitle, transform=axis.transAxes, color="#555555", fontsize=10)
            axis.set(xlim=(seed, returns.index[-1]), ylabel=ylabel)
            axis.grid(axis="y", alpha=.18)
            if filename == "growth.png":
                axis.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
                axis.legend(loc="upper left", frameon=False, fontsize=10)
            else:
                axis.yaxis.set_major_formatter(PercentFormatter(1))
                axis.axhline(0, color="#777777", linewidth=.7)
                axis.legend(loc="upper center", bbox_to_anchor=(.5, -.13),
                            ncol=3, frameon=False, fontsize=10)
            figure.tight_layout()
            _save_figure(figure, output / filename)

        figure, axis = plt.subplots(figsize=(11.2, max(4.5, .9 * len(episodes) + 1.7)))
        axis.set_title(f"Losses across identical {benchmark} decline dates", loc="left", pad=33)
        axis.text(0, 1.025, f"{period} · every {benchmark} episode reaching a 10% drawdown",
                  transform=axis.transAxes, color="#555555", fontsize=10)
        if episodes.empty:
            axis.text(.5, .5, f"No {benchmark} decline reached the 10% threshold.",
                      transform=axis.transAxes, ha="center", va="center")
            axis.set_axis_off()
        else:
            positions = np.arange(len(episodes))
            for offset, name in enumerate(names):
                selected = comparisons.loc[comparisons.asset.eq(name)].set_index("episode")
                losses = selected.reindex(episodes.episode).peak_to_trough_return.to_numpy()
                bars = axis.barh(positions + (offset - 1) * .23, losses,
                                 height=.21, color=colors[name], label=labels[name])
                axis.bar_label(bars, labels=[f"{loss:.1%}" for loss in losses], padding=4, fontsize=9)
            axis.set_yticks(positions, [f"{_date(row.peak_date)}\nto {_date(row.trough_date)}"
                                        for row in episodes.itertuples()])
            axis.invert_yaxis()
            axis.set_xlabel(f"Return from the common {benchmark} peak close to its trough close")
            axis.xaxis.set_major_formatter(PercentFormatter(1))
            axis.axvline(0, color="#777777", linewidth=.7)
            axis.grid(axis="x", alpha=.18)
            axis.margins(x=.24)
            axis.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3,
                        frameon=False, fontsize=10)
        figure.tight_layout()
        _save_figure(figure, output / "episode_losses.png")


def _recovery_text(row) -> str:
    if row.recovery_status == "not_recovered":
        return f"not recovered by {_date(row.observation_end)}"
    if row.recovery_status == "at_or_above_baseline":
        return f"0 days; at/above baseline on {_date(row.recovery_date)}"
    return f"{int(row.recovery_trading_days)} trading days; {_date(row.recovery_date)}"


def _difference(value: float, positive: str, negative: str) -> str:
    if abs(value) < 1e-12:
        return "unchanged"
    return f"{abs(value) * 100:.2f} percentage points {positive if value > 0 else negative}"


def write_comparison_outputs(study: Study, result: ComparisonResult,
                             metrics: pd.DataFrame, episodes: pd.DataFrame,
                             comparisons: pd.DataFrame, manifest: dict, output: Path) -> None:
    """Export supplied calculations with explicit split and portfolio conventions."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    calibration = result.calibration
    benchmark, control = calibration["benchmark"], calibration["control_name"]
    returns = result.evaluation_returns
    for name, table, indexed in [
        ("evaluation_returns", returns, True), ("performance", metrics, True),
        ("stock_exposure", result.exposure, False), ("episodes", episodes, False),
        ("episode_comparison", comparisons, False),
    ]:
        table.to_csv(output / f"{name}.csv", index=indexed, float_format="%.12g",
                     date_format="%Y-%m-%d", na_rep="NA")
    (output / "calibration.json").write_text(
        json.dumps(calibration, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _figures(result, episodes, comparisons, output)

    run = {
        "study": study.settings(), "calibration": calibration,
        "snapshot_files": manifest["files"],
        "sources": manifest.get("sources", {}),
        "software_versions": {"python": platform.python_version(), **{
            name: version(name) for name in ["financial-economics", "numpy", "pandas", "statsmodels", "matplotlib"]}},
        "portfolio": {
            "mix": "Calculate monthly equal-weight returns over the complete snapshot before slicing the evaluation.",
            "control": f"Set {benchmark} target weight to clipped training beta; cash target weight is one minus that weight.",
            "control_initialization": "Initialize at target weights before the first evaluation return; reset before each later month's first return.",
            "control_training_diagnostic": "Hypothetical training-period holdings initialized at the fitted target, then reset monthly; the target was known only after training.",
            "between_resets": "Hold units fixed; beginning-of-day weights drift with preceding asset returns.",
            "cash": "Kenneth French daily RF in decimal units, a conceptual cash-return proxy rather than a traded fund.",
            "trading_costs": 0, "taxes": 0,
        },
        "definitions": {
            "calibration": f"OLS with an intercept: mix return minus RF = intercept + beta * ({benchmark} return minus RF) + residual; training observations only.",
            "target_weight": "min(1, max(0, training beta)); target weights are frozen throughout evaluation.",
            "realized_exposure": f"Separate OLS slopes and daily intercepts against {benchmark} excess returns, reported for all three portfolios in both periods; descriptive estimates without significance tests.",
            "cagr": f"(1 + cumulative return)^({study.trading_days} / observations) - 1.",
            "annualized_volatility": f"Sample standard deviation of daily total returns times sqrt({study.trading_days}).",
            "sharpe": f"sqrt({study.trading_days}) times mean daily excess return divided by its sample standard deviation; undefined for constant excess returns.",
            "max_drawdown": "Minimum wealth / running peak wealth - 1, including initial capital in the running peak.",
            "episodes": f"Phase 2 rule: non-overlapping {benchmark} peak-to-first-recovery episodes reaching a 10% decline; an unfinished final episode ends at evaluation end.",
            "episode_calendar": f"Episodes are selected on evaluation-period {benchmark} wealth, seeded at the actual preceding close; all portfolios use identical dates.",
            "peak_to_trough_return": f"Each portfolio's compounded return between the common {benchmark} peak and trough closes; not independently dated maximum drawdown.",
            "window_min_return": f"Lowest return from common starting capital before or on the common {benchmark} trough, including the anchor.",
            "recovery": f"First close on or after the common {benchmark} trough at or above each portfolio's own wealth at the common {benchmark} peak; 1e-12 relative tolerance.",
            "recovery_clock": f"Trading observations and calendar days from the common {benchmark} trough; zero only when already at or above baseline at that trough.",
            "censoring": "Unobserved recovery date and duration are missing, not zero. Follow-up through evaluation end is saved separately.",
        },
        "limitations": [
            "Chronological historical evaluation: the full dataset and selected funds had already been inspected; this is not an untouched out-of-sample strategy test.",
            "Stock-sensitivity matching is approximate because holdings drift and realized exposures can change between periods.",
            f"A clipped training beta outside [0, 1] cannot be matched exactly by an unlevered long-only {benchmark}/cash allocation.",
            "Retrospectively selected surviving funds; benchmark episodes, troughs, and recoveries are dated with later observations.",
            "Zero trading costs and taxes; no optimized weights, cost model, causal attribution, or evidence of statistical superiority.",
        ],
    }
    (output / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = ["# Phase 3: a fairer diversification comparison", "",
             f"Evaluation: {calibration['evaluation_start']} to {calibration['evaluation_end']}, "
             f"{calibration['evaluation_observations']:,} daily observations. "
             f"Compare the {len(study.assets)}-fund mix with a {benchmark}/cash control and {benchmark} alone.", "",
             "## Allocation chosen from earlier returns", "",
             f"Use {calibration['training_start']} to {calibration['training_end']} "
             f"({calibration['training_observations']:,} observations) to regress the mix's daily excess return "
             f"on {benchmark}'s daily excess return, including an intercept. "
             f"The estimated slope is {calibration['raw_beta']:.6f}; the daily intercept is "
             f"{calibration['intercept_daily']:.6%}. This uses {benchmark} excess returns, not French's Mkt-RF factor.", "",
             f"The frozen target is **{calibration['stock_weight']:.2%} {benchmark} and "
             f"{calibration['cash_weight']:.2%} cash**. "
             + ("The slope was clipped to the permitted 0%–100% stock range. "
                if calibration['weight_clipped'] else "The slope lies inside the permitted 0%–100% stock range. ")
             + "Cash earns French's daily RF return. It is a conceptual cash proxy, not a traded ETF.", "",
             "The control starts at target weights before the first evaluation return, resets monthly, "
             "and drifts between resets. Targets never use evaluation returns. The mix is calculated over its "
             "full history before the evaluation dates are sliced, preserving existing holdings. "
             "Trading costs and taxes are zero.", "", "## Evaluation results", "",
             "| Portfolio | $1,000 grows to | CAGR | Annual volatility | Sharpe | Maximum drawdown |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, row in metrics.iterrows():
        lines.append(f"| {name} | ${1000 * (1 + row.cumulative_return):,.2f} | {row.cagr:.2%} | "
                     f"{row.annualized_volatility:.2%} | {_number(row.sharpe)} | {row.max_drawdown:.2%} |")
    mix, comparison = metrics.loc["Equal weight"], metrics.loc[control]
    interpretation = (f"Relative to {control}, the mix's CAGR was "
                      f"{_difference(mix.cagr - comparison.cagr, 'higher', 'lower')}, "
                      f"its volatility was {_difference(mix.annualized_volatility - comparison.annualized_volatility, 'higher', 'lower')}, "
                      f"and its maximum drawdown was {_difference(mix.max_drawdown - comparison.max_drawdown, 'shallower', 'deeper')}. "
                      f"Its Sharpe was {_number(mix.sharpe)}, compared with {_number(comparison.sharpe)} for the control. "
                      "These are historical differences, without a test of statistical significance.")
    if (mix.cagr > comparison.cagr
            and mix.annualized_volatility > comparison.annualized_volatility
            and mix.max_drawdown < comparison.max_drawdown
            and mix.sharpe < comparison.sharpe):
        interpretation += (" The mix's additional growth came with greater volatility, a deeper maximum "
                           "drawdown, and a lower Sharpe. It did not improve all these measures relative "
                           "to the simpler control.")
    lines += ["", interpretation, "",
              f"CAGR annualizes compounded growth using {study.trading_days} trading observations per year. "
              "Volatility uses the sample standard deviation of daily total returns. Sharpe uses daily excess "
              "returns and their sample standard deviation; it is undefined for constant excess returns. "
              "Maximum drawdown uses each portfolio's own prior wealth peak, including initial capital.", "",
              "![Growth of the same initial investment](growth.png)", "",
              "![Each portfolio's own drawdown](drawdowns.png)", "",
              "## Did stock sensitivity stay matched?", "",
              f"| Period | Portfolio | Realized {benchmark} beta | Daily intercept | Observations |",
              "|---|---|---:|---:|---:|"]
    for row in result.exposure.itertuples():
        lines.append(f"| {row.period} | {row.asset} | {row.beta:.4f} | {row.intercept_daily:.6%} | {row.n_obs:,} |")
    lines += ["", f"Beta is the slope from an intercept regression of portfolio excess returns on {benchmark} excess "
              f"returns. It measures co-movement with {benchmark}, rather than total risk. The target matches the mix's "
              "estimated training beta; monthly drift makes the control's realized beta differ from its target, "
              "and the mix's exposure can change after training. Compare the realized evaluation betas before "
              "attributing a performance difference to diversification.", "",
              "The control's training-period returns are a hypothetical diagnostic calculated after its target "
              "has been fitted. They are not the returns of an allocation that was known at the start of training. "
              "The intercepts are descriptive; this report does not test alpha or claim superior skill.", "",
              "## The same downturns and recovery rule", "",
              f"Use Phase 2's 10% {benchmark} rule inside the evaluation period: track each {benchmark} peak until first recovery, "
              "retain the episode if its decline reaches 10%, and keep an unfinished final episode through the "
              f"sample end. Every portfolio is measured from the same {benchmark} peak close to the same {benchmark} trough close. "
              f"The initial capital anchor is the preceding observed close, {calibration['evaluation_seed_date']}.", ""]
    if episodes.empty:
        lines += [f"No {benchmark} episode reached the threshold in this evaluation.", ""]
    else:
        selected = {name: comparisons.loc[comparisons.asset.eq(name)].set_index("episode") for name in returns}
        lines += [f"{len(episodes)} qualifying episodes. These common-date returns differ from the "
                  "maximum drawdowns in the full-period table.", "",
                  f"| {benchmark} peak | {benchmark} trough | {benchmark} | Equal weight | {control} |",
                  "|---|---|---:|---:|---:|"]
        for row in episodes.itertuples():
            losses = " | ".join(f"{selected[name].loc[row.episode, 'peak_to_trough_return']:.2%}" for name in returns)
            lines.append(f"| {_date(row.peak_date)} | {_date(row.trough_date)} | {losses} |")
        loss_difference = (selected["Equal weight"].peak_to_trough_return
                           - selected[control].peak_to_trough_return)
        lines += ["", f"The mix's common-window return was lower than {control} in "
                  f"{int((loss_difference < 0).sum())} of {len(episodes)} episodes, higher in "
                  f"{int((loss_difference > 0).sum())}, and equal in {int((loss_difference == 0).sum())}.", "",
                  f"Recovery means regaining each portfolio's own value at the common {benchmark} peak. "
                  f"The search and clock start at the common {benchmark} trough; a portfolio already at or above "
                  "its starting value has zero elapsed days. A 1e-12 relative tolerance handles equality. "
                  "The CSV also records calendar days and the lowest value before the common trough.", "",
                  f"| {benchmark} trough | {benchmark}: recovery from trough | Equal weight: recovery from trough | {control}: recovery from trough |",
                  "|---|---|---|---|"]
        for row in episodes.itertuples():
            recoveries = " | ".join(_recovery_text(selected[name].loc[row.episode]) for name in returns)
            lines.append(f"| {_date(row.trough_date)} | {recoveries} |")
        paired = pd.DataFrame({"mix": selected["Equal weight"].recovery_trading_days,
                               "control": selected[control].recovery_trading_days}).dropna()
        if not paired.empty:
            difference = paired["mix"] - paired["control"]
            lines += ["", f"Across {len(paired)} episodes where both recoveries were observed, the mix recovered "
                      f"sooner than {control} in {int((difference < 0).sum())}, on the same date in "
                      f"{int((difference == 0).sum())}, and later in {int((difference > 0).sum())}. "
                      "This count excludes unfinished recoveries and does not estimate a future probability."]
        lines += ["", "Unobserved recovery dates and durations are NA, never zero. Follow-up ends at "
                  f"{calibration['evaluation_end']}; later recovery remains possible.", ""]
    lines += ["![Common-date downturn returns](episode_losses.png)", "",
              "## What this comparison can establish", "",
              "This is a chronological historical evaluation: the allocation uses earlier returns, but the "
              "full dataset and the selected surviving funds had already been inspected. It is not an untouched "
              "out-of-sample strategy test. Episode peaks, troughs, and recoveries are also dated retrospectively.", "",
              "The control makes the market-risk comparison more informative. It cannot fully isolate "
              "diversification: the mix's realized stock exposure can change, and its assets carry risks that "
              f"{benchmark} beta does not capture. A higher return alone does "
              "not establish a better risk-adjusted portfolio. No optimization, cost model, or significance "
              "claim is added in this phase.", "", "## Reproduce", "", "```sh", "uv run fecon compare", "```", "",
              "This command reads the retained, hash-checked snapshot offline. calibration.json saves the "
              "training fit and frozen targets; evaluation_returns.csv contains the three evaluated daily paths. "
              "performance.csv, stock_exposure.csv, episodes.csv, and episode_comparison.csv retain the numerical "
              "results. run.json records source hashes, versions, assumptions, and definitions. Generated data "
              "and figures stay local rather than being redistributed in Git.", "",
              "ETF source: " + price_source_description(run) + ". The cash proxy comes from "
              "[Kenneth French's daily factors](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
