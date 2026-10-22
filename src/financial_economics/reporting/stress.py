"""Export same-date downturn comparisons and changing correlations."""

import json
from importlib.metadata import version
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from financial_economics.settings.study import Study
from financial_economics.datasets.public import price_source_description
from financial_economics.reporting.baseline import _save_figure
from financial_economics.portfolio.declines import _wealth


COLORS = {"benchmark": "#B45232", "Equal weight": "#285E89",
          "GLD": "#237a71", "TLT": "#7b4b94"}


def _date(value) -> str:
    return "not observed" if pd.isna(value) else str(pd.Timestamp(value).date())



def _episode_figures(returns: pd.DataFrame, episodes: pd.DataFrame,
                     comparisons: pd.DataFrame, benchmark: str,
                     output: Path, seed_date) -> None:
    if episodes.empty:
        for name in ["episode_losses.png", "recovery_paths.png"]:
            (output / name).unlink(missing_ok=True)
        return

    names = [name for name in [benchmark, "Equal weight"] if name in returns]
    figure, axis = plt.subplots(figsize=(10, max(4, len(episodes) * .8 + 1.8)))
    positions = np.arange(len(episodes))
    width = .7 / len(names)
    for i, name in enumerate(names):
        selected = comparisons.loc[comparisons.asset.eq(name)].set_index("episode")
        losses = selected.reindex(episodes.episode).peak_to_trough_return.to_numpy()
        bars = axis.barh(positions + (i - (len(names) - 1) / 2) * width, losses,
                         height=width, color=COLORS.get(name, COLORS["benchmark"]), label=name)
        axis.bar_label(bars, labels=[f"{loss:.1%}" for loss in losses], padding=4, fontsize=10)
    labels = [f"{_date(row.peak_date)} to {_date(row.trough_date)}"
              for row in episodes.itertuples()]
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set(title="Returns across the same benchmark decline",
             xlabel="Return from the benchmark peak to its trough")
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.axvline(0, color="#666666", linewidth=.8)
    axis.grid(axis="x", alpha=.2)
    axis.margins(x=.25)
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2, frameon=False)
    _save_figure(figure, output / "episode_losses.png")

    examples = episodes.loc[pd.to_datetime(episodes.trough_date).dt.year.isin([2020, 2022])]
    if examples.empty:
        examples = episodes.nsmallest(2, "benchmark_drawdown").sort_values("peak_date")
    examples = examples.head(2)
    figure, axes = plt.subplots(len(examples), 1, figsize=(10, 4 * len(examples)), squeeze=False)
    wealth = _wealth(returns, seed_date)
    for axis, episode in zip(axes.flat, examples.itertuples(), strict=True):
        rows = comparisons.loc[comparisons.episode.eq(episode.episode)
                               & comparisons.asset.isin(names)]
        end = (returns.index[-1] if rows.recovery_status.eq("not_recovered").any()
               else pd.to_datetime(rows.recovery_date).max())
        start = pd.Timestamp(episode.peak_date)
        path = wealth.loc[start:end, names].div(wealth.loc[start, names]) - 1
        for i, name in enumerate(names):
            axis.plot(path.index.to_pydatetime(), path[name].to_numpy(),
                      color=COLORS.get(name, COLORS["benchmark"]),
                      linestyle=["-", "--"][i], label=name)
        axis.axhline(0, color="#666666", linewidth=.8)
        axis.axvline(pd.Timestamp(episode.trough_date), color="#777777", linestyle=":",
                     label=f"{benchmark} trough")
        axis.set(title=f"Common starting close: {_date(start)}", ylabel="Return since starting close")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=.2)
        axis.legend(loc="best")
    figure.tight_layout(h_pad=2)
    _save_figure(figure, output / "recovery_paths.png")


def _correlation_figure(correlations: pd.DataFrame, benchmark: str,
                        output: Path, rolling_window: int) -> None:
    preferred = [name for name in ["GLD", "TLT", "Equal weight"] if name in correlations]
    names = preferred or list(correlations.columns[:3])
    figure, axis = plt.subplots(figsize=(10, 5))
    for name in names:
        axis.plot(correlations.index.to_pydatetime(), correlations[name].to_numpy(),
                  linewidth=1.1, color=COLORS.get(name), label=name)
    axis.axhline(0, color="#666666", linewidth=.8)
    axis.set(title=f"{rolling_window}-observation rolling correlation with {benchmark}",
             ylabel="Correlation of daily returns", ylim=(-1.05, 1.05))
    axis.grid(axis="y", alpha=.2)
    if names:
        axis.legend(loc="lower left", ncol=min(3, len(names)))
    _save_figure(figure, output / "rolling_correlations.png")


def _recovery_text(row) -> str:
    if row.recovery_status == "not_recovered":
        return f"not recovered by {_date(row.observation_end)}"
    if row.recovery_status == "at_or_above_baseline":
        return "already at/above starting value at trough (0 days)"
    return f"{int(row.recovery_trading_days)} trading days ({_date(row.recovery_date)})"


def write_stress_outputs(study: Study, returns: pd.DataFrame, episodes: pd.DataFrame,
                         comparisons: pd.DataFrame, correlations: pd.DataFrame,
                         manifest: dict, output: Path, *, drawdown_threshold: float,
                         rolling_window: int, seed_date) -> None:
    """Save the supplied calculations without selecting or recomputing episodes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, table, include_index in [("episodes", episodes, False),
                                       ("episode_comparison", comparisons, False),
                                       ("rolling_correlations", correlations, True)]:
        table.to_csv(output / f"{name}.csv", index=include_index, float_format="%.12g",
                     date_format="%Y-%m-%d", na_rep="NA")
    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"}):
        _episode_figures(returns, episodes, comparisons, study.benchmark, output, seed_date)
        _correlation_figure(correlations, study.benchmark, output, rolling_window)

    run = {
        "study": study.settings(), "observations": len(returns),
        "actual_start": _date(returns.index[0]), "actual_end": _date(returns.index[-1]),
        "seed_date": _date(seed_date), "snapshot_files": manifest["files"],
        "sources": manifest.get("sources", {}),
        "drawdown_threshold": drawdown_threshold, "rolling_window": rolling_window,
        "episodes": len(episodes),
        "software_versions": {"python": platform.python_version(), **{
            name: version(name) for name in ["financial-economics", "numpy", "pandas", "matplotlib"]}},
        "portfolio": {"rule": "full-sample monthly equal weight; drift within month",
                      "trading_costs": 0, "taxes": 0,
                      "episode_rebalancing": "none beyond the existing monthly rule"},
        "definitions": {
            "episodes": "Non-overlapping benchmark peak-to-first-recovery episodes reaching the drawdown threshold. "
                        "An unrecovered final episode ends at the sample end.",
            "common_window": "Benchmark peak close to benchmark trough close, inclusive of the starting capital anchor.",
            "peak_to_trough_return": "Benchmark and equal-weight portfolio compounded returns across the common benchmark peak-to-trough dates.",
            "window_min_return": "Lowest return from common starting capital before or on the benchmark trough, including the anchor.",
            "recovery": "First close on or after the benchmark trough at or above each asset's wealth at the common benchmark peak.",
            "recovery_clock": "Trading observations and calendar days elapsed from the common benchmark trough. "
                              "An asset already at or above its starting value at the trough has zero elapsed days.",
            "numerical_tolerance": "Recovery equality permits a 1e-12 relative tolerance in wealth levels to avoid rounding-only delays.",
            "censoring": "Missing recovery date and duration mean not recovered by sample end, not zero days. "
                         "Observed follow-up after the trough is recorded separately.",
            "rolling_correlation": f"Pearson correlation of {rolling_window} trailing daily returns with "
                                   f"{study.benchmark}, including the current return. Earlier incomplete windows are missing.",
        },
        "limitations": ["Retrospectively selected surviving instruments and benchmark-defined episodes.",
                        "No causal inference, optimization, out-of-sample test, or prediction.",
                        "The comparison does not isolate diversification from differences in market exposure.",
                        "Zero trading costs and taxes; episode selection uses future recovery and trough information.",
                        "Overlapping rolling windows are descriptive, not independent statistical evidence."],
    }
    (output / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = ["# Phase 2: diversification in market declines", "",
             f"{run['actual_start']} to {run['actual_end']}; {len(returns):,} daily observations.", "",
             f"Select every non-overlapping {study.benchmark} decline reaching {drawdown_threshold * 100:g}%. "
             "Compare the benchmark and equal-weight mix over the same benchmark peak-to-trough dates. "
             "The existing monthly equal-weight portfolio continues through each episode; "
             "rebasing a chart does not reset its weights.", "", "## Same dates, different losses", ""]
    if episodes.empty:
        lines += ["No benchmark episode reached the selected threshold. Episode figures are omitted; "
                  "the correlation analysis still uses the full return sample.", ""]
    else:
        lines += [f"{len(episodes)} qualifying episodes. Negative values below are losses from a common starting close, "
                  "not each portfolio's own maximum drawdown.", "",
                  f"| {study.benchmark} peak | {study.benchmark} trough | {study.benchmark} return | Equal-weight return |",
                  "|---|---|---:|---:|"]
        benchmark_rows = comparisons.loc[comparisons.asset.eq(study.benchmark)].set_index("episode")
        mix_rows = comparisons.loc[comparisons.asset.eq("Equal weight")].set_index("episode")
        for row in episodes.itertuples():
            benchmark_return = benchmark_rows.loc[row.episode, "peak_to_trough_return"]
            mix = (f"{mix_rows.loc[row.episode, 'peak_to_trough_return']:.2%}"
                   if row.episode in mix_rows.index else "not included")
            lines.append(f"| {_date(row.peak_date)} | {_date(row.trough_date)} | {benchmark_return:.2%} | {mix} |")
        common = benchmark_rows.index.intersection(mix_rows.index)
        if len(common):
            difference = (mix_rows.loc[common, "peak_to_trough_return"]
                          - benchmark_rows.loc[common, "peak_to_trough_return"])
            lines += ["", f"The mix had a higher return over {int((difference > 0).sum())} of {len(common)} benchmark "
                      f"peak-to-trough windows; it had a lower return over {int((difference < 0).sum())}. "
                      "This count describes these episodes and does not estimate a future probability."]
        lines += ["", "![Returns over matching decline dates](episode_losses.png)", "",
                  "The full CSV also reports each portfolio's lowest value before the benchmark trough. "
                  "A portfolio can finish above its starting value after having fallen earlier in that window.", "",
                  "## Recovery after the benchmark trough", "",
                  "Recovery means reaching the portfolio's own value at the common benchmark peak. "
                  "The clock starts at the benchmark trough; a portfolio already at or above that value "
                  "has zero elapsed days. This is not the duration of an independently dated drawdown. "
                  "A 1e-12 relative tolerance avoids recovery delays caused only by numerical rounding.", "",
                  f"| {study.benchmark} trough | {study.benchmark} recovery from trough | Equal-weight recovery from trough |",
                  "|---|---|---|"]
        for row in episodes.itertuples():
            benchmark_recovery = _recovery_text(benchmark_rows.loc[row.episode])
            mix_recovery = (_recovery_text(mix_rows.loc[row.episode])
                            if row.episode in mix_rows.index else "not included")
            lines.append(f"| {_date(row.trough_date)} | {benchmark_recovery} | {mix_recovery} |")
        paired = pd.DataFrame({
            "benchmark": benchmark_rows.recovery_trading_days,
            "mix": mix_rows.recovery_trading_days,
        }).dropna()
        if len(paired):
            elapsed_difference = paired["mix"] - paired["benchmark"]
            lines += ["", f"Among {len(paired)} episodes where both recoveries were observed, the mix "
                      f"recovered sooner in {int((elapsed_difference < 0).sum())}, on the same date in "
                      f"{int((elapsed_difference == 0).sum())}, and later in "
                      f"{int((elapsed_difference > 0).sum())}. This excludes unfinished recoveries."]
        lines += ["", "An unobserved recovery is missing in the CSV, never zero. Follow-up ends on "
                  f"{run['actual_end']}; a portfolio may recover later. The CSV records the observed follow-up "
                  "after the trough so unfinished recoveries can be identified.", "",
                  "![Selected recovery paths from common starting values](recovery_paths.png)", "",
                  "Charts show 2020 and 2022 episodes when present, otherwise up to two deepest benchmark episodes. "
                  "Each chart continues until both displayed portfolios recover or the sample ends.", ""]
    lines += ["## Correlation changes", "",
              f"Each point uses the latest {rolling_window} daily returns, including that date, to measure "
              f"co-movement with {study.benchmark}. The first {rolling_window - 1} values are missing. "
              "A negative value means opposite daily movements on average within that window; "
              "it does not guarantee gains during a decline.", "",
              "![Trailing correlations with the benchmark](rolling_correlations.png)", "",
              "The figure highlights gold, long-duration bonds, and the equal-weight mix when available. "
              "The CSV contains every non-benchmark asset. Adjacent windows overlap, so the plotted points "
              f"are not independent observations. The mix contains {study.benchmark} itself, "
              "which contributes to their correlation.", "", "## Interpretation and limits", "",
              "- Episode dates come from the benchmark alone, which prevents choosing different dates to favor each asset.",
              "- The drawdown threshold and full episodes are selected retrospectively. A trough or recovery is known only afterward.",
              "- A calendar decline can contain several sell-offs before recovery; these remain one episode.",
              "- The selected ETF universe contains surviving funds. Results apply to these instruments and dates.",
              "- The comparison does not isolate diversification from differences in market exposure.",
              "- Trading costs and taxes remain zero. TLT carries interest-rate risk and is not the risk-free asset.",
              "- This phase describes diversification in past declines. It does not optimize weights or test a trading rule on unseen data.",
              "- Smaller losses alone do not establish better overall performance; consider the full-period return and risk comparison from Phase 1 as well.",
              "", "## Reproduce", "", "```sh",
              f"uv run fecon stress --drawdown-threshold {drawdown_threshold:g} --rolling-window {rolling_window}",
              "```", "", "The command uses the retained, hash-checked Phase 1 snapshot offline. "
              "run.json records the settings, source hashes, software versions, and exact definitions. "
              "ETF source: " + price_source_description(run) + "; the shared trading calendar comes from "
              "[Kenneth French's daily factors](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). "
              "Generated tables and figures stay local and are not redistributed in Git.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
