"""Save Phase 4 uncertainty ranges and fund contributions with plain-language notes."""

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

from .config import Study

LABELS = {"sharpe": "Sharpe ratio", "cagr": "Compounded annual growth",
          "annualized_volatility": "Annual volatility"}
LOSS, GAIN, INK, MUTED = "#B45232", "#30766B", "#252925", "#62665F"


def _date(value) -> str:
    return str(pd.Timestamp(value).date())


def _value(measure: str, value: float, signed: bool = True) -> str:
    sign = "+" if signed else ""
    if measure == "sharpe":
        return format(value, f"{sign}.3f")
    return format(100 * value, f"{sign}.2f") + (" pp" if signed else "%")


def _save(figure, destination: Path) -> None:
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _figures(uncertainty: pd.DataFrame, contributions: pd.DataFrame,
             windows: list[str], benchmark: str, output: Path) -> None:
    style = {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
             "figure.facecolor": "white", "axes.titleweight": "bold"}
    with plt.rc_context(style):
        shown = uncertainty.loc[uncertainty.measure.isin(["sharpe", "cagr"])]
        figure, axes = plt.subplots(1, 2, figsize=(11.2, 3.6))
        for axis, (measure, rows) in zip(axes, shown.groupby("measure", sort=False)):
            positions = np.arange(len(rows))
            scale = 1 if measure == "sharpe" else 100
            axis.hlines(positions, scale * rows.ci_low, scale * rows.ci_high, color=MUTED, linewidth=3)
            axis.plot(scale * rows.estimate, positions, "o", color=INK)
            axis.axvline(0, color=LOSS, linewidth=1, linestyle="--")
            axis.set_yticks(positions, [f"Mix minus {name}" for name in rows.reference])
            axis.invert_yaxis()
            axis.set_title(f"{LABELS[measure]} difference", loc="left")
            axis.set_xlabel("Difference" if measure == "sharpe" else "Percentage points per year")
            axis.margins(y=.45)
            axis.grid(axis="x", alpha=.18)
        figure.suptitle("Dots: observed differences. Lines: 95% ranges from resampled histories.",
                        x=.01, ha="left", fontsize=10, color="#555555")
        figure.tight_layout()
        _save(figure, output / "uncertainty.png")

        figure, axes = plt.subplots(1, max(1, len(windows)), sharex=True, sharey=True,
                                    figsize=(11.2, 4.6), squeeze=False)
        if not windows:
            axes[0, 0].text(.5, .5, f"No {benchmark} decline reached the threshold.",
                            transform=axes[0, 0].transAxes, ha="center", va="center")
            axes[0, 0].set_axis_off()
        for axis, window in zip(axes[0], windows):
            rows = contributions.loc[contributions.window.eq(window)]
            values = rows.contribution.to_numpy()
            axis.barh(rows.asset, values, color=np.where(values < 0, LOSS, GAIN))
            axis.axvline(0, color=MUTED, linewidth=.7)
            axis.set_title(f"{window}\nmix {values.sum():+.1%}", loc="left", fontsize=10)
            axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            axis.grid(axis="x", alpha=.18)
        axes[0, 0].invert_yaxis()
        figure.suptitle(f"Each fund's gain or loss per $1 in the mix at the {benchmark} peak",
                        x=.01, ha="left", fontweight="bold")
        figure.tight_layout()
        _save(figure, output / "contributions.png")


def write_explain_outputs(study: Study, calibration: dict, uncertainty: pd.DataFrame,
                          contributions: pd.DataFrame, episode_windows: list[str],
                          correlation: pd.Series, bond_fund: str, manifest: dict,
                          output: Path) -> None:
    """Export Phase 4 calculations, settings, and an interpretation of their limits."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [("uncertainty", uncertainty), ("contributions", contributions)]:
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.12g",
                     date_format="%Y-%m-%d", na_rep="NA")
    correlation.to_csv(output / "bond_correlation.csv", float_format="%.12g", header=True)
    _figures(uncertainty, contributions, episode_windows, calibration["benchmark"], output)
    benchmark, control = calibration["benchmark"], calibration["control_name"]
    first = uncertainty.iloc[0]
    settings = {"evaluation_start": calibration["requested_evaluation_start"],
                "resamples": int(first.resamples), "mean_block_length": float(first.mean_block_length),
                "seed": int(first.seed), "bond_fund": bond_fund}
    run = {
        "study": study.settings(), "calibration": calibration, "settings": settings,
        "snapshot_files": manifest["files"],
        "software_versions": {"python": platform.python_version(), **{
            name: version(name) for name in ["financial-economics", "numpy", "pandas", "matplotlib"]}},
        "portfolio": {"mix": "Existing monthly equal-weight mix with drifting holdings, computed over the full history.",
                      "control": f"Phase 3 frozen {benchmark}/cash targets.",
                      "trading_costs": 0, "taxes": 0},
        "definitions": {
            "resampling": "Stationary bootstrap (Politis and Romano, 1994) of evaluation dates: random starting days, "
                          "geometric block lengths with the stated mean, wrapping at the sample end. The same dates "
                          "are drawn for every portfolio and the risk-free rate.",
            "interval": "2.5th to 97.5th percentiles of the resampled difference; a percentile interval, not a "
                        "studentized test.",
            "share_favoring_portfolio": "Share of resamples where the mix had the higher Sharpe or CAGR, or the "
                                        "lower volatility. Not a probability about future returns.",
            "contribution": "Sum over the window of portfolio value before each day times the fund's beginning-of-day "
                            "weight times its return, per $1 at the starting close; contributions add up to the mix's "
                            "compounded window return.",
            "windows": f"Phase 3 {benchmark} peak-to-trough closes, plus the whole evaluation period.",
            "bond_correlation": f"Correlation of daily {benchmark} and {settings['bond_fund']} returns, by calendar year.",
        },
        "limitations": [
            "Resampling assumes the evaluation period's behaviour is representative; it cannot add information "
            "the eight years did not contain, and the funds and dates had already been inspected.",
            "Maximum drawdown and recovery are path-dependent and are not resampled.",
            "Contributions describe where gains and losses arose; they do not show what a different mix would have done.",
            "Zero trading costs and taxes.",
        ],
    }
    (output / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    main = uncertainty.loc[uncertainty.reference.eq(control)].set_index("measure")
    lines = ["# Phase 4: how sure are we, and why did it happen?", "",
             f"Evaluation: {calibration['evaluation_start']} to {calibration['evaluation_end']}, "
             f"{calibration['evaluation_observations']:,} daily observations, the same dates as Phase 3.", "",
             "## How sure are we?", "",
             "Eight years is one history. To see how much the comparison could move by chance, "
             f"rebuild the period {settings['resamples']:,} times from month-like blocks of real trading days "
             f"(average {settings['mean_block_length']:g} days), keeping every portfolio on the same days. "
             "Each rebuilt history gives new differences; the middle 95% of them is the range below.", "",
             "| Comparison | Measure | Observed difference | 95% range | Mix better in |",
             "|---|---|---:|---:|---:|"]
    for row in uncertainty.itertuples():
        lines.append(f"| Mix minus {row.reference} | {LABELS[row.measure]} | {_value(row.measure, row.estimate)} | "
                     f"{_value(row.measure, row.ci_low)} to {_value(row.measure, row.ci_high)} | "
                     f"{row.share_favoring_portfolio:.0%} |")
    sharpe = main.loc["sharpe"]
    crosses = sharpe.ci_low < 0 < sharpe.ci_high
    lines += ["", (f"Against {control}, the Sharpe range runs from {_value('sharpe', sharpe.ci_low)} to "
                   f"{_value('sharpe', sharpe.ci_high)} and "
                   + ("includes zero. The observed gap is small relative to what chance alone could produce, "
                      "so these eight years cannot tell the two portfolios apart on risk-adjusted return."
                      if crosses else "excludes zero in this resampling."))
              + " A lower volatility counts as better in the last column.", "",
              "![Uncertainty ranges](uncertainty.png)", "",
              "## Which funds drove the falls?", "",
              f"For each Phase 3 {benchmark} decline, from the common peak close to the trough close, this "
              "measures every fund's own gain or loss per $1 held in the mix at the peak. The fund amounts add "
              "up to the mix's loss over the same dates.", ""]
    names = list(dict.fromkeys(contributions.asset))
    lines += ["| Window | " + " | ".join(names) + " | Mix |",
              "|---|" + "---:|" * (len(names) + 1)]
    for window, rows in contributions.groupby("window", sort=False):
        values = rows.set_index("asset").contribution
        lines.append(f"| {window} | " + " | ".join(f"{100 * values[name]:+.1f}" for name in names)
                     + f" | {100 * values.sum():+.1f} |")
    lines += ["", "Values are percentage points of the mix's value at the start of each window. "
              "The final row covers the whole evaluation period.", "",
              "![Fund contributions in each decline](contributions.png)", "",
              "## Did the bond fund keep hedging?", "",
              f"| Year | {benchmark}/{settings['bond_fund']} daily correlation |", "|---|---:|"]
    lines += [f"| {year} | {value:+.2f} |" for year, value in correlation.items()]
    lines += ["", "A negative correlation means the bond fund tended to rise on days stocks fell. "
              "Published research links positive stock-bond correlations to high inflation and tight monetary "
              "policy, the conditions of 2022.", "",
              "## What this can and cannot show", "",
              "Resampling only reuses the days this sample contains; it is not new evidence, and the funds and "
              "dates had already been inspected. Maximum drawdown and recovery depend on the order of days and "
              "are not resampled. Contributions describe where the mix's gains and losses came from, not what a "
              "different mix would have earned. Trading costs and taxes are zero.", "",
              "## Reproduce", "", "```sh", "uv run fecon explain", "```", "",
              "This reads the retained, hash-checked snapshot offline. uncertainty.csv, contributions.csv, and "
              "bond_correlation.csv hold the numbers; run.json records settings, source hashes, and definitions.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
