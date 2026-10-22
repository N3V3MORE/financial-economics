"""Write the Phase 4 tables, figures, metadata, and summary."""

import json
from importlib.metadata import version
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from financial_economics.settings.study import Study
from financial_economics.evaluation.attribution import ExplainResult
from financial_economics.reporting.baseline import _save_figure

LABELS = {"sharpe": "Sharpe ratio", "cagr": "Yearly growth",
          "annualized_volatility": "Yearly fluctuation"}
LOSS, GAIN, INK, MUTED = "#B45232", "#30766B", "#252925", "#62665F"


def _format(measure: str, value: float) -> str:
    if measure == "sharpe":
        return f"{value:+.3f}"
    return f"{100 * value:+.2f} pp"


def _write_figures(result: ExplainResult, output: Path) -> None:
    benchmark = result.calibration["benchmark"]
    style = {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
             "figure.facecolor": "white", "axes.titleweight": "bold"}
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(11.2, 3.6))
        for axis, measure in zip(axes, ["sharpe", "cagr"], strict=True):
            rows = result.uncertainty.loc[result.uncertainty.measure.eq(measure)]
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
        figure.suptitle("Dots: observed differences. Lines: 95% ranges from rebuilt histories.",
                        x=.01, ha="left", fontsize=10, color="#555555")
        figure.tight_layout()
        _save_figure(figure, output / "uncertainty.png")

        windows = result.episode_windows
        figure, axes = plt.subplots(1, max(1, len(windows)), sharex=True, sharey=True,
                                    figsize=(11.2, 4.6), squeeze=False)
        if not windows:
            axes[0, 0].text(.5, .5, f"No {benchmark} decline reached 10%.",
                            transform=axes[0, 0].transAxes, ha="center", va="center")
            axes[0, 0].set_axis_off()
        # With no qualifying decline, one placeholder panel stays without a window.
        for axis, window in zip(axes[0], windows, strict=False):
            rows = result.contributions.loc[result.contributions.window.eq(window)]
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
        _save_figure(figure, output / "contributions.png")


def write_explain_outputs(study: Study, result: ExplainResult, bond_fund: str,
                          manifest: dict, output: Path) -> None:
    """Write the tables, figures, metadata, and summary."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [("uncertainty", result.uncertainty), ("contributions", result.contributions)]:
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.12g",
                     date_format="%Y-%m-%d", na_rep="NA")
    result.correlation.to_csv(output / "bond_correlation.csv", float_format="%.12g")
    _write_figures(result, output)

    first = result.uncertainty.iloc[0]
    settings = {"evaluation_start": result.calibration["requested_evaluation_start"],
                "resamples": int(first.resamples), "mean_block_length": float(first.mean_block_length),
                "seed": int(first.seed), "bond_fund": bond_fund}
    software_versions = {"python": platform.python_version()}
    for package in ("financial-economics", "numpy", "pandas", "matplotlib"):
        software_versions[package] = version(package)
    run = {
        "study": study.settings(), "calibration": result.calibration, "settings": settings,
        "snapshot_files": manifest["files"],
        "software_versions": software_versions,
        "portfolio": {"rule": "Phase 3 mix and SPY/cash control", "trading_costs": 0, "taxes": 0},
        "resampling": "Stationary bootstrap of evaluation days; the same days for every portfolio "
                      "and the risk-free rate. Ranges are 2.5th to 97.5th percentiles.",
        "contribution": "Mix value before each day x fund weight x fund return, summed over the "
                        "window, per $1 at the starting close.",
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")

    _write_summary(result, settings, output)


def _write_summary(result: ExplainResult, settings: dict, output: Path) -> None:
    """Write the summary using the results exported to the CSV tables."""
    calibration = result.calibration
    benchmark, control = calibration["benchmark"], calibration["control_name"]
    lines = ["# Phase 4: how sure are we, and why did it happen?", "",
             f"{calibration['evaluation_start']} to {calibration['evaluation_end']}; "
             f"{calibration['evaluation_observations']:,} daily observations, the same dates as Phase 3.", "",
             "## Could chance explain the differences?", "",
             f"The {calibration['evaluation_start']} to {calibration['evaluation_end']} evaluation "
             f"period is one history. We rebuilt it {settings['resamples']:,} times from runs of "
             f"real trading days, about {settings['mean_block_length']:g} days long on average, keeping "
             "every portfolio on the same days. The range is the middle 95% of the rebuilt differences.", "",
             "| Comparison | Measure | Observed | 95% range | Mix better in |",
             "|---|---|---:|---:|---:|"]
    for row in result.uncertainty.itertuples():
        lines.append(f"| Mix minus {row.reference} | {LABELS[row.measure]} | "
                     f"{_format(row.measure, row.estimate)} | {_format(row.measure, row.ci_low)} to "
                     f"{_format(row.measure, row.ci_high)} | {row.share_favoring_portfolio:.0%} |")
    sharpe = result.uncertainty.set_index(["reference", "measure"]).loc[(control, "sharpe")]
    if sharpe.ci_low < 0 < sharpe.ci_high:
        verdict = (f"The Sharpe range against {control} includes zero, so this evaluation period cannot "
                   "tell the two portfolios apart on risk-adjusted return.")
    else:
        verdict = f"The Sharpe range against {control} does not include zero."
    lines += ["", verdict + " For fluctuation, lower counts as better.", "",
              "![Uncertainty ranges](uncertainty.png)", "",
              "## Which funds drove the falls?", "",
              f"For each {benchmark} decline from Phase 3, this shows each fund's gain or loss per $1 in "
              f"the mix at the {benchmark} peak, through the trough. The amounts add up to the mix's loss.", ""]
    funds = list(dict.fromkeys(result.contributions.asset))
    lines += ["| Window | " + " | ".join(funds) + " | Mix |", "|---|" + "---:|" * (len(funds) + 1)]
    for window, rows in result.contributions.groupby("window", sort=False):
        values = rows.set_index("asset").contribution
        cells = " | ".join(f"{100 * values[fund]:+.1f}" for fund in funds)
        lines.append(f"| {window} | {cells} | {100 * values.sum():+.1f} |")
    lines += ["", "Values are percentage points. The last row covers the whole evaluation period.", "",
              "![Fund contributions in each decline](contributions.png)", "",
              "## Did the bond fund keep cushioning stocks?", "",
              f"| Year | {benchmark}/{settings['bond_fund']} daily correlation |", "|---|---:|"]
    lines += [f"| {year} | {value:+.2f} |" for year, value in result.correlation.items()]
    lines += ["", "A negative correlation means the bond fund tended to rise on days stocks fell.", "",
              "## Limits", "",
              "The rebuilt histories reuse the same days, and the funds and dates had already been "
              "examined, so this is not new evidence. Worst declines and recoveries depend on the order "
              "of days and are not resampled. Contributions show where the mix's gains and losses came "
              "from, not what a different mix would have earned. Trading costs and taxes are zero.", "",
              "## Reproduce", "", "```sh", "uv run fecon explain", "```", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
