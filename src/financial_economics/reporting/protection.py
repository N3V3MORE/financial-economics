"""Inspectable CSV evidence, three static research figures, and complete run metadata."""
from importlib.metadata import version
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from financial_economics.provenance import file_digest, git_state, package_code_digests, utc_now


def _save(fig, output, name):
    fig.savefig(output / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _figures(output, study, settings, source):
    """Read the saved figure tables, so plotted values have an exact inspectable source."""
    period = f"{study.start[:4]}–{study.end[:4]}"
    note = f"{period} • {source} • French daily RF compounded monthly"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#222222", "text.color": "#222222"})
    protection = pd.read_csv(output / "figure_protection.csv")
    fig, ax = plt.subplots(figsize=(10, 7))
    positions = np.arange(len(protection))
    if "ci_low" in protection:
        ax.hlines(positions, protection.ci_low, protection.ci_high, color="#31688e", linewidth=2)
    ax.scatter(protection.mean_minus_cash, positions, color="#31688e", s=45, zorder=3)
    ax.axvline(0, color="#555555", linewidth=1)
    labels = [f"{r.fund}  (n={r.months})" for r in protection.itertuples()]
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Average fund return minus cash in negative-SPY months")
    ax.set_title("Which funds helped relative to cash?", loc="left", fontsize=16, pad=18)
    ax.grid(axis="x", color="#e6e6e6")
    fig.text(.02, .02, note + "\n" + ("Bars: pointwise 95% bootstrap intervals, mean block 6 months; Holm tests are in primary_tests.csv."
             if settings.primary_inference else "Longer sample is descriptive; it is not part of the 13-test primary family."), fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, 1))
    _save(fig, output, "protection.png")

    tradeoff = pd.read_csv(output / "figure_tradeoffs.csv")
    fig, axes = plt.subplots(1, len(settings.allocations), figsize=(15, 8), sharex=True, sharey=True, squeeze=False)
    funds = tradeoff.fund.drop_duplicates().tolist()
    positions = np.arange(len(funds))
    for ax, allocation in zip(axes[0], settings.allocations, strict=True):
        subset = tradeoff.loc[np.isclose(tradeoff.candidate_allocation, allocation)].set_index("fund").loc[funds]
        ax.axvline(0, color="#999999", linewidth=1)
        ax.scatter(subset.cagr_difference, positions-.12, color="#31688e", marker="o", s=42, label="CAGR")
        ax.scatter(subset.volatility_difference, positions+.12, color="#a66f00", marker="s", s=36, label="Volatility")
        ax.set_yticks(positions, funds)
        ax.set_title(f"{allocation:.0%} addition / {1-allocation:.0%} SPY")
        ax.set_xlabel("Change (percentage points)")
        ax.xaxis.set_major_formatter(PercentFormatter(1, symbol=""))
        ax.grid(color="#eeeeee")
        ax.set_axisbelow(True)
        ax.margins(x=.12, y=.06)
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Growth and risk relative to the same-SPY-weight cash portfolio", x=.03, ha="left", fontsize=16)
    cost = tradeoff.cost_bps.iloc[0]
    fig.text(.03, .02, note + f"\n{cost:g} bps per ETF dollar bought/sold; entry and monthly rebalancing included; no exit or taxes."
             "\nPositive CAGR difference means more growth; positive volatility difference means more variability.", fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .93))
    _save(fig, output, "tradeoffs.png")

    declines = pd.read_csv(output / "figure_declines.csv")
    funds = [fund for fund in study.tickers if fund != study.benchmark]
    pivot = declines.pivot(index="fund", columns="peak_date", values="return_minus_cash").reindex(funds)
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.05), max(5, len(funds)*.48 + 2)))
    if len(pivot.columns):
        limit = max(.01, np.abs(pivot.to_numpy()).max())
        image = ax.imshow(pivot, cmap="PuOr", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), [x[:7] for x in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(funds)), funds)
        for y in range(len(funds)):
            for x in range(len(pivot.columns)):
                value = pivot.iloc[y, x]
                ax.text(x, y, f"{100*value:+.1f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(value) > .6*limit else "#222222")
        bar = fig.colorbar(image, ax=ax, pad=.02)
        bar.ax.yaxis.set_major_formatter(PercentFormatter(1, symbol=""))
        bar.set_label("Fund return minus cash (pp)")
    else:
        ax.text(.5, .5, "No SPY drawdowns reached 10%", transform=ax.transAxes, ha="center")
    ax.set_title("Funds versus cash during SPY drawdowns of at least 10%", loc="left", fontsize=15, pad=16)
    ax.set_xlabel("SPY peak month; each window ends at its SPY trough")
    fig.text(.02, .015, note + "\nBuy-and-hold fund return minus cash over each fixed peak-to-trough window; not a timing strategy or a monthly mean.", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, 1))
    _save(fig, output, "declines.png")


def write_protection(result, study, settings, manifest, directory, config_path):
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(f"Protection output already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    tables = dict(result.tables)
    full = f"{study.start[:4]}:{study.end[:4]}"
    conditions = tables["fund_conditions"]
    figures = conditions.loc[conditions.window.eq(full) & conditions.condition.eq("negative_spy")].copy()
    if settings.primary_inference:
        ci = tables["bootstrap_intervals"].query("mean_block_months == 6")
        if ci.empty:
            raise ValueError("The primary figure requires the prespecified six-month bootstrap.")
        figures = figures.merge(ci[["fund", "ci_low", "ci_high"]], on="fund", validate="one_to_one")
    tables["figure_protection"] = figures
    cost = 10 if 10 in settings.costs_bps else settings.costs_bps[0]
    perf = tables["performance"].query("window == @full and cost_bps == @cost").set_index("portfolio")
    definitions = tables["portfolio_definitions"].query("kind == 'candidate'")
    trades = []
    for row in definitions.itertuples():
        control = f"{1-row.candidate_allocation:.0%} {study.benchmark} + {row.candidate_allocation:.0%} cash"
        trades.append({"fund": row.fund, "candidate_allocation": row.candidate_allocation, "cost_bps": cost,
                       "cagr_difference": perf.at[row.portfolio, "cagr"] - perf.at[control, "cagr"],
                       "volatility_difference": perf.at[row.portfolio, "annualized_volatility"] - perf.at[control, "annualized_volatility"],
                       "drawdown_difference": perf.at[row.portfolio, "max_drawdown"] - perf.at[control, "max_drawdown"]})
    tables["figure_tradeoffs"] = pd.DataFrame(trades)
    episode = tables["fund_episode_comparisons"]
    cash = episode.loc[episode.asset.eq("RF"), ["episode", "peak_to_trough_return"]].rename(columns={"peak_to_trough_return": "cash_return"})
    decline = episode.loc[~episode.asset.isin([study.benchmark, "RF"])].merge(cash, on="episode", validate="many_to_one")
    decline["return_minus_cash"] = decline.peak_to_trough_return - decline.cash_return
    tables["figure_declines"] = decline.rename(columns={"asset": "fund"})
    with TemporaryDirectory(prefix=f".{directory.name}-", dir=directory.parent) as temporary:
        output = Path(temporary)
        for name, table in tables.items():
            indexed = isinstance(table.index, pd.DatetimeIndex) or table.index.name == "portfolio"
            table.to_csv(output / f"{name}.csv", index=indexed, float_format="%.17g", na_rep="NA")
        provider = manifest.get("sources", {}).get("prices", {}).get("provider", "Unspecified provider")
        _figures(output, study, settings, provider)
        commit, dirty = git_state()
        run = {"created_at_utc": utc_now(), "study": study.settings(),
               "protection": settings.settings(), "source_manifest": manifest, **result.metadata,
               "git_commit": commit, "git_dirty": dirty,
               "code_sha256": package_code_digests(),
               "config_sha256": file_digest(Path(config_path)),
               "versions": {"python": platform.python_version(), **{name: version(name) for name in
                            ["numpy", "pandas", "statsmodels", "matplotlib", "financial-economics"]}},
               "inference": {"primary_family_size": len(study.tickers)-1 if settings.primary_inference else 0,
                             "HAC": "Bartlett; small-sample correction; Student t; two-sided; Holm",
                             "bootstrap": "paired whole monthly sequence; down mask reidentified per draw; pointwise percentile intervals"},
               "cash": "French daily RF compounded; Ibbotson through May 2024, ICE BofA from June 2024",
               "fees": "scenario bps per ETF purchase + sale notional; entry included; exit/taxes excluded",
               "undefined_values": "NA, never zero-imputed",
               "files": {p.name: file_digest(p) for p in sorted(output.iterdir())}}
        (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        if directory.exists():
            raise FileExistsError(f"Protection output already exists: {directory}")
        output.rename(directory)
